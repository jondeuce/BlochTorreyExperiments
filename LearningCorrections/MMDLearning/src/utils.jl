# ---------------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------------- #

# Saving, formatting
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
savebson(filename, data::Dict) = @elapsed BSON.bson(filename, data)
# gitdir() = realpath(DrWatson.projectdir(".."))

# Mini batching
make_minibatch(features, labels, idxs) = (features[.., idxs], labels[.., idxs])
function training_batches(features, labels, minibatchsize; overtrain = false)
    @assert batchsize(features) == batchsize(labels)
    batches = Iterators.partition(1:batchsize(features), minibatchsize)
    if overtrain
        train_set = [make_minibatch(features, labels, batches[1])]
    else
        train_set = [make_minibatch(features, labels, b) for b in batches]
    end
end
testing_batches(features, labels) = make_minibatch(features, labels, :)
features(batch) = batch[1]
labels(batch) = batch[2]

# Lazy mini batching
struct LazyMiniBatches{xType,yType,X,Y}
    len::Int
    x_sampler::X
    y_sampler::Y
    function LazyMiniBatches(len::Int, x_sampler::X, y_sampler::Y) where {X,Y}
        x = x_sampler()
        y = y_sampler(x)
        new{typeof(x), typeof(y), X, Y}(len, x_sampler, y_sampler)
    end
end
function Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{<:LazyMiniBatches{xType,yType}}) where {xType,yType}
    x = d[].x_sampler()  :: xType
    y = d[].y_sampler(x) :: yType
    return (x,y)
end
Base.length(S::LazyMiniBatches) = S.len
Base.eltype(::Type{<:LazyMiniBatches{xType,yType}}) where {xType,yType} = Tuple{xType,yType}
Base.iterate(S::LazyMiniBatches, state = 1) = state > S.len ? nothing : (rand(S), state+1)
Base.iterate(rS::Iterators.Reverse{LazyMiniBatches}, state = rS.itr.len) = state < 1 ? nothing : (rand(rS.itr), state-1)
# Base.firstindex(::LazyMiniBatches) = 1
# Base.lastindex(S::LazyMiniBatches) = S.len
# Base.getindex(S::LazyMiniBatches, i::Number) = rand(S)
# Base.getindex(S::LazyMiniBatches, I) = [rand(S) for i in I]

linearsampler(a,b) = a + rand() * (b - a)
rangesampler(a,b,s=1) = rand(a:s:b)
log10sampler(a,b) = 10^linearsampler(log10(a), log10(b))
acossampler(a,b) = acosd(linearsampler(cosd(b), cosd(a)))

function param_summary(model, train_set, test_set)
    test_dofs = length(test_set[2])
    train_dofs = sum(batch -> length(batch[2]), train_set)
    param_dofs = sum(length, Flux.params(model))
    test_param_density = param_dofs / test_dofs
    train_param_density = param_dofs / train_dofs
    @info @sprintf(" Testing parameter density: %d/%d (%.2f %%)", param_dofs, test_dofs, 100 * test_param_density)
    @info @sprintf("Training parameter density: %d/%d (%.2f %%)", param_dofs, train_dofs, 100 * train_param_density)
end

# Losses
function make_losses(model, losstype, weights = nothing)
    l1 = weights == nothing ? @λ((x,y) -> sum(abs, model(x) .- y))  : @λ((x,y) -> sum(abs, weights .* (model(x) .- y)))
    l2 = weights == nothing ? @λ((x,y) -> sum(abs2, model(x) .- y)) : @λ((x,y) -> sum(abs2, weights .* (model(x) .- y)))
    crossent = @λ((x,y) -> Flux.crossentropy(model(x), y))
    mae = @λ((x,y) -> l1(x,y) * 1 // length(y))
    mse = @λ((x,y) -> l2(x,y) * 1 // length(y))
    rmse = @λ((x,y) -> sqrt(mse(x,y)))
    mincrossent = @λ (y) -> -sum(y .* log.(y))

    lossdict = Dict("l1" => l1, "l2" => l2, "crossent" => crossent, "mae" => mae, "mse" => mse, "rmse" => rmse, "mincrossent" => mincrossent)
    if losstype ∉ keys(lossdict)
        @warn "Unknown loss $(losstype); defaulting to mse"
        losstype = "mse"
    end

    loss = lossdict[losstype]
    accloss = losstype == "crossent" ? @λ((x,y) -> loss(x,y) - mincrossent(y)) : rmse # default
    accuracy = @λ((x,y) -> 100 - 100 * accloss(x,y))
    labelacc = @λ((x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ (maximum(abs.(y); dims = 2) .- minimum(abs.(y); dims = 2))))
    # labelacc = @λ((x,y) -> 100 .* vec(mean(abs.((model(x) .- y) ./ y); dims = 2)))
    # labelacc = @λ((x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ maximum(abs.(y); dims = 2)))
    
    return @ntuple(loss, accuracy, labelacc)
end

# Optimizer
lr(opt) = opt.eta
lr!(opt, α) = (opt.eta = α; opt.eta)
lr(opt::Flux.Optimiser) = lr(opt[1])
lr!(opt::Flux.Optimiser, α) = lr!(opt[1], α)

fixedlr(e, opt) = lr(opt) # Fixed learning rate
geometriclr(e, opt; rate = 100, factor = √10) = mod(e, rate) == 0 ? lr(opt) / factor : lr(opt) # Drop lr every `rate` epochs
clamplr(e, opt; lower = -Inf, upper = Inf) = clamp(lr(opt), lower, upper) # Clamp learning rate in [lower, upper]
findlr(e, opt; epochs = 100, minlr = 1e-6, maxlr = 0.5) = e <= epochs ? logspace(1, epochs, minlr, maxlr)(e) : maxlr # Learning rate finder
cyclelr(e, opt; lrstart = 1e-5, lrmin = 1e-6, lrmax = 1e-2, lrwidth = 50, lrtail = 5) = # Learning rate cycling
                     e <=   lrwidth          ? linspace(        1,            lrwidth, lrstart,   lrmax)(e) :
      lrwidth + 1 <= e <= 2*lrwidth          ? linspace(  lrwidth,          2*lrwidth,   lrmax, lrstart)(e) :
    2*lrwidth + 1 <= e <= 2*lrwidth + lrtail ? linspace(2*lrwidth, 2*lrwidth + lrtail, lrstart,   lrmin)(e) :
    lrmin

function make_variancelr(state, opt; rate = 250, factor = √10, stdthresh = Inf)
    last_lr_update = 0
    function lrfun(e)
        if isempty(state)
            return lr(opt)
        elseif e > 2 * rate && e - last_lr_update > rate
            min_epoch = max(1, min(state[end, :epoch] - rate, rate))
            df = dropmissing(state[(state.dataset .== :test) .& (min_epoch .<= state.epoch), [:epoch, :loss]])
            if !isempty(df) && std(df[!, :loss]) > stdthresh
                last_lr_update = e
                return lr(opt) / √10
            else
                return lr(opt)
            end
        else
            return lr(opt)
        end
    end
end

"""
    batchsize(x::AbstractArray)

Returns the length of the last dimension of the data `x`.
`x` must have dimension of at least 2, otherwise an error is thrown.
"""
# batchsize(x::AbstractVector) = 1
batchsize(x::AbstractVector) = error("x must have dimension of at least 2, but x is a $(typeof(x))")
# batchsize(x::AbstractVecOrMat) = error("x must have dimension of at least 3, but x is a $(typeof(x))")
batchsize(x::AbstractArray{T,N}) where {T,N} = size(x, N)

"""
    channelsize(x::AbstractArray)

Returns the length of the second-last dimension of the data `x`.
`x` must have dimension of at least 3, otherwise an error is thrown.
"""
# Old docstring:
# Returns the length of the second-last dimension of the data `x`, unless:
#     `x` is a `Matrix`, in which case 1 is returned.
#     `x` is a `Vector`, in which case an error is thrown.
# channelsize(x::AbstractVector) = error("Channel size undefined for AbstractVector's")
# channelsize(x::AbstractMatrix) = 1
channelsize(x::AbstractVecOrMat) = error("x must have dimension of at least 3, but x is a $(typeof(x))")
channelsize(x::AbstractArray{T,N}) where {T,N} = size(x, N-1)

"""
    heightsize(x::AbstractArray)

Returns the length of the first dimension of the data `x`.
`x` must have dimension of at least 3, otherwise an error is thrown.
"""
# heightsize(x::AbstractVector) = error("heightsize undefined for vectors")
heightsize(x::AbstractVecOrMat) = error("x must have dimension of at least 3, but x is a $(typeof(x))")
heightsize(x::AbstractArray) = size(x, 1)

"""
    log10range(a, b; length = 10)

Returns a `length`-element vector with log-linearly spaced data
between `a` and `b`
"""
log10range(a, b; length = 10) = 10 .^ range(log10(a), log10(b); length = length)

"""
    linspace(x1,x2,y1,y2) = x -> (y2 - y1) / (x2 - x1) * (x - x1) + y1
"""
@inline linspace(x1,x2,y1,y2) = x -> (y2 - y1) / (x2 - x1) * (x - x1) + y1

"""
    logspace(x1,x2,y1,y2) = x -> 10^linspace(x1, x2, log10(y1), log10(y2))(x)
"""
@inline logspace(x1,x2,y1,y2) = x -> 10^linspace(x1, x2, log10(y1), log10(y2))(x)

"""
    unitsum(x; dims = :) = x ./ sum(x; dims = dims)
"""
unitsum(x; dims = :) = x ./ sum(x; dims = dims)
unitsum!(x; dims = :) = x ./= sum(x; dims = dims)

"""
    to_float_type_T(T, x)

Convert a number or collection `x` to have floating point type `T`.
"""
to_float_type_T(T, x) = map(T, x) # fallback
to_float_type_T(T, x::Number) = T(x)
to_float_type_T(T, x::AbstractVector) = convert(Vector{T}, x)
to_float_type_T(T, x::AbstractMatrix) = convert(Matrix{T}, x)
to_float_type_T(T, x::AbstractVector{C}) where {C <: Complex} = convert(Vector{Complex{T}}, x)
to_float_type_T(T, x::AbstractMatrix{C}) where {C <: Complex} = convert(Matrix{Complex{T}}, x)

"""
Extract `nTE` complex signal echoes from data `z`.
Assume that `z` is sampled every `TE/n` seconds for some positive integer `n`.
The output is the magnitude of the last `nTE` points sampled at a multiple of `TE`.
"""
function cplx_signal(z::AbstractVecOrMat{C}, nTE::Int = size(z,1) - 1) where {C <: Complex}
    n = size(z,1)
    dt = (n-1) ÷ nTE
    @assert n == 1 + dt * nTE
    return z[n - dt * (nTE-1) : dt : n, ..]
end

"""
    snr(x, n)

Signal-to-noise ratio of the signal `x` relative to the noise `n`.
"""
snr(x, n; dims = 1) = 10 .* log10.(sum(abs2, x; dims = dims) ./ sum(abs2, n; dims = dims))

"""
    noise_level(z, SNR)

Standard deviation of gaussian noise with a given `SNR` level, proportional to the first time point.
    Note: `SNR` ≤ 0 is special cased to return a noise level of zero.
"""
noise_level(z::AbstractArray{T}, SNR::Number) where {T} =
    SNR ≤ 0 ? 0 .* z[1:1, ..] : abs.(z[1:1, ..]) ./ T(10^(SNR/20)) # Works for both real and complex

gaussian_noise(z::AbstractArray, SNR) = noise_level(z, SNR) .* randn(eltype(z), size(z))

"""
    add_gaussian(z, SNR)

Add gaussian noise with signal-to-noise ratio `SNR` proportional to the first time point.
"""
add_gaussian!(out::AbstractArray, z::AbstractArray, SNR) = out .= z .+ gaussian_noise(z, SNR)
add_gaussian!(z::AbstractArray, SNR) = z .+= gaussian_noise(z, SNR)
add_gaussian(z::AbstractArray, SNR) = z .+ gaussian_noise(z, SNR)

"""
    add_rician(z, SNR)

Add rician noise with signal-to-noise ratio `SNR` proportional to the first time point.
Always returns a real array.
"""
add_rician(m::AbstractArray{<:Real}, SNR) = add_rician(complex.(m), SNR)
add_rician(z::AbstractArray{<:Complex}, SNR) = abs.(add_gaussian(z, SNR))
# add_rician(m::AbstractArray{<:Real}, SNR) = add_rician!(copy(m), SNR)
# add_rician!(m::AbstractArray{<:Real}, SNR) = (gr = inv(√2) * gaussian_noise(m, SNR); gi = inv(√2) * gaussian_noise(m, SNR); m .= sqrt.(abs2.(m.+gr) .+ abs2.(gi)); return m)

"""
Kaiming uniform initialization.
"""
function kaiming_uniform(T::Type, dims; gain = 1)
   fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
   bound = sqrt(3) * gain / sqrt(fan_in)
   return rand(Uniform(-bound, bound), dims) |> Array{T}
end
kaiming_uniform(T::Type, dims...; kwargs...) = kaiming_uniform(T::Type, dims; kwargs...)
kaiming_uniform(args...; kwargs...) = kaiming_uniform(Float64, args...; kwargs...)

"""
Kaiming normal initialization.
"""
function kaiming_normal(T::Type, dims; gain = 1)
   fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
   std = gain / sqrt(fan_in)
   return rand(Normal(0, std), dims) |> Array{T}
end
kaiming_normal(T::Type, dims...; kwargs...) = kaiming_normal(T::Type, dims; kwargs...)
kaiming_normal(args...; kwargs...) = kaiming_normal(Float64, args...; kwargs...)

"""
Xavier uniform initialization.
"""
function xavier_uniform(T::Type, dims; gain = 1)
   fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
   fan_out = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end-1]
   bound = sqrt(3) * gain * sqrt(2 / (fan_in + fan_out))
   return rand(Uniform(-bound, bound), dims) |> Array{T}
end
xavier_uniform(T::Type, dims...; kwargs...) = xavier_uniform(T::Type, dims; kwargs...)
xavier_uniform(args...; kwargs...) = xavier_uniform(Float64, args...; kwargs...)

"""
Xavier normal initialization.
"""
function xavier_normal(T::Type, dims; gain = 1)
   fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
   fan_out = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end-1]
   std = gain * sqrt(2 / (fan_in + fan_out))
   return rand(Normal(0, std), dims) |> Array{T}
end
xavier_normal(T::Type, dims...; kwargs...) = xavier_normal(T::Type, dims; kwargs...)
xavier_normal(args...; kwargs...) = xavier_normal(Float64, args...; kwargs...)

# Override flux defaults
# Flux.glorot_uniform(dims...) = xavier_uniform(Float64, dims...)
# Flux.glorot_uniform(T::Type, dims...) = xavier_uniform(T, dims...)
# Flux.glorot_normal(dims...) = xavier_normal(Float64, dims...)
# Flux.glorot_normal(T::Type, dims...) = xavier_normal(T, dims...)

####
#### Callbacks
####

function epochthrottle(f, state, epoch_rate)
    last_epoch = 0
    function epochthrottled(args...; kwargs...)
        isempty(state) && return nothing
        epoch = state[end, :epoch]
        if epoch >= last_epoch + epoch_rate
            last_epoch = epoch
            f(args...; kwargs...)
        else
            nothing
        end
    end
end

function log10ticks(a, b; baseticks = 1:10)
    l, u = floor(Int, log10(a)), ceil(Int, log10(b))
    ticks = unique!(vcat([10.0^x .* baseticks for x in l:u]...))
    return filter!(x -> a ≤ x ≤ b, ticks)
end
function slidingindices(epoch, window = 100)
    i1_window = findfirst(e -> e ≥ epoch[end] - window + 1, epoch)
    i1_first  = findfirst(e -> e ≥ window, epoch)
    (i1_first === nothing) && (i1_first = 1)
    return min(i1_window, i1_first) : length(epoch)
end

function make_test_err_cb(state, lossfun, accfun, laberrfun, test_set)
    function()
        update_time = @elapsed begin
            if !isempty(state)
                row = findlast(==(:test), state.dataset)
                state[row, :loss]     = Flux.cpu(lossfun(test_set...))
                state[row, :acc]      = Flux.cpu(accfun(test_set...))
                state[row, :labelerr] = Flux.cpu(laberrfun(test_set...))
            end
        end
        # @info @sprintf("[%d] -> Updating testing error... (%d ms)", state[row, :epoch], 1000 * update_time)
    end
end
function make_train_err_cb(state, lossfun, accfun, laberrfun, train_set)
    function()
        update_time = @elapsed begin
            if !isempty(state)
                row = findlast(==(:train), state.dataset)
                state[row, :loss]     = mean([Flux.cpu(lossfun(b...))   for b in train_set])
                state[row, :acc]      = mean([Flux.cpu(accfun(b...))    for b in train_set])
                state[row, :labelerr] = mean([Flux.cpu(laberrfun(b...)) for b in train_set])
            end
        end
        # @info @sprintf("[%d] -> Updating training error... (%d ms)", state[row, :epoch], 1000 * update_time)
    end
end
function make_plot_errs_cb(state, filename = nothing; labelnames = "")
    function err_subplots()
        ps = Any[]
        for dataset in unique(state.dataset)
            window = 100
            min_epoch = max(1, min(state[end, :epoch] - window, window))
            df = state[(state.dataset .== dataset) .& (min_epoch .<= state.epoch), :]

            commonkw = (xscale = :log10, xticks = log10ticks(df[1, :epoch], df[end, :epoch]), xrotation = 75.0, xformatter = x->string(round(Int,x)), lw = 3, titlefontsize = 8, tickfontsize = 6, legend = :best, legendfontsize = 6)
            logspacing!(dfp) = isempty(dfp) ? dfp : unique(round.(Int, 10.0 .^ range(log10.(dfp.epoch[[1,end]])...; length = 10000))) |> I -> length(I) ≥ 5000 ? deleterows!(dfp, findall(!in(I), dfp.epoch)) : dfp

            dfp = logspacing!(dropmissing(df[!, [:epoch, :loss]]))
            p1 = plot()
            if !isempty(dfp)
                minloss = round(minimum(dfp.loss); sigdigits = 4)
                p1 = @df dfp plot(:epoch, :loss; title = "Loss ($dataset): min = $minloss)", label = "loss", ylim = (minloss, quantile(dfp.loss, 0.99)), commonkw...)
            end

            dfp = logspacing!(dropmissing(df[!, [:epoch, :acc]]))
            p2 = plot()
            if !isempty(dfp)
                maxacc = round(maximum(dfp.acc); sigdigits = 4)
                p2 = @df dfp plot(:epoch, :acc; title = "Accuracy ($dataset): peak = $maxacc%)", label = "acc", yticks = 50:0.1:100, ylim = (clamp(maxacc, 50, 99) - 1.5, min(maxacc + 0.5, 100.0)), commonkw...)
            end

            dfp = logspacing!(dropmissing(df[!, [:epoch, :labelerr]]))
            p3 = plot()
            if !isempty(dfp)
                labelerr = permutedims(reduce(hcat, dfp[!, :labelerr]))
                labcol = size(labelerr,2) == 1 ? :blue : permutedims(RGB[cgrad(:darkrainbow)[z] for z in range(0.0, 1.0, length = size(labelerr,2))])
                p3 = @df dfp plot(:epoch, labelerr; title = "Label Error ($dataset): rel. %)", label = labelnames, c = labcol, yticks = 0:100, ylim = (0, min(50, maximum(labelerr[end,:]) + 3.0)), commonkw...)
            end

            push!(ps, plot(p1, p2, p3; layout = (1,3)))
        end
        plot(ps...; layout = (length(ps), 1))
    end
    function()
        try
            plot_time = @elapsed begin
                fig = err_subplots()
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            # @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[end, :epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_checkpoint_state_cb(state, filename = nothing; filtermissings = false, filternans = false)
    function()
        save_time = @elapsed let state = deepcopy(state)
            if !isnothing(filename)
                filtermissings && dropmissing!(state) # drop rows with missings
                filternans && filter!(r -> all(x -> !((x isa Number && isnan(x)) || (x isa AbstractArray{<:Number} && any(isnan, x))), r), state) # drop rows with NaNs
                savebson(filename, @dict(state))
            end
        end
        # @info @sprintf("[%d] -> Error checkpoint... (%d ms)", state[end, :epoch], 1000 * save_time)
    end
end
function make_plot_gan_losses_cb(state, filename = nothing)
    function()
        try
            plot_time = @elapsed begin
                fig = @df state plot(:epoch, [:Gloss :Dloss :D_x :D_G_z]; label = ["G Loss" "D loss" "D(x)" "D(G(z))"], lw = 3)
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            # @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[end, :epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_plot_ligocvae_losses_cb(state, filename = nothing)
    function()
        try
            plot_time = @elapsed begin
                ps = Any[]
                for dataset in unique(state.dataset)
                    window = 100
                    min_epoch = max(1, min(state[end, :epoch] - window, window))
                    logspacing!(dfp) = isempty(dfp) ? dfp : unique(round.(Int, 10.0 .^ range(log10.(dfp.epoch[[1,end]])...; length = 10000))) |> I -> length(I) ≥ 5000 ? deleterows!(dfp, findall(!in(I), dfp.epoch)) : dfp

                    dfp = logspacing!(dropmissing(state[(state.dataset .== dataset) .& (min_epoch .<= state.epoch), [:epoch, :ELBO, :KL, :loss]]))
                    p = plot()
                    if !isempty(dfp)
                        commonkw = (xaxis = (:log10, log10ticks(dfp[1, :epoch], dfp[end, :epoch])), xrotation = 60.0, legend = :best, lw = 3, xformatter = x->string(round(Int,x)))
                        pKL   = @df dfp plot(:epoch, :KL;   title =   "KL vs. epoch ($dataset): max = $(round(maximum(dfp.KL);   sigdigits = 4))", lab = "KL",   c = :orange, commonkw...)
                        pELBO = @df dfp plot(:epoch, :ELBO; title = "ELBO vs. epoch ($dataset): min = $(round(minimum(dfp.ELBO); sigdigits = 4))", lab = "ELBO", c = :blue,   commonkw...)
                        pH    = @df dfp plot(:epoch, :loss; title =    "H vs. epoch ($dataset): min = $(round(minimum(dfp.loss); sigdigits = 4))", lab = "loss", c = :green,  commonkw...)
                        p     = plot(pKL, pELBO, pH; layout = (3,1))
                    end
                    push!(ps, p)
                end
                fig = plot(ps...)
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            # @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[end, :epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_update_lr_cb(state, opt, lrfun; lrcutoff = 1e-6)
    last_lr = nothing
    curr_lr = lr(opt)
    function()
        # Update learning rate and exit if it has become to small
        if !isempty(state)
            curr_lr = lr!(opt, lrfun(state[end, :epoch]))
        end
        if curr_lr < lrcutoff
            @info(" -> Early-exiting: Learning rate has dropped below cutoff = $lrcutoff")
            Flux.stop()
        end
        if last_lr === nothing
            @info(" -> Initial learning rate: " * @sprintf("%.2e", curr_lr))
        elseif last_lr != curr_lr
            @info(" -> Learning rate updated: " * @sprintf("%.2e", last_lr) * " --> "  * @sprintf("%.2e", curr_lr))
        end
        last_lr = curr_lr
        return nothing
    end
end
function make_save_best_model_cb(state, model, opt, filename = nothing)
    function()
        # If this is the best accuracy we've seen so far, save the model out
        isempty(state) && return nothing
        df = state[state.dataset .== :test, :]
        ismissing(df.acc[end]) && return nothing
        isempty(skipmissing(df.acc)) && return nothing

        best_acc = maximum(skipmissing(df.acc))
        if df[end, :acc] == best_acc
            try
                save_time = @elapsed let model = Flux.cpu(deepcopy(model)) #, opt = Flux.cpu(deepcopy(opt))
                    # weights = collect(Flux.params(model))
                    # !(filename === nothing) && savebson(filename * "weights-best.bson", @dict(weights))
                    !(filename === nothing) && savebson(filename * "model-best.bson", @dict(model))
                    # !(filename === nothing) && savebson(filename * "opt-best.bson", @dict(opt)) #TODO BSON optimizer saving broken
                end
                @info @sprintf("[%d] -> New best accuracy %.4f; model saved (%4d ms)", df[end, :epoch], df[end, :acc], 1000 * save_time)
            catch e
                @warn "Error saving best model..."
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
        nothing
    end
end
function make_checkpoint_model_cb(state, model, opt, filename = nothing)
    function()
        try
            save_time = @elapsed let model = Flux.cpu(deepcopy(model)) #, opt = Flux.cpu(deepcopy(opt))
                # weights = collect(Flux.params(model))
                # !(filename === nothing) && savebson(filename * "weights-checkpoint.bson", @dict(weights))
                !(filename === nothing) && savebson(filename * "model-checkpoint.bson", @dict(model))
                # !(filename === nothing) && savebson(filename * "opt-checkpoint.bson", @dict(opt)) #TODO BSON optimizer saving broken
            end
            # @info @sprintf("[%d] -> Model checkpoint... (%d ms)", state[end, :epoch], 1000 * save_time)
        catch e
            @warn "Error checkpointing model..."
            @warn sprint(showerror, e, catch_backtrace())
        end
    end
end

####
#### Gradient testing
####

function ngradient(f, xs::AbstractArray...)
    grads = zeros.(eltype.(xs), size.(xs))
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        δ = cbrt(eps(eltype(x))) # cbrt seems to be slightly better than sqrt
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
        display.(Δ[i]) #TODO FIXME
    end
    return grads
end

function gradcheck(f, m, xs::AbstractArray...; onlyfirst = true, seed = 0)
    ps = Flux.params(m)
    !isnothing(seed) && Random.seed!(seed)
    g0 = Flux.gradient(() -> f(m, xs...), ps) |> g -> [g[p] for p in ps]
    onlyfirst && (g0 = first.(g0))
    display.(g0) #TODO FIXME

    m  = Flux.paramtype(BigFloat, m)
    ys = Flux.paramtype(BigFloat, xs)
    ps = Flux.params(m)
    onlyfirst && (ps = [@views(p[1:1]) for p in ps])
    g1 = MWFLearning.ngradient(ps...) do (args...)
        !isnothing(seed) && Random.seed!(seed)
        f(m, ys...)
    end
    onlyfirst && (g1 = first.(g1))
    display.(g1) #TODO FIXME

    display.(g0 .- g1) #TODO FIXME
    map(g0, g1) do g0, g1
        display(abs.(g0 .- g1) .< cbrt.(eps.(g0))^2) #TODO FIXME
    end
end;

#=
let
    m = Flux.Dense(2,2,Flux.relu) |> Flux.f32
    x = 100*rand(2) |> Flux.f32
    MWFLearning.gradcheck((m,x) -> sum(abs2, m(x)), m, x)
end;
let m = Flux.f32(m), xy = Flux.f32(test_data)
    # MWFLearning.H_LIGOCVAE(m, xy...) |> display
    MWFLearning.gradcheck((m,xy...) -> MWFLearning.H_LIGOCVAE(m, xy...), m, xy...)
end;
let m = Flux.f64(m), xy = Flux.f64(test_data)
    # MWFLearning.H_LIGOCVAE(m, xy...) |> display
    MWFLearning.gradcheck((m,xy...) -> MWFLearning.H_LIGOCVAE(m, xy...), m, xy...)
end;
=#

function gradcheck(f, xs::AbstractArray...)
    dx0 = Flux.gradient(f, xs...)
    dx1 = ngradient(f, xs...)
    @show maximum.(abs, dx0)
    @show maximum.(abs, dx1)
    @show maximum.(abs, (dx0 .- dx1) ./ dx0)
    all(isapprox.(dx0, dx1, rtol = 1e-4, atol = 0))
end
