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
    batches = partition(1:batchsize(features), minibatchsize)
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

fixedlr(e,opt) = lr(opt) # Fixed learning rate
geometriclr(e,opt,rate=100,factor=10^(1/4)) = mod(e, rate) == 0 ? lr(opt) / factor : lr(opt) # Drop lr every `rate` epochs
findlr(e,opt,epochs=100,minlr=1e-6,maxlr=0.5) = e <= epochs ? logspace(1,epochs,minlr,maxlr)(e) : maxlr # Learning rate finder
cyclelr(e,opt,lrstart=1e-5,lrmin=1e-6,lrmax=1e-2,lrwidth=50,lrtail=5) = # Learning rate cycling
                     e <=   lrwidth          ? linspace(        1,            lrwidth, lrstart,   lrmax)(e) :
      lrwidth + 1 <= e <= 2*lrwidth          ? linspace(  lrwidth,          2*lrwidth,   lrmax, lrstart)(e) :
    2*lrwidth + 1 <= e <= 2*lrwidth + lrtail ? linspace(2*lrwidth, 2*lrwidth + lrtail, lrstart,   lrmin)(e) :
    lrmin

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
    unitsum(x) = x ./ sum(x)
"""
unitsum(x) = x ./ sum(x)

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
Normalize input complex signal data `z`.
Assume that `z` is sampled every `TE/n` seconds for some positive integer `n`.
The output is the magnitude of the last `nTE` points sampled at a multiple of `TE`.
to have the first point equal to 1.
"""
function cplx_signal(z::AbstractVecOrMat{C}, nTE::Int = size(z,1) - 1) where {C <: Complex}
    n = size(z,1)
    dt = (n-1) ÷ nTE
    @assert n == 1 + dt * nTE
    # Extract last nTE echoes, and normalize by the first point |S0| = |S(t=0)|.
    # This sets |S(t=0)| = 1 for any domain, but allows all measurable points,
    # i.e. S(t=TE), S(T=2TE), ..., to be unnormalized.
    Z = z[n - dt * (nTE-1) : dt : n, ..]
    Z ./= abs.(z[1:1, ..])
    return Z
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
add_gaussian!(out::AbstractArray, z::AbstractArray, SNR) = (out .= z .+ gaussian_noise(z, SNR); return out)
add_gaussian!(z::AbstractArray, SNR) = (z .+= gaussian_noise(z, SNR); return z)
add_gaussian(z::AbstractArray, SNR) = add_gaussian!(copy(z), z, SNR)

"""
    add_rician(z, SNR)

Add rician noise with signal-to-noise ratio `SNR` proportional to the first time point.
Always returns a real array.
"""
add_rician(m::AbstractArray{<:Real}, SNR) = add_rician(complex(m), SNR)
add_rician(z::AbstractArray{<:Complex}, SNR) = abs.(add_gaussian(z, SNR))
# add_rician(m::AbstractArray{<:Real}, SNR) = add_rician!(copy(m), SNR)
# add_rician!(m::AbstractArray{<:Real}, SNR) = (gr = inv(√2) * gaussian_noise(m, SNR); gi = inv(√2) * gaussian_noise(m, SNR); m .= sqrt.(abs2.(m.+gr) .+ abs2.(gi)); return m)

"""
    myelin_prop(...)
"""
function myelin_prop(
        mwf::T    = T(0.2),
        iewf::T   = T(1-mwf),
        rT2iew::T = T(63e-3/10e-3),
        rT2mw::T  = T(15e-3/10e-3),
        alpha::T  = T(170.0),
        rT1iew::T = T(10_000e-3/10e-3), # By default, assume T1 effects are negligeable
        rT1mw::T  = T(10_000e-3/10e-3), # By default, assume T1 effects are negligeable
        nTE::Int  = 32,
    ) where {T}

    M = mwf  .* forward_prop(rT2mw, rT1mw, alpha, nTE) .+
        iewf .* forward_prop(rT2iew, rT1iew, alpha, nTE)
    
    return (m -> √(m[1]^2 + m[2]^2)).(M)
end

"""
    forward_prop(...)
"""
function forward_prop!(
        M::AbstractVector{Vec{3,T}},
        rT2::T   = T(65e-3 / 10e-3),
        rT1::T   = T(10_000e-3 / 10e-3), # By default, assume T1 effects are negligeable
        alpha::T = T(170.0),
        nTE::Int = 32
    ) where {T}

    @assert length(M) == nTE

    m₀ = Vec{3,T}((0, -1, 0))
    m∞ = Vec{3,T}((0, 0, 1))

    # By Tensors.jl convention, this specifies the transpose rotation matrix
    At = Tensor{2,3,T}((
        one(T),  zero(T),      zero(T),
        zero(T), cosd(alpha), -sind(alpha),
        zero(T), sind(alpha),  cosd(alpha)))
    A  = transpose(At)
    R  = Vec{3,T}((exp(-inv(2*rT2)), exp(-inv(2*rT2)), exp(-inv(2*rT1))))

    step1 = (m) -> m∞ - R ⊙ (m∞ - A  ⋅ (m∞ - R ⊙ (m∞ - m)))
    step2 = (m) -> m∞ - R ⊙ (m∞ - A' ⋅ (m∞ - R ⊙ (m∞ - m)))

    M[1] = step1(m₀)
    M[2] = step2(M[1])
    for ii = 3:2:nTE
        M[ii  ] = step1(M[ii-1])
        M[ii+1] = step2(M[ii  ])
    end

    return M
end
forward_prop(rT2::T, rT1::T, alpha::T, nTE::Int) where {T} =
    forward_prop!(zeros(Vec{3,T}, nTE), rT2, rT1, alpha, nTE)

"""
Kaiming uniform initialization.
"""
function kaiming_uniform(T::Type, dims; gain = 1)
   fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
   bound = sqrt(3) * gain / sqrt(fan_in)
   return rand(Uniform(-bound, bound), dims) |> Array{T}
end
kaiming_uniform(T::Type, dims...; kwargs...) = kaiming_uniform(T::Type, dims; kwargs...)
kaiming_uniform(args...; kwargs...) = kaiming_uniform(Float32, args...; kwargs...)

"""
Kaiming normal initialization.
"""
function kaiming_normal(T::Type, dims; gain = 1)
   fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
   std = gain / sqrt(fan_in)
   return rand(Normal(0, std), dims) |> Array{T}
end
kaiming_normal(T::Type, dims...; kwargs...) = kaiming_normal(T::Type, dims; kwargs...)
kaiming_normal(args...; kwargs...) = kaiming_normal(Float32, args...; kwargs...)

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
xavier_uniform(args...; kwargs...) = xavier_uniform(Float32, args...; kwargs...)

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
xavier_normal(args...; kwargs...) = xavier_normal(Float32, args...; kwargs...)

# Override flux defaults
Flux.glorot_uniform(dims...) = xavier_uniform(Float32, dims...)
Flux.glorot_uniform(T::Type, dims...) = xavier_uniform(T, dims...)
Flux.glorot_normal(dims...) = xavier_normal(Float32, dims...)
Flux.glorot_normal(T::Type, dims...) = xavier_normal(T, dims...)

####
#### Callbacks
####

function epochthrottle(f, state, epoch_rate)
    last_epoch = 0
    function epochthrottled(args...; kwargs...)
        epoch = state[:epoch]
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
            pushx!(d) = x -> push!(d, x)
            push!(state[:callbacks][:testing][:epoch], state[:epoch])
            Flux.cpu(Flux.data(lossfun(test_set...)))   |> pushx!(state[:callbacks][:testing][:loss])
            Flux.cpu(Flux.data(accfun(test_set...)))    |> pushx!(state[:callbacks][:testing][:acc])
            Flux.cpu(Flux.data(laberrfun(test_set...))) |> pushx!(state[:callbacks][:testing][:labelerr])
        end
        @info @sprintf("[%d] -> Updating testing error... (%d ms)", state[:epoch], 1000 * update_time)
    end
end
function make_train_err_cb(state, lossfun, accfun, laberrfun, train_set)
    function()
        update_time = @elapsed begin
            pushx!(d) = x -> push!(d, x)
            push!(state[:callbacks][:training][:epoch], state[:epoch])
            mean([Flux.cpu(Flux.data(lossfun(b...)))   for b in train_set]) |> pushx!(state[:callbacks][:training][:loss])
            mean([Flux.cpu(Flux.data(accfun(b...)))    for b in train_set]) |> pushx!(state[:callbacks][:training][:acc])
            mean([Flux.cpu(Flux.data(laberrfun(b...))) for b in train_set]) |> pushx!(state[:callbacks][:training][:labelerr])
        end
        @info @sprintf("[%d] -> Updating training error... (%d ms)", state[:epoch], 1000 * update_time)
    end
end
function make_plot_errs_cb(state, filename = nothing; labelnames = "")
    function err_subplots(k,v)
        @unpack epoch, loss, acc, labelerr = v
        idx = slidingindices(epoch)
        epoch, loss, acc, labelerr = epoch[idx], loss[idx], acc[idx], labelerr[idx,:]
        
        laberr = permutedims(reduce(hcat, labelerr))
        labcol = permutedims(RGB[cgrad(:darkrainbow)[z] for z in range(0.0, 1.0, length = size(laberr,2))])
        minloss, maxacc = round(minimum(loss); sigdigits = 4), round(maximum(acc); sigdigits = 4)
        commonkw = (xscale = :log10, xticks = log10ticks(epoch[1], epoch[end]), xrotation = 75.0, xformatter = x->string(round(Int,x)), lw = 3, legend = :best, titlefontsize = 8, tickfontsize = 6, legendfontsize = 6)

        p1 = plot(epoch, loss;   title = "Loss ($k: min = $minloss)", label = "loss", ylim = (minloss, quantile(loss, 0.95)), commonkw...)
        p2 = plot(epoch, acc;    title = "Accuracy ($k: peak = $maxacc%)", label = "acc", yticks = 50:0.1:100, ylim = (clamp(maxacc, 50, 99) - 1.0, min(maxacc + 0.5, 100.0)), commonkw...)
        p3 = plot(epoch, laberr; title = "Label Error ($k: rel. %)", label = labelnames, c = labcol, yticks = 0:100, ylim = (max(0, minimum(laberr) - 1.0), min(50, 1.2 * maximum(laberr[end,:]))), commonkw...) #min(50, quantile(vec(laberr), 0.90))
        if k == :testing
            idxloop = slidingindices(state[:loop][:epoch])
            plot!(p2, state[:loop][:epoch][idxloop], state[:loop][:acc][idxloop]; label = "loop acc", lw = 1)
        end
        plot(p1, p2, p3; layout = (1,3))
    end
    function()
        try
            plot_time = @elapsed begin
                fig = plot([err_subplots(k,v) for (k,v) in state[:callbacks]]...; layout = (length(state[:callbacks]), 1))
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[:epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[:epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_checkpoint_model_opt_cb(state, model, opt, filename)
    function()
        save_time = @elapsed let opt = MWFLearning.opt_to_cpu(opt, Flux.params(model)), model = Flux.cpu(model)
            savebson(filename, @dict(model, opt))
        end
        @info @sprintf("[%d] -> Model checkpoint... (%d ms)", state[:epoch], 1000 * save_time)
    end
end
function make_checkpoint_state_cb(state, filename)
    function()
        save_time = @elapsed let state = deepcopy(state)
            savebson(filename, @dict(state))
        end
        @info @sprintf("[%d] -> Error checkpoint... (%d ms)", state[:epoch], 1000 * save_time)
    end
end
function make_plot_gan_losses_cb(state, filename = nothing)
    function()
        try
            plot_time = @elapsed begin
                @unpack epoch, Gloss, Dloss, D_x, D_G_z = state[:loop]
                fig = plot(epoch, [Gloss Dloss D_x D_G_z]; label = ["G Loss" "D loss" "D(x)" "D(G(z))"], lw = 3)
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[:epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[:epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_plot_ligocvae_losses_cb(state, filename = nothing)
    function()
        try
            plot_time = @elapsed begin
                @unpack epoch, ELBO, KL, loss = state[:loop]
                idx = slidingindices(epoch)
                fig = plot(epoch[idx], [ELBO[idx], KL[idx], loss[idx]];
                    title = L"Cross-entropy Loss $H$ vs. Epoch",
                    label = [L"ELBO" L"KL" L"H = ELBO + KL"],
                    xaxis = (:log10, log10ticks(epoch[idx[1]], epoch[idx[end]])), xrotation = 60.0,
                    legend = :best, lw = 3, c = [:blue :orange :green], formatter = x->string(round(Int,x)))
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[:epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[:epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_update_lr_cb(state, opt, lrfun; lrcutoff = 1e-6)
    last_lr = nothing
    curr_lr = 0.0
    function()
        # Update learning rate and exit if it has become to small
        curr_lr = lr!(opt, lrfun(state[:epoch]))
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
function make_save_best_model_cb(state, model, filename)
    function()
        # If this is the best accuracy we've seen so far, save the model out
        epoch, acc = state[:epoch], state[:loop][:acc][end]
        if acc >= state[:best_acc]
            state[:best_acc] = acc
            state[:last_improved_epoch] = epoch
            try
                save_time = @elapsed let weights = Flux.cpu.(Flux.data.(Flux.params(model)))
                    savebson(filename, @dict(weights, epoch, acc))
                end
                # @info " -> New best accuracy; weights saved ($(round(1000*save_time; digits = 2)) ms)"
                @info @sprintf("[%d] -> New best accuracy; weights saved (%d ms)", epoch, 1000 * save_time)
            catch e
                @warn "Error saving weights"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
        nothing
    end
end
