# ---------------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------------- #

function load_settings(
        default_settings_file,
    )
    # Load default settings + merge in custom settings, if given
    settings = TOML.parsefile(default_settings_file)
    mergereducer!(x, y) = deepcopy(y) # fallback
    mergereducer!(x::AbstractDict, y::AbstractDict) = merge!(mergereducer!, x, y)
    haskey(ENV, "SETTINGSFILE") && merge!(mergereducer!, settings, TOML.parsefile(ENV["SETTINGSFILE"]))

    # Save + print resulting settings
    outpath = settings["data"]["out"]
    !isdir(outpath) && mkpath(outpath)
    open(joinpath(outpath, "settings.toml"); write = true) do io
        TOML.print(io, settings)
    end
    TOML.print(stdout, settings)

    return settings
end

####
#### Saving and formatting
####

getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
savebson(filename, data::AbstractDict) = @elapsed BSON.bson(filename, data)
# gitdir() = realpath(DrWatson.projectdir(".."))

function handleinterrupt(e; msg = "Error")
    if e isa InterruptException
        @info "User interrupt"
    elseif e isa Flux.Optimise.StopException
        @info "Training stopped Flux callback"
    else
        !isempty(msg) && @warn msg
        @warn sprint(showerror, e, catch_backtrace())
    end
    return nothing
end

# previously: saveprogress(savefolder, prefix, suffix)
function saveprogress(savedata::AbstractDict; savefolder, prefix = "", suffix = "")
    !isdir(savefolder) && mkpath(savefolder)
    for (key, data) in savedata
        try
            BSON.bson(
                joinpath(savefolder, "$(prefix)$(key)$(suffix).bson"),
                Dict{String,Any}(string(key) => deepcopy(Flux.cpu(data))),
            )
        catch e
            handleinterrupt(e; msg = "Error saving progress")
        end
    end
end

# previously: saveplots(savefolder, prefix, suffix, plothandles)
function saveplots(plothandles::AbstractDict; savefolder, prefix = "", suffix = "", ext = ".png")
    !isdir(savefolder) && mkpath(savefolder)
    for (name, p) in plothandles
        isnothing(p) && continue
        try
            savefig(p, joinpath(savefolder, prefix * string(name) * suffix * ext))
        catch e
            handleinterrupt(e; msg = "Error saving plot ($name)")
        end
    end
end

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
validation_batches(features, labels) = make_minibatch(features, labels, :)
features(batch) = batch[1]
labels(batch) = batch[2]

function param_summary(model, train_set, val_set)
    val_dofs = length(val_set[2])
    train_dofs = sum(batch -> length(batch[2]), train_set)
    param_dofs = sum(length, Flux.params(model))
    val_param_density = param_dofs / val_dofs
    train_param_density = param_dofs / train_dofs
    @info @sprintf("  Training parameter density: %d/%d (%.2f %%)", param_dofs, train_dofs, 100 * train_param_density)
    @info @sprintf("Validation parameter density: %d/%d (%.2f %%)", param_dofs, val_dofs,   100 * val_param_density)
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
            df = dropmissing(state[(state.dataset .=== :val) .& (min_epoch .<= state.epoch), [:epoch, :loss]])
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

function make_val_err_cb(state, lossfun, accfun, laberrfun, val_set)
    function()
        update_time = @elapsed begin
            if !isempty(state)
                row = findlast(===(:val), state.dataset)
                state[row, :loss]     = Flux.cpu(lossfun(val_set...))
                state[row, :acc]      = Flux.cpu(accfun(val_set...))
                state[row, :labelerr] = Flux.cpu(laberrfun(val_set...))
            end
        end
        # @info @sprintf("[%d] -> Updating validation error... (%d ms)", state[row, :epoch], 1000 * update_time)
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

            commonkw = (xscale = :log10, xticks = log10ticks(df[1, :epoch], df[end, :epoch]), xrotation = 75.0, lw = 3, titlefontsize = 8, tickfontsize = 6, legend = :best, legendfontsize = 6)
            logspacing!(dfp) = isempty(dfp) ? dfp : unique(round.(Int, 10.0 .^ range(log10.(dfp.epoch[[1,end]])...; length = 10000))) |> I -> length(I) ≥ 5000 ? deleterows!(dfp, findall(!in(I), dfp.epoch)) : dfp

            dfp = logspacing!(dropmissing(df[!, [:epoch, :loss]]))
            p1 = plot()
            if !isempty(dfp)
                minloss = round(minimum(dfp.loss); sigdigits = 4)
                p1 = plot(dfp.epoch, dfp.loss; title = "Loss ($dataset): min = $minloss)", label = "loss", ylim = (minloss, quantile(dfp.loss, 0.99)), commonkw...)
            end

            dfp = logspacing!(dropmissing(df[!, [:epoch, :acc]]))
            p2 = plot()
            if !isempty(dfp)
                maxacc = round(maximum(dfp.acc); sigdigits = 4)
                p2 = plot(dfp.epoch, dfp.acc; title = "Accuracy ($dataset): peak = $maxacc%)", label = "acc", yticks = 50:0.1:100, ylim = (clamp(maxacc, 50, 99) - 1.5, min(maxacc + 0.5, 100.0)), commonkw...)
            end

            dfp = logspacing!(dropmissing(df[!, [:epoch, :labelerr]]))
            p3 = plot()
            if !isempty(dfp)
                labelerr = permutedims(reduce(hcat, dfp[!, :labelerr]))
                labcol = size(labelerr,2) == 1 ? :blue : permutedims(RGB[cgrad(:darkrainbow)[z] for z in range(0.0, 1.0, length = size(labelerr,2))])
                p3 = plot(dfp.epoch, labelerr; title = "Label Error ($dataset): rel. %)", label = labelnames, c = labcol, yticks = 0:100, ylim = (0, min(50, maximum(labelerr[end,:]) + 3.0)), commonkw...)
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
            handleinterrupt(e; msg = @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch]))
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
                fig = plot(state.epoch, [state.Gloss state.Dloss state.D_x state.D_G_z]; label = ["G Loss" "D loss" "D(x)" "D(G(z))"], lw = 3)
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            # @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[end, :epoch], 1000 * plot_time)
        catch e
            handleinterrupt(e; msg = @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch]))
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
                        commonkw = (xaxis = (:log10, log10ticks(dfp[1, :epoch], dfp[end, :epoch])), xrotation = 60.0, legend = :best, lw = 3)
                        pKL   = plot(dfp.epoch, dfp.KL;   title =   "KL vs. epoch ($dataset): max = $(round(maximum(dfp.KL);   sigdigits = 4))", lab = "KL",   c = :orange, commonkw...)
                        pELBO = plot(dfp.epoch, dfp.ELBO; title = "ELBO vs. epoch ($dataset): min = $(round(minimum(dfp.ELBO); sigdigits = 4))", lab = "ELBO", c = :blue,   commonkw...)
                        pH    = plot(dfp.epoch, dfp.loss; title =    "H vs. epoch ($dataset): min = $(round(minimum(dfp.loss); sigdigits = 4))", lab = "loss", c = :green,  commonkw...)
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
            handleinterrupt(e; msg = @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch]))
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
        df = state[state.dataset .=== :val, :]
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
#### GAN Callbacks
####

function initialize_callback(phys::ToyModel; nsamples::Int)
    cb_state = Dict{String,Any}()
    cb_state["θ"]  = sampleθ(phys, nsamples; dataset = :val)
    cb_state["Xθ"] = sampleX(phys, cb_state["θ"]) # sampleX == signal_model when given explicit θ
    cb_state["Yθ"] = sampleX(ClosedForm(phys), cb_state["θ"])
    cb_state["Yθhat"] = sampleX(ClosedForm(phys), cb_state["θ"], noiselevel(ClosedForm(phys)))
    cb_state["Y"]  = sampleY(phys, nsamples; dataset = :val) # Y is deliberately sampled with different θ values
    cb_state["curr_time"] = time()
    cb_state["last_time"] = -Inf
    cb_state["last_fit_time"] = -Inf
    cb_state["last_curr_checkpoint"] = -Inf
    cb_state["last_best_checkpoint"] = -Inf
    # cb_state["fits"] = nothing
    # cb_state["last_θbb"] = Union{AbstractVecOrMat{Float64}, Nothing}[nothing]
    # cb_state["last_i_fit"] = Union{Vector{Int}, Nothing}[nothing]
    return cb_state
end

#=
function initialize_callback!(cb_state::AbstractDict, phys::EPGModel)
    # Sample X and θ randomly, but choose Ys + corresponding fits consistently in order to compare models, and choose Ys with reasonable agreeance with data in order to not be overconfident in improving terrible fits
    Xval = sampleX(settings["gan"]["batchsize"]; dataset = :val)
    θval = sampleθ(settings["gan"]["batchsize"]; dataset = :val)
    iY = if settings["data"]["normalize"]::Bool
        iY = filter(i -> fits_val.loss[i] <= -200.0 && fits_val.rmse[i] <= 0.002, 1:nrow(fits_val)) #TODO
    else
        iY = filter(i -> fits_val.loss[i] <= -125.0 && fits_val.rmse[i] <= 0.15, 1:nrow(fits_val)) #TODO
    end
    iY = sample(MersenneTwister(0), iY, settings["gan"]["batchsize"]; replace = false)
    Yval = sampleY(nothing; dataset = :val)[..,iY]
    valfits = fits_val[iY,:]

    cb_state["Xθ"] = sampleX(phys, nsamples; dataset = :val)
    cb_state["Y"] = sampleY(phys, nsamples; dataset = :val)
    cb_state["θ"] = sampleθ(phys, nsamples; dataset = :val)
    cb_state["last_time"] = Ref(time())
    cb_state["last_curr_checkpoint"] = Ref(-Inf)
    cb_state["last_best_checkpoint"] = Ref(-Inf)
    cb_state["fits"] = valfits
    cb_state["last_θbb"] = Union{AbstractVecOrMat{Float64}, Nothing}[nothing]
    cb_state["last_i_fit"] = Union{Vector{Int}, Nothing}[nothing]
    return cb_state
end
=#

function update_callback!(
        cb_state::AbstractDict,
        phys::PhysicsModel,
        G::RicianCorrector;
        ninfer::Int,
        infermethod::Symbol = :mle,
        inferperiod::Real = 300.0,
        forceinfer::Bool = false,
    )

    cb_state["last_time"] = get!(cb_state, "curr_time", time())
    cb_state["curr_time"] = time()

    # Compute signal correction, noise instances, etc.
    @unpack θ, Xθ, Yθ, Yθhat, Y = cb_state
    δθ, ϵθ = correction_and_noiselevel(G, Xθ)
    Xθδ = add_correction(G, Xθ, δθ)
    Xθhat = add_noise_instance(G, Xθδ, ϵθ)
    @pack! cb_state = δθ, ϵθ, Xθδ, Xθhat

    # Record useful metrics
    metrics = Dict{String,Any}()
    metrics["rmse"] = hasclosedform(phys) ? sqrt(mean(abs2, Yθ - Xθδ)) : missing

    if infermethod === :mle
        # If closed form model is known, fit to Yθhat (which has known θ); otherwise, fit to Y samples
        Ydata = hasclosedform(phys) ? Yθhat : Y

        if forceinfer || cb_state["curr_time"] - get!(cb_state, "last_fit_time", -Inf) >= inferperiod
            cb_state["last_fit_time"] = cb_state["curr_time"]

            # Sample `ninfer` new Ydata to fit to
            i_fit = sample(1:size(Ydata,2), ninfer; replace = false)
            Yfit = Ydata[:,i_fit]
            all_infer_results = @timeit "theta inference" let
                #TODO: Optimizers fail without Float64:
                #   - BlackBoxOptim hardcodes Float64 internally
                #   - Optim should work, but the optimizer tends to error with Float32... floating point issues?
                _G = G |> Flux.cpu |> Flux.f64
                _model = X -> rician_params(_G, X)
                _signal_fun = θ -> signal_model(phys, θ) # `phys` only used for dispatch; type of θ is used
                _θbounds = NTuple{2,Float64}.(θbounds(phys))
                signal_loglikelihood_inference(Yfit, nothing, _model, _signal_fun; objective = :mle, bounds = _θbounds)
            end

            # Extract and sort best results
            best_infer_results = map(all_infer_results) do (bbopt_res, optim_res)
                x1, ℓ1 = BlackBoxOptim.best_candidate(bbopt_res), BlackBoxOptim.best_fitness(bbopt_res)
                x2, ℓ2 = Optim.minimizer(optim_res), Optim.minimum(optim_res)
                (ℓ1 < ℓ2) && @warn "BlackBoxOptim results less than Optim result" #TODO
                T = eltype(phys) #TODO BlackBoxOptim/Optim return Float64
                return ℓ1 < ℓ2 ? (x = T.(x1), loss = T(ℓ1)) : (x = T.(x2), loss = T(ℓ2))
            end

            θfit = mapreduce(r -> r.x, hcat, best_infer_results)
            i_fit, Yfit, θfit = sortperm(map(r -> r.loss, best_infer_results)) |> I -> (i_fit[I], Yfit[:,I], θfit[:,I])

            # Record sorted fit results in cb state
            @pack! cb_state = i_fit, Yfit, θfit
        else
            # Use θfit results from previous inference on Yfit; θfit is a proxy for the current "best guess" θ
            @unpack i_fit, Yfit, θfit = cb_state
        end

        # Compute signal correction, noise instances, etc.
        Xθfit = signal_model(phys, θfit)
        δθfit, ϵθfit = correction_and_noiselevel(G, Xθfit)
        Xθδfit = add_correction(G, Xθfit, δθfit)
        Xθhatfit = add_noise_instance(G, Xθδfit, ϵθfit)
        Yθfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit) : missing
        Yθhatfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit, noiselevel(ClosedForm(phys))) : missing
        @pack! cb_state = Xθfit, Yθfit, Yθhatfit, δθfit, ϵθfit, Xθδfit, Xθhatfit

        # Compute error metrics
        all_Xhat_rmse = sqrt.(mean(abs2, Yfit .- Xθhatfit; dims = 1)) |> vec
        all_Xhat_logL = .-sum(logpdf.(Rician.(Xθδfit, ϵθfit), Yfit); dims = 1) |> vec
        Xhat_rmse = mean(all_Xhat_rmse)
        Xhat_logL = mean(all_Xhat_logL)
        @pack! metrics = all_Xhat_rmse, all_Xhat_logL, Xhat_rmse, Xhat_logL

        if hasclosedform(phys)
            # Evaluate error in recovered θ if closed form is known
            Xhat_theta_err = mean(θerror(phys, θ[:,i_fit], θfit); dims = 2) |> vec |> copy
            @pack! metrics = theta_err
        end
    end

    # Save metrics
    @pack! cb_state = metrics

    return cb_state
end

####
#### Plots
####

function plot_gan_loss(logger, cb_state, phys; window = 100, lrdroprate = typemax(Int), lrdrop = 1.0, showplot = false)
    @timeit "gan loss plot" try
        !all(c -> c ∈ propertynames(logger), (:epoch, :dataset, :Gloss, :Dloss, :D_Y, :D_G_X)) && return
        dfp = logger[logger.dataset .=== :val, :]
        epoch = dfp.epoch[end]
        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, dfp)
        dfp = dropmissing(dfp[:, [:epoch, :Gloss, :Dloss, :D_Y, :D_G_X]])
        if !isempty(dfp)
            s = x -> (x == round(Int, x) ? round(Int, x) : round(x; sigdigits = 4)) |> string
            ps = [
                plot(dfp.epoch, dfp.Dloss; label = "D loss", lw = 2, c = :red),
                plot(dfp.epoch, dfp.Gloss; label = "G loss", lw = 2, c = :blue),
                plot(dfp.epoch, [dfp.D_Y dfp.D_G_X]; label = ["D(Y)" "D(G(X))"], c = [:red :blue], lw = 2),
                plot(dfp.epoch, dfp.D_Y - dfp.D_G_X; label = "D(Y) - D(G(X))", c = :green, lw = 2),
            ]
            (epoch >= lrdroprate) && map(ps) do p
                plot!(p, lrdroprate : lrdroprate : epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)", seriestype = :vline)
                plot!(p; xscale = ifelse(epoch < 10*window, :identity, :log10))
            end
            p = plot(ps...)
        else
            p = nothing
        end
        showplot && !isnothing(p) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making gan loss plot")
    end
end

function plot_rician_model(logger, cb_state, phys; bandwidths = nothing, showplot = false)
    @timeit "model plot" try
        @unpack δθ, ϵθ = cb_state
        _subplots = []
        let
            δmid, δlo, δhi = eachrow(δθ) |> δ -> (mean.(δ), quantile.(δ, 0.25), quantile.(δ, 0.75))
            ϵmid, ϵlo, ϵhi = eachrow(ϵθ) |> ϵ -> (mean.(ϵ), quantile.(ϵ, 0.25), quantile.(ϵ, 0.75))
            push!(_subplots, plot(
                plot(δmid; yerr = (δmid - δlo, δhi - δmid), label = L"signal correction $g_\delta(X)$", c = :red, title = "model outputs vs. data channel"),
                plot(ϵmid; yerr = (ϵmid - ϵlo, ϵhi - ϵmid), label = L"noise amplitude $\exp(g_\epsilon(X))$", c = :blue);
                layout = (2,1),
            ))
        end
        bandwidth_plot(logσ::AbstractVector) = bandwidth_plot(permutedims(logσ))
        bandwidth_plot(logσ::AbstractMatrix) = size(logσ,1) == 1 ?
            scatter(1:length(logσ), vec(logσ); label = "logσ", title = "logσ") :
            plot(logσ; leg = :none, marker = (1,:circle), title = "logσ vs. data channel")
        if !isnothing(bandwidths)
            push!(_subplots,
                eltype(bandwidths) <: AbstractArray ?
                    plot(bandwidth_plot.(bandwidths)...; layout = (length(bandwidths), 1)) :
                    plot(bandwidths)
            )
        end
        p = plot(_subplots...)
        showplot && !isnothing(p) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making Rician model plot")
    end
end

function plot_rician_signals(logger, cb_state, phys; showplot = false, nsignals = 4)
    @timeit "signal plot" try
        @unpack Y, Xθ, Xθhat, δθ, Yθ = cb_state
        Yplot = hasclosedform(phys) ? Yθ : Y
        θplotidx = sample(1:size(Xθ,2), nsignals; replace = false)
        p = plot(
            [plot(hcat(Yplot[:,j], Xθhat[:,j]); c = [:blue :red], lab = [L"$Y$" L"\hat{X} \sim G(X(\hat{\theta}))"]) for j in θplotidx]...,
            [plot(hcat(Yplot[:,j] - Xθ[:,j], δθ[:,j]); c = [:blue :red], lab = [L"$Y - X(\hat{\theta})$" L"$g_\delta(X(\hat{\theta}))$"]) for j in θplotidx]...,
            [plot(Yplot[:,j] - Xθ[:,j] - δθ[:,j]; lab = L"$Y - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$") for j in θplotidx]...;
            layout = (3, nsignals),
        )
        showplot && !isnothing(p) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making Rician signal plot")
    end
end

function plot_rician_model_fits(logger, cb_state, phys; showplot = false)
    @timeit "signal plot" try
        @unpack Yfit, Xθfit, Xθhatfit, δθfit = cb_state
        nsignals = 4 # number of θ sets to draw for plotting simulated signals
        θplotidx = sample(1:size(Xθfit,2), nsignals; replace = false)
        p = plot(
            [plot(hcat(Yfit[:,j], Xθhatfit[:,j]); c = [:blue :red], lab = [L"$\hat{X}$" L"\hat{X} \sim G(X(\hat{\theta}))"]) for j in θplotidx]...,
            [plot(hcat(Yfit[:,j] - Xθfit[:,j], δθfit[:,j]); c = [:blue :red], lab = [L"$\hat{X} - X(\hat{\theta})$" L"$g_\delta(X(\hat{\theta}))$"]) for j in θplotidx]...,
            [plot(Yfit[:,j] - Xθfit[:,j] - δθfit[:,j]; lab = L"$\hat{X} - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$") for j in θplotidx]...;
            layout = (3, nsignals),
        )
        showplot && !isnothing(p) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making Rician signal plot")
    end
end

function plot_mmd_losses(logger, cb_state, phys; window = 100, lrdroprate = typemax(Int), lrdrop = 1.0, showplot = false)
    @timeit "mmd loss plot" try
        s = x -> x == round(x) ? round(Int, x) : round(x; sigdigits = 4)
        dfp = logger[logger.dataset .=== :val, :]
        epoch = dfp.epoch[end]
        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, dfp)
        if !isempty(dfp)
            tstat_nan_outliers = map((_tstat, _mmdvar) -> _mmdvar > eps() ? _tstat : NaN, dfp.tstat, dfp.MMDvar)
            tstat_drop_outliers = filter(!isnan, tstat_nan_outliers)
            tstat_median = isempty(tstat_drop_outliers) ? NaN : median(tstat_drop_outliers)
            tstat_ylim = isempty(tstat_drop_outliers) ? nothing : quantile(tstat_drop_outliers, [0.01, 0.99])
            p1 = plot(dfp.epoch, dfp.MMDsq; label = "m*MMD²", title = "median loss = $(s(median(dfp.MMDsq)))") # ylim = quantile(dfp.MMDsq, [0.01, 0.99])
            p2 = plot(dfp.epoch, dfp.MMDvar; label = "m²MMDvar", title = "median m²MMDvar = $(s(median(dfp.MMDvar)))") # ylim = quantile(dfp.MMDvar, [0.01, 0.99])
            p3 = plot(dfp.epoch, tstat_nan_outliers; title = "median t = $(s(tstat_median))", label = "t = MMD²/MMDσ", ylim = tstat_ylim)
            p4 = plot(dfp.epoch, dfp.P_alpha; label = "P_α", title = "median P_α = $(s(median(dfp.P_alpha)))", ylim = (0,1))
            foreach([p1,p2,p3,p4]) do p
                (epoch >= lrdroprate) && vline!(p, lrdroprate : lrdroprate : epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                plot!(p; xscale = ifelse(epoch < 10*window, :identity, :log10))
            end
            p = plot(p1, p2, p3, p4)
        else
            p = nothing
        end
        showplot && !isnothing(p) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making MMD losses plot")
    end
end

function plot_rician_inference(logger, cb_state, phys; window = 100, showplot = false)
    @timeit "theta inference plot" try
        s = x -> (x == round(Int, x) ? round(Int, x) : round(x; sigdigits = 4)) |> string
        dfp = logger[logger.dataset .=== :val, :]
        epoch = dfp.epoch[end]
        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, dfp)
        df_inf = filter(dfp) do r
            !ismissing(r.Xhat_rmse) && !ismissing(r.Xhat_logL) && (
                hasclosedform(phys) ?
                    (!ismissing(r.Yhat_rmse_true) && !ismissing(r.Yhat_theta_err)) :
                    (!ismissing(r.Yhat_rmse) && !ismissing(r.Xhat_theta_err)))
        end

        if !isempty(dfp) && !isempty(df_inf)
            @unpack all_Xhat_logL, all_Xhat_rmse = cb_state["metrics"]
            @unpack Xθ, Y = cb_state
            p = plot(
                plot(
                    plot(hcat(Y[:,end÷2], Xθ[:,end÷2]); c = [:blue :red], lab = [L"Y data" L"$X(\hat{\theta})$ fit"]),
                    sticks(sort(all_Xhat_rmse[sample(1:end, min(128,end))]); m = (:circle,4), lab = "rmse"),
                    sticks(sort(all_Xhat_logL[sample(1:end, min(128,end))]); m = (:circle,4), lab = "-logL"),
                    layout = @layout([a{0.25h}; b{0.375h}; c{0.375h}]),
                ),
                let
                    _subplots = Any[]
                    if hasclosedform(phys)
                        plogL = plot(df_inf.epoch, df_inf.Yhat_logL_true; title = L"True $-logL(Y)$: min = %$(s(minimum(df_inf.Yhat_logL_true)))", label = L"-logL(Y)", xscale = ifelse(epoch < 10*window, :identity, :log10)) # $Y(\hat{\theta}) - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$
                        pθerr = plot(df_inf.epoch, permutedims(reduce(hcat, df_inf.Yhat_theta_err)); title = L"$\hat{\theta}(Y)$: min max = %$(s(minimum(maximum.(df_inf.Yhat_theta_err))))", label = permutedims(θlabels(phys)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                        append!(_subplots, [plogL, pθerr])
                    else
                        plogL = plot(df_inf.epoch, df_inf.Yhat_logL; title = L"$-logL(Y)$: min = %$(s(minimum(df_inf.Yhat_logL)))", label = L"-logL(Y)", xscale = ifelse(epoch < 10*window, :identity, :log10)) # $Y(\hat{\theta}) - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$
                        pθerr = plot(df_inf.epoch, permutedims(reduce(hcat, df_inf.Xhat_theta_err)); title = L"$\hat{\theta}(\hat{X})$: min max = %$(s(minimum(maximum.(df_inf.Xhat_theta_err))))", label = permutedims(θlabels(phys)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                        append!(_subplots, [plogL, pθerr])
                    end

                    if false #TODO MRI model
                        rmselab *= "\nrmse prior: $(round(mean(phys.valfits.rmse); sigdigits = 4))"
                        logLlab *= "\n-logL prior: $(round(mean(phys.valfits.loss); sigdigits = 4))"
                    end
                    logLlab = [L"-logL(\hat{X})" L"-logL(Y)"]
                    rmselab = [L"rmse(\hat{X})" L"rmse(Y)"] # \hat{X}(\hat{\theta})
                    plogL = plot(df_inf.epoch, [df_inf.Xhat_logL df_inf.Yhat_logL]; title = "min = $(s(minimum(df_inf.Xhat_logL))), $(s(minimum(df_inf.Yhat_logL)))", lab = logLlab, xscale = ifelse(epoch < 10*window, :identity, :log10))
                    prmse = plot(df_inf.epoch, [df_inf.Xhat_rmse df_inf.Yhat_rmse]; title = "min = $(s(minimum(df_inf.Xhat_rmse))), $(s(minimum(df_inf.Yhat_rmse)))", lab = rmselab, xscale = ifelse(epoch < 10*window, :identity, :log10))
                    append!(_subplots, [plogL, prmse])

                    plot(_subplots...)
                end;
                layout = @layout([a{0.25w} b{0.75w}]),
            )
        else
            p = nothing
        end

        showplot && !isnothing(p) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making θ inference plot")
    end
end

function plot_all_logger_losses(logger, cb_state, phys;
        colnames = setdiff(propertynames(logger), [:epoch, :iter, :dataset, :time]),
        dataset = :val,
        window = 100,
        showplot = false,
    )
    @timeit "signal plot" try
        s = x -> (x == round(Int, x) ? round(Int, x) : round(x; sigdigits = 4)) |> string
        dfp = logger[logger.dataset .=== dataset, :]
        epoch = dfp.epoch[end]
        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, dfp)
        ps = map(sort(colnames)) do colname
            i = (!ismissing).(dfp[!, colname])
            if any(i)
                xdata = dfp.epoch[i]
                ydata = dfp[i, colname]
                if ydata[1] isa AbstractArray
                    ydata = permutedims(reduce(hcat, vec.(ydata)))
                end
                plot(xdata, ydata; lab = :none, title = string(colname), titlefontsize = 6, ytickfontsize = 6, xtickfontsize = 6, xscale = ifelse(epoch < 10*window, :identity, :log10))
            else
                nothing
            end
        end
        p = plot(filter(!isnothing, ps)...)
        showplot && !isnothing(p) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making Rician signal plot")
    end
end

####
#### Histogram
####

function fast_hist_1D(y, edges; normalize = nothing)
    #=
    @assert !isempty(y) && length(edges) >= 2
    _y = sort!(copy(y))
    @assert _y[1] >= edges[1]
    h = Histogram((edges,), zeros(Int, length(edges)-1), :left)
    j, done = 1, false
    @inbounds for i = 1:length(_y)
        while _y[i] >= edges[j+1]
            j += 1
            done = (j+1 > length(edges))
            done && break
        end
        done && break
        h.weights[j] += 1
    end
    =#

    # Faster just to call fit(Histogram, ...) in parallel
    hist(w) = Histogram((copy(edges),), w, :left)
    hs = [hist(zeros(Int, length(edges)-1)) for _ in 1:Threads.nthreads()]
    Is = collect(Iterators.partition(1:length(y), div(length(y), Threads.nthreads(), RoundUp)))
    Threads.@threads for I in Is
        @inbounds begin
            yi = view(y, I) # chunk of original y vector
            hi = hs[Threads.threadid()]
            hi = fit(Histogram, yi, UnitWeights{Float64}(length(yi)), hi.edges[1]; closed = :left)
            hs[Threads.threadid()] = hi
        end
    end
    h = hist(sum([hi.weights for hi in hs]))

    !isnothing(normalize) && (h = Plots.normalize(h, mode = normalize))
    return h
end

function fast_hist_1D(y, edges::AbstractRange; normalize = nothing)
    @assert length(edges) >= 2
    lo, hi, dx, n = first(edges), last(edges), step(edges), length(edges)
    hist(w) = Histogram((copy(edges),), w, :left)
    hs = [hist(zeros(Int, n-1)) for _ in 1:Threads.nthreads()]
    Threads.@threads for i in eachindex(y)
        @inbounds begin
            j = 1 + floor(Int, (y[i] - lo) / dx)
            (1 <= j <= n-1) && (hs[Threads.threadid()].weights[j] += 1)
        end
    end
    h = hist(sum([hi.weights for hi in hs]))
    !isnothing(normalize) && (h = Plots.normalize(h, mode = normalize))
    return h
end

function _fast_hist_test()
    _make_hist(x, edges) = fit(Histogram, x, UnitWeights{Float64}(length(x)), edges; closed = :left)
    for _ in 1:100
        n = rand([1:10; 32; 512; 1024])
        x = 100 .* rand(n)
        edges = rand(Bool) ? [0.0; sort(100 .* rand(n))] : (rand(1:10) : rand(1:10) : 100-rand(1:10))
        try
            @assert fast_hist_1D(x, edges) == _make_hist(x, edges)
        catch e
            if e isa InterruptException
                break
            else
                fast_hist_1D(x, edges).weights' |> x -> (display(x); display((first(x), last(x), sum(x))))
                _make_hist(x, edges).weights' |> x -> (display(x); display((first(x), last(x), sum(x))))
                rethrow(e)
            end
        end
    end

    x = rand(10^7)
    for ne in [4, 64, 1024]
        edges = range(0, 1; length = ne)
        @info "range (ne = $ne)"; @btime fast_hist_1D($x, $edges)
        @info "array (ne = $ne)"; @btime fast_hist_1D($x, $(collect(edges)))
        @info "plots (ne = $ne)"; @btime $_make_hist($x, $edges)
    end
end

_ChiSquared(Pi::T, Qi::T) where {T} = ifelse(Pi + Qi <= eps(T), zero(T), (Pi - Qi)^2 / (2 * (Pi + Qi)))
_KLDivergence(Pi::T, Qi::T) where {T} = ifelse(Pi <= eps(T) || Qi <= eps(T), zero(T), Pi * log(Pi / Qi))
ChiSquared(P::Histogram, Q::Histogram) = sum(_ChiSquared.(MMDLearning.unitsum(P.weights), MMDLearning.unitsum(Q.weights)))
KLDivergence(P::Histogram, Q::Histogram) = sum(_KLDivergence.(MMDLearning.unitsum(P.weights), MMDLearning.unitsum(Q.weights)))
CityBlock(P::Histogram, Q::Histogram) = sum(abs, MMDLearning.unitsum(P.weights) .- MMDLearning.unitsum(Q.weights))
Euclidean(P::Histogram, Q::Histogram) = sqrt(sum(abs2, MMDLearning.unitsum(P.weights) .- MMDLearning.unitsum(Q.weights)))

function signal_histograms(Y::AbstractMatrix; nbins = nothing, edges = nothing, normalize = nothing)
    make_edges(x) = ((lo,hi) = extrema(vec(x)); return range(lo, hi; length = nbins)) # mid = median(vec(x)); length = ceil(Int, (hi - lo) * nbins / max(hi - mid, mid - lo))
    hists = Dict{Int, Histogram}()
    hists[0] = fast_hist_1D(vec(Y), isnothing(edges) ? make_edges(Y) : edges[0]; normalize)
    for i in 1:size(Y,1)
        hists[i] = fast_hist_1D(Y[i,:], isnothing(edges) ? make_edges(Y[i,:]) : edges[i]; normalize)
    end
    return hists
end

####
#### Evaluating MRI model
####

function pyheatmap(
        imdata::AbstractMatrix;
        formatter = nothing,
        filename = nothing,
        clim = nothing,
        cticks = nothing,
        title = nothing,
        savetypes = [".png", ".pdf"]
    )

    plt.figure(figsize = (8.0, 8.0), dpi = 150.0)
    plt.set_cmap("plasma")
    fig, ax = plt.subplots()
    img = ax.imshow(imdata, aspect = "equal", interpolation = "nearest")
    plt.title(title)
    ax.set_axis_off()

    (formatter isa Function) && (formatter = plt.matplotlib.ticker.FuncFormatter(formatter))
    cbar = fig.colorbar(img, ticks = cticks, format = formatter, aspect = 40)
    cbar.ax.tick_params(labelsize = 10)

    !isnothing(clim) && img.set_clim(clim...)
    !isnothing(filename) && foreach(ext -> plt.savefig(filename * ext, bbox_inches = "tight", dpi = 150.0), savetypes)
    plt.close("all")

    return nothing
end

# Compute discrete CDF
discrete_cdf(x) = (t = sort(x; dims = 2); c = cumsum(t; dims = 2) ./ sum(t; dims = 2); return permutedims.((t, c))) # return (t[1:12:end, :]', c[1:12:end, :]')

# Unzip array of structs into struct of arrays
unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

function bin_sorted(X, Y; binsize::Int)
    X_sorted, Y_sorted = unzip(sort(collect(zip(X, Y)); by = first))
    X_binned, Y_binned = unzip(map(is -> (mean(X_sorted[is]), mean(Y_sorted[is])), Iterators.partition(1:length(X), binsize)))
    return X_binned, Y_binned
end

function bin_edges(X, Y, edges)
    X_binned, Y_binned = map(1:length(edges)-1) do i
        Is = @. edges[i] <= X <= edges[i+1]
        mean(X[Is]), mean(Y[Is])
    end |> unzip
    return X_binned, Y_binned
end

function eval_mri_model(
        phys::BiexpEPGModel,
        models,
        derived;
        zslices = 24:24,
        naverage = 10,
        savefolder = ".",
        savetypes = [".png"],
        mle_image_path = ".",
        mle_sim_path = ".",
        force_decaes = false,
        force_histograms = false,
        posterior_mode = :maxlikelihood,
        quiet = false,
        dataset = :val, # :val or (for final model comparison) :test
    )

    inverter(Ysamples; kwargs...) = posterior_state(derived["cvae"], derived["prior"], Ysamples; verbose = !quiet, alpha = 0.0, miniter = 1, maxiter = naverage, mode = posterior_mode, kwargs...)
    saveplot(p, name, folder = savefolder) = map(suf -> savefig(p, joinpath(mkpath(folder), name * suf)), savetypes)

    flat_test(x) = flat_indices(x, phys.image[Symbol(dataset, :_indices)])
    flat_train(x) = flat_indices(x, phys.image[:train_indices])
    flat_indices(x, indices) =
        x isa AbstractMatrix ? (@assert(size(x,2) == length(indices)); return x) : # matrix with length(indices) columns
        x isa AbstractTensor4D ?
            (size(x)[1:3] == (length(indices), 1, 1)) ? permutedims(reshape(x, :, size(x,4))) : # flattened 4D array with first three dimensions (length(indices), 1, 1)
            (size(x)[1:3] == size(phys.image[:data])[1:3]) ? permutedims(x[indices,:]) : # 4D array with first three dimensions equal to image size
            error("4D array has wrong shape") :
        error("x must be an $AbstractMatrix or an $AbstractTensor4D")

    flat_image_to_flat_test(x) = flat_image_to_flat_indices(x, phys.image[Symbol(dataset, :_indices)])
    flat_image_to_flat_train(x) = flat_image_to_flat_indices(x, phys.image[:train_indices])
    function flat_image_to_flat_indices(x, indices)
        _x = similar(x, size(x,1), size(phys.image[:data])[1:3]...)
        _x[:, phys.image[:mask_indices]] = x
        return _x[:, indices]
    end

    # Compute decaes on the image data if necessary
    if !haskey(phys.meta, :decaes)
        @info "Recomputing T2 distribution for image data..."
        @time t2_distributions!(phys)
    end

    mle_image_state = let
        mle_image_results = Glob.readdir(Glob.glob"mle-image-mask-results-final-*.mat", mle_image_path) |> last |> DECAES.MAT.matread
        θ = mle_image_results["theta"] |> to32
        ϵ = reshape(exp.(mle_image_results["epsilon"] |> to32), 1, :) #TODO: should save as "logepsilon"
        ℓ = reshape(mle_image_results["loss"] |> to32, 1, :) # negative log-likelihood loss
        X = signal_model(phys, θ)
        ν, δ, Z = X, nothing, nothing
        Y = add_noise_instance(phys, X, ϵ)
        (; Y, θ, Z, X, δ, ϵ, ν, ℓ)
    end

    let
        Y_test = phys.Y[dataset] |> to32
        Y_train = phys.Y[:train] |> to32
        Y_train_edges = Dict([k => v.edges[1] for (k,v) in phys.meta[:histograms][:train]])
        cvae_image_state = inverter(Y_test; maxiter = 1, mode = posterior_mode)

        # Compute decaes on the image data if necessary
        Xs = Dict{Symbol,Dict{Symbol,Any}}()
        Xs[:Y_test]    = Dict(:label => L"Y_{TEST}",       :colour => :grey,   :data => Y_test)
        Xs[:Y_train]   = Dict(:label => L"Y_{TRAIN}",      :colour => :black,  :data => Y_train)
        Xs[:Yhat_mle]  = Dict(:label => L"\hat{Y}_{MLE}",  :colour => :red,    :data => flat_image_to_flat_test(mle_image_state.Y))
        Xs[:Yhat_cvae] = Dict(:label => L"\hat{Y}_{CVAE}", :colour => :blue,   :data => add_noise_instance(derived["ricegen"], cvae_image_state.ν, cvae_image_state.ϵ))
        Xs[:X_decaes]  = Dict(:label => L"X_{DECAES}",     :colour => :orange, :data => flat_test(phys.meta[:decaes][:t2maps][:Y]["decaycurve"]))
        Xs[:X_mle]     = Dict(:label => L"X_{MLE}",        :colour => :green,  :data => flat_image_to_flat_test(mle_image_state.ν))
        Xs[:X_cvae]    = Dict(:label => L"X_{CVAE}",       :colour => :purple, :data => cvae_image_state.ν)

        commonkwargs = Dict{Symbol,Any}(
            # :titlefontsize => 16, :labelfontsize => 14, :xtickfontsize => 12, :ytickfontsize => 12, :legendfontsize => 11,
            :titlefontsize => 10, :labelfontsize => 10, :xtickfontsize => 10, :ytickfontsize => 10, :legendfontsize => 10, #TODO
            :legend => :topright,
        )

        for (key, X) in Xs
            get!(phys.meta[:histograms], :inference, Dict{Symbol, Any}())
            X[:hist] =
                key === :Y_test ? phys.meta[:histograms][dataset] :
                key === :Y_train ? phys.meta[:histograms][:train] :
                (force_histograms || !haskey(phys.meta[:histograms][:inference], key)) ?
                    let
                        @info "Computing signal histogram for $(key) data..."
                        @time signal_histograms(Flux.cpu(X[:data]); edges = Y_train_edges, nbins = nothing)
                    end :
                    phys.meta[:histograms][:inference][key]

            X[:t2dist] =
                key === :Y_test ? flat_test(phys.meta[:decaes][:t2dist][:Y]) :
                key === :Y_train ? flat_train(phys.meta[:decaes][:t2dist][:Y]) :
                key === :X_decaes ? flat_test(phys.meta[:decaes][:t2dist][:Y]) : # decaes signal gives identical t2 distbn by definition, as it consists purely of EPG basis functions
                let
                    if (force_decaes || !haskey(phys.meta[:decaes][:t2maps], key))
                        @info "Computing T2 distribution for $(key) data..."
                        @time t2_distributions!(phys, key => convert(Matrix{Float64}, X[:data]))
                    end
                    flat_test(phys.meta[:decaes][:t2dist][key])
                end

            phys.meta[:histograms][:inference][key] = X[:hist] # update phys metadata
        end

        @info "Plotting histogram distances compared to $dataset data..." # Compare histogram distances for each echo and across all-signal for test data and simulated data
        phist = @time plot(
            map(collect(pairs((; ChiSquared, KLDivergence, CityBlock, Euclidean)))) do (distname, dist)
                echoes = 0:size(phys.image[:data],4)
                Xplots = [X for (k,X) in Xs if k !== :Y_test]
                logdists = mapreduce(hcat, Xplots) do X
                    (i -> log10(dist(X[:hist][i], Xs[:Y_test][:hist][i]))).(echoes)
                end
                plot(
                    echoes, logdists;
                    label = permutedims(getindex.(Xplots, :label)) .* map(x -> L" ($d_0$ = %$(round(x; sigdigits = 3)))", logdists[1:1,:]),
                    line = (2, permutedims(getindex.(Xplots, :colour))), title = string(distname),
                    commonkwargs...,
                )
            end...;
            commonkwargs...,
        )
        saveplot(phist, "signal-hist-distances")

        @info "Plotting T2 distributions compared to $dataset data..."
        pt2dist = @time let
            Xplots = [X for (k,X) in Xs if k !== :X_decaes]
            # Xplots = [Xs[k] for k ∈ (:Y_test, :Y_train, :Yhat_mle, :Yhat_cvae)] #TODO
            T2dists = mapreduce(X -> mean(X[:t2dist]; dims = 2), hcat, Xplots)
            plot(
                1000 .* phys.meta[:decaes][:t2maps][:Y]["t2times"], T2dists;
                label = permutedims(getindex.(Xplots, :label)),
                line = (2, permutedims(getindex.(Xplots, :colour))),
                xscale = :log10,
                xlabel = L"$T_2$ [ms]",
                ylabel = L"$T_2$ Amplitude [a.u.]",
                # title = L"$T_2$-distributions",
                commonkwargs...,
            )
        end
        saveplot(pt2dist, "decaes-T2-distbn")

        @info "Plotting T2 distribution differences compared to $dataset data..."
        pt2diff = @time let
            Xplots = [X for (k,X) in Xs if k ∉ (:X_decaes, :Y_test)]
            # Xplots = [Xs[k] for k ∈ (:Y_train, :Yhat_mle, :Yhat_cvae)] #TODO
            T2diffs = mapreduce(X -> mean(X[:t2dist]; dims = 2) .- mean(Xs[:Y_test][:t2dist]; dims = 2), hcat, Xplots)
            logL2 = log10.(sum(abs2, T2diffs; dims = 1))
            plot(
                1000 .* phys.meta[:decaes][:t2maps][:Y]["t2times"], T2diffs;
                label = permutedims(getindex.(Xplots, :label)) .* L" $-$ " .* Xs[:Y_test][:label] .* map(x -> L" ($\log_{10}\ell_2$ = %$(round(x; sigdigits = 3)))", logL2), #TODO
                line = (2, permutedims(getindex.(Xplots, :colour))),
                ylim = (-0.06, 0.1), #TODO
                xscale = :log10,
                xlabel = L"$T_2$ [ms]",
                # ylabel = L"$T_2$ Amplitude [a.u.]", #TODO
                # title = L"$T_2$-distribution Differences",
                commonkwargs...,
            )
        end
        saveplot(pt2diff, "decaes-T2-distbn-diff")

        @info "Plotting signal distributions compared to $dataset data..."
        psignaldist = @time let
            Xplots = [X for (k,X) in Xs if k !== :X_decaes]
            # Xplots = [Xs[k] for k ∈ (:Y_test, :Y_train, :Yhat_mle, :Yhat_cvae)] #TODO
            p = plot(;
                xlabel = "Signal magnitude [a.u.]",
                ylabel = "Density [a.u.]",
                commonkwargs...
            )
            for X in Xplots
                plot!(p, normalize(X[:hist][0]); alpha = 0.1, label = X[:label], line = (2, X[:colour]), commonkwargs...)
            end
            p
        end
        saveplot(psignaldist, "decaes-signal-distbn")

        @info "Plotting signal distribution differences compared to $dataset data..."
        psignaldiff = @time let
            Xplots = [X for (k,X) in Xs if k ∉ (:X_decaes, :Y_test)]
            # Xplots = [Xs[k] for k ∈ (:Y_train, :Yhat_mle, :Yhat_cvae)] #TODO
            histdiffs = mapreduce(X -> unitsum(X[:hist][0].weights) .- unitsum(Xs[:Y_test][:hist][0].weights), hcat, Xplots)
            plot(
                Xs[:Y_test][:hist][0].edges[1][2:end], histdiffs;
                label = permutedims(getindex.(Xplots, :label)) .* L" $-$ " .* Xs[:Y_test][:label],
                series = :steppost, line = (2, permutedims(getindex.(Xplots, :colour))),
                xlabel = "Signal magnitude [a.u.]",
                ylim = (-0.002, 0.0035), #TODO
                # ylabel = "Density [a.u.]", #TODO
                commonkwargs...,
            )
        end
        saveplot(psignaldiff, "decaes-signal-distbn-diff")

        saveplot(plot(pt2dist, pt2diff; layout = (1,2), commonkwargs...), "decaes-T2-distbn-and-diff")
        saveplot(plot(psignaldist, psignaldiff, pt2dist, pt2diff; layout = (2,2), commonkwargs...), "decaes-distbn-ensemble")

        @info "Plotting signal distributions compared to $dataset data..." # Compare per-echo and all-signal cdf's of test data and simulated data
        pcdf = plot(; commonkwargs...)
        @time for X in [X for (k,X) in Xs if k ∈ (:Y_test, :Yhat_cvae)]
            plot!(pcdf, discrete_cdf(Flux.cpu(X[:data]))...; line = (1, X[:colour]), legend = :none, commonkwargs...)
            plot!(pcdf, discrete_cdf(reshape(Flux.cpu(X[:data]),1,:))...; line = (1, X[:colour]), legend = :none, commonkwargs...)
        end
        saveplot(pcdf, "signal-cdf-compare")
    end

    function θderived_cpu(θ)
        # named tuple of misc. parameters of interest derived from θ
        map(arr64, θderived(phys, θ))
    end

    function infer_θderived(Y)
        @info "Computing named tuple of θ values, averaging over $naverage samples..."
        θ = @time map(_ -> θderived_cpu(inverter(Y; maxiter = 1, mode = posterior_mode).θ), 1:naverage)
        θ = map((θs...,) -> mean(θs), θ...) # mean over each named tuple field
    end

    let
        mle_sim_data = Glob.readdir(Glob.glob"mle-simulated-mask-data-*.mat", mle_sim_path) |> last |> DECAES.MAT.matread
        mle_sim_results = Glob.readdir(Glob.glob"mle-simulated-mask-results-final-*.mat", mle_sim_path) |> last |> DECAES.MAT.matread

        Ytrue, X̂true, Xtrue, θtrue, Ztrue = getindex.(Ref(mle_sim_data), ("Y", "Xhat", "X", "theta", "Z"))
        θtrue_derived = θtrue |> θderived_cpu
        θmle_derived = mle_sim_results["theta"] |> θderived_cpu

        @info "Computing DECAES inference error..."
        if (force_decaes || !haskey(phys.meta[:decaes][:t2maps], :Yhat_cvae_decaes))
            @time t2_distributions!(phys, :Yhat_cvae_decaes => convert(Matrix{Float64}, X̂true))
        end
        @info 0, :decaes, [
            L"\alpha" => mean(abs, θtrue_derived.alpha - vec(phys.meta[:decaes][:t2maps][:Yhat_cvae_decaes]["alpha"])),
            L"\bar{T}_2" => mean(abs, θtrue_derived.T2bar - vec(phys.meta[:decaes][:t2maps][:Yhat_cvae_decaes]["ggm"])),
            L"T_{2,SGM}" => mean(abs, filter(!isnan, θtrue_derived.T2sgm - vec(phys.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["sgm"]))), # "sgm" is set to NaN if all T2 components within SPWin are zero; be generous with error measurement
            L"T_{2,MGM}" => mean(abs, filter(!isnan, θtrue_derived.T2mgm - vec(phys.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["mgm"]))), # "mgm" is set to NaN if all T2 components within MPWin are zero; be generous with error measurement
            L"MWF" => mean(abs, filter(!isnan, θtrue_derived.mwf - vec(phys.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["sfr"]))),
        ]

        @info "Computing MLE inference error..."
        @info 0, :mle, [lab =>round(mean(abs, θt - θi); sigdigits = 3) for (lab, θt, θi) in zip(θderivedlabels(phys), θtrue_derived, θmle_derived)]

        for mode in [:mean], maxiter in [1,2,5]
            inf_time = @elapsed cvae_state = inverter(X̂true |> to32; maxiter, mode, verbose = false)
            θcvae_derived = cvae_state.θ |> θderived_cpu
            @info maxiter, mode, inf_time, [lab => round(mean(abs, θt - θi); sigdigits = 3) for (lab, θt, θi) in zip(θderivedlabels(phys), θtrue_derived, θcvae_derived)]
        end
    end

    # Heatmaps
    Y = phys.image[:data][:,:,zslices,:] # (nx, ny, nslice, nTE)
    Islices = findall(!isnan, Y[..,1]) # entries within Y mask
    Imaskslices = filter(I -> I[3] ∈ zslices, phys.image[:mask_indices])
    makemaps(x) = (out = fill(NaN, size(Y)[1:3]); out[Islices] .= Flux.cpu(x); return permutedims(out, (2,1,3)))

    θcvae = infer_θderived(permutedims(Y[Islices,:]) |> to32)
    θmle = θderived_cpu(flat_image_to_flat_indices(mle_image_state.θ, Imaskslices))

    # DECAES heatmaps
    @time let
        θdecaes = (
            alpha   = (phys.meta[:decaes][:t2maps][:Y]["alpha"], L"\alpha",     (50.0, 180.0)),
            T2bar   = (phys.meta[:decaes][:t2maps][:Y]["ggm"],   L"\bar{T}_2",  (0.0, 0.25)),
            T2sgm   = (phys.meta[:decaes][:t2parts][:Y]["sgm"],  L"T_{2,SGM}",  (0.0, 0.1)),
            T2mgm   = (phys.meta[:decaes][:t2parts][:Y]["mgm"],  L"T_{2,MGM}",  (0.0, 1.0)),
            mwf     = (phys.meta[:decaes][:t2parts][:Y]["sfr"],  L"MWF",        (0.0, 0.4)),
        )
        for (θname, (θk, θlabel, θbd)) in pairs(θdecaes), (j,zj) in enumerate(zslices)
            pyheatmap(permutedims(θk[:,:,zj]); title = θlabel * " (slice $zj)", clim = θbd, filename = joinpath(mkpath(joinpath(savefolder, "decaes")), "$θname-$zj"), savetypes)
        end
    end

    # CVAE and MLE heatmaps
    for (θfolder, θ) ∈ [:cvae => θcvae, :mle => θmle]
        @info "Plotting heatmap plots for mean θ values..."
        @time let
            for (k, ((θname, θk), θlabel, θbd)) in enumerate(zip(pairs(θ), θderivedlabels(phys), θderivedbounds(phys)))
                θmaps = makemaps(θk)
                for (j,zj) in enumerate(zslices)
                    pyheatmap(θmaps[:,:,j]; title = θlabel * " (slice $zj)", clim = θbd, filename = joinpath(mkpath(joinpath(savefolder, string(θfolder))), "$θname-$zj"), savetypes)
                end
            end
        end

        @info "Plotting T2-distribution over test data..."
        @time let
            T2 = 1000 .* vcat(θ.T2short, θ.T2long)
            A = vcat(θ.Ashort, θ.Along)
            p = plot(
                # bin_edges(T2, A, exp.(range(log.(phys.T2bd)...; length = 100)))...;
                bin_sorted(T2, A; binsize = 100)...;
                label = "T2 Distribution", ylabel = "T2 Amplitude [a.u.]", xlabel = "T2 [ms]",
                xscale = :log10, xlim = 1000 .* phys.T2bd, xticks = 10 .^ (0.5:0.25:3),
            )
            saveplot(p, "T2distbn-$(zslices[1])-$(zslices[end])", joinpath(savefolder, string(θfolder)))
        end
    end
end

function mle_mri_model(
        phys::BiexpEPGModel,
        models,
        derived;
        data_source   = :image, # One of :image, :simulated
        data_subset   = :mask,  # One of :mask, :val, :train, :test
        batch_size    = 128 * Threads.nthreads(),
        initial_iter  = 10,
        verbose       = true,
        checkpoint    = false,
        dryrun        = false,
        dryrunsamples = dryrun ? batch_size : nothing,
        opt_alg       = :LD_SLSQP, # Rough algorithm ranking: [:LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG, :LD_MMA] (Note: :LD_LBFGS fails to converge with tolerance looser than ~ 1e-4)
        opt_args      = Dict{Symbol,Any}(),
    )
    @assert data_source ∈ (:image, :simulated)
    @assert data_subset ∈ (:mask, :val, :train, :test)

    # MLE for whole image of simulated data
    Y, Yc = let
        image_ind  = phys.image[Symbol(data_subset, :_indices)]
        image_data = phys.image[:data][image_ind,:] |> to32 |> permutedims
        if data_source === :image
            !dryrun && DECAES.MAT.matwrite("mle-$data_source-$data_subset-data-$(getnow()).mat", Dict{String,Any}("Y" => arr64(image_data)))
            arr64(image_data), image_data
        else # data_source === :simulated
            X, θ, Z = sampleXθZ(derived["cvae"], derived["prior"], image_data; posterior_θ = true, posterior_Z = true)
            X̂       = sampleX̂(derived["ricegen"], X, Z)
            !dryrun && DECAES.MAT.matwrite("mle-$data_source-$data_subset-data-$(getnow()).mat", Dict{String,Any}("X" => arr64(X), "theta" => arr64(θ), "Z" => arr64(Z), "Xhat" => arr64(X̂), "Y" => arr64(image_data)))
            arr64(X̂), X̂
        end
    end
    !isnothing(dryrunsamples) && (I = sample(MersenneTwister(0), 1:size(Y,2), dryrunsamples; replace = dryrunsamples > size(Y,2)); Y = Y[:,I]; Yc = Yc[:,I])

    initial_guess = posterior_state(
        derived["cvae"],
        derived["prior"],
        Yc;
        miniter = 1,
        maxiter = initial_iter,
        alpha   = 0.0,
        verbose = verbose,
        mode    = :maxlikelihood,
    )
    initial_guess = map(arr64, initial_guess)

    logϵlo, logϵhi = log.(extrema(vec(initial_guess.ϵ))) .|> Float64
    initial_logϵ   = log.(vec(mean(initial_guess.ϵ; dims = 1))) |> arr64
    lower_bounds   = [θlower(phys); logϵlo] |> arr64
    upper_bounds   = [θupper(phys); logϵhi] |> arr64

    #= Test random initial guess
    if initial_iter == 0
        initial_guess.θ .= sampleθprior(phys, size(Y,2)) |> arr64
        initial_logϵ    .= logϵlo .+ (logϵhi - logϵlo) .* rand(size(Y,2)) |> arr64
    end
    =#

    work_spaces = map(1:Threads.nthreads()) do _
        (
            Y   = zeros(nsignal(phys)),
            x0  = zeros(ntheta(phys) + 1),
            epg = BiexpEPGModelWork(phys),
            opt = let
                opt = NLopt.Opt(opt_alg, ntheta(phys) + 1)
                opt.lower_bounds  = lower_bounds
                opt.upper_bounds  = upper_bounds
                opt.xtol_rel      = 1e-8
                opt.ftol_rel      = 1e-8
                opt.maxeval       = 250
                opt.maxtime       = 1.0
                for (k,v) in opt_args
                    setproperty!(opt, k, v)
                end
                opt
            end,
        )
    end

    results = (
        theta     = fill(NaN, ntheta(phys), size(Y,2)),
        epsilon   = fill(NaN, size(Y,2)),
        loss      = fill(NaN, size(Y,2)),
        retcode   = fill(NaN, size(Y,2)),
        numevals  = fill(NaN, size(Y,2)),
        solvetime = fill(NaN, size(Y,2)),
    )

    function f(work, x::Vector{Float64})
        @inbounds begin
            nθ = ntheta(phys)
            θ  = ntuple(i -> x[i], nθ)
            ϵ  = exp(x[nθ+1])
            ψ  = θsignalmodel(phys, θ...)
            X  = _signal_model_f64(phys, work.epg, ψ)
            μX = -Inf
            for i in eachindex(X)
                μX = max(μX, _rician_mean_cuda(X[i], ϵ)) # Signal normalization factor
            end
            ℓ  = 0.0
            ϵ  = ϵ / μX # Hoist outside loop
            for i in eachindex(X)
                ℓ -= _rician_logpdf_cuda(work.Y[i], X[i] / μX, ϵ) # Rician negative log likelihood
            end
            return ℓ
        end
    end

    function fg!(work, x::Vector{Float64}, g::Vector{Float64})
        if length(g) > 0
            simple_fd_gradient!(g, y -> f(work, y), x, lower_bounds, upper_bounds)
        else
            f(work, x)
        end
    end

    #= Benchmarking
    work_spaces[1].Y .= rand(nsignal(phys))
    @info "Timing function..."; @btime( $f( $( work_spaces[1] ), $( (lower_bounds .+ upper_bounds) ./ 2 ) ) ) |> display
    @info "Timing gradient..."; @btime( $fg!( $( work_spaces[1] ), $( (lower_bounds .+ upper_bounds) ./ 2 ), $( zeros(ntheta(phys) + 1) ) ) ) |> display
    =#

    start_time = time()
    batches = Iterators.partition(1:size(Y,2), batch_size)
    LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    @time for (batchnum, batch) in enumerate(batches)
        batchtime = @elapsed Threads.@sync for j in batch
            Threads.@spawn @inbounds begin
                work = work_spaces[Threads.threadid()]
                work.Y                .= Y[:,j]
                work.x0[1:end-1]      .= initial_guess.θ[:,j]
                work.x0[end]           = initial_logϵ[j]
                work.opt.min_objective = (x, g) -> fg!(work, x, g)

                solvetime              = @elapsed (minf, minx, ret) = NLopt.optimize(work.opt, work.x0)
                results.theta[:,j]    .= minx[1:end-1]
                results.epsilon[j]     = minx[end]
                results.loss[j]        = minf
                results.retcode[j]     = Base.eval(NLopt, ret) |> Int #TODO cleaner way to convert Symbol to enum?
                results.numevals[j]    = work.opt.numevals
                results.solvetime[j]   = solvetime
            end
        end

        # Checkpoint results
        elapsed_time   = time() - start_time
        remaining_time = (elapsed_time / batchnum) * (length(batches) - batchnum)
        mle_per_second = batch[end] / elapsed_time
        savetime       = @elapsed !dryrun && checkpoint && DECAES.MAT.matwrite("mle-$data_source-$data_subset-results-checkpoint-$(getnow()).mat", Dict{String,Any}(string(k) => copy(v) for (k,v) in pairs(results))) # checkpoint progress
        verbose && @info "$batchnum / $(length(batches))" *
            " -- batch: $(DECAES.pretty_time(batchtime))" *
            " -- elapsed: $(DECAES.pretty_time(elapsed_time))" *
            " -- remaining: $(DECAES.pretty_time(remaining_time))" *
            " -- save: $(round(savetime; digits = 2))s" *
            " -- rate: $(round(mle_per_second; digits = 2))Hz" *
            " -- initial loss: $(round(mean(initial_guess.ℓ[1,1:batch[end]]); digits = 2))" *
            " -- loss: $(round(mean(results.loss[1:batch[end]]); digits = 2))"
    end

    !dryrun && DECAES.MAT.matwrite("mle-$data_source-$data_subset-results-final-$(getnow()).mat", Dict{String,Any}(string(k) => copy(v) for (k,v) in pairs(results))) # save final results
    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return initial_guess, results
end

function simple_fd_gradient!(g, f, x, lo = nothing, hi = nothing)
    δ = cbrt(eps(float(eltype(x))))
    f₀ = f(x)
    @inbounds for i in 1:length(x)
        x₀ = x[i]
        if !isnothing(lo) && (x₀ - δ/2 <= lo[i]) # near LHS boundary; use second-order forward: (-3 * f(x) + 4 * f(x + δ/2) - f(x + δ)) / δ
            x[i] = x₀ + δ/2
            f₊   = f(x)
            x[i] = x₀ + δ
            f₊₊  = f(x)
            g[i] = (-3f₀ + 4f₊ - f₊₊)/δ
        elseif !isnothing(hi) && (x₀ + δ/2 >= hi[i]) # near RHS boundary; use second-order backward: (3 * f(x) - 4 * f(x - δ/2) + f(x - δ)) / δ
            x[i] = x₀ - δ/2
            f₋   = f(x)
            x[i] = x₀ - δ
            f₋₋  = f(x)
            g[i] = (3f₀ - 4f₋ + f₋₋)/δ
        else # safely within boundary; use second-order central: (f(x + δ/2) - f(x - δ/2)) / δ
            x[i] = x₀ - δ/2
            f₋   = f(x)
            x[i] = x₀ + δ/2
            f₊   = f(x)
            g[i] = (f₊ - f₋)/δ
        end
        x[i] = x₀
    end
    return f₀
end
