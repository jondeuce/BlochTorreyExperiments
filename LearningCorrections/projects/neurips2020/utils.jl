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
#### PyPlot
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
