####
#### Settings
####

using DrWatson: @quickactivate
@quickactivate "Playground"
using Playground
lib.initenv()

lib.settings_template() = TOML.parse(
"""
[data]
    image_folders = [
        "2019-10-28_48echo_8msTE_CPMG",
        "2019-09-22_56echo_7msTE_CPMG",
        "2021-05-07_NeurIPS2021_64echo_10msTE_MockBiexpEPG_CPMG",
    ]

[train]
    timeout     = 1e9
    epochs      = 50_000
    batchsize   = 256   # 1024
    nbatches    = 100   # number of batches per epoch
    [train.labels]
        simulated     = false
        pseudo        = true
        mcmc          = false
        mle           = false
        pretrained    = false
    [train.augment]
        mask          = 32 # Randomly zero CVAE training signals starting from the `mask`th echo (i.e. Y[i+1:end] .= 0 where i >= `mask`; if `mask` <= 0, no masking is done)

[eval]
    batchsize        = 10240 # batch size for evaluation
    nbatches         = 1     # number of val batches for each image
    stepsaveperiod   = 60.0
    valsaveperiod    = 120.0
    trainsaveperiod  = 120.0
    printperiod      = 120.0
    checkpointperiod = 600.0

[opt]
    lr       = 1e-4    # 3.16e-4 # Initial learning rate
    lrthresh = 1e-6    # Absolute minimum learning rate
    lrdrop   = 3.16    # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    lrrate   = 5_000   # 10_000 # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    gclip    = 0.0     # Gradient clipping
    wdecay   = 0.0     # Weight decay
    [opt.cvae]
        INHERIT = "%PARENT%"
        gclip          = 0.0
        lambda_vae     = 0.0 # Weighting of vae decoder regularization loss on simulated signals
        lambda_latent  = 1.0 # Weighting of latent space regularization

[arch]
    posterior = "TruncatedGaussian" # "TruncatedGaussian", "Gaussian", "Kumaraswamy"
    nlatent   = 0   # number of latent variables Z
    zdim      = 12  # embedding dimension of z
    hdim      = 512 # size of hidden layers
    skip      = false # skip connection
    layernorm = false # layer normalization following dense layer
    nhidden   = 2   # number of hidden layers
    [arch.enc1]
        INHERIT = "%PARENT%"
    [arch.enc2]
        INHERIT = "%PARENT%"
    [arch.dec]
        INHERIT = "%PARENT%"
    [arch.vae_dec]
        INHERIT = "%PARENT%"
        regtype = "Rician" # VAE regularization ("L1", "Gaussian", "Rician", or "None")
"""
)

_DEBUG_ = false
_WANDB_ = false
settings = lib.load_settings(force_new_settings = true)
wandb_logger = lib.init_wandb_logger(settings; activate = _WANDB_, dryrun = false, wandb_dir = lib.projectdir())

lib.set_logdirname!()
lib.clear_checkpointdir!()
lib.set_checkpointdir!(lib.projectdir("log", "2021-05-13-T-18-23-18-980"))
# derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = only(Glob.glob("run-*-1p14e3na/files", lib.projectdir("wandb"))), modelprefix = "best-")

lib.@save_expression lib.logdir("build_physics.jl") function build_physics()
    isdefined(Main, :phys) ? Main.phys : lib.load_epgmodel_physics()
end

lib.@save_expression lib.logdir("build_models.jl") function build_models(
        phys,
        settings,
        models = Dict{String,Any}(),
        derived = Dict{String,Any}(),
    )
    kws(keys...) = lib.make_kwargs(settings, keys...)

    lib.initialize!(phys; seed = 0, kws("data")...)
    get!(models, "enc1") do; lib.init_mlp_cvae_enc1(phys; kws("arch", "enc1")...); end
    get!(models, "enc2") do; lib.init_mlp_cvae_enc2(phys; kws("arch", "enc2")...); end
    get!(models, "dec") do; lib.init_mlp_cvae_dec(phys; kws("arch", "dec")...); end
    get!(models, "vae_dec") do; lib.init_mlp_cvae_vae_dec(phys; kws("arch", "vae_dec")...); end

    derived["cvae"] = lib.derived_cvae(phys, models["enc1"], models["enc2"], models["dec"]; kws("arch")...)
    derived["vae_reg_loss"] = lib.VAEReg(models["vae_dec"]; regtype = settings["arch"]["vae_dec"]["regtype"])

    # # Estimate mle labels using maximum likelihood estimation from pretrained cvae
    # derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = lib.checkpointdir(), modelprefix = "best-")
    # lib.compute_mle_labels!(phys, derived["pretrained_cvae"]; force_recompute = true)
    # lib.verify_mle_labels(phys)

    # # Estimate mle labels using random initial guess form the prior
    # lib.compute_mle_labels!(phys; force_recompute = false)
    # lib.verify_mle_labels(phys)

    # load mcmc labels, if they exist
    lib.load_mcmc_labels!(phys; force_reload = true)
    lib.verify_mcmc_labels(phys)

    # Initial pseudo labels
    lib.initialize_pseudo_labels!(phys; labelset = :mcmc)
    lib.verify_pseudo_labels(phys)

    return models, derived
end

lib.@save_expression lib.logdir("build_optimizers.jl") function build_optimizers(
        phys,
        settings,
        models,
        optimizers = Dict{String,Any}(),
    )
    kws(keys...) = lib.make_kwargs(settings, keys...)
    optimizers["cvae"] = lib.init_optimizer(Flux.ADAM; kws("opt", "cvae")...)
    return optimizers
end

phys = build_physics()
models, derived = build_models(phys, settings)
# models, derived = build_models(phys, settings, lib.load_checkpoint("current-models.jld2"))
# models, derived = build_models(phys, settings, lib.load_checkpoint("failure-models.jld2"))
optimizers = build_optimizers(phys, settings, models)
lib.save_snapshot(settings, models)

####
#### CVAE
####

# Conditional variational autoencoder losses
function CVAElosses(Y, θ)
    λ_vae_sim  = Zygote.@ignore settings["opt"]["cvae"]["lambda_vae"]::Float64 |> eltype(Y)
    λ_latent   = Zygote.@ignore settings["opt"]["cvae"]["lambda_latent"]::Float64 |> eltype(Y)
    minkept    = Zygote.@ignore settings["train"]["augment"]["mask"]::Int

    # Cross-entropy loss function components
    Ymasked, Ymask = lib.pad_and_mask_signal(Y, lib.nsignal(derived["cvae"]); minkept = minkept, maxkept = lib.nsignal(derived["cvae"]))
    Ystate = lib.CVAETrainingState(derived["cvae"], Ymasked, θ)
    KLDiv, ELBO = lib.KL_and_ELBO(Ystate)
    ℓ = (; KLDiv, ELBO)

    if λ_latent > 0
        # Regularize latent variables such that their distribution *across the dataset* tends toward unit normal
        Reg = lib.EnsembleKLDivUnitGaussian(Ystate.μq0, Ystate.logσq)
        ℓ = push!!(ℓ, :LatentReg => λ_latent * Reg)
    end

    if λ_vae_sim > 0
        # Regularize latent variables using a variational autoencoder penalty, i.e. train a VAE to be able to reconstruct the input signal from the latent space representation
        zq  = lib.sample_mv_normal(Ystate.μq0, exp.(Ystate.logσq))
        VAE = derived["vae_reg_loss"](lib.signal(Ystate), Ymask, zq)
        ℓ   = push!!(ℓ, :VAE => λ_vae_sim * VAE)
    end

    return ℓ
end

function train_step(engine, batch)
    outputs = Dict{Any,Any}()
    trainer.should_terminate && return outputs

    @timeit "sample batch" begin
        @unpack img_idx, Y, θ = sample_batch(batch; dataset = :train, batchsize = settings["train"]["batchsize"]::Int)
    end

    # Model parameters
    ps = Flux.params(models["enc1"], models["enc2"], models["dec"], models["vae_dec"])

    # Forward pass, logging metrics
    @timeit "forward" ℓ, back = Zygote.pullback(ps) do
        ℓs = CVAElosses(Y, θ)
        Zygote.@ignore let
            for (k,v) in pairs(ℓs)
                outputs["$k"] = v
            end
            outputs["CVAE"] = sum(ℓs)
        end
        return sum(ℓs)
    end

    # Reverse pass
    @timeit "reverse" gs = back(one(ℓ))

    # Save model and abort in case of NaN and/or Inf parameters and/or gradients
    _DEBUG_ && @timeit "failure check" let
        lib.on_bad_params_or_gradients(models, ps, gs) do
            batchdata = Dict("img_idx" => img_idx, "Y" => lib.cpu(Y), "θ" => lib.cpu(θ))
            lib.save_progress(@dict(models, logger, batchdata); savefolder = lib.logdir(), prefix = "failure-", ext = ".jld2")
            engine.terminate()
        end
        trainer.should_terminate && return outputs
    end

    # Update parameters
    @timeit "update!" Flux.Optimise.update!(optimizers["cvae"], ps, gs)

    return outputs
end

####
#### Training
####

# Global logger
logger = DataFrame(
    :epoch      => Int[],
    :iter       => Int[],
    :dataset    => Symbol[],
    :img_idx    => Int[],
    :time       => Float64[],
)

tensor_dataset(x) = torch.utils.data.TensorDataset(lib.j2p_array(x))
image_indices_dataset(nbatches, use_simulated = true) = repeat(!use_simulated : length(phys.images), nbatches) |> tensor_dataset
train_loader = torch.utils.data.DataLoader(image_indices_dataset(settings["train"]["nbatches"], settings["train"]["labels"]["simulated"]); shuffle = true)
val_eval_loader = torch.utils.data.DataLoader(image_indices_dataset(settings["eval"]["nbatches"]); shuffle = false)
train_eval_loader = torch.utils.data.DataLoader(image_indices_dataset(settings["eval"]["nbatches"]); shuffle = false)

function sample_batch(batch; dataset::Symbol, batchsize::Int)

    # Batch contains an index into the list of images
    img_idx = batch isa Int ? batch : only(lib.p2j_array(only(batch))) |> Int

    if img_idx == 0
        # Sentinel value signaling to use simulated data, generated from θ sampled from the prior
        img = img_cols = Ymeta = nothing
        θ = lib.sampleθprior(phys, CuMatrix{Float32}, batchsize)
        X = lib.signal_model(phys, θ)
        Y = lib.add_noise_instance(phys, X, θ)
        Ymax = maximum(Y; dims = 1)
        Y ./= Ymax
        X ./= Ymax
        θ[7:7, ..] .= clamp.(θ[7:7, ..] .- log.(Ymax), -2.5f0, 2.5f0)

    else
        # Sample signals from the image
        img = phys.images[img_idx]
        Y, img_cols = lib.sample_columns(img.partitions[dataset], batchsize; replace = false)
        Y = Y |> to32
        Ymeta = lib.MetaCPMGSignal(phys, img, Y)

        if settings["train"]["labels"]["pseudo"]::Bool
            # Train using pseudo labels from the Metropolis-Hastings sampler
            @unpack mh_sampler = img.meta[:pseudo_labels][dataset]
            X, θ, _, _ = lib.update!(mh_sampler, phys, derived["cvae"], Ymeta, img_cols)

        elseif settings["train"]["labels"]["mcmc"]::Bool
            # Train using labels drawn from precomputed MCMC chains
            θ = img.meta[:mcmc_labels][dataset][:theta][:, img_cols, rand(1:end)] |> to32
            X = lib.signal_model(phys, Ymeta, θ)

        elseif settings["train"]["labels"]["mle"]::Bool
            # Train using precomputed MLE labels
            θ = img.meta[:mle_labels][dataset][:theta][:, img_cols] |> to32
            X = img.meta[:mle_labels][dataset][:signalfit][:, img_cols] |> to32

        elseif settings["train"]["labels"]["pretrained"]::Bool
            # Train using pseudo labels drawn from a pretrained CVAE
            @assert haskey(derived, "pretrained_cvae")
            θ, _ = lib.sampleθZ(phys, derived["pretrained_cvae"], Ymeta; posterior_θ = true, posterior_Z = true, posterior_mode = false)
            X = lib.signal_model(phys, Ymeta, θ)

        else
            error("No labels chosen")
        end
    end

    return out = (; img, img_idx, img_cols, Ymeta, Y, X, θ)
end

function compute_metrics(engine, batch; dataset)
    outputs = Dict{Any,Any}()
    trainer.should_terminate && return outputs

    @timeit "sample batch" begin
        @unpack img, img_idx, img_cols, Ymeta, Y, X, θ = sample_batch(batch; dataset = dataset, batchsize = settings["eval"]["batchsize"]::Int)
    end

    # Update callback state
    metrics = Dict{Symbol,Any}()
    metrics[:epoch]   = trainer.state.epoch
    metrics[:iter]    = trainer.state.iteration
    metrics[:dataset] = dataset
    metrics[:img_idx] = img_idx
    metrics[:time]    = time()

    # Inference using CVAE
    @timeit "sample cvae" begin
        θ′, _ = lib.sampleθZposterior(derived["cvae"], Y)
        X′ = img_idx == 0 ? lib.signal_model(phys, θ′) : lib.signal_model(phys, Ymeta, θ′)
        X′ = lib.clamp_dim1(Y, X′)
        # Y′ = add_noise_instance(phys, X′, θ′)
    end

    # CVAE losses
    @timeit "cvae losses" let
        ℓs = CVAElosses(Y, θ)
        metrics[:CVAE] = sum(ℓs)
        for (k,v) in pairs(ℓs)
            metrics[k] = v
        end
    end

    # Goodness of fit metrics for CVAE posterior samples
    @timeit "goodness of fit metrics" let
        metrics[:CVAE_rmse] = sqrt(mean(abs2, Y .- X′))
        metrics[:CVAE_logL] = mean(lib.negloglikelihood(phys, Y, X′, θ′))
    end

    # Error metrics w.r.t true labels
    img_idx === 0 && @timeit "true label metrics" let
        # Record cdf distance metrics
        θwidths = lib.θupper(phys) .- lib.θlower(phys)
        θerrs   = abs.(θ .- θ′) |> lib.cpu
        for (i, lab) in enumerate(lib.θasciilabels(phys))
            metrics[Symbol("$(lab)_err")] = 100 * θerrs[i] / θwidths[i]
        end
    end

    # Pseudo label metrics (note: these are over whole validation sets, not just this batch)
    img_idx !== 0 && @timeit "pseudo label metrics" let
        # Record current negative log likelihood of pseudo labels
        @unpack mh_sampler = img.meta[:pseudo_labels][dataset]
        neglogPXθ = mh_sampler.neglogPXθ[:, lib.buffer_indices(mh_sampler)]
        metrics[:Pseudo_logL] = mean(filter(!isinf, vec(neglogPXθ)))

        # Record cdf distance metrics
        θwidths  = lib.θupper(phys) .- lib.θlower(phys)
        θ_mcmc   = img.meta[:mcmc_labels][dataset][:theta]
        θ_pseudo = mh_sampler.θ
        dists    = lib.cdf_distance((1, 2), θ_mcmc, θ_pseudo)
        dists    = mean(reinterpret(DECAES.SVector{2,eltype(θ_mcmc)}, dists); dims = 2) # reinterpret since mean over tuples isn't defined, but mean over SVector's is
        ℓ₁       = (x->x[1]).(dists) ./ θwidths # scale to range [0,1]
        ℓ₂       = (x->x[2]).(dists) ./ sqrt.(θwidths) # scale to range [0,1]
        for (i, lab) in enumerate(lib.θasciilabels(phys))
            metrics[Symbol("$(lab)_L1")] = ℓ₁[i]
            metrics[Symbol("$(lab)_L2")] = ℓ₂[i]
        end
    end

    # Update logger dataframe
    push!(logger, metrics; cols = :union)

    # Return metrics for W&B logging
    for (k,v) in metrics
        k ∈ [:epoch, :iter, :dataset, :time] && continue # output non-housekeeping metrics
        ismissing(v) && continue # return non-missing metrics (wandb cannot handle missing)
        outputs["$k"] = v
    end

    return outputs
end

function make_plots(;showplot = false)
    trainer.should_terminate && return Dict{Symbol,Any}()
    try
        plots  = Dict{String, Any}()
        groups = DataFrames.groupby(logger, [:dataset, :img_idx]) |> pairs
        for (group_key, group) in groups
            ps = Any[]
            cols_iterator = group[!, DataFrames.Not([:epoch, :iter, :dataset, :time, :img_idx])] |> eachcol |> pairs |> collect
            for (colname, col) in cols_iterator
                I = findall(x -> !ismissing(x) && !isnan(x) && !isinf(x), col)
                isempty(I) && continue
                epochs, col = group.epoch[I], col[I]
                p = plot(epochs, col; label = :none, title = "$colname", titlefontsize = 10)
                !isempty(epochs) && first(epochs) >= 100 && Plots.xaxis!(p, xlims = (100, Inf))
                !isempty(epochs) && last(epochs) >= 1000 && Plots.xaxis!(p, scale = :log10)
                push!(ps, p)
            end
            plots["$(group_key.dataset)-losses-$(group_key.img_idx)"] = plot(ps...)
        end
        return plots
    catch e
        lib.handle_interrupt(e; msg = "Error plotting")
    end
end

trainer = ignite.engine.Engine(@j2p (args...) -> @timeit "train step" train_step(args...))
trainer.logger = ignite.utils.setup_logger("trainer")

val_evaluator = ignite.engine.Engine(@j2p (args...) -> @timeit "val metrics" compute_metrics(args...; dataset = :val))
val_evaluator.logger = ignite.utils.setup_logger("val_evaluator")

train_evaluator = ignite.engine.Engine(@j2p (args...) -> @timeit "train metrics" compute_metrics(args...; dataset = :train))
train_evaluator.logger = ignite.utils.setup_logger("train_evaluator")

####
#### Events
####

Events = ignite.engine.Events

# Force terminate
trainer.add_event_handler(
    Events.STARTED | Events.ITERATION_STARTED | Events.ITERATION_COMPLETED,
    @j2p lib.terminate_file_event(file = lib.logdir("stop"))
)

# Timeout
trainer.add_event_handler(
    Events.EPOCH_COMPLETED(event_filter = @j2p lib.timeout_event_filter(settings["train"]["timeout"])),
    @j2p function (engine)
        @info "Exiting: training time exceeded $(DECAES.pretty_time(settings["train"]["timeout"]))"
        engine.terminate()
    end
)

# Compute callback metrics
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["valsaveperiod"])),
    @j2p (engine) -> val_evaluator.run(val_eval_loader)
)
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["trainsaveperiod"])),
    @j2p (engine) -> train_evaluator.run(train_eval_loader)
)

# Checkpoint current model + logger + make plots
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["checkpointperiod"])),
    @j2p function (engine)
        @timeit "checkpoint" let models = lib.cpu(models)
            lib.on_bad_params_or_gradients(engine.terminate, models) && return nothing
            @timeit "save current model" lib.save_progress(@dict(models, logger); savefolder = lib.logdir(), prefix = "current-", ext = ".jld2")
            @timeit "make current plots" plothandles = make_plots()
            @timeit "save current plots" lib.save_plots(plothandles; savefolder = lib.logdir(), prefix = "current-")
        end
    end
)

# Check for + save best model + logger + make plots
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["checkpointperiod"])),
    @j2p function (engine)
        loss_metric = :CVAE
        losses = logger[logger.dataset .=== :val, loss_metric] |> skipmissing |> collect
        if !isempty(losses) && (length(losses) == 1 || losses[end] < minimum(losses[1:end-1]))
            @timeit "save best progress" let models = lib.cpu(models)
                lib.on_bad_params_or_gradients(engine.terminate, models) && return nothing
                @timeit "save best model" lib.save_progress(@dict(models, logger); savefolder = lib.logdir(), prefix = "best-", ext = ".jld2")
                @timeit "make best plots" plothandles = make_plots()
                @timeit "save best plots" lib.save_plots(plothandles; savefolder = lib.logdir(), prefix = "best-")
            end
        end
    end
)

# Drop learning rate
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p lib.droplr_file_event(optimizers;
        file = lib.logdir("droplr"),
        lrrate = settings["opt"]["lrrate"]::Int,
        lrdrop = settings["opt"]["lrdrop"]::Float64,
        lrthresh = settings["opt"]["lrthresh"]::Float64,
    )
)

# User input
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p(lib.file_event(lib.user_input_event(); file = lib.logdir("user"))),
)

# Print TimerOutputs timings
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["printperiod"])),
    @j2p function (engine)
        @info sprint() do io
            println(io, "Log folder: $(lib.logdir())"); println(io, "\n")
            show(io, TimerOutputs.get_defaulttimer()); println(io, "\n")
            show(io, last(logger[!,[names(logger)[1:4]; sort(names(logger)[5:end])]], 10)); println(io, "\n")
        end
        (engine.state.epoch == 1) && TimerOutputs.reset_timer!() # throw out compilation timings
    end
)

# Reset logging/timer/etc.
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p lib.file_event(file = lib.logdir("reset")) do engine
        TimerOutputs.reset_timer!()
        empty!(cb_state)
        empty!(logger)
    end
)

####
#### Weights & biases logger
####

# Attach training/validation output handlers
(wandb_logger !== nothing) && for (tag, engine, event_name) in [
        (tag = "step",  engine = trainer,         event_name = Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["stepsaveperiod"]))), # computed each iteration; throttle recording
        (tag = "train", engine = train_evaluator, event_name = Events.EPOCH_COMPLETED), # throttled above; record every epoch
        (tag = "val",   engine = val_evaluator,   event_name = Events.EPOCH_COMPLETED), # throttled above; record every epoch
    ]
    wandb_logger.attach_output_handler(
        engine;
        tag = tag,
        event_name = event_name,
        output_transform = @j2p(metrics -> metrics),
        global_step_transform = @j2p((args...; kwargs...) -> trainer.state.epoch),
    )
end

####
#### Run trainer
####

TimerOutputs.reset_timer!()
trainer.run(train_loader, settings["train"]["epochs"])
(wandb_logger !== nothing) && wandb.run.finish()
