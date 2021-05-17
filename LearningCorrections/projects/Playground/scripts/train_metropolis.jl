####
#### Settings
####

using DrWatson: @quickactivate
@quickactivate "Playground"
using Playground
lib.initenv()
Plots.gr()

lib.settings_template() = TOML.parse(
"""
[data]
    image_folders = [
        "2019-10-28_48echo_8msTE_CPMG",
        "2019-09-22_56echo_7msTE_CPMG",
        "2021-05-07_NeurIPS2021_64echo_10msTE_MockBiexpEPG_CPMG",
    ]
    [data.labels]
        train_indices     = [0]          # image folders to use for training (0 = simulated data generated on the fly w/ CVAE trained using true labels)
        eval_indices      = [0, 1, 2, 3] # image folders to use for evaluation (0 = simulated data generated on the fly)
        image_labelset    = "pseudo"     # label set used for the images is one of "pseudo", precomputed "mcmc", or "cvae"
        initialize_pseudo = "prior"      # initialize pseudo labels from "prior" or precomputed "mcmc"

[train]
    timeout   = 1e9
    epochs    = 5_000
    batchsize = 1024
    nbatches  = 1_000 # number of batches per epoch
    [train.augment]
        mask  = 32 # Randomly zero CVAE training signals starting from the `mask`th echo (i.e. Y[i+1:end] .= 0 where i >= `mask`; if `mask` <= 0, no masking is done)

[eval]
    batchsize          = 8192
    evalmetricsperiod  = 300.0
    checkpointperiod   = 600.0

[opt]
    lr       = 1e-4  # 3.16e-4 # Initial learning rate
    lrthresh = 1e-6  # Minimum learning rate
    lrdrop   = 3.16  # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    lrrate   = 1_000 # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    gclip    = 0.0   # Gradient clipping
    wdecay   = 0.0   # Weight decay
    [opt.cvae]
        INHERIT       = "%PARENT%"
        gclip         = 0.0
        lambda_vae    = 0.0 # Weighting of vae decoder regularization loss on simulated signals
        lambda_latent = 1.0 # Weighting of latent space regularization

[arch]
    posterior   = "TruncatedGaussian" # "TruncatedGaussian", "Gaussian", "Kumaraswamy"
    zdim        = 12    # embedding dimension of z
    hdim        = 512   # size of hidden layers
    skip        = false # skip connection
    layernorm   = false # layer normalization following dense layer
    nhidden     = 2     # number of internal hidden layers, i.e. there are `nhidden + 2` total `Dense` layers
    nlatent     = 0     # number of marginalized latent variables Z
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

settings = lib.load_settings(force_new_settings = true)
wandb_logger = lib.init_wandb_logger(settings; dryrun = false, wandb_dir = lib.projectdir())

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

    lib.initialize!(phys; seed = 0, image_folders = settings["data"]["image_folders"])
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

    # Load true labels
    lib.load_true_labels!(phys; force_reload = false)
    lib.verify_true_labels(phys)

    # Load mcmc labels
    lib.load_mcmc_labels!(phys; force_reload = false)
    lib.verify_mcmc_labels(phys)

    # Initial pseudo labels
    lib.initialize_pseudo_labels!(phys; force_recompute = true, labelset = settings["data"]["labels"]["initialize_pseudo"])
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

# Global logger
logger = DataFrame(
    :epoch      => Int[],
    :iter       => Int[],
    :time       => Float64[],
    :dataset    => Symbol[],
    :img_idx    => Int[],
)

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
    Ystate         = lib.CVAETrainingState(derived["cvae"], Ymasked, θ)
    KLDiv, ELBO    = lib.KL_and_ELBO(Ystate)
    ℓ = (; KLDiv, ELBO)

    if λ_latent > 0
        # Regularize latent variables such that their distribution *across the dataset* tends toward unit normal
        Reg = lib.EnsembleKLDivUnitGaussian(Ystate.μq0, Ystate.logσq)
        ℓ   = push!!(ℓ, :LatentReg => λ_latent * Reg)
    end

    if λ_vae_sim > 0
        # Regularize latent variables using a variational autoencoder penalty, i.e. train a VAE to be able to reconstruct the input signal from the latent space representation
        zq  = lib.sample_mv_normal(Ystate.μq0, exp.(Ystate.logσq))
        VAE = derived["vae_reg_loss"](lib.signal(Ystate), Ymask, zq)
        ℓ   = push!!(ℓ, :VAE => λ_vae_sim * VAE)
    end

    return ℓ
end

# Update image pseudolabels
function update_pseudolabels(; dataset::Symbol)
    for img in phys.images
        @unpack mh_sampler = img.meta[:pseudo_labels][dataset]
        lib.update!(mh_sampler, phys, derived["cvae"], img; dataset, gpu_batch_size = 32768)
    end
end

####
#### Training
####

function sample_batch(batch; dataset::Symbol, batchsize::Int, signalfit::Bool = false)

    # Batch contains an index into the list of images
    img_idx = batch isa Int ? batch : only(lib.p2j_array(batch)) |> Int

    if img_idx == 0
        # Sentinel value signaling to use simulated data, generated from θ sampled from the prior
        img = img_cols = Ymeta = nothing
        θ = lib.sampleθprior(phys, Matrix{Float32}, batchsize) # signal_model is faster on cpu; move to gpu afterward
        X = lib.signal_model(phys, θ) # faster on cpu; move to gpu afterward
        θ = θ |> to32
        X = X |> to32
        Y = lib.add_noise_instance(phys, X, θ)

        # Normalize Y, X, θ for CVAE
        Ymax = maximum(Y; dims = 1)
        Y ./= Ymax
        X ./= Ymax
        θ[7:7, ..] .= clamp.(θ[7:7, ..] .- log.(Ymax), lib.θlower(phys)[7], lib.θupper(phys)[7])

    else
        # Sample signals from the image
        img   = phys.images[img_idx]
        Y_cpu, img_cols = lib.sample_columns(img.partitions[dataset], batchsize; replace = false)
        Y     = Y_cpu |> to32
        Ymeta = lib.MetaCPMGSignal(phys, img, Y)

        if settings["data"]["labels"]["image_labelset"]::String == "pseudo"
            # Use pseudo labels from the Metropolis-Hastings sampler
            @unpack mh_sampler = img.meta[:pseudo_labels][dataset]
            let
                θ′, _      = lib.sampleθZposterior(derived["cvae"], Y)
                θ′_cpu     = θ′ |> lib.cpu
                X′_cpu     = lib.signal_model(phys, img, θ′_cpu)
                neglogPXθ′ = lib.negloglikelihood(phys, Y_cpu, X′_cpu, θ′_cpu)
                neglogPθ′  = lib.neglogprior(phys, θ′_cpu)
                lib.update!(mh_sampler, θ′_cpu, neglogPXθ′, neglogPθ′, img_cols)
            end
            θ = mh_sampler.θ[:, lib.buffer_indices(mh_sampler, img_cols)]
            X = !signalfit ? nothing : lib.signal_model(phys, θ) # faster on cpu; move to gpu afterward
            θ = θ |> to32
            X = !signalfit ? nothing : (X |> to32)

        elseif settings["data"]["labels"]["image_labelset"]::String == "mcmc"
            # Use labels drawn from precomputed MCMC chains
            θ = img.meta[:mcmc_labels][dataset][:theta][:, img_cols, rand(1:end)]
            X = !signalfit ? nothing : lib.signal_model(phys, img, θ) # faster on cpu; move to gpu afterward
            θ = θ |> to32
            X = !signalfit ? nothing : (X |> to32)

        elseif settings["data"]["labels"]["image_labelset"]::String == "mle"
            # Use precomputed MLE labels
            θ = img.meta[:mle_labels][dataset][:theta][:, img_cols] |> to32
            X = !signalfit ? nothing : img.meta[:mle_labels][dataset][:signalfit][:, img_cols] |> to32

        elseif (cvae_key = settings["data"]["labels"]["image_labelset"]::String) ∈ ("cvae", "pretrained_cvae")
            # Use pseudo labels drawn from a pretrained CVAE
            @assert haskey(derived, cvae_key)
            θ, _ = lib.sampleθZposterior(derived[cvae_key], signal(Ymeta))
            X = !signalfit ? nothing : lib.signal_model(phys, img, θ) # theta already on gpu

        else
            error("No labels chosen")
        end
    end

    return (; img, img_idx, img_cols, Ymeta, Y, X, θ)
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
                outputs["$(k)_$(img_idx)"] = v # tag output metric with image index
            end
            outputs["CVAE_$(img_idx)"] = sum(ℓs) # tag output metric with image index
        end
        return sum(ℓs)
    end

    # Reverse pass
    @timeit "reverse" gs = back(one(ℓ))

    # Save model and abort in case of NaN and/or Inf parameters and/or gradients
    ENV["JL_TRAIN_DEBUG"] == "1" && @timeit "failure check" let
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

function compute_metrics(engine, batch; dataset::Symbol)
    outputs = Dict{Any,Any}()
    trainer.should_terminate && return outputs

    @timeit "sample batch" begin
        @unpack img, img_idx, img_cols, Ymeta, Y, X, θ = sample_batch(batch; dataset = dataset, batchsize = settings["eval"]["batchsize"]::Int)
    end

    # Initialize metrics
    metrics = Dict{Symbol,Any}()
    metrics[:epoch]   = trainer.state.epoch
    metrics[:iter]    = trainer.state.iteration
    metrics[:time]    = time()
    metrics[:dataset] = dataset
    metrics[:img_idx] = img_idx

    # helper function for cdf distances
    function mcmc_cdf_distances(θ₁, θ₂)
        dists    = lib.cdf_distance((1, 2), θ₁, θ₂) # array of tuples of ℓ₁ and ℓ₂ distances between the empirical cdf's
        dists    = mean(reinterpret(DECAES.SVector{2,eltype(eltype(dists))}, dists); dims = 2) # mean over SVector's is defined, but not over Tuple's, hence the reinterpret
        θ_widths = lib.θupper(phys) .- lib.θlower(phys)
        ℓ₁       = (x->x[1]).(dists) ./ θ_widths # scale to range [0,1]
        ℓ₂       = (x->x[2]).(dists) ./ sqrt.(θ_widths) # scale to range [0,1]
        return (L1 = ℓ₁, L2 = ℓ₂)
    end

    # Update Metropolis-Hastings pseudo labels
    (img_idx != 0) && @timeit "update pseudo labels" begin
        @unpack mh_sampler = img.meta[:pseudo_labels][dataset]
        lib.update!(mh_sampler, phys, derived["cvae"], img; dataset, gpu_batch_size = 32768)
    end

    # Inference using CVAE
    @timeit "sample cvae" begin
        θ′_sampler = lib.θZposterior_sampler(derived["cvae"], Y)
        θ′_mode, _ = θ′_sampler(mode = true)
        X′_mode    = img_idx != 0 ? lib.signal_model(phys, img, θ′_mode) : lib.signal_model(phys, θ′_mode)
        X′_mode    = lib.clamp_dim1(Y, X′_mode)
        Y′         = lib.add_noise_instance(phys, X′_mode, θ′_mode)
        θ′_samples = zeros(Float32, size(θ′_mode)..., 100)
        for i in 1:size(θ′_samples, 3)
            local θ′, _ = θ′_sampler()
            θ′_samples[:,:,i] .= lib.cpu(θ′)
        end
    end

    # MLE using CVAE mode as initial guess
    (img_idx != 0) && @timeit "compute mle" begin
        mle_init   = (; Y = lib.arr64(Y), θ = lib.arr64(θ′_mode))
        _, mle_res = lib.mle_biexp_epg(phys, img; initial_guess = mle_init, batch_size = Colon(), verbose = false)
        metrics[:logL_MLE] = mean(mle_res.loss)
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
    @timeit "cvae goodness of fit" let
        metrics[:rmse_CVAE] = sqrt(mean(abs2, Y .- X′_mode))
        metrics[:logL_CVAE] = mean(lib.negloglikelihood(phys, Y, X′_mode, θ′_mode))
    end

    # CVAE parameter cdf distances w.r.t. ground truth mcmc
    (img_idx != 0) && @timeit "cvae cdf distances" let
        θ_mcmc  = img.meta[:mcmc_labels][dataset][:theta]
        θ_dists = mcmc_cdf_distances(θ_mcmc[:, img_cols, :], θ′_samples)
        for (i, lab) in enumerate(lib.θasciilabels(phys)), (ℓ_name, ℓ) in pairs(θ_dists)
            metrics[Symbol("$(lab)_$(ℓ_name)_CVAE")] = ℓ[i]
        end

        θ_dists = mcmc_cdf_distances(mh_sampler.θ[:, img_cols, :], θ′_samples)
        for (i, lab) in enumerate(lib.θasciilabels(phys)), (ℓ_name, ℓ) in pairs(θ_dists)
            metrics[Symbol("$(lab)_$(ℓ_name)_Self")] = ℓ[i]
        end
    end

    # Pseudo label metrics (note: these are over whole validation sets, not just this batch)
    (img_idx != 0) && @timeit "pseudo label metrics" let
        # Record current negative log likelihood of pseudo labels
        neglogPXθ = mh_sampler.neglogPXθ[:, lib.buffer_indices(mh_sampler)]
        metrics[:logL_Pseudo] = mean(filter(!isinf, vec(neglogPXθ)))

        # Record cdf distance metrics
        θ_mcmc  = img.meta[:mcmc_labels][dataset][:theta]
        θ_dists = mcmc_cdf_distances(θ_mcmc, mh_sampler.θ)
        for (i, lab) in enumerate(lib.θasciilabels(phys)), (ℓ_name, ℓ) in pairs(θ_dists)
            metrics[Symbol("$(lab)_$(ℓ_name)_Pseudo")] = ℓ[i]
        end
    end

    # Error metrics w.r.t true labels
    (img_idx == 0 || haskey(img.meta, :true_labels)) && @timeit "true label metrics" let
        θ_true = img_idx == 0 ? lib.cpu(θ) : img.meta[:true_labels][dataset][:theta][:, img_cols]
        lib.θ_errs_dict!(metrics, phys, θ_true, lib.cpu(θ′_mode); suffix = "CVAE_mode") # error of CVAE posterior mode
        lib.θ_errs_dict!(metrics, phys, θ_true, dropdims(mean(θ′_samples; dims = 3); dims = 3); suffix = "CVAE_mean") # error of CVAE posterior mean
        if img_idx != 0
            lib.θ_errs_dict!(metrics, phys, θ_true, mle_res.theta .|> Float32; suffix = "CVAE_mle") # error of MLE initialized with CVAE posterior sample
            lib.θ_errs_dict!(metrics, phys, θ_true, dropdims(mean(mh_sampler.θ[:, img_cols, :]; dims = 3); dims = 3); suffix = "Pseudo_mean") # error of mean pseudo labels
        end
    end

    # Update logger dataframe
    push!(logger, metrics; cols = :union)

    # Return metrics for W&B logging
    for (k,v) in metrics
        k ∈ [:epoch, :iter, :time, :dataset] && continue # output non-housekeeping metrics
        ismissing(v) && continue # return non-missing metrics (wandb cannot handle missing)
        outputs["$(k)_$(img_idx)"] = v # tag output metric with image index
    end

    return outputs
end

function make_plots(; showplot::Bool = false)
    trainer.should_terminate && return Dict{Symbol,Any}()
    plots  = Dict{String, Any}()
    groups = DataFrames.groupby(logger, [:dataset, :img_idx]) |> pairs
    try
        for (group_key, group) in groups
            ps = Any[]
            cols_iterator = group[!, DataFrames.Not([:epoch, :iter, :time, :dataset, :img_idx])] |> eachcol |> pairs |> collect |> iter -> sort(iter; by = first)
            for (colname, col) in cols_iterator
                I = (x -> !ismissing(x) && !isnan(x) && !isinf(x)).(col)
                I .&= max(first(group.epoch), min(100, last(group.epoch)-100)) .<= group.epoch
                !any(I) && continue
                epochs, col = group.epoch[I], col[I]
                p = Plots.plot(
                    epochs, col;
                    label = :none, title = "$colname",
                    titlefontsize = 10, xtickfontsize = 8, ytickfontsize = 8,
                    xlims = (min(first(epochs), 100), last(epochs)),
                    xscale = last(epochs) < 1000 ? :identity : :log10,
                    xformatter = x -> round(Int, x),
                )
                if group_key.img_idx != 0 && contains(String(colname), "_err_")
                    img              = phys.images[group_key.img_idx]
                    mcmc_errs_dict   = lib.recursive_try_get(img.meta, [:mcmc_labels, group_key.dataset, :theta_errs])
                    lab, _           = split(String(colname), "_err_")
                    mcmc_errs_dict !== missing && Plots.hline!(p, [mcmc_errs_dict[Symbol(lab * "_err_MCMC_mean")]]; colour = :red, label = :none)
                    mcmc_errs_dict !== missing && Plots.hline!(p, [mcmc_errs_dict[Symbol(lab * "_err_MCMC_mle")]]; colour = :green, label = :none)
                end
                push!(ps, p)
            end
            p = Plots.plot(ps...; size = max.((800,600), ceil(sqrt(length(ps))) .* (240,180)))
            showplot && display(p)
            plots["$(group_key.dataset)-losses-$(group_key.img_idx)"] = p
        end
        return plots
    catch e
        lib.handle_interrupt(e; msg = "Error plotting")
    end
end

####
#### Data loaders, callbacks, and events
####

function data_loader(; loader_type::Symbol)
    # Batch sampling is done on the julia side, but the image which is sampled from is chosen by Ignite
    if loader_type === :train
        train_nbatches = settings["train"]["nbatches"]::Int
        train_indices  = settings["data"]["labels"]["train_indices"]::Vector{Int}
        sampler        = torch.utils.data.RandomSampler(collect(0:length(train_indices)-1), replacement = true, num_samples = train_nbatches)
        torch.utils.data.DataLoader(train_indices, sampler = sampler)
    elseif loader_type === :eval
        eval_indices   = settings["data"]["labels"]["eval_indices"]::Vector{Int}
        sampler        = torch.utils.data.SequentialSampler(collect(0:length(eval_indices)-1))
        torch.utils.data.DataLoader(eval_indices, sampler = sampler)
    else
        error("loader_type must be :train or :eval, got: $(repr(loader_type))")
    end
end

train_loader = data_loader(; loader_type = :train)
val_eval_loader = data_loader(; loader_type = :eval)
train_eval_loader = data_loader(; loader_type = :eval)

trainer = ignite.engine.Engine(@j2p (args...) -> @timeit "train step" train_step(args...))
trainer.logger = ignite.utils.setup_logger("trainer")

val_evaluator = ignite.engine.Engine(@j2p (args...) -> @timeit "val metrics" compute_metrics(args...; dataset = :val))
val_evaluator.logger = ignite.utils.setup_logger("val_evaluator")

train_evaluator = ignite.engine.Engine(@j2p (args...) -> @timeit "train metrics" compute_metrics(args...; dataset = :train))
train_evaluator.logger = ignite.utils.setup_logger("train_evaluator")

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

# Evaluator events
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["evalmetricsperiod"])),
    @j2p (engine) -> val_evaluator.run(val_eval_loader)
)

trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["evalmetricsperiod"])),
    @j2p (engine) -> train_evaluator.run(train_eval_loader)
)

# Checkpoint current model + logger + make plots
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["checkpointperiod"])),
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
    Events.TERMINATE | Events.EPOCH_COMPLETED,
    @j2p function (engine)
        is_best = false
        for (group_key, group) in pairs(DataFrames.groupby(logger[!, [:dataset, :img_idx, :CVAE]], [:dataset, :img_idx]))
            group_key.dataset === :val || continue
            is_best |= length(group.CVAE) >= 2 && group.CVAE[end] < minimum(group.CVAE[1:end-1])
        end
        if is_best
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
        file     = lib.logdir("droplr"),
        lrrate   = settings["opt"]["lrrate"]::Int,
        lrdrop   = settings["opt"]["lrdrop"]::Float64,
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
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["evalmetricsperiod"])),
    @j2p function (engine)
        @info "Log folder: $(lib.logdir())"
        show(TimerOutputs.get_defaulttimer()); println("")
        show_all_cols = false
        start_cols    = [:epoch, :iter, :time, :dataset, :img_idx, :CVAE, :KLDiv, :ELBO, :LatentReg, :VAE, :logL_CVAE, :logL_MLE, :logL_Pseudo]
        start_cols    = filter(c -> String(c) ∈ names(logger), start_cols)
        show_order    = [start_cols; sort([Symbol(col) for col in names(logger) if Symbol(col) ∉ start_cols])]
        display(last(logger[!, (show_all_cols ? show_order : start_cols)], 12)); println("")
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
        (tag = "step",  engine = trainer,         event_name = Events.EPOCH_COMPLETED),
        (tag = "train", engine = train_evaluator, event_name = Events.ITERATION_COMPLETED),
        (tag = "val",   engine = val_evaluator,   event_name = Events.ITERATION_COMPLETED),
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
