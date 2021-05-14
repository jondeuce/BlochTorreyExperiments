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
    timeout     = 1e9 #TODO 10800.0
    epochs      = 50_000
    batchsize   = 256   # 1024
    nbatches    = 100   # number of batches per epoch
    MMDCVAErate = 0     # Train combined MMD+CVAE loss every `MMDCVAErate` epochs
    CVAErate    = 1     # Train CVAE loss every `CVAErate` iterations
    CVAEsteps   = 1     # Train CVAE losses with `CVAEsteps` updates per iteration
    CVAEmask    = 32    # Randomly mask cvae training signals up to `CVAEmask` echoes (<=0 performs no masking)
    MMDrate     = 0     # Train MMD loss every `MMDrate` epochs
    GANrate     = 0     # Train GAN losses every `GANrate` iterations
    Dsteps      = 5     # Train GAN losses with `Dsteps` discrim updates per genatr update
    kernelrate  = 0     # Train kernel every `kernelrate` iterations
    kernelsteps = 0     # Gradient updates per kernel train
    DeepThetaPrior  = false # Learn deep prior
    DeepLatentPrior = false # Learn deep prior
    [train.augment]
        signal        = true  # Plain input signal
        gradient      = false # Gradient of input signal (1D central difference)
        laplacian     = false # Laplacian of input signal (1D second order)
        fdcat         = 0     # Concatenated finite differences up to order `fdcat`
        encoderspace  = false # Discriminate encoder space representations
        residuals     = false # Discriminate residual vectors
        fftcat        = false # Fourier transform of input signal, concatenating real/imag
        fftsplit      = false # Fourier transform of input signal, treating real/imag separately
    [train.transform]
        flipsignals   = false # Randomly reverse signals
        chunk         = 0     # Random chunks of size `chunk` (0 uses whole signal)
        nsamples      = 1     # Average over `nsamples` instances of corrected signals

[eval]
    batchsize        = 10240 # batch size for evaluation
    nbatches         = 2     # number of eval batches
    stepsaveperiod   = 60.0
    valsaveperiod    = 120.0
    trainsaveperiod  = 120.0
    printperiod      = 120.0
    checkpointperiod = 600.0

[opt]
    lr       = 1e-4    # 3.16e-4 # Initial learning rate
    lrthresh = 1e-6    # Absolute minimum learning rate
    lrdrop   = 3.16    # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    lrrate   = 10_000  # 5_000   # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    gclip    = 0.0     # Gradient clipping
    wdecay   = 0.0     # Weight decay
    [opt.cvae]
        INHERIT = "%PARENT%"
        gclip               = 0.0
        lambda_vae_sim      = 0.0 # Weighting of vae decoder regularization loss on simulated signals
        lambda_vae_data     = 0.0 # Weighting of vae decoder regularization loss on real signals
        lambda_latent       = 1.0 # Weighting of latent space regularization
        lambda_pseudo       = 1.0 # Weighting of pseudo label loss
        tau_lambda_pseudo   = 0.0 # Time constant for `lambda_pseudo` factor (units of iterations)
        delta_lambda_pseudo = 0.0 # Time delay for `lambda_pseudo` factor (units of iterations)
        tau_cvae            = 0.0 # Time constant for CVAE moving average (units of iterations)
    [opt.genatr]
        INHERIT = "%PARENT%" #TODO: 0.01 train generator more slowly
    [opt.discrim]
        INHERIT = "%PARENT%"
    [opt.mmd]
        INHERIT = "%PARENT%"
        gclip           = 1.0
        lambda_0        = 0.0 # MMD loss weighting relative to CVAE
        lambda_eps      = 0.0 # Regularize noise amplitude epsilon
        lambda_deps_dz  = 0.0 # Regularize gradient of epsilon w.r.t. latent variables
    [opt.kernel]
        INHERIT = "%PARENT%" # Kernel learning rate 
        loss  = "mmd"        # Kernel loss ("mmd", "tstatistic", or "mmd_diff")

[arch]
    posterior = "TruncatedGaussian" # "TruncatedGaussian", "Gaussian", "Kumaraswamy"
    nlatent   = 0   # number of latent variables Z
    zdim      = 12  # embedding dimension of z
    hdim      = 512 # size of hidden layers
    skip      = false # skip connection
    layernorm = false # layer normalization following dense layer
    nhidden   = 2   # number of hidden layers
    esize     = 32  # transformer input embedding size
    nheads    = 4   # number of attention heads
    headsize  = 16  # projection head size
    seqlength = 64  # number of signal projection shards
    qseqlength= 16  # number of query tokens for Perceiver
    share     = false # share parameters between recurrent Perceiver layers
    [arch.enc1]
        INHERIT = "%PARENT%"
    [arch.enc2]
        INHERIT = "%PARENT%"
    [arch.dec]
        INHERIT = "%PARENT%"
    [arch.vae_dec]
        INHERIT = "%PARENT%"
        regtype = "Rician" # VAE regularization ("L1", "Gaussian", "Rician", or "None")
    [arch.genatr]
        INHERIT     = "%PARENT%"
        hdim        = 64   #TODO
        nhidden     = 2    #TODO
        ktheta      = 16   #TODO Dimension of domain of theta prior space
        klatent     = 4    #TODO Dimension of domain of latent prior space
        prior_mix   = 0.0  #TODO Mix (possibly learned) genatr prior with `prior_mix` fraction of default prior
        leakyslope  = 0.0
        maxcorr     = 0.1
        noisebounds = [-11.51292546497023, -2.302585092994046] # (natural-)log noise amplitude bounds; equivalent to 20 <= SNR <= 100, where log(eps) = -(SNR/20)*log(10)
    [arch.discrim]
        INHERIT     = "%PARENT%"
        hdim      = 0     #TODO
        nhidden   = 0     #TODO
        dropout   = 0.1
    [arch.kernel]
        INHERIT      = "%PARENT%"
        nbandwidth   = 32            #TODO
        channelwise  = false         #TODO
        embeddingdim = 0             #TODO
        bwbounds     = [-8.0, 4.0]   # Bounds for kernel bandwidths (logsigma)
        clampnoise   = 0.0           #TODO
"""
)

_DEBUG_ = true
_WANDB_ = false
settings = lib.load_settings(force_new_settings = true)
wandb_logger = lib.init_wandb_logger(settings; activate = _WANDB_, dryrun = false, wandb_dir = lib.projectdir())

lib.set_logdirname!()
lib.clear_checkpointdir!()
lib.set_checkpointdir!(lib.projectdir("log", "2021-05-13-T-18-23-18-980"))

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

    # Estimate mle labels using maximum likelihood estimation from pretrained cvae
    derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = lib.checkpointdir(), modelprefix = "best-")
    lib.compute_mle_labels!(phys, derived["pretrained_cvae"]; force_recompute = false)
    lib.verify_mle_labels(phys)

    # # Estimate mle labels using random initial guess form the prior
    # lib.compute_mle_labels!(phys; force_recompute = false)
    # lib.verify_mle_labels(phys)

    # load mcmc labels, if they exist
    lib.load_mcmc_labels!(phys; force_reload = false)
    lib.verify_mcmc_labels(phys)

    # Initial pseudo labels
    lib.initialize_pseudo_labels!(phys; labelset = :prior)
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
    optimizers["genatr"] = lib.init_optimizer(Flux.ADAM; kws("opt", "genatr")...)
    return optimizers
end

phys = build_physics()
models, derived = build_models(phys, settings)
# models, derived = build_models(phys, settings, lib.load_checkpoint("current-models.jld2"))
# models, derived = build_models(phys, settings, lib.load_checkpoint("failure-models.jld2"))
optimizers = build_optimizers(phys, settings, models)
lib.save_snapshot(settings, models)

####
#### Augmentations
####

function augment_and_transform(Xs::AbstractArray...)
end

####
#### CVAE
####

# Conditional variational autoencoder losses
function CVAElosses(Ymeta::lib.AbstractMetaDataSignal, θPseudo = nothing)
    λ_vae_sim  = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "lambda_vae_sim", 0.0)::Float64)
    λ_vae_data = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "lambda_vae_data", 0.0)::Float64)
    λ_latent   = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "lambda_latent", 0.0)::Float64)
    λ_pseudo   = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "lambda_pseudo", 0.0)::Float64)
    τ_pseudo   = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "tau_lambda_pseudo", 0.0)::Float64)
    δ_pseudo   = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "delta_lambda_pseudo", 0.0)::Float64)
    minkept    = Zygote.@ignore get!(settings["train"], "CVAEmask", 0)::Int

    X̂, θ = Zygote.@ignore let
        # Sample X̂,θ from priors
        θ = lib.sampleθprior(phys, lib.signal(Ymeta))
        X = lib.signal_model(phys, Ymeta, θ)
        X̂ = lib.add_noise_instance(phys, X, θ)
        X̂, θ
    end

    # Cross-entropy loss function components
    X̂masked, X̂mask = lib.pad_and_mask_signal(X̂, lib.nsignal(derived["cvae"]); minkept, maxkept = lib.nsignal(derived["cvae"]))
    X̂state = lib.CVAETrainingState(derived["cvae"], X̂masked, θ)
    KLDiv, ELBO = lib.KL_and_ELBO(X̂state)
    ℓ = (; KLDiv, ELBO)

    if λ_vae_sim > 0
        ℓ = push!!(ℓ, :VAE => λ_vae_sim * derived["vae_reg_loss"](lib.signal(X̂state), X̂mask, lib.sample_mv_normal(X̂state.μq0, exp.(X̂state.logσq))))
    end
    if λ_latent > 0
        ℓ = push!!(ℓ, :LatentReg => λ_latent * lib.EnsembleKLDivUnitGaussian(X̂state.μq0, X̂state.logσq))
    end
    #=
    ℓ = (; KLDiv = zero(eltype(Ymeta)), ELBO = zero(eltype(Ymeta)))
    =#

    if λ_pseudo > 0 && θPseudo !== nothing
        # CVAE pseudo-state
        Ymasked, Ymask = lib.pad_and_mask_signal(lib.signal(Ymeta), lib.nsignal(derived["cvae"]); minkept, maxkept = lib.nsignal(Ymeta))
        Ystate = lib.CVAETrainingState(derived["cvae"], Ymasked, θPseudo)

        # CVAE loss from pseudo labels
        KLDivPseudo, ELBOPseudo = lib.KL_and_ELBO(Ystate) # recover pseudo θ
        ℓ = push!!(ℓ, :KLDivPseudo => λ_pseudo * KLDivPseudo, :ELBOPseudo => λ_pseudo * ELBOPseudo)

        if λ_vae_data > 0
            ℓ = push!!(ℓ, :VAEPseudo => λ_pseudo * λ_vae_data * derived["vae_reg_loss"](lib.signal(Ystate), Ymask, lib.sample_mv_normal(Ystate.μq0, exp.(Ystate.logσq))))
        end
        if λ_latent > 0
            ℓ = push!!(ℓ, :LatentRegPseudo => λ_pseudo * λ_latent * lib.EnsembleKLDivUnitGaussian(Ystate.μq0, Ystate.logσq))
        end
    end

    return ℓ
end

####
#### Training
####

# Global state
cb_state = Dict{String,Any}()
logger = DataFrame(
    :epoch      => Int[], # mandatory field
    :iter       => Int[], # mandatory field
    :dataset    => Symbol[], # mandatory field
    :time       => Union{Float64, Missing}[],
)

make_dataset_indices(n) = torch.utils.data.TensorDataset(lib.j2p_array(collect(1:n)))
train_loader = torch.utils.data.DataLoader(make_dataset_indices(settings["train"]["nbatches"]))
val_eval_loader = torch.utils.data.DataLoader(make_dataset_indices(settings["eval"]["nbatches"]))
train_eval_loader = torch.utils.data.DataLoader(make_dataset_indices(settings["eval"]["nbatches"]))

function sample_batch(dataset::Symbol; batchsize::Int, img_idx = nothing)
    (img_idx === nothing) && (img_idx = rand(1:length(phys.images)))
    img = phys.images[img_idx]
    Y, J = lib.sample_columns(img.partitions[dataset], batchsize; replace = false)
    Y = Y |> to32
    Ymeta = lib.MetaCPMGSignal(phys, img, Y)
    if true
        XPseudo, θPseudo, _, _ = lib.update!(img.meta[:pseudo_labels][dataset][:mh_sampler], phys, derived["cvae"], Ymeta, J)
    elseif false #haskey(derived, "pretrained_cvae")
        θPseudo, _ = lib.sampleθZ(phys, derived["pretrained_cvae"], Ymeta; posterior_θ = true, posterior_Z = true, posterior_mode = false)
        XPseudo = lib.signal_model(phys, Ymeta, θPseudo)
    elseif false
        θPseudo = img.meta[:mle_labels][dataset][:theta][:,J] |> to32
        XPseudo = img.meta[:mle_labels][dataset][:signalfit][:,J] |> to32
    else
        θPseudo = XPseudo = nothing
    end
    return out = (; img_idx, img, Y, Ymeta, XPseudo, θPseudo)
end

function train_step(engine, batch)
    trainer.should_terminate && return Dict{Any,Any}() #TODO

    @unpack img_idx, img, Y, Ymeta, θPseudo = sample_batch(:train; batchsize = settings["train"]["batchsize"])
    outputs = Dict{Any,Any}()

    @timeit "train batch" begin #TODO CUDA.@sync
        every(rate) = rate <= 0 ? false : mod(engine.state.iteration-1, rate) == 0
        train_MMDCVAE = every(settings["train"]["MMDCVAErate"]::Int)
        train_CVAE = every(settings["train"]["CVAErate"]::Int)
        train_MMD = every(settings["train"]["MMDrate"]::Int)
        train_GAN = train_discrim = train_genatr = every(settings["train"]["GANrate"]::Int)
        train_k = every(settings["train"]["kernelrate"]::Int)

        # Train CVAE loss
        train_CVAE && @timeit "cvae" let #TODO CUDA.@sync
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"], models["vae_dec"])
            for _ in 1:settings["train"]["CVAEsteps"]
                @timeit "forward" ℓ, back = Zygote.pullback(() -> sum(CVAElosses(Ymeta, θPseudo)), ps) #TODO CUDA.@sync
                @timeit "reverse" gs = back(one(ℓ)) #TODO CUDA.@sync
                if _DEBUG_
                    lib.on_bad_params_or_gradients(models, ps, gs) do
                        batchdata = Dict(
                            "img_idx" => img_idx,
                            "Y" => lib.cpu(Y),
                            "θ" => lib.cpu(θPseudo),
                        )
                        lib.save_progress(@dict(models, logger, batchdata); savefolder = lib.logdir(), prefix = "failure-", ext = ".jld2")
                        engine.terminate()
                    end
                end
                @timeit "update!" Flux.Optimise.update!(optimizers["cvae"], ps, gs) #TODO CUDA.@sync
                outputs["CVAE"] = ℓ
            end
        end
    end

    return deepcopy(outputs)
end

function cvae_posterior(Ymeta; kwargs...)
    return lib.posterior_state(phys, derived["cvae"], Ymeta; kwargs...)
end

function fit_metrics(Ymeta, Ymeta_fit_state, θtrue)
    @unpack X̂, θ, X, ℓ = Ymeta_fit_state
    all_rmse = sqrt.(mean(abs2, lib.signal(Ymeta) .- X; dims = 1)) |> lib.cpu |> vec |> copy
    all_logL = ℓ |> lib.cpu |> vec |> copy
    rmse, logL = mean(all_rmse), mean(all_logL)
    theta_err = (θtrue === nothing) ? missing : mean(abs, lib.θerror(phys, θtrue, θ); dims = 2) |> lib.cpu |> vec |> copy

    @timeit "mle noise fit" begin
        _, results = lib.mle_biexp_epg_noise_only(X, lib.signal(Ymeta); batch_size = :, verbose = false)
        logL_opt = mean(results.loss)
    end

    metrics = (; all_rmse, all_logL, rmse, logL, logL_opt, theta_err, rmse_true = missing, logL_true = missing)

    return metrics
end

function compute_metrics(engine, batch; dataset)
    trainer.should_terminate && return Dict{Any,Any}()

    @timeit "compute metrics" begin #TODO CUDA.@sync
        # Update callback state
        get!(cb_state, "start_time", time())
        get!(cb_state, "log_metrics", Dict{Symbol,Any}())
        get!(cb_state, "histograms", Dict{Symbol,Any}())
        get!(cb_state, "all_log_metrics", Dict{Symbol,Any}(:train => Dict{Symbol,Any}(), :test => Dict{Symbol,Any}(), :val => Dict{Symbol,Any}()))
        get!(cb_state, "all_histograms", Dict{Symbol,Any}(:train => Dict{Symbol,Any}(), :test => Dict{Symbol,Any}(), :val => Dict{Symbol,Any}()))
        cb_state["last_time"] = get!(cb_state, "curr_time", 0.0)
        cb_state["curr_time"] = time() - cb_state["start_time"]
        cb_state["metrics"] = Dict{String,Any}()

        # Initialize output metrics dictionary
        is_consecutive = !isempty(logger) && (logger.epoch[end] == trainer.state.epoch && logger.iter[end] == trainer.state.iteration && logger.dataset[end] === dataset)
        accum!(k, v) = (d = cb_state["all_log_metrics"][dataset]; (!is_consecutive || !haskey(d, k)) ? (d[Symbol(k)] = Any[Float64.(v)]) : push!(d[Symbol(k)], Float64.(v)))
        accum!(k, v::Histogram) = (d = cb_state["all_histograms"][dataset]; (!is_consecutive || !haskey(d, k)) ? (d[Symbol(k)] = v) : (d[Symbol(k)].weights .+= v.weights))
        accum!(iter) = foreach(((k,v),) -> accum!(k, v), collect(pairs(iter)))

        cb_state["log_metrics"][:epoch]   = trainer.state.epoch
        cb_state["log_metrics"][:iter]    = trainer.state.iteration
        cb_state["log_metrics"][:dataset] = dataset
        cb_state["log_metrics"][:time]    = cb_state["curr_time"]

        # Invert Y and make Xs
        @unpack img_idx, img, Y, Ymeta, θPseudo = sample_batch(dataset; batchsize = settings["eval"]["batchsize"], img_idx = mod1(engine.state.iteration, length(phys.images)))
        Y_fit_state = cvae_posterior(Ymeta)
        X̂meta = lib.MetaCPMGSignal(phys, img, Y_fit_state.X̂) # Perform inference on X̂, the noisy best fit to Y
        X̂_fit_state = cvae_posterior(X̂meta)

        # TODO: sampling likelihood
        let
            s = img.meta[:pseudo_labels][dataset][:mh_sampler]
            ℓ = s.neglogPXθ[:, lib.buffer_indices(s)]
            accum!((; Ysample_logL = mean(filter(!isinf, vec(ℓ)))))
        end

        let
            ℓ_CVAE = CVAElosses(Ymeta, θPseudo)
            ℓ_CVAE = push!!(ℓ_CVAE, :CVAE => sum(ℓ_CVAE))
            accum!(ℓ_CVAE)
        end

        # Cache values for evaluating CVAE performance for estimating parameters of Y
        let
            Y_metrics = fit_metrics(Ymeta, Y_fit_state, θPseudo)
            cb_state["metrics"]["all_Yhat_rmse"] = Y_metrics.all_rmse
            cb_state["metrics"]["all_Yhat_logL"] = Y_metrics.all_logL
            accum!(Dict(Symbol(:Yhat_, k) => v for (k,v) in pairs(Y_metrics) if k ∉ (:all_rmse, :all_logL) && !ismissing(v)))
        end

        # Cache values for evaluating CVAE performance for estimating parameters of X̂
        let
            X̂_metrics = fit_metrics(X̂meta, X̂_fit_state, Y_fit_state.θ)
            cb_state["metrics"]["all_Xhat_rmse"] = X̂_metrics.all_rmse
            cb_state["metrics"]["all_Xhat_logL"] = X̂_metrics.all_logL
            accum!(Dict(Symbol(:Xhat_, k) => v for (k,v) in pairs(X̂_metrics) if k ∉ (:all_rmse, :all_logL) && !ismissing(v)))

            # img_key = Symbol(:img, img_idx)
            # accum!(img_key, lib.fast_hist_1D(lib.cpu(vec(Y_fit_state.X̂)), img.meta[:histograms][dataset][0].edges[1]))
            # Dist_L1 = lib.CityBlock(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            # Dist_ChiSq = lib.ChiSquared(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            # Dist_KLDiv = lib.KLDivergence(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            # accum!((; Dist_L1, Dist_ChiSq, Dist_KLDiv))
        end

        # Update logger dataframe and return metrics for logging
        foreach(((k,v),) -> (cb_state["log_metrics"][k] = mean(v)), cb_state["all_log_metrics"][dataset]) # update log metrics
        !is_consecutive ? push!(logger, cb_state["log_metrics"]; cols = :union) : foreach(((k,v),) -> (logger[end,k] = v), pairs(cb_state["log_metrics"]))
        output_metrics = Dict{Any,Any}(string(k) => deepcopy(v) for (k,v) in cb_state["log_metrics"] if k ∉ [:epoch, :iter, :dataset, :time]) # output non-housekeeping metrics
        merge!(cb_state["metrics"], output_metrics) # merge all log metrics into cb_state
        filter!(((k,v),) -> !ismissing(v), output_metrics) # return non-missing metrics (wandb cannot handle missing)
        for (k,v) in output_metrics
            if endswith(k, "_err") && v isa AbstractVector
                for (i,vi) in enumerate(v)
                    output_metrics[k * "_$i"] = vi
                end
                delete!(output_metrics, k)
            end
        end

        return output_metrics
    end
end

function makeplots(;showplot = false)
    trainer.should_terminate && return Dict{Symbol,Any}()
    try
        Dict{Symbol, Any}(
            # :ricemodel    => lib.plot_rician_model(logger, cb_state, phys; showplot, bandwidths = (filter(((k,v),) -> startswith(k, "logsigma"), collect(models)) |> logσs -> isempty(logσs) ? nothing : (x->lib.cpu(permutedims(x[2]))).(logσs))),
            # :signals      => lib.plot_rician_signals(logger, cb_state, phys; showplot),
            # :signalmodels => lib.plot_rician_model_fits(logger, cb_state, phys; showplot),
            # :infer        => lib.plot_rician_inference(logger, cb_state, phys; showplot),
            # :ganloss      => lib.plot_gan_loss(logger, cb_state, phys; showplot, lrdroprate = settings["opt"]["lrrate"], lrdrop = settings["opt"]["lrdrop"]),
            :vallosses    => lib.plot_all_logger_losses(logger, cb_state, phys; showplot, dataset = :val),
            :trainlosses  => lib.plot_all_logger_losses(logger, cb_state, phys; showplot, dataset = :train),
            # # :epsline      => lib.plot_epsilon(phys, derived; showplot, seriestype = :line), #TODO
            # # :epscontour   => lib.plot_epsilon(phys, derived; showplot, seriestype = :contour), #TODO
            # :priors       => lib.plot_priors(phys, derived; showplot), #TODO
            # :cvaepriors   => lib.plot_cvaepriors(phys, derived; showplot), #TODO
            # # :posteriors   => lib.plot_posteriors(phys, derived; showplot), #TODO
        )
    catch e
        lib.handle_interrupt(e; msg = "Error plotting")
    end
end

trainer = ignite.engine.Engine(@j2p train_step)
trainer.logger = ignite.utils.setup_logger("trainer")

val_evaluator = ignite.engine.Engine(@j2p (args...) -> compute_metrics(args...; dataset = :val))
val_evaluator.logger = ignite.utils.setup_logger("val_evaluator")

train_evaluator = ignite.engine.Engine(@j2p (args...) -> compute_metrics(args...; dataset = :train))
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
            @timeit "make current plots" plothandles = makeplots()
            @timeit "save current plots" lib.save_plots(plothandles; savefolder = lib.logdir(), prefix = "current-")
        end
    end
)

# Check for + save best model + logger + make plots
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["checkpointperiod"])),
    @j2p function (engine)
        loss_metric = :CVAE # :Yhat_logL
        losses = logger[logger.dataset .=== :val, loss_metric] |> skipmissing |> collect
        if !isempty(losses) && (length(losses) == 1 || losses[end] < minimum(losses[1:end-1]))
            @timeit "save best progress" let models = lib.cpu(models)
                lib.on_bad_params_or_gradients(engine.terminate, models) && return nothing
                @timeit "save best model" lib.save_progress(@dict(models, logger); savefolder = lib.logdir(), prefix = "best-", ext = ".jld2")
                @timeit "make best plots" plothandles = makeplots()
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
