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
    ntrain = "auto" # 102_400
    ntest  = "auto" # 10_240
    nval   = "auto" # 10_240

[train]
    timeout     = 1e9 #TODO 10800.0
    epochs      = 50_000
    batchsize   = 256   #256 #512 #1024 #2048 #3072 #4096
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
    lr       = 1e-4    # Initial learning rate
    lrthresh = 1e-6    # Absolute minimum learning rate
    lrdrop   = 3.16    # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    lrrate   = 10_000  # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    gclip    = 0.0     # Gradient clipping
    wdecay   = 0.0     # Weight decay
    [opt.cvae]
        INHERIT = "%PARENT%"
        gclip               = 0.0
        lambda_vae_sim      = 0.0 # Weighting of vae decoder regularization loss on simulated signals
        lambda_vae_data     = 0.0 # Weighting of vae decoder regularization loss on real signals
        lambda_latent       = 0.0 # Weighting of latent space regularization
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
    posterior = "Gaussian" # "TruncatedGaussian", "Gaussian", "Kumaraswamy"
    nlatent   = 1   # number of latent variables Z
    zdim      = 12  # embedding dimension of z
    hdim      = 512 # size of hidden layers
    skip      = false # skip connection
    layernorm = false # layer normalization following dense layer
    nhidden   = 4    # number of hidden layers
    esize     = 128 # transformer input embedding size
    nheads    = 4   # number of attention heads
    headsize  = 4   # projection head size
    seqlength = 8   # number of signal projection shards
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

_DEBUG_ = false
settings = lib.load_settings(force_new_settings = true)
wandb_logger = lib.init_wandb_logger(settings; activate = !_DEBUG_, dryrun = false, wandb_dir = lib.projectdir())

lib.set_logdirname!()
lib.clear_checkpointdir!()
# lib.set_checkpointdir!(lib.projectdir("log", "ignite-cvae-2021-01-05-T-13-45-23-425"))
# lib.set_checkpointdir!(lib.projectdir("wandb", "latest-run", "files"))
# lib.set_checkpointdir!(only(Glob.glob("run-*-3igrugah/files", lib.projectdir("wandb"))))
# lib.set_checkpointdir!(only(Glob.glob("run-*-2e9prscm/files", lib.projectdir("wandb"))))
# lib.set_checkpointdir!(only(Glob.glob("run-*-373tj1p0/files", lib.projectdir("wandb")))) # failed run

lib.@save_expression lib.logdir("build_physics.jl") function build_physics()
    # isdefined(Main, :phys) ? Main.phys : lib.EPGModel{Float32,false}(n = 64)
    isdefined(Main, :phys) ? Main.phys : lib.load_epgmodel_physics()
end

lib.@save_expression lib.logdir("build_models.jl") function build_models(
        phys,
        settings,
        models = Dict{String,Any}(),
        derived = Dict{String,Any}(),
    )
    kws(keys...) = lib.make_kwargs(settings, keys...)

    get!(models, "genatr") do; lib.init_isotropic_rician_generator(phys; kws("arch", "genatr")...); end
    get!(models, "theta_prior") do; !settings["train"]["DeepThetaPrior"] ? nothing : lib.init_deep_theta_prior(phys; kws("arch", "genatr")...); end
    get!(models, "latent_prior") do; !settings["train"]["DeepLatentPrior"] ? nothing : lib.init_deep_latent_prior(phys; kws("arch", "genatr")...); end
    get!(models, "enc1") do; lib.init_mlp_cvae_enc1(phys; kws("arch", "enc1")...); end
    get!(models, "enc2") do; lib.init_mlp_cvae_enc2(phys; kws("arch", "enc2")...); end
    get!(models, "dec") do; lib.init_mlp_cvae_dec(phys; kws("arch", "dec")...); end
    # get!(models, "enc1") do; lib.init_xformer_cvae_enc1(phys; kws("arch", "enc1")...); end
    # get!(models, "enc2") do; lib.init_xformer_cvae_enc2(phys; kws("arch", "enc2")...); end
    # get!(models, "dec") do; lib.init_xformer_cvae_dec(phys; kws("arch", "dec")...); end
    get!(models, "vae_dec") do; lib.init_mlp_cvae_vae_dec(phys; kws("arch", "vae_dec")...); end
    get!(models, "discrim") do; lib.init_mlp_discrim(phys; kws("arch", "discrim")..., ninput = lib.domain_transforms_sum_outlengths(phys; kws("train", "augment")...)); end
    get!(models, "kernel") do; lib.init_mmd_kernels(phys; kws("arch", "kernel")..., bwsizes = lib.domain_transforms_outlengths(phys; kws("train", "augment")...)); end

    derived["domain_transforms"] = lib.DomainTransforms(phys; kws("train", "augment")...)
    derived["augmentations"] = lib.Augmentations(; kws("train", "transform")...)
    derived["forwarddiff"] = lib.ForwardDifference() |> to32
    derived["laplacian"] = lib.Laplacian() |> to32
    derived["L1grad"] = lib.DepthwiseSmoothReg(; type = :L1grad) |> to32
    derived["L2diff"] = lib.ChannelwiseSmoothReg(; type = :L2diff) |> to32
    # derived["encoderspace"] = lib.NotTrainable(lib.flattenchain(Flux.Chain(models["enc1"], lib.split_mean_exp_std, lib.sample_mv_normal))) # non-trainable sampling of encoder signal representations
    derived["genatr_theta_prior"] = !settings["train"]["DeepThetaPrior"] ? lib.init_default_theta_prior(phys; kws("arch", "genatr")...) : models["theta_prior"]
    derived["genatr_latent_prior"] = !settings["train"]["DeepLatentPrior"] ? lib.init_default_latent_prior(phys; kws("arch", "genatr")...) : models["latent_prior"]
    derived["cvae_theta_prior"] = lib.derived_cvae_theta_prior(phys, derived["genatr_theta_prior"]; kws("arch", "genatr")...)
    derived["cvae_latent_prior"] = lib.derived_cvae_latent_prior(phys, derived["genatr_latent_prior"]; kws("arch", "genatr")...)
    derived["cvae"] = lib.derived_cvae(phys, models["enc1"], models["enc2"], models["dec"]; kws("arch")...)
    derived["mean_cvae"] = deepcopy(derived["cvae"])
    derived["vae_reg_loss"] = lib.VAEReg(models["vae_dec"]; regtype = settings["arch"]["vae_dec"]["regtype"])

    # Pseudo labels for Y data
    # derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = lib.checkpointdir(), modelprefix = "best-")
    # derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = only(Glob.glob("run-*-1jt2z1ry/files", lib.projectdir("wandb"))), modelprefix = "best-")
    # derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = only(Glob.glob("run-*-2e9prscm/files", lib.projectdir("wandb"))), modelprefix = "best-")

    # derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = only(Glob.glob("run-*-38wajxo4/files", lib.projectdir("wandb"))), modelprefix = "best-")
    # lib.pseudo_labels!(
    #     phys, lib.NormalizedRicianCorrector(lib.corrector(models["genatr"]), lib.ApplyOverDims(maximum; dims = 1), lib.ApplyOverDims(mean; dims = 1)), derived["pretrained_cvae"],
    #     new_noisescale = nothing, force_recompute = true, initial_guess_only = true, sigma_reg = 0.5, noisebounds = settings["arch"]["genatr"]["noisebounds"],
    # )

    # derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = only(Glob.glob("run-*-13pptz2h/files", lib.projectdir("wandb"))), modelprefix = "best-")
    # derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = only(Glob.glob("run-*-15v0mdjw/files", lib.projectdir("wandb"))), modelprefix = "best-")
    # derived["pretrained_cvae"] = lib.load_pretrained_cvae(phys; modelfolder = only(Glob.glob("run-*-1p14e3na/files", lib.projectdir("wandb"))), modelprefix = "best-")

    # lib.pseudo_labels!(phys, models["genatr"], derived["pretrained_cvae"]; force_recompute = false, initial_guess_only = false, sigma_reg = 0.5, noisebounds = settings["arch"]["genatr"]["noisebounds"])
    # lib.verify_pseudo_labels(phys, models["genatr"])

    # initialize pseudo labels as junk using initial guess of untrained cvae
    lib.pseudo_labels!(phys, models["genatr"], derived["cvae"]; force_recompute = true, initial_guess_only = true, sigma_reg = 0.5, noisebounds = settings["arch"]["genatr"]["noisebounds"])
    lib.verify_pseudo_labels(phys, models["genatr"])

    return models, derived
end

lib.@save_expression lib.logdir("build_optimizers.jl") function build_optimizers(
        phys,
        settings,
        models,
        optimizers = Dict{String,Any}(),
    )
    kws(keys...) = lib.make_kwargs(settings, keys...)
    optimizers["mmd"] = lib.init_optimizer(Flux.ADAM; kws("opt", "mmd")...)
    optimizers["cvae"] = lib.init_optimizer(Flux.ADAM; kws("opt", "cvae")...)
    optimizers["genatr"] = lib.init_optimizer(Flux.ADAM; kws("opt", "genatr")...)
    optimizers["discrim"] = lib.init_optimizer(Flux.ADAM; kws("opt", "discrim")...)
    optimizers["kernel"] = map(_ -> lib.init_optimizer(Flux.ADAM; kws("opt", "kernel")...), models["kernel"])
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
    Ts = map(derived["domain_transforms"], Xs) # tuple of named tuples of domain transformations
    Ys = lib.unzipnamedtuple(map(derived["augmentations"], lib.zipnamedtuples(Ts)))
end

####
#### GANs
####

function Gloss(X,Z)
    X̂ = lib.sampleX̂(models["genatr"], X, Z)
    X̂s, = augment_and_transform(X̂)
    σ⁻¹PX̂ = models["discrim"](reduce(vcat, X̂s))
    BCE_GX = lib.LogitBCEOne(σ⁻¹PX̂) # -log(D(G(Z))) (analogous to log(1-D(G(Z))) in Goodfellow et al.)
    return (; BCE_GX)
end

function Dloss(X,Y,Z)
    X̂ = Zygote.@ignore lib.sampleX̂(models["genatr"], X, Z)
    X̂s, Ys = augment_and_transform(X̂, Y)
    σ⁻¹PY = models["discrim"](reduce(vcat, Ys))
    σ⁻¹PX̂ = models["discrim"](reduce(vcat, X̂s))
    BCE_DY = lib.LogitBCEOne(σ⁻¹PY) # -log(D(Y))
    BCE_DX = lib.LogitBCEZero(σ⁻¹PX̂) # -log(1-D(G(Z)))
    return (; BCE_DY, BCE_DX)
end

####
#### MMD
####

# Maximum mean discrepency (m*MMD^2) loss
function MMDlosses(Ymeta::lib.AbstractMetaDataSignal)
    Y = lib.signal(Ymeta)
    λ_ϵ = Zygote.@ignore eltype(Y)(get!(settings["opt"]["mmd"], "lambda_eps", 0.0)::Float64)
    λ_∂ϵ∂Z = Zygote.@ignore eltype(Y)(get!(settings["opt"]["mmd"], "lambda_deps_dz", 0.0)::Float64)

    if settings["train"]["DeepThetaPrior"]::Bool && settings["train"]["DeepLatentPrior"]::Bool
        X, θ, Z = lib.sampleXθZ(phys, derived["cvae"], derived["genatr_theta_prior"], derived["genatr_latent_prior"], Ymeta; posterior_θ = false, posterior_Z = false) # sample θ and Z from the learned deep priors, differentiating through the sampling process and the physics model
    else
        Z = settings["train"]["DeepLatentPrior"]::Bool ? sample(derived["genatr_latent_prior"], Y) : Zygote.@ignore(sample(derived["genatr_latent_prior"], Y)) # differentiate through deep Z prior, else ignore
        θ = settings["train"]["DeepThetaPrior"]::Bool ? sample(derived["genatr_theta_prior"], Y) : Zygote.@ignore(lib.sampleθZ(phys, derived["cvae"], derived["genatr_theta_prior"], derived["genatr_latent_prior"], Ymeta; posterior_θ = true, posterior_Z = true)[1]) # sample θ from CVAE posterior; ignore Z posterior sample
        X = settings["train"]["DeepThetaPrior"]::Bool ? lib.signal_model(phys, θ) : Zygote.@ignore(lib.signal_model(phys, θ)) # differentiate through physics model if learning deep θ prior, else ignore
    end

    # Differentiate through generator corrections `sampleX̂`
    @unpack X̂, ϵ = lib.sample_rician_state(models["genatr"], X, Z)
    X̂s, Ys = augment_and_transform(lib.clamp_dim1(Y, X̂), Y)
    ℓ = map(models["kernel"], X̂s, Ys) do k, X, Y
        size(Y,2) * lib.mmd(k, X, Y)
    end

    # Regularization
    (λ_ϵ > 0) && (ℓ = push!!(ℓ, :reg_eps => λ_ϵ * derived["L1grad"](ϵ)))
    (λ_∂ϵ∂Z > 0) && (ℓ = push!!(ℓ, :reg_Z => λ_∂ϵ∂Z * derived["L2diff"](ϵ, Z)))

    return ℓ
end

####
#### CVAE
####

# Conditional variational autoencoder losses
function CVAElosses(Ymeta::lib.AbstractMetaDataSignal, θPseudo = nothing, ZPseudo = nothing; marginalize_Z)
    λ_vae_sim  = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "lambda_vae_sim", 0.0)::Float64)
    λ_vae_data = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "lambda_vae_data", 0.0)::Float64)
    λ_latent   = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "lambda_latent", 0.0)::Float64)
    λ_pseudo   = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "lambda_pseudo", 0.0)::Float64)
    τ_pseudo   = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "tau_lambda_pseudo", 0.0)::Float64)
    δ_pseudo   = Zygote.@ignore eltype(Ymeta)(get!(settings["opt"]["cvae"], "delta_lambda_pseudo", 0.0)::Float64)
    minkept    = Zygote.@ignore get!(settings["train"], "CVAEmask", 0)::Int

    #=
    # Sample X̂,θ,Z from priors
    θ = Zygote.@ignore sample(derived["cvae_theta_prior"], lib.signal(Ymeta))
    Z = Zygote.@ignore sample(derived["cvae_latent_prior"], lib.signal(Ymeta))
    X = Zygote.@ignore lib.signal_model(phys, θ)
    X̂ = Zygote.@ignore lib.sampleX̂(models["genatr"], X, Z)

    # Cross-entropy loss function components
    X̂masked, X̂mask = lib.pad_and_mask_signal(X̂, lib.nsignal(derived["cvae"]); minkept, maxkept = lib.nsignal(derived["cvae"]))
    X̂state = lib.CVAETrainingState(derived["cvae"], X̂masked, θ, Z)
    KLDiv, ELBO = lib.KL_and_ELBO(X̂state; marginalize_Z)
    ℓ = (; KLDiv, ELBO)

    if λ_vae_sim > 0
        ℓ = push!!(ℓ, :VAE => λ_vae_sim * derived["vae_reg_loss"](lib.signal(X̂state), X̂mask, lib.sample_mv_normal(X̂state.μq0, exp.(X̂state.logσq))))
    end
    if λ_latent > 0
        ℓ = push!!(ℓ, :LatentReg => λ_latent * lib.EnsembleKLDivUnitGaussian(X̂state.μq0, X̂state.logσq))
    end
    =#
    ℓ = (; KLDiv = zero(eltype(Ymeta)), ELBO = zero(eltype(Ymeta)))

    if λ_pseudo > 0
        λ_pseudo = Zygote.@ignore λ_pseudo * (τ_pseudo <= 0 || !@isdefined(trainer) ? one(λ_pseudo) : lib.cos_warmup(oftype(λ_pseudo, trainer.state.iteration), τ_pseudo, δ_pseudo))
        if λ_pseudo == 0
            ℓ = push!!(ℓ, :KLDivPseudo => zero(eltype(Ymeta)), :ELBOPseudo => zero(eltype(Ymeta)))
            (λ_vae_data > 0) && (ℓ = push!!(ℓ, :VAEPseudo => zero(eltype(Ymeta))))
            (λ_latent > 0) && (ℓ = push!!(ℓ, :LatentRegPseudo => zero(eltype(Ymeta))))
            return ℓ
        end

        #=
        # Generate pseudolabels for Ymeta from exponential moving average of CVAEs with time constant `τ_cvae`
        θPseudo, ZPseudo = Zygote.@ignore lib.sampleθZ(phys, derived["mean_cvae"], Ymeta; posterior_mode = true) # pseudo θ and Z labels for Y are given by cvae posterior modes
        =#

        # CVAE pseudo-state
        Ymasked, Ymask = lib.pad_and_mask_signal(lib.signal(Ymeta), lib.nsignal(derived["cvae"]); minkept, maxkept = lib.nsignal(Ymeta))
        Ystate = lib.CVAETrainingState(derived["cvae"], Ymasked, θPseudo, ZPseudo)

        # CVAE loss from pseudolabels
        KLDivPseudo, ELBOPseudo = lib.KL_and_ELBO(Ystate; marginalize_Z) # recover pseudo θ and Z labels
        ℓ = push!!(ℓ, :KLDivPseudo => λ_pseudo * KLDivPseudo, :ELBOPseudo => λ_pseudo * ELBOPseudo)

        if λ_vae_data > 0
            ℓ = push!!(ℓ, :VAEPseudo => λ_pseudo * λ_vae_data * derived["vae_reg_loss"](lib.signal(Ystate), Ymask, lib.sample_mv_normal(Ystate.μq0, exp.(Ystate.logσq))))
        end
        if λ_latent > 0
            ℓ = push!!(ℓ, :LatentRegPseudo => λ_pseudo * λ_latent * lib.EnsembleKLDivUnitGaussian(Ystate.μq0, Ystate.logσq))
        end

        #=
        if λ_latent > 0
            # Ymasked, Ymask = lib.pad_and_mask_signal(lib.signal(Ymeta), lib.nsignal(derived["cvae"]); minkept = lib.nsignal(Ymeta), maxkept = lib.nsignal(Ymeta))
            Ymasked, Ymask = lib.pad_and_mask_signal(lib.signal(Ymeta), lib.nsignal(derived["cvae"]); minkept, maxkept = lib.nsignal(Ymeta))
            Ystate = lib.CVAEInferenceState(derived["cvae"], Ymasked)

            #=
            Ŷstate = Zygote.@ignore begin
                ŶPseudo = lib.sampleX̂(models["genatr"], lib.signal_model(phys, θPseudo), ZPseudo) # Noisy instance of fitted signal from pseudo labels
                lib.CVAEInferenceState(derived["cvae"], Ymask .* ŶPseudo) # Use same mask as Ystate above
            end

            ℓ = push!!(ℓ, :LatentRegPseudo => λ_pseudo * λ_latent * lib.EnsembleKLDivUnitGaussian(Ystate.μr0, Ystate.logσr))
            =#

            #=
            if λ_vae_data > 0
                # Reconstruct Ŷstate signal from Ystate encoder samples
                ℓ = push!!(ℓ, :VAEPseudo => λ_pseudo * λ_vae_data * derived["vae_reg_loss"](lib.signal(Ŷstate), Ymask, lib.sample_mv_normal(Ystate.μr0, exp.(Ystate.logσr))))
            end
            =#

            if λ_vae_data > 0
                Ystate_detached = lib.CVAEInferenceState(derived["mean_cvae"], Ystate.Y, Ystate.μr0, Ystate.logσr)
                θMode, ZMode = lib.sampleθZ(phys, derived["mean_cvae"], Ymeta, Ystate_detached; posterior_θ = true, posterior_Z = true, posterior_mode = true)
                post_state = lib.posterior_state(phys, models["genatr"], lib.signal(Ymeta), θMode, ZMode)
                Ystate_VAEFitLogL = sum(Ymask .* lib.pad_signal(post_state.ℓ, size(Ymask,1))) / sum(Ymask)
                ℓ = push!!(ℓ, :VAEPseudo => λ_pseudo * λ_vae_data * Ystate_VAEFitLogL)
            end
        end
        =#
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

# make_dataset(dataset) = torch.utils.data.TensorDataset(lib.j2p_array(lib.sampleY(phys, :all; dataset)))
# train_loader = torch.utils.data.DataLoader(make_dataset(:train); batch_size = settings["train"]["batchsize"], shuffle = true, drop_last = true)
# val_eval_loader = torch.utils.data.DataLoader(make_dataset(:val); batch_size = settings["train"]["batchsize"], shuffle = false, drop_last = true) #Note: drop_last=true and batch_size=train_batchsize for MMD (else, batch_size = settings["data"]["nval"] is fine)

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
        get!(img.meta[:pseudolabels][dataset], :negloglikelihood) do
            θ = img.meta[:pseudolabels][dataset][:theta]
            lib.zeros_similar(θ, 1, size(θ,2)) .= Inf
        end
        θPseudo = img.meta[:pseudolabels][dataset][:theta][:,J] |> to32
        ZPseudo = img.meta[:pseudolabels][dataset][:latent][:,J] |> to32
        XPseudo = img.meta[:pseudolabels][dataset][:signalfit][:,J] |> to32
        ℓPseudo = img.meta[:pseudolabels][dataset][:negloglikelihood][:,J] |> to32
        state = lib.posterior_state(phys, models["genatr"], derived["cvae"], Ymeta; miniter = 1, maxiter = 1, alpha = 0.0, verbose = false, mode = :maxlikelihood)
        logα = ℓPseudo .- state.ℓ # α = L(Y|θnew) / L(Y|θold) = exp(-logL(Y|θold) - -logL(Y|θnew))
        logu = log.(lib.rand_similar(ℓPseudo)) # acceptance threshold: u <= α, where u ~ Uniform(0,1)
        mask = logu .<= logα
        θPseudo = ifelse.(mask, state.θ, θPseudo)
        ZPseudo = ifelse.(mask, state.Z, ZPseudo)
        XPseudo = ifelse.(mask, state.X, XPseudo)
        ℓPseudo = ifelse.(mask, state.ℓ, ℓPseudo)
        img.meta[:pseudolabels][dataset][:theta][:,J] = θPseudo |> Flux.cpu
        img.meta[:pseudolabels][dataset][:latent][:,J] = ZPseudo |> Flux.cpu
        img.meta[:pseudolabels][dataset][:signalfit][:,J] = XPseudo |> Flux.cpu
        img.meta[:pseudolabels][dataset][:negloglikelihood][:,J] = ℓPseudo |> Flux.cpu
    elseif false #haskey(derived, "pretrained_cvae")
        θPseudo, ZPseudo = lib.sampleθZ(phys, derived["pretrained_cvae"], Ymeta; posterior_θ = true, posterior_Z = true, posterior_mode = false)
        XPseudo = lib.signal_model(phys, θPseudo)
    elseif false
        θPseudo = img.meta[:pseudolabels][dataset][:theta][:,J] |> to32
        ZPseudo = img.meta[:pseudolabels][dataset][:latent][:,J] |> to32
        XPseudo = img.meta[:pseudolabels][dataset][:signalfit][:,J] |> to32
    else
        θPseudo = ZPseudo = XPseudo = nothing
    end
    return out = (; img_idx, img, Y, Ymeta, XPseudo, θPseudo, ZPseudo)
end

function train_step(engine, batch)
    trainer.should_terminate && return Dict{Any,Any}() #TODO

    @unpack Y, Ymeta, θPseudo, ZPseudo = sample_batch(:train; batchsize = settings["train"]["batchsize"])
    outputs = Dict{Any,Any}()

    @timeit "train batch" begin #TODO CUDA.@sync
        every(rate) = rate <= 0 ? false : mod(engine.state.iteration-1, rate) == 0
        train_MMDCVAE = every(settings["train"]["MMDCVAErate"]::Int)
        train_CVAE = every(settings["train"]["CVAErate"]::Int)
        train_MMD = every(settings["train"]["MMDrate"]::Int)
        train_GAN = train_discrim = train_genatr = every(settings["train"]["GANrate"]::Int)
        train_k = every(settings["train"]["kernelrate"]::Int)

        # Train Self MMD CVAE loss
        train_MMDCVAE && @timeit "mmd + cvae" let #TODO CUDA.@sync
            deeppriors = [models["theta_prior"], models["latent_prior"]][[settings["train"]["DeepThetaPrior"]::Bool, settings["train"]["DeepLatentPrior"]::Bool]]
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"], models["genatr"], deeppriors...)
            λ_0 = eltype(Y)(get!(settings["opt"]["mmd"], "lambda_0", 0.0)::Float64)
            @timeit "forward" ℓ, back = Zygote.pullback(ps) do #TODO CUDA.@sync
                mmd = sum(MMDlosses(Ymeta))
                cvae = sum(CVAElosses(Ymeta, θPseudo, ZPseudo; marginalize_Z = false)) #TODO marginalize_Z
                return λ_0 * mmd + cvae
            end
            @timeit "reverse" gs = back(one(ℓ)) #TODO CUDA.@sync
            @timeit "update!" Flux.Optimise.update!(optimizers["mmd"], ps, gs) #TODO CUDA.@sync
            outputs["loss"] = ℓ
        end

        # Train CVAE loss
        train_CVAE && @timeit "cvae" let #TODO CUDA.@sync
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"], models["vae_dec"])
            for _ in 1:settings["train"]["CVAEsteps"]
                @timeit "forward" ℓ, back = Zygote.pullback(() -> sum(CVAElosses(Ymeta, θPseudo, ZPseudo; marginalize_Z = false)), ps) #TODO CUDA.@sync #TODO marginalize_Z
                @timeit "reverse" gs = back(one(ℓ)) #TODO CUDA.@sync
                #=
                if _DEBUG_
                    lib.on_bad_params_or_gradients(models, ps, gs) do
                        lib.save_progress(@dict(models, logger); savefolder = lib.logdir(), prefix = "failure-", ext = ".jld2")
                        engine.terminate()
                    end
                end
                mod(trainer.state.iteration, 10 * 100) == 0 && let
                    log∇ϵ = -6
                    ∇ϵ = 10f0 ^ log∇ϵ
                    # Gradient histograms
                    plot(
                        map(["enc1", "enc2", "dec", "vae_dec"]) do name
                            ps_name = Flux.params(models[name])
                            gs_gpu = [gs[p] for p in ps_name if gs[p] !== nothing]
                            isempty(gs_gpu) && return plot(; title = L"%$name: $\log_{10}(10^{%$log∇ϵ} + |g|)$", titlefontsize = 14)
                            gs_cpu = mapreduce(vec, vcat, gs_gpu) |> Flux.cpu
                            log_gs_cpu = log10.(∇ϵ .+ abs.(gs_cpu))
                            histogram(log_gs_cpu; title = L"%$name: $\log_{10}(10^{%$log∇ϵ} + |g|)$", titlefontsize = 14, xlim = (log∇ϵ - 0.5, 3.0))
                        end...
                    ) |> display
                    #=
                    # Check for zero gradients
                    for name in ["enc1", "enc2", "dec", "vae_dec"], p in Flux.params(models[name])
                        if any(iszero, gs[p])
                            @info "Zero gradient: $(lib.find_model_param(models, p))"
                        end
                    end
                    =#
                end
                =#
                @timeit "update!" Flux.Optimise.update!(optimizers["cvae"], ps, gs) #TODO CUDA.@sync
                outputs["CVAE"] = ℓ
            end
            if haskey(derived, "mean_cvae")
                τ_cvae = get!(settings["opt"]["cvae"], "tau_cvae", 0.0)::Float64
                (τ_cvae > 0) && lib.movingaverage!(derived["mean_cvae"], derived["cvae"], τ_cvae) # exponential moving average of cvaes
            end
        end

        # Train MMD loss
        train_MMD && @timeit "mmd" let #TODO CUDA.@sync
            @timeit "genatr" let #TODO CUDA.@sync
                deeppriors = [models["theta_prior"], models["latent_prior"]][[settings["train"]["DeepThetaPrior"]::Bool, settings["train"]["DeepLatentPrior"]::Bool]]
                ps = Flux.params(models["genatr"], deeppriors...)
                @timeit "forward" ℓ, back = Zygote.pullback(() -> sum(MMDlosses(Ymeta)), ps) #TODO CUDA.@sync
                @timeit "reverse" gs = back(one(ℓ)) #TODO CUDA.@sync
                @timeit "update!" Flux.Optimise.update!(optimizers["mmd"], ps, gs) #TODO CUDA.@sync
                outputs["MMD"] = ℓ
            end
        end

        # Train GAN loss
        train_GAN && @timeit "gan" let #TODO CUDA.@sync
            @timeit "sampleXθZ" X, θ, Z = lib.sampleXθZ(phys, derived["cvae"], derived["genatr_theta_prior"], derived["genatr_latent_prior"], Ymeta; posterior_θ = true, posterior_Z = false) # learn to map whole Z domain via `posterior_Z = false` #TODO CUDA.@sync
            train_discrim && @timeit "discrim" let #TODO CUDA.@sync
                ps = Flux.params(models["discrim"])
                for _ in 1:settings["train"]["Dsteps"]
                    @timeit "forward" ℓ, back = Zygote.pullback(() -> sum(Dloss(X, Y, Z)), ps) #TODO CUDA.@sync
                    @timeit "reverse" gs = back(one(ℓ)) #TODO CUDA.@sync
                    @timeit "update!" Flux.Optimise.update!(optimizers["discrim"], ps, gs) #TODO CUDA.@sync
                    outputs["Dloss"] = ℓ
                end
            end
            train_genatr && @timeit "genatr" let #TODO CUDA.@sync
                deeppriors = [models["theta_prior"], models["latent_prior"]][[settings["train"]["DeepThetaPrior"]::Bool, settings["train"]["DeepLatentPrior"]::Bool]]
                ps = Flux.params(models["genatr"], deeppriors...)
                @timeit "forward" ℓ, back = Zygote.pullback(() -> sum(Gloss(X, Z)), ps) #TODO CUDA.@sync
                @timeit "reverse" gs = back(one(ℓ)) #TODO CUDA.@sync
                @timeit "update!" Flux.Optimise.update!(optimizers["genatr"], ps, gs) #TODO CUDA.@sync
                outputs["Gloss"] = ℓ
            end
        end

        # Train MMD kernels
        train_k && @timeit "kernel" let #TODO CUDA.@sync
            noisyclamp!(x::AbstractArray{T}, lo, hi, ϵ) where {T} = clamp!(x .+ T(ϵ) .* lib.randn_similar(x, size(x)...), T(lo), T(hi))
            restrict!(k) = noisyclamp!(lib.logbandwidths(k), -Inf, Inf, settings["arch"]["kernel"]["clampnoise"])
            aug_types, aug_Ys = augment_and_transform(Y) |> first |> Y -> (keys(Y), values(Y)) # augment data
            opts = (get_kernel_opt(aug) for aug in aug_types) # unique optimizer per augmentation
            kernels = (get_mmd_kernel(aug, size(Y,1)) for (aug, Y) in zip(aug_types, aug_Ys)) # unique kernel per augmentation
            for _ in 1:settings["train"]["kernelsteps"]::Int
                @timeit "sample G(X)" X̂ = lib.sampleX̂(phys, derived["cvae"], derived["genatr_theta_prior"], derived["genatr_latent_prior"], Ymeta; posterior_θ = true, posterior_Z = false) # # sample unique X̂ per step (TODO: posterior_Z = true? or posterior_Z = false to force learning of whole Z domain?)
                @timeit "sample Y2" Y2 = lib.sampleY(phys, settings["train"]["batchsize"]::Int; dataset = :train) |> to32 # draw another Y sample
                aug_X̂s, aug_Ys2 = augment_and_transform(X̂, Y2) .|> values # augment data + simulated data
                for (aug, kernel, aug_X̂, aug_Y, aug_Y2, opt) in zip(aug_types, kernels, aug_X̂s, aug_Ys, aug_Ys2, opts)
                    @timeit "$aug" lib.train_kernel!(
                        kernel, aug_X̂, aug_Y, opt, aug_Y2;
                        restrict! = restrict!, kernelloss = settings["opt"]["kernel"]["loss"]::String, kernelsteps = 1
                    )
                end
            end
        end
    end

    return deepcopy(outputs)
end

function cvae_posterior(Ymeta)
    # CVAE posterior state
    @timeit "posterior state" @unpack θ, Z, X, δ, ϵ, ν, ℓ = lib.posterior_state(
        phys, models["genatr"], derived["cvae"], Ymeta;
        miniter = 1, maxiter = 1, alpha = 0.0, verbose = false, mode = :maxlikelihood,
    )

    # Add noise (Z-dependent, i.e. ϵ = ϵ(Z))
    X̂ = lib.add_noise_instance(models["genatr"], ν, ϵ)

    return (; Y = lib.signal(Ymeta), X̂, θ, Z, X, δ, ϵ, ν, ℓ)
end

function fit_metrics(Ymeta, Ymeta_fit_state, θtrue, Ztrue)
    @unpack X̂, θ, Z, X, δ, ϵ, ν, ℓ = Ymeta_fit_state
    all_rmse = sqrt.(mean(abs2, lib.signal(Ymeta) .- ν; dims = 1)) |> lib.cpu |> vec |> copy
    all_logL = ℓ |> lib.cpu |> vec |> copy
    rmse, logL = mean(all_rmse), mean(all_logL)
    theta_err = (θtrue === nothing) ? missing : mean(abs, lib.θerror(phys, θtrue, θ); dims = 2) |> lib.cpu |> vec |> copy
    Z_err = (Ztrue === nothing) ? missing : mean(abs, Ztrue .- Z; dims = 2) |> lib.cpu |> vec |> copy

    @timeit "mle noise fit" begin
        _, results = lib.mle_biexp_epg_noise_only(ν, lib.signal(Ymeta); batch_size = :, verbose = false)
        logL_opt = mean(results.loss)
    end

    metrics = (; all_rmse, all_logL, rmse, logL, logL_opt, theta_err, Z_err, rmse_true = missing, logL_true = missing)
    cache_cb_args = (lib.signal(Ymeta), θ, Z, X, δ, ϵ, ν, X̂, missing) # νtrue

    return metrics, cache_cb_args
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
        @unpack img_idx, img, Y, Ymeta, θPseudo, ZPseudo = sample_batch(dataset; batchsize = settings["eval"]["batchsize"], img_idx = mod1(engine.state.iteration, length(phys.images)))
        Y_fit_state = cvae_posterior(Ymeta) #TODO marginalize_Z
        X̂meta = lib.MetaCPMGSignal(phys, img, Y_fit_state.X̂) # Perform inference on X̂, the noisy best fit to Y
        X̂_fit_state = cvae_posterior(X̂meta) #TODO marginalize_Z

        let
            ℓ_CVAE = CVAElosses(Ymeta, θPseudo, ZPseudo; marginalize_Z = false) #TODO marginalize_Z
            ℓ_CVAE = push!!(ℓ_CVAE, :CVAE => sum(ℓ_CVAE))
            accum!(ℓ_CVAE)

            Nbatch = size(Y,2)
            ℓ_MMD = MMDlosses(Ymeta[:, 1:min(Nbatch, 1024)])
            ℓ_MMD = NamedTuple{Symbol.(:MMD_, keys(ℓ_MMD))}(values(ℓ_MMD)) # prefix labels with "MMD_"
            ℓ_MMD = push!!(ℓ_MMD, :MMD => sum(ℓ_MMD))
            accum!(ℓ_MMD)

            λ_0 = eltype(Y)(get!(settings["opt"]["mmd"], "lambda_0", 0.0)::Float64)
            loss = ℓ_CVAE.CVAE + λ_0 * ℓ_MMD.MMD
            accum!((; loss))

            CVAE_Acts = let
                θ = sample(derived["cvae_theta_prior"], lib.signal(Ymeta))
                Z = sample(derived["cvae_latent_prior"], lib.signal(Ymeta))
                X = lib.signal_model(phys, θ)
                X̂ = lib.sampleX̂(models["genatr"], X, Z)
                @unpack μr0, logσr, μq0, logσq = lib.CVAETrainingState(derived["cvae"], X̂, θ, Z)
                activation_scale(z) = vcat(mean(z; dims=2), std(z; dims=2)) |> vec |> lib.cpu |> z -> @. log10(1e-3 + abs(z)) # distribution of mean/std of activations
                mapreduce(activation_scale, vcat, (μr0, logσr, μq0, logσq))
            end
            accum!((; CVAE_Acts))

            if settings["train"]["GANrate"]::Int > 0
                ℓ_GAN = Dloss(Y_fit_state.X, Y, Y_fit_state.Z)
                ℓ_GAN = push!!(ℓ_GAN, :Dloss => sum(ℓ_GAN))
                ℓ_GAN = push!!(ℓ_GAN, :Gloss => -ℓ_GAN.BCE_DX)
                accum!(ℓ_GAN)
            end
        end

        # Cache cb state variables using naming convention
        cache_cb_state!(Y, θ, Z, Xθ, δθ, ϵθ, Xθδ, Xθhat, Yθ; suf::String) = foreach(((k,v),) -> (cb_state[string(k) * suf] = lib.cpu(v)), pairs((; Y, θ, Z, Xθ, δθ, ϵθ, Xθδ, Xθhat, Yθ)))

        # Cache values for evaluating CVAE performance for estimating parameters of Y
        let
            Y_metrics, Y_cache_cb_args = fit_metrics(Ymeta, Y_fit_state, θPseudo, ZPseudo)
            cache_cb_state!(Y_cache_cb_args...; suf = "")
            cb_state["metrics"]["all_Yhat_rmse"] = Y_metrics.all_rmse
            cb_state["metrics"]["all_Yhat_logL"] = Y_metrics.all_logL
            accum!(Dict(Symbol(:Yhat_, k) => v for (k,v) in pairs(Y_metrics) if k ∉ (:all_rmse, :all_logL) && !ismissing(v)))
        end

        # Cache values for evaluating CVAE performance for estimating parameters of X̂
        let
            X̂_metrics, X̂_cache_cb_args = fit_metrics(X̂meta, X̂_fit_state, Y_fit_state.θ, Y_fit_state.Z) # X̂meta wraps `Y_fit_state.X̂`, and therefore true labels for θ, Z are also from `Y_fit_state`
            cache_cb_state!(X̂_cache_cb_args...; suf = "fit")
            cb_state["metrics"]["all_Xhat_rmse"] = X̂_metrics.all_rmse
            cb_state["metrics"]["all_Xhat_logL"] = X̂_metrics.all_logL
            accum!(Dict(Symbol(:Xhat_, k) => v for (k,v) in pairs(X̂_metrics) if k ∉ (:all_rmse, :all_logL) && !ismissing(v)))

            img_key = Symbol(:img, img_idx)
            accum!(img_key, lib.fast_hist_1D(lib.cpu(vec(Y_fit_state.X̂)), img.meta[:histograms][dataset][0].edges[1]))
            Dist_L1 = lib.CityBlock(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            Dist_ChiSq = lib.ChiSquared(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            Dist_KLDiv = lib.KLDivergence(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            accum!((; Dist_L1, Dist_ChiSq, Dist_KLDiv))
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
            :ricemodel    => lib.plot_rician_model(logger, cb_state, phys; showplot, bandwidths = (filter(((k,v),) -> startswith(k, "logsigma"), collect(models)) |> logσs -> isempty(logσs) ? nothing : (x->lib.cpu(permutedims(x[2]))).(logσs))),
            :signals      => lib.plot_rician_signals(logger, cb_state, phys; showplot),
            :signalmodels => lib.plot_rician_model_fits(logger, cb_state, phys; showplot),
            :infer        => lib.plot_rician_inference(logger, cb_state, phys; showplot),
            :ganloss      => lib.plot_gan_loss(logger, cb_state, phys; showplot, lrdroprate = settings["opt"]["lrrate"], lrdrop = settings["opt"]["lrdrop"]),
            :vallosses    => lib.plot_all_logger_losses(logger, cb_state, phys; showplot, dataset = :val),
            :trainlosses  => lib.plot_all_logger_losses(logger, cb_state, phys; showplot, dataset = :train),
            # :epsline      => lib.plot_epsilon(phys, derived; showplot, seriestype = :line), #TODO
            # :epscontour   => lib.plot_epsilon(phys, derived; showplot, seriestype = :contour), #TODO
            :priors       => lib.plot_priors(phys, derived; showplot), #TODO
            :cvaepriors   => lib.plot_cvaepriors(phys, derived; showplot), #TODO
            # :posteriors   => lib.plot_posteriors(phys, derived; showplot), #TODO
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

# Reset loging/timer/etc.
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
