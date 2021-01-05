####
#### Settings
####

using Working
using Working:
    new_settings_template, new_savepath, make_settings, make_physics, make_models!, load_checkpoint, make_optimizers, make_wandb_logger!, save_snapshot!,
    AbstractTensor3D, AbstractTensor4D, CuTensor3D, CuTensor4D,
    AbstractMetaDataSignal, MetaCPMGSignal, signal,
    MMDKernel, FunctionKernel, DeepExponentialKernel,
    mmd, mmdvar, mmd_and_mmdvar, tstat, logbandwidths, train_kernel!,
    CityBlock, ChiSquared, KLDivergence,
    map_dict, sum_dict, apply_dim1, clamp_dim1,
    arr_similar, arr32, arr64, zeros_similar, ones_similar, randn_similar, rand_similar, fill_similar,
    handleinterrupt, saveprogress, saveplots,
    corrector, generator, nsignal, nlatent, ninput, noutput, correction_and_noiselevel, correction, noiselevel, add_noise_instance, corrected_signal_instance, add_correction, rician_params,
    hasclosedform, physicsmodel,
    ntheta, nsignal, signal_model, noiselevel, sampleθprior, sampleZprior, sampleWprior, θlabels, θasciilabels, θunits, θlower, θupper, θerror, nnuissance, nmarginalized, nmodel, θmarginalized, θnuissance, θderived, θmodel, θsufficient,
    sampleθ, sampleX, sampleX̂, sampleY,
    sampleθZ, θZ_sampler, sampleXθZ, sampleX̂θZ, sampleθZposterior, θZposterior_sampler, make_state, posterior_state,
    KLDivUnitNormal, KLDivergence, EvidenceLowerBound, KL_and_ELBO, DeepPriorRicianPhysicsModel, NegLogLikelihood,
    apply_pad, mv_normal_parameters, sample_columns, sample_mv_normal,
    MLP, flattenchain,
    fast_hist_1D, signal_histograms, pyheatmap

pyplot(size=(800,600))

Working.new_settings_template() = TOML.parse(
"""
[data]
    out    = "$(new_savepath())"
    ntrain = "auto" # 102_400
    ntest  = "auto" # 10_240
    nval   = "auto" # 10_240

[train]
    timeout     = 1e9 #TODO 10800.0
    epochs      = 1000_000
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
    batchsize       = 10240 # batch size for evaluation
    nbatches        = 2     # number of eval batches
    valevalperiod   = 120.0
    trainevalperiod = 120.0
    printperiod     = 120.0
    saveperiod      = 600.0

[opt]
    lr       = 1e-4    # Initial learning rate
    lrrel    = 0.0     # Initial learning rate relative to batch size, i.e. lr = lrrel / batchsize
    lrthresh = 0.0     #1e-6 # Absolute minimum learning rate
    lrdrop   = 3.16    # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    lrrate   = 999_999 # Drop learning rate by factor `lrdrop` every `lrrate` epochs
    lrwarmup = 1000    # Linear learning rate warmup period (iterations, not epochs)
    gclip    = 0.0     # Gradient clipping
    wdecay   = 0.0     # Weight decay
    [opt.cvae]
        INHERIT = "%PARENT%"
        lambda_vae_reg = 1.0 # Weighting of vae decoder regularization loss
        lambda_pseudo  = 0.0 # Weighting of pseudo label loss
    [opt.genatr]
        INHERIT = "%PARENT%" #TODO: 0.01 train generator more slowly
    [opt.discrim]
        INHERIT = "%PARENT%"
    [opt.mmd]
        INHERIT = "%PARENT%"
        gclip = 1.0
        lambda_0        = 0.0 # MMD loss weighting relative to CVAE
        lambda_eps      = 0.0   # Regularize noise amplitude epsilon
        lambda_deps_dz  = 0.0   # Regularize gradient of epsilon w.r.t. latent variables
    [opt.kernel]
        INHERIT = "%PARENT%" # Kernel learning rate 
        loss  = "mmd"        # Kernel loss ("mmd", "tstatistic", or "mmd_diff")

[arch]
    nlatent   = 1   # number of latent variables Z
    zdim      = 12  # embedding dimension of z
    hdim      = 512 # size of hidden layers
    skip      = false # skip connection
    layernorm = false # layer normalization following dense layer
    nhidden   = 4    # number of hidden layers
    head      = 4    # number of attention heads
    psize     = 128 # transformer input size
    hsize     = 32  # hidden size of multihead attention (hsize == psize ÷ head keeps num. params constant w.r.t head)
    nshards   = 8   # number of signal projection shards
    chunksize = 0   # nshards  == (nsignals - chunksize) ÷ (chunksize - overlap) + 1
    overlap   = 0   # nsignals == nshards * (chunksize - overlap) + overlap
    [arch.enc1]
        INHERIT = "%PARENT%"
    [arch.enc2]
        INHERIT = "%PARENT%"
    [arch.dec]
        INHERIT = "%PARENT%"
    [arch.vae_dec]
        INHERIT = "%PARENT%"
    [arch.genatr]
        hdim        = 64   #TODO "%PARENT%"
        nhidden     = 2    #TODO "%PARENT%"
        ktheta      = 16   #TODO Dimension of domain of theta prior space
        klatent     = 4    #TODO Dimension of domain of latent prior space
        prior_mix   = 0.5  #TODO Mix learned deep prior with `prior_mix` fraction of default prior for robustness
        leakyslope  = 0.0
        maxcorr     = 0.1
        noisebounds = [-6.0, 0.0] #TODO
    [arch.discrim]
        hdim      = 0     #TODO "%PARENT%"
        nhidden   = 0     #TODO "%PARENT%"
        dropout   = 0.1
    [arch.kernel]
        nbandwidth  = 32            #TODO
        channelwise = false         #TODO
        deep        = false         #TODO
        bwbounds    = [-8.0, 4.0]   # Bounds for kernel bandwidths (logsigma)
        clampnoise  = 0.0           #TODO
"""
)

####
#### Setup
####

Working.JL_CHECKPOINT_FOLDER[] = "output/ignite-cvae-2021-01-05-T-12-20-53-131"

settings = make_settings()

# for (parent, (key, leaf)) in Ignite.breadth_first_iterator(settings), (k,v) in leaf
#    (k == "lr") && (leaf[k] = 1e-4)
#    (k == "lrwarmup") && (leaf[k] = 10000)
#    (k == "wdecay") && (leaf[k] = 0.0)
# end

!isdefined(Main, :phys) && (phys = make_physics(settings))
# phys = make_physics(settings)
models, derived = make_models!(phys, settings, load_checkpoint())
optimizers = make_optimizers(settings)
wandb_logger = make_wandb_logger!(settings)
save_snapshot!(settings, models)

Working.JL_CHECKPOINT_FOLDER[] = ""

# # Gradient testing
# _models, _derived = deepcopy(models), deepcopy(derived)
# models, derived = make_models!(phys, settings, BSON.load("/home/jdoucette/Documents/code/BlochTorreyExperiments-shared/LearningCorrections/projects/ismrm2021/output/ignite-cvae-2021-01-02-T-13-44-10-685/current-models.bson")["models"] |> deepcopy |> todevice |> to32)
# let
#     ymeta = sample_batch(:val; batchsize = settings["eval"]["batchsize"], img_idx = 2)[end]
#     m = derived["cvae"]
#     loss = () -> sum(CVAElosses(ymeta; marginalize_Z = false))
#     # m = Transformers.Basic.MultiheadAttention(2, 32, 16, 32) |> todevice
#     # x = CUDA.randn(32,4,1)
#     # loss = () -> sum(abs2, m(1f0 .* x, 1f0 .* x, 1f0 .* x))
#     Working.modelgradcheck(loss, m; extrapolate = true, subset = :random, verbose = true, rtol = 1e-2, atol = 1e-2, seed = 5)
# end

####
#### Augmentations
####

function augment_and_transform(Xs::AbstractArray...)
    chunk = settings["train"]["transform"]["chunk"]::Int
    flip = settings["train"]["transform"]["flipsignals"]::Bool
    Xaugs = map(augment, Xs) # tuple of named tuples of domain augmentations
    Xtrans = map(Xaugs...) do (Xaug...) # tuple of inputs over same domain
        nchannel = size(first(Xaug),1) # all Xs of same domain are equal size in first dimension
        if (0 < chunk < nchannel) || flip
            i = ifelse(!(0 < chunk < nchannel), 1:nchannel, rand(1:nchannel-chunk+1) .+ (0:chunk-1))
            i = ifelse(!flip, i, ifelse(rand(Bool), i, reverse(i)))
            map(Xi -> Xi[i,..], Xaug) # tuple of transformed augmentations
        else
            Xaug
        end
    end
    ks = Zygote.@ignore keys(Xtrans)
    return map((xs...,) -> NamedTuple{ks}(xs), Xtrans...) # named tuple (by domain) of tuples -> tuple of named tuples (by domain)
end

function augment(X::AbstractMatrix)
    X₀   = settings["train"]["augment"]["signal"]::Bool ? X : nothing # Plain signal
    ∇X   = settings["train"]["augment"]["gradient"]::Bool ? derived["forwarddiff"](X) : nothing # Signal gradient
    ∇²X  = settings["train"]["augment"]["laplacian"]::Bool ? derived["laplacian"](X) : nothing # Signal laplacian
    ∇ⁿX  = settings["train"]["augment"]["fdcat"]::Int > 0 ? derived["fdcat"](X) : nothing # Signal finite differences, concatenated
    Xres = nothing #TODO metadata: settings["train"]["augment"]["residuals"]::Bool ? X .- Zygote.@ignore(sampleXθZ(derived["cvae"], derived["prior"], X; posterior_θ = true, posterior_Z = true))[1] : nothing # Residual relative to different sample X̄(θ), θ ~ P(θ|X) (note: Z discarded, posterior_Z irrelevant)
    Xenc = settings["train"]["augment"]["encoderspace"]::Bool ? derived["encoderspace"](X) : nothing # Encoder-space signal
    Xfft = settings["train"]["augment"]["fftcat"]::Bool ? vcat(reim(rfft(X,1))...) : nothing # Concatenated real/imag fourier components
    Xrfft, Xifft = settings["train"]["augment"]["fftsplit"]::Bool ? reim(rfft(X,1)) : (nothing, nothing) # Separate real/imag fourier components

    ks = (:signal, :grad, :lap, :fd, :res, :enc, :fft, :rfft, :ifft)
    Xs = [X₀, ∇X, ∇²X, ∇ⁿX, Xres, Xenc, Xfft, Xrfft, Xifft]
    is = (x -> x !== nothing).(Xs)
    Xs = NamedTuple{ks[is]}(Xs[is])

    return Xs
end

function augment(X::AbstractArray)
    Xs = augment(reshape(X, size(X,1), :))
    return map(Xi -> reshape(Xi, size(Xi,1), Base.tail(size(X))...), Xs)
end

####
#### GANs
####

D_Y_prob(Y) = -sum(log.(apply_dim1(models["discrim"], Y) .+ eps(eltype(Y)))) / size(Y,2) # discrim learns toward Prob(Y) = 1
D_G_X_prob(X̂) = -sum(log.(1 .- apply_dim1(models["discrim"], X̂) .+ eps(eltype(X̂)))) / size(X̂,2) # discrim learns toward Prob(G(X)) = 0

function Dloss(X,Y,Z)
    X̂augs, Yaugs = augment_and_transform(sampleX̂(derived["ricegen"], X, Z), Y)
    X̂s, Ys = reduce(vcat, X̂augs), reduce(vcat, Yaugs)
    D_Y = D_Y_prob(Ys)
    D_G_X = D_G_X_prob(X̂s)
    return (; D_Y, D_G_X)
end

function Gloss(X,Z)
    X̂aug, = augment_and_transform(sampleX̂(derived["ricegen"], X, Z))
    X̂s = reduce(vcat, X̂aug)
    neg_D_G_X = -D_G_X_prob(X̂s) # genatr learns toward Prob(G(X)) = 1
    return (; neg_D_G_X)
end

####
#### MMD
####

function noiselevel_regularization(ϵ::AbstractMatrix, ::Val{type}) where {type}
    if type === :L2lap
        ∇²ϵ = derived["laplacian"](ϵ)
        √(sum(abs2, ∇²ϵ) / length(∇²ϵ))
    elseif type === :L1grad
        ∇ϵ = derived["forwarddiff"](ϵ)
        sum(abs, ∇ϵ) / length(∇ϵ)
    else
        nothing
    end
end

function noiselevel_gradient_regularization(ϵ::AbstractMatrix, Z::AbstractMatrix, ::Val{type}) where {type}
    Δϵ = derived["forwarddiff"](permutedims(ϵ, (2,1))) # (b-1) × n Differences
    ΔZ = derived["forwarddiff"](permutedims(Z, (2,1))) # (b-1) × nz Differences
    ΔZ² = sum(abs2, ΔZ; dims = 2) ./ size(ΔZ, 2) # (b-1) × 1 Mean squared distance
    ΔZ0² = eltype(Z)(1e-3)
    if type === :L2diff
        dϵ_dZ = @. abs2(Δϵ) / (ΔZ² + ΔZ0²)
        √(sum(dϵ_dZ) / length(dϵ_dZ))
    elseif type === :L1diff
        dϵ_dZ = @. abs(Δϵ) / √(ΔZ² + ΔZ0²)
        sum(dϵ_dZ) / length(dϵ_dZ)
    else
        nothing
    end
end

function get_kernel_opt(key)
    # Initialize optimizer, if necessary
    get!(optimizers, "kernel_$key") do
        make_optimizer(lr = initial_learning_rate!(settings, "kernel"))
    end
    return optimizers["kernel_$key"]
end

function get_mmd_kernel(key, nchannel)
    bwbounds = settings["arch"]["kernel"]["bwbounds"]::Vector{Float64}
    nbandwidth = settings["arch"]["kernel"]["nbandwidth"]::Int
    channelwise = settings["arch"]["kernel"]["channelwise"]::Bool
    deep = settings["arch"]["kernel"]["deep"]::Bool
    chunk = settings["train"]["transform"]["chunk"]::Int
    nz = settings["arch"]["zdim"]::Int # embedding dimension
    hdim = settings["arch"]["genatr"]["hdim"]::Int

    # Initialize MMD kernel bandwidths, if necessary
    get!(models, "logsigma_$key") do
        ndata = deep ? nz : nchannel
        ndata = !(0 < chunk < ndata) ? ndata : chunk
        logσ = range((deep ? (-5.0, 5.0) : bwbounds)...; length = nbandwidth+2)[2:end-1]
        logσ = repeat(logσ, 1, (channelwise ? ndata : 1))
        logσ |> to32
    end

    # MMD kernel wrapper
    get!(derived, "kernel_$key") do
        DeepExponentialKernel(
            models["logsigma_$key"],
            !deep ?
                identity :
                Flux.Chain(
                    MLP(nchannel => nz, 0, hdim, Flux.relu, identity; skip = false),
                    z -> Flux.normalise(z; dims = 1), # kernel bandwidths are sensitive to scale; normalize learned representations
                    z -> z .+ randn_similar(z, size(z)...), # stochastic embedding prevents overfitting to Y data
                ) |> flattenchain |> to32,
            )
    end

    return derived["kernel_$key"]
end

MMDloss(k, X::AbstractMatrix, Y::AbstractMatrix) = size(Y,2) * mmd(k, X, Y)
MMDloss(k, X::AbstractTensor3D, Y::AbstractMatrix) = mean(map(i -> MMDloss(k, X[:,:,i], Y), 1:size(X,3)))

# Maximum mean discrepency (m*MMD^2) loss
function MMDlosses(Ymeta::AbstractMetaDataSignal)
    if settings["train"]["DeepThetaPrior"]::Bool && settings["train"]["DeepLatentPrior"]::Bool
        X, θ, Z = sampleXθZ(derived["cvae"], derived["prior"], Ymeta; posterior_θ = false, posterior_Z = false) # sample θ and Z from the learned deep priors, differentiating through the sampling process and the physics model
    else
        Z = settings["train"]["DeepLatentPrior"]::Bool ? sampleZprior(derived["prior"], signal(Ymeta)) : Zygote.@ignore(sampleZprior(derived["prior"], signal(Ymeta))) # differentiate through deep Z prior, else ignore
        θ = settings["train"]["DeepThetaPrior"]::Bool ? sampleθprior(derived["prior"], signal(Ymeta)) : Zygote.@ignore(sampleθZ(derived["cvae"], derived["prior"], Ymeta; posterior_θ = true, posterior_Z = true)[1]) # use CVAE posterior for θ prior; Z is ignored w/ `posterior_Z` arbitrary
        X = settings["train"]["DeepThetaPrior"]::Bool ? signal_model(phys, θ) : Zygote.@ignore(signal_model(phys, θ)) # differentiate through physics model if learning deep θ prior, else ignore
    end

    # Differentiate through generator corrections `sampleX̂`
    nX̂ = settings["train"]["transform"]["nsamples"]::Int |> n -> ifelse(n > 1, n, nothing)
    ν, ϵ = rician_params(derived["ricegen"], X, Z)
    ν, ϵ = clamp_dim1(signal(Ymeta), ν, ϵ)
    X̂ = add_noise_instance(derived["ricegen"], ν, ϵ, nX̂)

    X̂s, Ys = augment_and_transform(X̂, signal(Ymeta))
    ks = Zygote.@ignore NamedTuple{keys(X̂s)}(get_mmd_kernel.(keys(X̂s), size.(values(X̂s),1)))
    ℓ  = map(MMDloss, ks, X̂s, Ys)

    λ_ϵ = Zygote.@ignore eltype(signal(Ymeta))(get!(settings["opt"]["mmd"], "lambda_eps", 0.0)::Float64)
    if λ_ϵ > 0
        R = λ_ϵ * noiselevel_regularization(ϵ, Val(:L1grad))
        ℓ = push!!(ℓ, :reg_eps => R)
    end

    λ_∂ϵ∂Z = Zygote.@ignore eltype(signal(Ymeta))(get!(settings["opt"]["mmd"], "lambda_deps_dz", 0.0)::Float64)
    if λ_∂ϵ∂Z > 0
        R = λ_∂ϵ∂Z * noiselevel_gradient_regularization(ϵ, Z, Val(:L2diff))
        ℓ = push!!(ℓ, :reg_Z => R)
    end

    return ℓ
end

####
#### CVAE
####

function make_signal_mask(Y::AbstractMatrix; minkept::Int = firstindex(Y,1), maxkept::Int = lastindex(Y,1))
    # Create mask which randomly zeroes the tails of the columns of Y. That is, for each column `j` the mask satisfies
    # `m[i <= nⱼ, j] = 1` and `m[i > nⱼ, j] = 0`, with `minkept <= nⱼ <= maxkept` chosen randomly per column `j`
    Irows   = arr_similar(Y, collect(axes(Y,1)))
    Icutoff = arr_similar(Y, collect(rand(minkept:maxkept, 1, size(Y,2))))
    mask    = arr_similar(Y, Irows .<= Icutoff)
    return mask
end

# Conditional variational autoencoder losses
function CVAElosses(Ymeta::AbstractMetaDataSignal; marginalize_Z)
    λ_vae_reg = Zygote.@ignore eltype(signal(Ymeta))(get!(settings["opt"]["cvae"], "lambda_vae_reg", 0.0)::Float64)
    λ_pseudo  = Zygote.@ignore eltype(signal(Ymeta))(get!(settings["opt"]["cvae"], "lambda_pseudo", 0.0)::Float64)
    minkept   = Zygote.@ignore get!(settings["train"], "CVAEmask", 0)::Int

    # Sample X̂,θ,Z from priors
    θ = Zygote.@ignore sampleθprior(derived["cvae_prior"], signal(Ymeta))
    Z = Zygote.@ignore sampleZprior(derived["cvae_prior"], signal(Ymeta))
    X = Zygote.@ignore signal_model(phys, θ)
    X̂ = Zygote.@ignore sampleX̂(derived["ricegen"], X, Z)

    # Cross-entropy loss function components
    X̂mask   = Zygote.@ignore make_signal_mask(X̂; minkept)
    X̂masked = X̂mask .* X̂
    KLDiv, ELBO = KL_and_ELBO(derived["cvae"], X̂masked, θ, Z; marginalize_Z)

    ℓ = if λ_vae_reg == 0
        (; KLDiv, ELBO)
    else
        Ypadded = apply_pad(derived["cvae"], signal(Ymeta))
        Ymask   = Zygote.@ignore make_signal_mask(Ypadded; minkept, maxkept = nsignal(Ymeta))
        Ymasked = Ymask .* Ypadded
        X̂dec    = models["vae_dec"](sample_mv_normal(mv_normal_parameters(derived["cvae"], X̂masked)))
        Ydec    = models["vae_dec"](sample_mv_normal(mv_normal_parameters(derived["cvae"], Ymasked)))
        VAEsim  = λ_vae_reg * sum(abs, X̂masked .- X̂mask .* X̂dec) / size(X̂masked, 2) # only penalize recon error within X̂mask
        VAEdata = λ_vae_reg * sum(abs, Ymasked .- Ymask .* Ydec) / size(Ymasked, 2) # only penalize recon error within Ymask
        (; KLDiv, ELBO, VAEsim, VAEdata)
    end

    #= Use inferred params as pseudolabels for Ymeta
    θY, ZY = Zygote.@ignore sampleθZ(derived["cvae"], derived["cvae_prior"], Ymeta; posterior_θ = true, posterior_Z = true) # draw pseudo θ and Z labels for Y
    Ymasked = signal(Ymeta) .* Zygote.@ignore(make_signal_mask(signal(Ymeta); minkept))
    KLDivY, ELBOY = KL_and_ELBO(derived["cvae"], Ymasked, θY, ZY; marginalize_Z) # recover pseudo θ and Z labels
    (; KLDiv, ELBO, KLDivPseudo = λ_pseudo * KLDivY, ELBOPseudo = λ_pseudo * ELBOY)
    =#

    return ℓ
end

# let
#     Ymeta = sample_batch(:val; batchsize = settings["eval"]["batchsize"], img_idx = 2)[end]
#     minkept = Zygote.@ignore get!(settings["train"], "CVAEmask", 0)::Int
#     θ = Zygote.@ignore sampleθprior(derived["cvae_prior"], signal(Ymeta))
#     Z = Zygote.@ignore sampleZprior(derived["cvae_prior"], signal(Ymeta))
#     X = Zygote.@ignore signal_model(phys, θ)
#     X̂ = Zygote.@ignore sampleX̂(derived["ricegen"], X, Z)
#     X̂mask   = Zygote.@ignore make_signal_mask(X̂; minkept)
#     X̂masked = X̂mask .* X̂
#     KLDiv, ELBO = KL_and_ELBO(derived["cvae"], X̂masked, θ, Z; marginalize_Z = false)
#     Ypadded = apply_pad(derived["cvae"], signal(Ymeta))
#     Ymask   = Zygote.@ignore make_signal_mask(Ypadded; minkept, maxkept = nsignal(Ymeta))
#     Ymasked = Ymask .* Ypadded
#     # inspect vae_dec
#     m = models["vae_dec"]
#     dropnan(x) = ifelse(isnan(x), zero(x), x)
#     zr_Y = sample_mv_normal(mv_normal_parameters(derived["cvae"], Ymasked))
#     @info "zr_Y"; zr_Y |> display
#     # @info "1:end-2"; m[1:end-2](zr_Y) |> display
#     # @info "1:end-1 (pre-softplus)"; Flux.Chain(m[1:end-2]..., Flux.Dense((m -> (m.W, m.b))(m[end-1])...))(zr_Y) |> display
#     @info "1:end-1 (pre-softplus, NaN removed)"; Flux.Chain(m[1:end-2]..., Flux.Dense(dropnan.(m[end-1].W), m[end-1].b))(zr_Y) |> display
#     Ydec = Flux.Chain(m[1:end-2]..., Flux.Dense(dropnan.(m[end-1].W), m[end-1].b, Flux.softplus), m[end])(zr_Y)
#     @info "1:end (with-softplus, NaN removed)"; Ydec |> display
#     # @info "1:end-1"; m[1:end-1](zr_Y) |> display
#     # @info "1:end-0"; m[1:end-0](zr_Y) |> display
#     @info "Y (padded + masked)"; Ymasked |> display
#     @info "ΔY (no mask)"; Ymasked .- Ydec |> display
#     @info "ΔY (with mask)"; Ymask .* (Ymasked .- Ydec) |> display
#     # check grad
#     _m = Flux.Chain(m[1:end-2]..., Flux.Dense(dropnan.(m[end-1].W), m[end-1].b, Flux.softplus), m[end])
#     Working.modelgradcheck(_m; extrapolate = true, subset = :random, verbose = true, rtol = 1e-2, atol = 1e-2, seed = 5) do
#         sum(abs, Ymask .* (Ymasked .- _m(zr_Y))) / size(Ymask,2)
#     end
# end;
# error("here")

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

# make_dataset(dataset) = torch.utils.data.TensorDataset(PyTools.array(sampleY(phys, :all; dataset)))
# train_loader = torch.utils.data.DataLoader(make_dataset(:train); batch_size = settings["train"]["batchsize"], shuffle = true, drop_last = true)
# val_eval_loader = torch.utils.data.DataLoader(make_dataset(:val); batch_size = settings["train"]["batchsize"], shuffle = false, drop_last = true) #Note: drop_last=true and batch_size=train_batchsize for MMD (else, batch_size = settings["data"]["nval"] is fine)

make_dataset_indices(n) = torch.utils.data.TensorDataset(PyTools.array(collect(1:n)))
train_loader = torch.utils.data.DataLoader(make_dataset_indices(settings["train"]["nbatches"]))
val_eval_loader = torch.utils.data.DataLoader(make_dataset_indices(settings["eval"]["nbatches"]))
train_eval_loader = torch.utils.data.DataLoader(make_dataset_indices(settings["eval"]["nbatches"]))

function sample_batch(dataset::Symbol; batchsize::Int, img_idx::Int = 0)
    img = phys.images[img_idx > 0 ? img_idx : rand(1:length(phys.images))]
    Y = sample_columns(img.partitions[dataset], batchsize; replace = false) |> todevice
    Ymeta = MetaCPMGSignal(phys, img, Y)
    return (; img_idx, img, Y, Ymeta)
end

function train_step(engine, batch)
    img_idx, img, Ytrain, Ytrainmeta = sample_batch(:train; batchsize = settings["train"]["batchsize"])
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
            λ_0 = eltype(Ytrain)(get!(settings["opt"]["mmd"], "lambda_0", 0.0)::Float64)
            @timeit "forward" ℓ, back = Zygote.pullback(ps) do #TODO CUDA.@sync
                mmd = sum(MMDlosses(Ytrainmeta))
                cvae = sum(CVAElosses(Ytrainmeta; marginalize_Z = false)) #TODO marginalize_Z
                return λ_0 * mmd + cvae
            end
            @timeit "reverse" gs = back(one(eltype(phys))) #TODO CUDA.@sync
            @timeit "update!" Flux.Optimise.update!(optimizers["mmd"], ps, gs) #TODO CUDA.@sync
            outputs["loss"] = ℓ
        end

        # Train CVAE loss
        train_CVAE && @timeit "cvae" let #TODO CUDA.@sync
            # ps = Flux.params(models["enc1"], models["enc2"], models["dec"])
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"], models["vae_dec"]) #TODO
            _found_nan_param = false
            for m in ["enc1", "enc2", "dec", "vae_dec"]
                pm = Flux.params(models[m])
                for (i,p) in enumerate(pm)
                    if any(isnan, p)
                        _found_nan_param = true
                        @info m, i, typeof(p), size(p), sum(isnan, p) / length(p)
                        J = findall(isnan.(p))
                        for j in 1:min(length(J), 5)
                            @info m, i, j, J[j:j], p[J[j:j]]
                        end
                    end
                end
            end
            if _found_nan_param
                @info "TERMINATING - FOUND NAN PARAM"
                engine.terminate()
            end
            for _ in 1:settings["train"]["CVAEsteps"]
                @timeit "forward"   ℓ, back = Zygote.pullback(() -> sum(CVAElosses(Ytrainmeta; marginalize_Z = false)), ps) #TODO CUDA.@sync #TODO marginalize_Z
                @timeit "reverse"   gs = back(one(eltype(phys))) #TODO CUDA.@sync
                _found_nan_grad = false
                for m in ["enc1", "enc2", "dec", "vae_dec"]
                    pm = Flux.params(models[m])
                    for (i,p) in enumerate(pm)
                        g = gs[p]
                        if any(isnan, g)
                            _found_nan_grad = true
                            @info m, i, typeof(g), size(g), sum(isnan, g) / length(g)
                            J = findall(isnan.(g))
                            for j in 1:min(length(J), 5)
                                @info m, i, j, J[j:j], p[J[j:j]], g[J[j:j]]
                            end
                        end
                        # @. g = ifelse(isnan(g), 0, g)
                    end
                end
                if _found_nan_grad
                    # @info "ZEROING NAN GRADIENT"
                    @info "TERMINATING - FOUND NAN GRADIENT"
                    engine.terminate()
                end
                @timeit "update!"   Flux.Optimise.update!(optimizers["cvae"], ps, gs) #TODO CUDA.@sync
                outputs["CVAE"] = ℓ
            end
        end

        # Train MMD loss
        train_MMD && @timeit "mmd" let #TODO CUDA.@sync
            @timeit "genatr" let #TODO CUDA.@sync
                deeppriors = [models["theta_prior"], models["latent_prior"]][[settings["train"]["DeepThetaPrior"]::Bool, settings["train"]["DeepLatentPrior"]::Bool]]
                ps = Flux.params(models["genatr"], deeppriors...)
                @timeit "forward" ℓ, back = Zygote.pullback(() -> sum(MMDlosses(Ytrainmeta)), ps) #TODO CUDA.@sync
                @timeit "reverse" gs = back(one(eltype(phys))) #TODO CUDA.@sync
                @timeit "update!" Flux.Optimise.update!(optimizers["mmd"], ps, gs) #TODO CUDA.@sync
                outputs["MMD"] = ℓ
            end
        end

        # Train GAN loss
        train_GAN && @timeit "gan" let #TODO CUDA.@sync
            @timeit "sampleXθZ" Xtrain, θtrain, Ztrain = sampleXθZ(derived["cvae"], derived["prior"], Ytrainmeta; posterior_θ = true, posterior_Z = false) # learn to map whole Z domain via `posterior_Z = false` #TODO CUDA.@sync
            train_discrim && @timeit "discrim" let #TODO CUDA.@sync
                ps = Flux.params(models["discrim"])
                for _ in 1:settings["train"]["Dsteps"]
                    @timeit "forward" ℓ, back = Zygote.pullback(() -> sum(Dloss(Xtrain, Ytrain, Ztrain)), ps) #TODO CUDA.@sync
                    @timeit "reverse" gs = back(one(eltype(phys))) #TODO CUDA.@sync
                    @timeit "update!" Flux.Optimise.update!(optimizers["discrim"], ps, gs) #TODO CUDA.@sync
                    outputs["Dloss"] = ℓ
                end
            end
            train_genatr && @timeit "genatr" let #TODO CUDA.@sync
                deeppriors = [models["theta_prior"], models["latent_prior"]][[settings["train"]["DeepThetaPrior"]::Bool, settings["train"]["DeepLatentPrior"]::Bool]]
                ps = Flux.params(models["genatr"], deeppriors...)
                @timeit "forward" ℓ, back = Zygote.pullback(() -> sum(Gloss(Xtrain, Ztrain)), ps) #TODO CUDA.@sync
                @timeit "reverse" gs = back(one(eltype(phys))) #TODO CUDA.@sync
                @timeit "update!" Flux.Optimise.update!(optimizers["genatr"], ps, gs) #TODO CUDA.@sync
                outputs["Gloss"] = ℓ
            end
        end

        # Train MMD kernels
        train_k && @timeit "kernel" let #TODO CUDA.@sync
            noisyclamp!(x::AbstractArray{T}, lo, hi, ϵ) where {T} = clamp!(x .+ T(ϵ) .* randn_similar(x, size(x)...), T(lo), T(hi))
            restrict!(k) = noisyclamp!(logbandwidths(k), -Inf, Inf, settings["arch"]["kernel"]["clampnoise"])
            aug_types, aug_Ytrains = augment_and_transform(Ytrain) |> first |> Y -> (keys(Y), values(Y)) # augment data
            opts = (get_kernel_opt(aug) for aug in aug_types) # unique optimizer per augmentation
            kernels = (get_mmd_kernel(aug, size(Y,1)) for (aug, Y) in zip(aug_types, aug_Ytrains)) # unique kernel per augmentation
            for _ in 1:settings["train"]["kernelsteps"]::Int
                @timeit "sample G(X)" X̂train = sampleX̂(derived["cvae"], derived["prior"], Ytrainmeta; posterior_θ = true, posterior_Z = false) # # sample unique X̂ per step (TODO: posterior_Z = true? or posterior_Z = false to force learning of whole Z domain?)
                @timeit "sample Y2" Ytrain2 = sampleY(phys, settings["train"]["batchsize"]::Int; dataset = :train) |> to32 # draw another Y sample
                aug_X̂trains, aug_Ytrains2 = augment_and_transform(X̂train, Ytrain2) .|> values # augment data + simulated data
                for (aug, kernel, aug_X̂train, aug_Ytrain, aug_Ytrain2, opt) in zip(aug_types, kernels, aug_X̂trains, aug_Ytrains, aug_Ytrains2, opts)
                    @timeit "$aug" train_kernel!(
                        kernel, aug_X̂train, aug_Ytrain, opt, aug_Ytrain2;
                        restrict! = restrict!, kernelloss = settings["opt"]["kernel"]["loss"]::String, kernelsteps = 1
                    )
                end
            end
        end
    end

    return deepcopy(outputs)
end

function fit_cvae(Ymeta; marginalize_Z)
    # CVAE posterior state
    @timeit "posterior state" @unpack θ, Z, X, δ, ϵ, ν, ℓ = posterior_state(
        derived["cvae"], derived["prior"], Ymeta;
        miniter = 1, maxiter = 1, alpha = 0.0, verbose = false, mode = :maxlikelihood,
    )

    mle_init, mle_res = nothing, nothing
    if false #TODO marginalize_Z
        # If Z is marginalized over, fit noise directly via MLE
        @timeit "mle noise fit" begin
            mle_init, mle_res = mle_biexp_epg_noise_only(ν, signal(Ymeta); verbose = true, batch_size = size(signal(Ymeta), 2), checkpoint = false, dryrun = true, dryrunsamples = nothing)
            ℓ .= arr_similar(ν, mle_res.loss)'
            ϵ .= arr_similar(ν, mle_res.logepsilon)' .|> exp
            # Z .= 0 #TODO marginalize_Z # Z is filled with garbage
            X̂ = add_noise_instance(derived["ricegen"], ν, ϵ)
        end
    else
        # Add noise (Z-dependent, i.e. ϵ = ϵ(Z))
        X̂ = add_noise_instance(derived["ricegen"], ν, ϵ)
    end

    return (; Y = signal(Ymeta), X̂, θ, Z, X, δ, ϵ, ν, ℓ, mle_init, mle_res)
end

function fit_metrics(Ymeta, Ymeta_fit_state, θtrue, Ztrue, Wtrue)
    #= TODO
    if hasclosedform(phys)
        W = sampleWprior(ClosedForm(phys), Y, size(Y, 2)) # Sample hidden latent variables
        Y_metrics, Y_cache_cb_args = fit_metrics(nothing, θ, nothing, W)
    else
        Y_metrics, Y_cache_cb_args = fit_metrics(Y, nothing, nothing, nothing)
    end
    =#
    #= TODO
    νtrue = ϵtrue = rmse_true = logL_true = missing
    if hasclosedform(phys) && (θtrue !== nothing) && (Wtrue !== nothing)
        νtrue, ϵtrue = rician_params(ClosedForm(phys), θtrue, Wtrue) # noiseless true signal and noise level
        Ytrue = add_noise_instance(phys, νtrue, ϵtrue) # noisy true signal
        rmse_true = sqrt(mean(abs2, Ytrue - νtrue))
        logL_true = mean(NegLogLikelihood(derived["ricegen"], Ytrue, νtrue, ϵtrue))
    end
    =#

    @unpack X̂, θ, Z, X, δ, ϵ, ν, ℓ = Ymeta_fit_state
    all_rmse = sqrt.(mean(abs2, signal(Ymeta) .- ν; dims = 1)) |> Flux.cpu |> vec |> copy
    all_logL = ℓ |> Flux.cpu |> vec |> copy
    rmse, logL = mean(all_rmse), mean(all_logL)
    theta_err = (θtrue === nothing) ? missing : mean(abs, θerror(phys, θtrue, θ); dims = 2) |> Flux.cpu |> vec |> copy
    Z_err = (Ztrue === nothing) ? missing : mean(abs, Ztrue .- Z; dims = 2) |> Flux.cpu |> vec |> copy

    metrics = (; all_rmse, all_logL, rmse, logL, theta_err, Z_err, rmse_true = missing, logL_true = missing)
    cache_cb_args = (signal(Ymeta), θ, Z, X, δ, ϵ, ν, X̂, missing) # νtrue

    return metrics, cache_cb_args
end

function compute_metrics(engine, batch; dataset)
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
        accum!(k, v) = (d = cb_state["all_log_metrics"][dataset]; (!is_consecutive || !haskey(d, k)) ? (d[Symbol(k)] = Any[v]) : push!(d[Symbol(k)], v))
        accum!(k, v::Histogram) = (d = cb_state["all_histograms"][dataset]; (!is_consecutive || !haskey(d, k)) ? (d[Symbol(k)] = v) : (d[Symbol(k)].weights .+= v.weights))
        accum!(iter) = foreach(((k,v),) -> accum!(k, v), collect(pairs(iter)))

        cb_state["log_metrics"][:epoch]   = trainer.state.epoch
        cb_state["log_metrics"][:iter]    = trainer.state.iteration
        cb_state["log_metrics"][:dataset] = dataset
        cb_state["log_metrics"][:time]    = cb_state["curr_time"]

        # Invert Y and make Xs
        img_idx, img, Y, Ymeta = sample_batch(:val; batchsize = settings["eval"]["batchsize"], img_idx = mod1(engine.state.iteration, length(phys.images)))
        Y_fit_state = fit_cvae(Ymeta; marginalize_Z = false) #TODO marginalize_Z
        X̂meta = MetaCPMGSignal(phys, img, Y_fit_state.X̂)
        X̂_fit_state = fit_cvae(X̂meta; marginalize_Z = false) #TODO marginalize_Z

        let
            ℓ_CVAE = CVAElosses(Ymeta; marginalize_Z = false) #TODO marginalize_Z
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

            if settings["train"]["GANrate"]::Int > 0
                ℓ_GAN = Dloss(Y_fit_state.X, Y, Y_fit_state.Z)
                ℓ_GAN = push!!(ℓ_GAN, :Dloss => sum(ℓ_GAN))
                ℓ_GAN = push!!(ℓ_GAN, :Gloss => -ℓ_GAN.D_G_X)
                accum!(ℓ_GAN)
            end
        end

        # Cache cb state variables using naming convention
        cache_cb_state!(Y, θ, Z, Xθ, δθ, ϵθ, Xθδ, Xθhat, Yθ; suf::String) = foreach(((k,v),) -> (cb_state[string(k) * suf] = Flux.cpu(v)), pairs((; Y, θ, Z, Xθ, δθ, ϵθ, Xθδ, Xθhat, Yθ)))

        # Cache values for evaluating CVAE performance for estimating parameters of Y
        let
            Y_metrics, Y_cache_cb_args = fit_metrics(Ymeta, Y_fit_state, nothing, nothing, nothing)
            cache_cb_state!(Y_cache_cb_args...; suf = "")
            cb_state["metrics"]["all_Yhat_rmse"] = Y_metrics.all_rmse
            cb_state["metrics"]["all_Yhat_logL"] = Y_metrics.all_logL
            accum!(Dict(Symbol(:Yhat_, k) => v for (k,v) in pairs(Y_metrics) if k ∉ (:all_rmse, :all_logL) && !ismissing(v)))
        end

        # Cache values for evaluating CVAE performance for estimating parameters of X̂
        let
            X̂_metrics, X̂_cache_cb_args = fit_metrics(X̂meta, X̂_fit_state, Y_fit_state.θ, Y_fit_state.Z, nothing)
            cache_cb_state!(X̂_cache_cb_args...; suf = "fit")
            cb_state["metrics"]["all_Xhat_rmse"] = X̂_metrics.all_rmse
            cb_state["metrics"]["all_Xhat_logL"] = X̂_metrics.all_logL
            accum!(Dict(Symbol(:Xhat_, k) => v for (k,v) in pairs(X̂_metrics) if k ∉ (:all_rmse, :all_logL) && !ismissing(v)))

            img_key = Symbol(:img, img_idx)
            accum!(img_key, fast_hist_1D(Flux.cpu(vec(Y_fit_state.X̂)), img.meta[:histograms][dataset][0].edges[1]))
            Dist_L1 = CityBlock(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            Dist_ChiSq = ChiSquared(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            Dist_KLDiv = KLDivergence(img.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][img_key])
            accum!((; Dist_L1, Dist_ChiSq, Dist_KLDiv))
        end

        # Update logger dataframe and return metrics for logging
        foreach(((k,v),) -> (cb_state["log_metrics"][k] = mean(v)), cb_state["all_log_metrics"][dataset]) # update log metrics
        !is_consecutive ? push!(logger, cb_state["log_metrics"]; cols = :union) : foreach(((k,v),) -> (logger[end,k] = v), pairs(cb_state["log_metrics"]))
        output_metrics = Dict{Any,Any}(string(k) => deepcopy(v) for (k,v) in cb_state["log_metrics"] if k ∉ [:epoch, :iter, :dataset, :time]) # output non-housekeeping metrics
        merge!(cb_state["metrics"], output_metrics) # merge all log metrics into cb_state
        filter!(((k,v),) -> !ismissing(v), output_metrics) # return non-missing metrics (wandb cannot handle missing)

        return output_metrics
    end
end

function makeplots(;showplot = false)
    try
        Dict{Symbol, Any}(
            :ricemodel    => Working.plot_rician_model(logger, cb_state, phys; showplot, bandwidths = (filter(((k,v),) -> startswith(k, "logsigma"), collect(models)) |> logσs -> isempty(logσs) ? nothing : (x->Flux.cpu(permutedims(x[2]))).(logσs))),
            :signals      => Working.plot_rician_signals(logger, cb_state, phys; showplot),
            :signalmodels => Working.plot_rician_model_fits(logger, cb_state, phys; showplot),
            :infer        => Working.plot_rician_inference(logger, cb_state, phys; showplot),
            :ganloss      => Working.plot_gan_loss(logger, cb_state, phys; showplot, lrdroprate = settings["opt"]["lrrate"], lrdrop = settings["opt"]["lrdrop"]),
            :vallosses    => Working.plot_all_logger_losses(logger, cb_state, phys; showplot, dataset = :val),
            :trainlosses  => Working.plot_all_logger_losses(logger, cb_state, phys; showplot, dataset = :train),
            # :epsline      => Working.plot_epsilon(phys, derived; showplot, seriestype = :line), #TODO
            # :epscontour   => Working.plot_epsilon(phys, derived; showplot, seriestype = :contour), #TODO
            :priors       => Working.plot_priors(phys, derived; showplot), #TODO
            :cvaepriors   => Working.plot_cvaepriors(phys, derived; showplot), #TODO
            # :posteriors   => Working.plot_posteriors(phys, derived; showplot), #TODO
        )
    catch e
        handleinterrupt(e; msg = "Error plotting")
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
    @j2p Ignite.terminate_file_event(file = joinpath(settings["data"]["out"], "stop"))
)

# Timeout
trainer.add_event_handler(
    Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.timeout_event_filter(settings["train"]["timeout"])),
    @j2p function (engine)
        @info "Exiting: training time exceeded $(DECAES.pretty_time(settings["train"]["timeout"]))"
        engine.terminate()
    end
)

# Compute callback metrics
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.throttler_event_filter(settings["eval"]["valevalperiod"])),
    @j2p (engine) -> val_evaluator.run(val_eval_loader)
)
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.throttler_event_filter(settings["eval"]["trainevalperiod"])),
    @j2p (engine) -> train_evaluator.run(train_eval_loader)
)

# Checkpoint current model + logger + make plots
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.throttler_event_filter(settings["eval"]["saveperiod"])),
    @j2p function (engine)
        @timeit "checkpoint" let models = map_dict(Flux.cpu, models)
            @timeit "save current model" saveprogress(@dict(models, logger); savefolder = settings["data"]["out"], prefix = "current-")
            @timeit "make current plots" plothandles = makeplots()
            @timeit "save current plots" saveplots(plothandles; savefolder = settings["data"]["out"], prefix = "current-")
        end
    end
)

# Check for + save best model + logger + make plots
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.throttler_event_filter(settings["eval"]["saveperiod"])),
    @j2p function (engine)
        loss_metric = :CVAE # :Yhat_logL
        losses = logger[logger.dataset .=== :val, loss_metric] |> skipmissing |> collect
        if !isempty(losses) && (length(losses) == 1 || losses[end] < minimum(losses[1:end-1]))
            @timeit "save best progress" let models = map_dict(Flux.cpu, models)
                @timeit "save best model" saveprogress(@dict(models, logger); savefolder = settings["data"]["out"], prefix = "best-")
                @timeit "make best plots" plothandles = makeplots()
                @timeit "save best plots" saveplots(plothandles; savefolder = settings["data"]["out"], prefix = "best-")
            end
        end
    end
)

#=
# Linear warmup
trainer.add_event_handler(
    Events.STARTED | Events.ITERATION_COMPLETED,
    @j2p function (engine)
        Ignite.update_optimizers!(optimizers; field = :eta) do opt, name
            lr_warmup = settings["opt"][name]["lrwarmup"]
            lr_initial = Working.initial_learning_rate!(settings, name)
            !(engine.state.iteration <= lr_warmup > 0) && return
            new_eta = range(lr_initial / lr_warmup, lr_initial; length = lr_warmup + 1)[engine.state.iteration + 1]
            opt.eta = new_eta
        end
    end
)
=#

# Drop learning rate
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p Ignite.droplr_file_event(optimizers;
        file = joinpath(settings["data"]["out"], "droplr"),
        lrrate = settings["opt"]["lrrate"]::Int,
        lrdrop = settings["opt"]["lrdrop"]::Float64,
        lrthresh = settings["opt"]["lrthresh"]::Float64,
    )
)

# User input
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p(Ignite.file_event(Ignite.user_input_event(); file = joinpath(settings["data"]["out"], "user"))),
)

# Print TimerOutputs timings
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.throttler_event_filter(settings["eval"]["printperiod"])),
    @j2p function (engine)
        @info sprint() do io
            println(io, "Log folder: $(settings["data"]["out"])"); println(io, "\n")
            show(io, TimerOutputs.get_defaulttimer()); println(io, "\n")
            show(io, last(logger[!,[names(logger)[1:4]; sort(names(logger)[5:end])]], 10)); println(io, "\n")
        end
        (engine.state.epoch == 1) && TimerOutputs.reset_timer!() # throw out compilation timings
    end
)

# Reset loging/timer/etc.
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p Ignite.file_event(file = joinpath(settings["data"]["out"], "reset")) do engine
        TimerOutputs.reset_timer!()
        empty!(cb_state)
        empty!(logger)
    end
)

####
#### Weights & biases logger
####

# Attach training/validation output handlers
if (wandb_logger !== nothing)
    for (tag, engine, event) in [
            ("step",  trainer,         Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.timeout_event_filter(settings["eval"]["trainevalperiod"]))), # computed each iteration; throttle recording
            ("train", train_evaluator, Events.EPOCH_COMPLETED), # throttled above; record every epoch
            ("val",   val_evaluator,   Events.EPOCH_COMPLETED), # throttled above; record every epoch
        ]
        wandb_logger.attach_output_handler(engine;
            tag = tag,
            event_name = event,
            output_transform = @j2p(metrics -> metrics),
            global_step_transform = @j2p((args...; kwargs...) -> trainer.state.epoch),
        )
    end
end

####
#### Run trainer
####

TimerOutputs.reset_timer!()
trainer.run(train_loader, settings["train"]["epochs"])
