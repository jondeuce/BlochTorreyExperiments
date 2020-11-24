####
#### Setup
####

using MMDLearning
using PyCall
pyplot(size=(800,600))
Ignite.init()

# Init python resources
const torch = pyimport("torch")
const wandb = pyimport("wandb")
const ignite = pyimport("ignite")
const logging = pyimport("logging")
py"""
from ignite.contrib.handlers.wandb_logger import *
"""

using MMDLearning: map_dict, sum_dict, apply_dim1
const Events = ignite.engine.Events

# Parse command line arguments into default settings
settings = if haskey(ENV, "JL_CHECKPOINT_FOLDER")
    # Load settings from past run
    TOML.parsefile(joinpath(ENV["JL_CHECKPOINT_FOLDER"], "settings.toml"))
else
    # Load default settings with command line arguments
    Ignite.parse_command_line!(TOML.parse(
    """
    [data]
        out    = "./output/ignite-cvae-$(MMDLearning.getnow())"
        ntrain = "auto" # 102_400
        ntest  = "auto" # 10_240
        nval   = "auto" # 10_240

    [train]
        timeout     = 1e9 #TODO 10800.0
        epochs      = 1000_000
        batchsize   = 1024 #256 #512 #1024 #2048 #3072 #4096
        PseudoCVAE  = false # Label data signals with pseudolabels from the CVAE
        MMDCVAErate = 0     # Train combined MMD+CVAE loss every `MMDCVAErate` epochs
        CVAErate    = 1     # Train CVAE loss every `CVAErate` iterations
        CVAEsteps   = 1     # Train CVAE losses with `CVAEsteps` updates per iteration
        MMDrate     = 0     # Train MMD loss every `MMDrate` epochs
        GANrate     = 0     # Train GAN losses every `GANrate` iterations
        Dsteps      = 5     # Train GAN losses with `Dsteps` discrim updates per genatr update
        # GANcycle    = 1   # CVAE and GAN take turns training for `GANcycle` consecutive epochs (0 trains both each iteration)
        # Dcycle      = 0   # Train for `Dcycle` epochs of discrim only, followed by `Dcycle` epochs of CVAE and GAN together
        # Dheadstart  = 0   # Train discriminator for `Dheadstart` epochs before training generator
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
        valevalperiod   = 300.0
        trainevalperiod = 600.0
        saveperiod      = 300.0
        printperiod     = 300.0

    [opt]
        lrrel    = 0.03    #0.1  # Learning rate relative to batch size, i.e. lr = lrrel / batchsize
        lrthresh = 0.0     #1e-6 # Absolute minimum learning rate
        lrdrop   = 3.16    # Drop learning rate by factor `lrdrop` every `lrrate` epochs
        lrrate   = 999_999 # Drop learning rate by factor `lrdrop` every `lrrate` epochs
        [opt.cvae]
            lrrel = "%PARENT%"
        [opt.genatr]
            lrrel = "%PARENT%" #TODO: 0.01 train generator more slowly
        [opt.discrim]
            lrrel = "%PARENT%"
        [opt.mmd]
            lrrel = "%PARENT%"
            gclip = 1.0
            lambda_0        = 100.0 # MMD loss weighting relative to CVAE
            lambda_eps      = 0.0   # Regularize noise amplitude epsilon
            lambda_deps_dz  = 0.0   # Regularize gradient of epsilon w.r.t. latent variables
        [opt.kernel]
            lrrel = "%PARENT%" # Kernel learning rate 
            loss  = "mmd"      # Kernel loss ("mmd", "tstatistic", or "mmd_diff")

    [arch]
        physics   = "epg"
        nlatent   = 1   # number of latent variables Z
        zdim      = 12  # embedding dimension of z
        hdim      = 256 # size of hidden layers
        nhidden   = 4   # number of hidden layers
        skip      = false # skip connection
        layernorm = false # layer normalization following dense layer
        [arch.enc1]
            psize     = 32
            head      = 4
            hdim      = "%PARENT%"
            nhidden   = "%PARENT%"
        [arch.enc2]
            psize     = 32
            head      = 4
            hdim      = "%PARENT%"
            nhidden   = "%PARENT%"
        [arch.dec]
            hdim      = "%PARENT%"
            nhidden   = "%PARENT%"
        [arch.genatr]
            ktheta      = 16  # Dimension of domain of theta prior space
            klatent     = 4   # Dimension of domain of latent prior space
            hdim        = 64  #TODO "%PARENT%"
            nhidden     = 2   #TODO "%PARENT%"
            leakyslope  = 0.0
            maxcorr     = 0.1
            noisebounds = [-6.0, 0.0]
        [arch.discrim]
            dropout   = 0.1
            hdim      = 0     #TODO "%PARENT%"
            nhidden   = 0     #TODO "%PARENT%"
        [arch.kernel]
            nbandwidth  = 32            #TODO
            channelwise = true          #TODO
            deep        = false         #TODO
            bwbounds    = [-8.0, 4.0]   # Bounds for kernel bandwidths (logsigma)
            clampnoise  = 0.0           #TODO
    """))
end

# Init wandb_logger and save settings
wandb_logger = Ignite.init_wandb_logger(settings)
!isnothing(wandb_logger) && (settings["data"]["out"] = wandb.run.dir)
Ignite.save_and_print(settings; filename = !haskey(ENV, "JL_CHECKPOINT_FOLDER") ? joinpath(settings["data"]["out"], "settings.toml") : nothing)

####
#### Models
####

# phys = initialize!(MMDLearning.ToyEPGModel{Float32,true}(); ntrain = settings["data"]["ntrain"]::Int, ntest = settings["data"]["ntest"]::Int, nval = settings["data"]["nval"]::Int)
phys = initialize!(MMDLearning.EPGModel{Float32,true}(); seed = 0, imagepath = "/home/jdoucette/Documents/code/MWI-Julia-Paper/Example_48echo_8msTE/data-in/ORIENTATION_B0_08_WIP_MWF_CPMG_CS_AXIAL_5_1.masked-image.nii.gz")
models_checkpoint = !haskey(ENV, "JL_CHECKPOINT_FOLDER") ? Dict{String, Any}() : map_dict(to32, deepcopy(BSON.load(joinpath(ENV["JL_CHECKPOINT_FOLDER"], "current-models.bson"))["models"]))
models, derived = MMDLearning.make_mmd_cvae_models(phys, settings, models_checkpoint)
make_opt(otype = Flux.ADAM; lr, gclip = 0, wdecay = 0) = (os = Any[otype(lr)]; (gclip > 0) && pushfirst!(os, Flux.ClipValue(gclip)); (wdecay > 0) && push!(os, Flux.WeightDecay(wdecay)); return Flux.Optimiser(os))
optimizers = Dict{String,Any}(
    "mmd"     => make_opt(lr = settings["opt"]["discrim"]["lrrel"] / settings["train"]["batchsize"], gclip = settings["opt"]["mmd"]["gclip"]),
    "cvae"    => make_opt(lr = settings["opt"]["cvae"]["lrrel"] / settings["train"]["batchsize"]),
    "genatr"  => make_opt(lr = settings["opt"]["genatr"]["lrrel"] / settings["train"]["batchsize"]),
    "discrim" => make_opt(lr = settings["opt"]["discrim"]["lrrel"] / settings["train"]["batchsize"]),
)
MMDLearning.model_summary(models, !haskey(ENV, "JL_CHECKPOINT_FOLDER") ? joinpath(settings["data"]["out"], "model-summary.txt") : nothing)

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
    Xres = settings["train"]["augment"]["residuals"]::Bool ? X .- Zygote.@ignore(sampleX(derived["cvae"], derived["prior"], X; posterior_θ = true, posterior_Z = true)) : nothing # Residual relative to different sample X̄(θ), θ ~ P(θ|X) (note: Z discarded, posterior_Z irrelevant)
    Xenc = settings["train"]["augment"]["encoderspace"]::Bool ? derived["encoderspace"](X) : nothing # Encoder-space signal
    Xfft = settings["train"]["augment"]["fftcat"]::Bool ? vcat(reim(rfft(X,1))...) : nothing # Concatenated real/imag fourier components
    Xrfft, Xifft = settings["train"]["augment"]["fftsplit"]::Bool ? reim(rfft(X,1)) : (nothing, nothing) # Separate real/imag fourier components

    ks = (:signal, :grad, :lap, :fd, :res, :enc, :fft, :rfft, :ifft)
    Xs = [X₀, ∇X, ∇²X, ∇ⁿX, Xres, Xenc, Xfft, Xrfft, Xifft]
    is = (!isnothing).(Xs)
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
        make_opt(lr = settings["opt"]["kernel"]["lrrel"] / settings["train"]["batchsize"])
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
        MMDLearning.DeepExponentialKernel(
            models["logsigma_$key"],
            !deep ?
                identity :
                Flux.Chain(
                    MMDLearning.MLP(nchannel => nz, 0, hdim, Flux.relu, identity; skip = false),
                    z -> Flux.normalise(z; dims = 1), # kernel bandwidths are sensitive to scale; normalize learned representations
                    z -> z .+ randn_similar(z, size(z)...), # stochastic embedding prevents overfitting to Y data
                ) |> MMDLearning.flattenchain |> to32,
            )
    end

    return derived["kernel_$key"]
end

MMDloss(k, X::AbstractMatrix, Y::AbstractMatrix) = size(Y,2) * mmd(k, X, Y)
MMDloss(k, X::AbstractTensor3D, Y::AbstractMatrix) = mean(map(i -> MMDloss(k, X[:,:,i], Y), 1:size(X,3)))

# Maximum mean discrepency (m*MMD^2) loss
function MMDlosses(Y)
    if settings["train"]["DeepThetaPrior"]::Bool && settings["train"]["DeepLatentPrior"]::Bool
        X, θ, Z = sampleXθZ(derived["cvae"], derived["prior"], Y; posterior_θ = false, posterior_Z = false) # sample θ and Z from the learned deep priors, differentiating through the sampling process and the physics model
    else
        Z = settings["train"]["DeepLatentPrior"]::Bool ? sampleZprior(derived["prior"], Y) : Zygote.@ignore(sampleZprior(derived["prior"], Y)) # differentiate through deep Z prior, else ignore
        θ = settings["train"]["DeepThetaPrior"]::Bool ? sampleθprior(derived["prior"], Y) : Zygote.@ignore(sampleθZ(derived["cvae"], derived["prior"], Y; posterior_θ = true, posterior_Z = true)[1]) # use CVAE posterior for θ prior; Z is ignored w/ `posterior_Z` arbitrary
        X = settings["train"]["DeepThetaPrior"]::Bool ? signal_model(phys, θ) : Zygote.@ignore(signal_model(phys, θ)) # differentiate through physics model if learning deep θ prior, else ignore
    end

    # Differentiate through generator corrections `sampleX̂`
    nX̂ = settings["train"]["transform"]["nsamples"]::Int |> n -> ifelse(n > 1, n, nothing)
    ν, ϵ = rician_params(derived["ricegen"], X, Z)
    X̂ = add_noise_instance(derived["ricegen"], ν, ϵ, nX̂)

    X̂s, Ys = augment_and_transform(X̂, Y)
    ks = Zygote.@ignore NamedTuple{keys(X̂s)}(get_mmd_kernel.(keys(X̂s), size.(values(X̂s),1)))
    ℓ  = map(MMDloss, ks, X̂s, Ys)

    λ_ϵ = eltype(Y)(settings["opt"]["mmd"]["lambda_eps"]::Float64)
    if λ_ϵ > 0
        R = λ_ϵ * noiselevel_regularization(ϵ, Val(:L1grad))
        ℓ = push!!(ℓ, :reg_eps => R)
    end

    λ_∂ϵ∂Z = eltype(Y)(settings["opt"]["mmd"]["lambda_deps_dz"]::Float64)
    if λ_∂ϵ∂Z > 0
        R = λ_∂ϵ∂Z * noiselevel_gradient_regularization(ϵ, Z, Val(:L2diff))
        ℓ = push!!(ℓ, :reg_Z => R)
    end

    return ℓ
end

####
#### CVAE
####

# Conditional variational autoencoder losses
function CVAElosses(Y; marginalize_Z)
    # Sample X̂,θ,Z from priors
    X̂, θ, Z = Zygote.@ignore sampleX̂θZ(derived["cvae"], derived["prior"], Y; posterior_θ = false, posterior_Z = false) # sample θ and Z priors

    # Cross-entropy loss function components
    KLDiv, ELBO = KL_and_ELBO(derived["cvae"], X̂, θ, Z; marginalize_Z)

    ℓ = if !settings["train"]["PseudoCVAE"]::Bool
        (; KLDiv, ELBO)
    else
        # Use inferred params as pseudolabels for Y
        θY, ZY = Zygote.@ignore sampleθZ(derived["cvae"], derived["prior"], Y; posterior_θ = true, posterior_Z = true) # draw pseudo θ and Z labels for Y
        KLDivPseudo, ELBOPseudo = KL_and_ELBO(derived["cvae"], Y, θY, ZY; marginalize_Z) # recover pseudo θ and Z labels
        (; KLDiv, ELBO, KLDivPseudo, ELBOPseudo)
    end

    return ℓ
end

####
#### Training
####

# Global state
const cb_state = Dict{String,Any}()
const logger = DataFrame(
    :epoch      => Int[], # mandatory field
    :iter       => Int[], # mandatory field
    :dataset    => Symbol[], # mandatory field
    :time       => Union{Float64, Missing}[],
)

make_data_tuples(dataset) = tuple.(copy.(eachcol(sampleY(phys, :all; dataset = dataset))))
train_loader = torch.utils.data.DataLoader(make_data_tuples(:train); batch_size = settings["train"]["batchsize"], shuffle = true, drop_last = true)
val_loader = torch.utils.data.DataLoader(make_data_tuples(:val); batch_size = settings["train"]["batchsize"], shuffle = false, drop_last = true) #Note: drop_last=true and batch_size=train_batchsize for MMD (else, batch_size = settings["data"]["nval"] is fine)

function train_step(engine, batch)
    Ytrain_cpu, = Ignite.array.(batch)
    Ytrain = Ytrain_cpu |> todevice
    outputs = Dict{Any,Any}()

    @timeit "train batch" CUDA.@sync begin
        every(rate) = rate <= 0 ? false : mod(engine.state.iteration-1, rate) == 0
        train_MMDCVAE = every(settings["train"]["MMDCVAErate"]::Int)
        train_CVAE = every(settings["train"]["CVAErate"]::Int)
        train_MMD = every(settings["train"]["MMDrate"]::Int)
        train_GAN = train_discrim = train_genatr = every(settings["train"]["GANrate"]::Int)
        train_k = every(settings["train"]["kernelrate"]::Int)

        # Train Self MMD CVAE loss
        train_MMDCVAE && @timeit "mmd + cvae" CUDA.@sync let
            deeppriors = [models["theta_prior"], models["latent_prior"]][[settings["train"]["DeepThetaPrior"]::Bool, settings["train"]["DeepLatentPrior"]::Bool]]
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"], models["genatr"], deeppriors...)
            λ_0 = eltype(Ytrain)(settings["opt"]["mmd"]["lambda_0"]::Float64)
            @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(ps) do
                mmd = sum(MMDlosses(Ytrain))
                cvae = sum(CVAElosses(Ytrain; marginalize_Z = false))
                return λ_0 * mmd + cvae
            end
            @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
            @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["mmd"], ps, gs)
            outputs["loss"] = ℓ
        end

        # Train CVAE loss
        train_CVAE && @timeit "cvae" CUDA.@sync let
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"])
            for _ in 1:settings["train"]["CVAEsteps"]
                @timeit "forward"   CUDA.@sync ℓ, back = Zygote.pullback(() -> sum(CVAElosses(Ytrain; marginalize_Z = false)), ps)
                @timeit "reverse"   CUDA.@sync gs = back(one(eltype(phys)))
                @timeit "update!"   CUDA.@sync Flux.Optimise.update!(optimizers["cvae"], ps, gs)
                outputs["CVAE"] = ℓ
            end
        end

        # Train MMD loss
        train_MMD && @timeit "mmd" CUDA.@sync let
            @timeit "genatr" CUDA.@sync let
                deeppriors = [models["theta_prior"], models["latent_prior"]][[settings["train"]["DeepThetaPrior"]::Bool, settings["train"]["DeepLatentPrior"]::Bool]]
                ps = Flux.params(models["genatr"], deeppriors...)
                @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> sum(MMDlosses(Ytrain)), ps)
                @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["mmd"], ps, gs)
                outputs["MMD"] = ℓ
            end
        end

        # Train GAN loss
        train_GAN && @timeit "gan" CUDA.@sync let
            @timeit "sampleXθZ" CUDA.@sync Xtrain, θtrain, Ztrain = sampleXθZ(derived["cvae"], derived["prior"], Ytrain; posterior_θ = true, posterior_Z = false) # learn to map whole Z domain via `posterior_Z = false`
            train_discrim && @timeit "discrim" CUDA.@sync let
                ps = Flux.params(models["discrim"])
                for _ in 1:settings["train"]["Dsteps"]
                    @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> sum(Dloss(Xtrain, Ytrain, Ztrain)), ps)
                    @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                    @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["discrim"], ps, gs)
                    outputs["Dloss"] = ℓ
                end
            end
            train_genatr && @timeit "genatr" CUDA.@sync let
                deeppriors = [models["theta_prior"], models["latent_prior"]][[settings["train"]["DeepThetaPrior"]::Bool, settings["train"]["DeepLatentPrior"]::Bool]]
                ps = Flux.params(models["genatr"], deeppriors...)
                @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> sum(Gloss(Xtrain, Ztrain)), ps)
                @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["genatr"], ps, gs)
                outputs["Gloss"] = ℓ
            end
        end

        # Train MMD kernels
        train_k && @timeit "kernel" CUDA.@sync let
            noisyclamp!(x::AbstractArray{T}, lo, hi, ϵ) where {T} = clamp!(x .+ T(ϵ) .* randn_similar(x, size(x)...), T(lo), T(hi))
            restrict!(k) = noisyclamp!(MMDLearning.logbandwidths(k), -Inf, Inf, settings["arch"]["kernel"]["clampnoise"])
            aug_types, aug_Ytrains = augment_and_transform(Ytrain) |> first |> Y -> (keys(Y), values(Y)) # augment data
            opts = (get_kernel_opt(aug) for aug in aug_types) # unique optimizer per augmentation
            kernels = (get_mmd_kernel(aug, size(Y,1)) for (aug, Y) in zip(aug_types, aug_Ytrains)) # unique kernel per augmentation
            for _ in 1:settings["train"]["kernelsteps"]::Int
                @timeit "sample G(X)" X̂train = sampleX̂(derived["cvae"], derived["prior"], Ytrain; posterior_θ = true, posterior_Z = false) # # sample unique X̂ per step (TODO: posterior_Z = true? or posterior_Z = false to force learning of whole Z domain?)
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

function fit_metrics(Ytrue, θtrue, Ztrue, Wtrue)
    νtrue = ϵtrue = rmse_true = logL_true = missing
    if hasclosedform(phys) && !isnothing(θtrue) && !isnothing(Wtrue)
        νtrue, ϵtrue = rician_params(ClosedForm(phys), θtrue, Wtrue) # noiseless true signal and noise level
        Ytrue = add_noise_instance(phys, νtrue, ϵtrue) # noisey true signal
        rmse_true = sqrt(mean(abs2, Ytrue - νtrue))
        logL_true = mean(MMDLearning.NegLogLikelihood(derived["ricegen"], Ytrue, νtrue, ϵtrue))
    end

    @unpack θ, Z, X, δ, ϵ, ν, ℓ = MMDLearning.posterior_state(
        derived["cvae"], derived["prior"], Ytrue;
        miniter = 1, maxiter = 1, alpha = 0.0, verbose = false, mode = :maxlikelihood,
    )
    X̂ = add_noise_instance(derived["ricegen"], ν, ϵ)

    all_rmse = sqrt.(mean(abs2, Ytrue .- ν; dims = 1)) |> Flux.cpu |> vec |> copy
    all_logL = ℓ |> Flux.cpu |> vec |> copy
    rmse, logL = mean(all_rmse), mean(all_logL)
    theta_err = isnothing(θtrue) ? missing : mean(abs, θerror(phys, θtrue, θ); dims = 2) |> Flux.cpu |> vec |> copy
    Z_err = isnothing(Ztrue) ? missing : mean(abs, Ztrue .- Z; dims = 2) |> Flux.cpu |> vec |> copy

    metrics = (; rmse_true, logL_true, all_rmse, all_logL, rmse, logL, theta_err, Z_err)
    cache_cb_args = (Ytrue, θ, Z, X, δ, ϵ, ν, X̂, νtrue)
    # cache_cb_args = (; θ, Z, X, δ, ν, ϵ, X̂, Ytrue, θtrue, Ztrue, νtrue, ϵtrue) #TODO renaming...

    return metrics, cache_cb_args
end

function compute_metrics(engine, batch; dataset)
    @timeit "compute metrics" CUDA.@sync begin
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
        accum!(k, v) = !is_consecutive ? (cb_state["all_log_metrics"][dataset][Symbol(k)] = Any[v]) : push!(cb_state["all_log_metrics"][dataset][Symbol(k)], v)
        accum!(k, v::Histogram) = !is_consecutive ? (cb_state["all_histograms"][dataset][Symbol(k)] = v) : (cb_state["all_histograms"][dataset][Symbol(k)].weights .+= v.weights)
        accum!(iter) = foreach(((k,v),) -> accum!(k, v), collect(pairs(iter)))

        cb_state["log_metrics"][:epoch]   = trainer.state.epoch
        cb_state["log_metrics"][:iter]    = trainer.state.iteration
        cb_state["log_metrics"][:dataset] = dataset
        cb_state["log_metrics"][:time]    = cb_state["curr_time"]

        # Invert Y and make Xs
        Y, = Ignite.array.(batch) .|> todevice
        X, θ, Z = sampleXθZ(derived["cvae"], derived["prior"], Y; posterior_θ = true, posterior_Z = true)
        X̂ = sampleX̂(derived["ricegen"], X, Z)
        Nbatch = size(Y,2)

        let
            ℓ_CVAE = CVAElosses(Y; marginalize_Z = false)
            ℓ_CVAE = push!!(ℓ_CVAE, :CVAE => sum(ℓ_CVAE))
            accum!(ℓ_CVAE)

            ℓ_MMD = MMDlosses(Y)
            ℓ_MMD = NamedTuple{Symbol.(:MMD_, keys(ℓ_MMD))}(values(ℓ_MMD)) # prefix labels with "MMD_"
            ℓ_MMD = push!!(ℓ_MMD, :MMD => sum(ℓ_MMD))
            accum!(ℓ_MMD)

            λ_0 = eltype(Y)(settings["opt"]["mmd"]["lambda_0"]::Float64)
            loss = ℓ_CVAE.CVAE + λ_0 * ℓ_MMD.MMD
            Zreg = sum(abs2, Z; dims = 2) / (2*Nbatch) |> Flux.cpu
            Zdiv = [KLDivUnitNormal(mean_and_std(Z[i,:])...) for i in 1:size(Z,1)] |> Flux.cpu
            accum!((; loss, Zreg, Zdiv))

            if settings["train"]["GANrate"]::Int > 0
                ℓ_GAN = Dloss(X,Y,Z)
                ℓ_GAN = push!!(ℓ_GAN, :Dloss => sum(ℓ_GAN))
                ℓ_GAN = push!!(ℓ_GAN, :Gloss => -ℓ_GAN.D_G_X)
                accum!(ℓ_GAN)
            end
        end

        # Cache cb state variables using naming convention
        cache_cb_state!(Y, θ, Z, Xθ, δθ, ϵθ, Xθδ, Xθhat, Yθ; suf::String) = foreach(((k,v),) -> (cb_state[string(k) * suf] = Flux.cpu(v)), pairs((; Y, θ, Z, Xθ, δθ, ϵθ, Xθδ, Xθhat, Yθ)))

        # Cache values for evaluating CVAE performance for estimating parameters of Y
        let
            if hasclosedform(phys)
                W = sampleWprior(ClosedForm(phys), Y, size(Y, 2)) # Sample hidden latent variables
                Y_metrics, Y_cache_cb_args = fit_metrics(nothing, θ, nothing, W)
            else
                Y_metrics, Y_cache_cb_args = fit_metrics(Y, nothing, nothing, nothing)
            end
            cache_cb_state!(Y_cache_cb_args...; suf = "")
            cb_state["metrics"]["all_Yhat_rmse"] = Y_metrics.all_rmse
            cb_state["metrics"]["all_Yhat_logL"] = Y_metrics.all_logL
            accum!(Dict(Symbol(:Yhat_, k) => v for (k,v) in pairs(Y_metrics) if k ∉ (:all_rmse, :all_logL) && !ismissing(v)))
        end

        # Cache values for evaluating CVAE performance for estimating parameters of X̂
        let
            X̂_metrics, X̂_cache_cb_args = fit_metrics(X̂, θ, Z, nothing)
            cache_cb_state!(X̂_cache_cb_args...; suf = "fit")
            cb_state["metrics"]["all_Xhat_rmse"] = X̂_metrics.all_rmse
            cb_state["metrics"]["all_Xhat_logL"] = X̂_metrics.all_logL
            accum!(Dict(Symbol(:Xhat_, k) => v for (k,v) in pairs(X̂_metrics) if k ∉ (:all_rmse, :all_logL) && !ismissing(v)))

            accum!(:signal, MMDLearning.fast_hist_1D(Flux.cpu(vec(X̂)), phys.meta[:histograms][dataset][0].edges[1]))
            Dist_L1 = MMDLearning.CityBlock(phys.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][:signal])
            Dist_ChiSq = MMDLearning.ChiSquared(phys.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][:signal])
            Dist_KLDiv = MMDLearning.KLDivergence(phys.meta[:histograms][dataset][0], cb_state["all_histograms"][dataset][:signal])
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
    function plot_epsilon(; knots = (-1.0, 1.0), seriestype = :line, showplot = false)
        function plot_epsilon_inner(; start, stop, zlen = 256, levels = 50)
            #TODO fixed knots, start, stop
            n, nθ, nz = nsignal(phys)::Int, ntheta(phys)::Int, nlatent(derived["ricegen"])::Int
            Y = sampleY(phys, zlen * nz; dataset = :train) |> to32
            X, _, Z = sampleXθZ(derived["cvae"], derived["prior"], Y; posterior_θ = true, posterior_Z = false)
            Z = Z[:,1] |> Flux.cpu |> z -> repeat(z, 1, zlen, nz) |> z -> (foreach(i -> z[i,:,i] .= range(start, stop; length = zlen), 1:nz); z)
            _, ϵ = rician_params(derived["ricegen"], X, reshape(Z, nz, :) |> Flux.gpu)
            (size(ϵ,1) == 1) && (ϵ = repeat(ϵ, n, 1))
            log10ϵ = log10.(reshape(ϵ, :, zlen, nz)) |> Flux.cpu
            ps = map(1:nz) do i
                zlabs = nz == 1 ? "" : latexstring(" (" * join(map(j -> L"$Z_%$(j)$ = %$(round(Z[1,1,j]; digits = 2))", setdiff(1:nz, i)), ", ") * ")")
                kwcommon = (; leg = :none, colorbar = :right, color = cgrad(:cividis), xlabel = L"$t$", title = L"$\log_{10}\epsilon$ vs. $t$ and $Z_{%$(i)}$%$(zlabs)")
                if seriestype === :surface
                    surface(reshape(1:n,n,1), Z[i,:,i]', log10ϵ[:,:,i]; ylabel = L"$Z_{%$(i)}$", fill_z = log10ϵ[:,:,i], camera = (60.0, 30.0), kwcommon...)
                elseif seriestype === :contour
                    contourf(repeat(1:n,1,zlen), repeat(Z[i,:,i]',n,1), log10ϵ[:,:,i]; ylabel = L"$Z_{%$(i)}$", levels, kwcommon...)
                else
                    plot(log10ϵ[:,:,i]; line_z = Z[i,:,i]', ylabel = L"$\log_{10}\epsilon$", lw = 2, alpha = 0.3, kwcommon...)
                end
            end
            return ps
        end
        ps = mapreduce(vcat, 1:length(knots)-1; init = Any[]) do i
            plot_epsilon_inner(; start = knots[i], stop = knots[i+1])
        end
        p = plot(ps...)
        if showplot; display(p); end
        return p
    end

    function plot_priors(;showplot = false)
        θ = sampleθprior(derived["prior"], 10000) |> Flux.cpu
        Z = sampleZprior(derived["prior"], 10000) |> Flux.cpu
        pθs = [histogram(θ[i,:]; nbins = 100, label = θlabels(phys)[i], xlim = θbounds(phys)[i]) for i in 1:size(θ,1)]
        pZs = [histogram(Z[i,:]; nbins = 100, label = L"Z_%$i") for i in 1:size(Z,1)] #TODO
        p = plot(pθs..., pZs...)
        if showplot; display(p); end
        return p
    end

    function plot_posteriors(;showplot = false)
        Y = sampleY(phys, 10000; dataset = :train) |> to32
        θ, Z = sampleθZ(derived["cvae"], derived["prior"], Y; posterior_θ = true, posterior_Z = true) .|> Flux.cpu
        pθs = [histogram(θ[i,:]; nbins = 100, label = θlabels(phys)[i], xlim = θbounds(phys)[i]) for i in 1:size(θ,1)]
        pZs = [histogram(Z[i,:]; nbins = 100, label = L"Z_%$i") for i in 1:size(Z,1)] #TODO
        p = plot(pθs..., pZs...)
        if showplot; display(p); end
        return p
    end

    try
        Dict{Symbol, Any}(
            :ricemodel    => MMDLearning.plot_rician_model(logger, cb_state, phys; showplot, bandwidths = (filter(((k,v),) -> startswith(k, "logsigma"), collect(models)) |> logσs -> isempty(logσs) ? nothing : (x->Flux.cpu(permutedims(x[2]))).(logσs))),
            :signals      => MMDLearning.plot_rician_signals(logger, cb_state, phys; showplot),
            :signalmodels => MMDLearning.plot_rician_model_fits(logger, cb_state, phys; showplot),
            :infer        => MMDLearning.plot_rician_inference(logger, cb_state, phys; showplot),
            :ganloss      => MMDLearning.plot_gan_loss(logger, cb_state, phys; showplot, lrdroprate = settings["opt"]["lrrate"], lrdrop = settings["opt"]["lrdrop"]),
            :vallosses    => MMDLearning.plot_all_logger_losses(logger, cb_state, phys; showplot, dataset = :val),
            :trainlosses  => MMDLearning.plot_all_logger_losses(logger, cb_state, phys; showplot, dataset = :train),
            :epsline      => plot_epsilon(; showplot, seriestype = :line), #TODO
            :epscontour   => plot_epsilon(; showplot, seriestype = :contour), #TODO
            :priors       => plot_priors(; showplot), #TODO
            :posteriors   => plot_posteriors(; showplot), #TODO
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
    @j2p (engine) -> val_evaluator.run(val_loader)
)
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.throttler_event_filter(settings["eval"]["trainevalperiod"])),
    @j2p (engine) -> train_evaluator.run(train_loader)
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
        losses = logger.Yhat_logL[logger.dataset .=== :val] |> skipmissing |> collect
        if !isempty(losses) && (length(losses) == 1 || losses[end] < minimum(losses[1:end-1]))
            @timeit "save best progress" let models = map_dict(Flux.cpu, models)
                @timeit "save best model" saveprogress(@dict(models, logger); savefolder = settings["data"]["out"], prefix = "best-")
                @timeit "make best plots" plothandles = makeplots()
                @timeit "save best plots" saveplots(plothandles; savefolder = settings["data"]["out"], prefix = "best-")
            end
        end
    end
)

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

# Print TimerOutputs timings
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p Ignite.throttler_event_filter(settings["eval"]["printperiod"])),
    @j2p function (engine)
        show(stdout, TimerOutputs.get_defaulttimer()); println("\n")
        show(stdout, last(logger[!,[names(logger)[1:4]; sort(names(logger)[5:end])]], 10)); println("\n")
        (engine.state.epoch == 1) && TimerOutputs.reset_timer!() # throw out compilation timings
    end
)

####
#### Weights & biases logger
####

# Attach training/validation output handlers
if !isnothing(wandb_logger)
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
