####
#### Setup
####

using MMDLearning
using PyCall
pyplot(size=(800,600))
Threads.@threads for i in 1:Threads.nthreads(); set_zero_subnormals(true); end
if CUDA.functional() && !haskey(ENV, "JL_DISABLE_GPU")
    CUDA.allowscalar(false)
    CUDA.device!(parse(Int, get(ENV, "JL_CUDA_DEVICE", "0")))
    @eval todevice(x) = Flux.gpu(x)
else
    @eval todevice(x) = Flux.cpu(x)
end
to32(x) = x |> Flux.f32 |> todevice

const torch = pyimport("torch")
const wandb = pyimport("wandb")
const ignite = pyimport("ignite")
const logging = pyimport("logging")
py"""
from ignite.contrib.handlers.wandb_logger import *
"""

const Events = ignite.engine.Events
const WandBLogger = ignite.contrib.handlers.wandb_logger.WandBLogger

const settings = TOML.parse("""
    [data]
        out    = "./output/ignite-cvae-$(MMDLearning.getnow())"
        ntrain = 102_400
        ntest  = 10_240
        nval   = 10_240

    [train]
        timeout     = 1e9 #TODO 10800.0
        epochs      = 999_999
        batchsize   = 1024 # 256 2048
        # kernelrate  = 10 # Train kernel every `kernelrate` iterations
        # kernelsteps = 1 # Gradient updates per kernel train
        # GANcycle    = 1 # CVAE and GAN take turns training for `GANcycle` consecutive epochs (0 trains both each iteration)
        # Dcycle      = 0 # Train for `Dcycle` epochs of discrim only, followed by `Dcycle` epochs of CVAE and GAN together
        # GANrate     = 10 # Train GAN losses every `GANrate` iterations
        # Dsteps      = 5  # Train GAN losses with `Dsteps` discrim updates per genatr update
        # Dheadstart  = 0  # Train discriminator for `Dheadstart` epochs before training generator
        MMDrate     = 10 # Train MMD loss every `MMDrate` iterations
        [train.augment]
            fftcat        = false # Fourier transform of input signal, concatenating real/imag
            fftsplit      = false # Fourier transform of input signal, treating real/imag separately
            gradient      = true  # Gradient of input signal (1D central difference)
            laplacian     = false # Laplacian of input signal (1D second order)
            encoderspace  = true  # Discriminate encoder space representations
            residuals     = false # Discriminate residual vectors
            flipsignals   = false # Randomly reverse signals
            scaleandshift = false # Randomly scale and shift signals
            # Gsamples      = 1   # Discriminator averages over `Gsamples` instances of corrected signals
            Dchunk        = 0     # Discriminator looks at random chunks of size `Dchunk` (0 uses whole signal)

    [eval]
        valevalperiod   = 60.0
        trainevalperiod = 120.0
        saveperiod      = 300.0 #TODO
        printperiod     = 60.0

    [opt]
        lr       = 1e-4 #3e-5 #TODO
        lrdrop   = 1.0 #3.17
        lrthresh = 1e-5
        lrrate   = 1000
        [opt.cvae]
            lr = "%PARENT%" #TODO 1e-4
        [opt.genatr]
            lr = "%PARENT%" #TODO 1e-5
        [opt.discrim]
            lr = "%PARENT%" #TODO 3e-4
        [opt.mmd]
            lr = "%PARENT%"
        # [opt.kernel]
        #     loss = "mmd" #"tstatistic"
        #     lr = 1e-2

    [arch]
        physics = "$(get(ENV, "JL_PHYS_MODEL", "toy"))" # "toy" or "mri"
        nlatent = 5 # number of latent variables Z
        zdim    = 8 # embedding dimension of z
        hdim    = 256 # size of hidden layers
        nhidden = 4 # number of hidden layers
        skip    = false # skip connection
        [arch.enc1]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
            skip    = "%PARENT%"
        [arch.enc2]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
            skip    = "%PARENT%"
        [arch.dec]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
            skip    = "%PARENT%"
        [arch.genatr]
            hdim        = "%PARENT%" #TODO 32
            nhidden     = "%PARENT%" #TODO 2
            skip        = "%PARENT%"
            maxcorr     = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? 0.1 : 0.025) # correction amplitude
            noisebounds = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? [-8.0, -2.0] : [-6.0, -3.0]) # noise amplitude
        [arch.discrim]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
            skip    = "%PARENT%"
            dropout = 0.1
        [arch.kernel]
            nbandwidth  = 32
            channelwise = false
            bwbounds    = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? [-8.0, 4.0] : [-10.0, 4.0]) # bounds for kernel bandwidths (logsigma)
""")
Ignite.parse_command_line!(settings)

# Initialize WandBLogger and save settings
const wandb_logger = !haskey(ENV, "JL_WANDB_LOGGER") ? nothing : isempty(ARGS) ? WandBLogger() : WandBLogger(config = filter(((k,v),) -> any(startswith("--" * k), ARGS), Ignite.flatten_dict(settings)))
!isnothing(wandb_logger) && (settings["data"]["out"] = wandb.run.dir)
Ignite.save_and_print(settings; outpath = settings["data"]["out"], filename = "settings.toml")

# Initialize generator + discriminator + kernel
function make_models(phys::PhysicsModel{Float32})
    models = Dict{String, Any}()
    derived = Dict{String, Any}()
    n   = nsignal(phys) # input signal length
    nθ  = ntheta(phys) # number of physics variables
    θbd = θbounds(phys)
    k   = settings["arch"]["nlatent"]::Int # number of latent variables Z
    nz  = settings["arch"]["zdim"]::Int # embedding dimension

    RiceGenType = LatentVectorRicianCorrector{n,k}
    # RiceGenType = LatentVectorRicianNoiseCorrector{n,k}
    # RiceGenType = VectorRicianCorrector{n,k}

    # Rician generator. First `n` elements for `δX` scaled to (-δ, δ), second `n` elements for `logϵ` scaled to (noisebounds[1], noisebounds[2])
    models["genatr"] = let
        hdim = settings["arch"]["genatr"]["hdim"]::Int
        nhidden = settings["arch"]["genatr"]["nhidden"]::Int
        skip = settings["arch"]["genatr"]["skip"]::Bool
        maxcorr = settings["arch"]["genatr"]["maxcorr"]::Float64
        noisebounds = settings["arch"]["genatr"]["noisebounds"]::Vector{Float64}
        OutputScale =
            RiceGenType <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? MMDLearning.CatScale([(-maxcorr, maxcorr), (noisebounds...,)], [n,n]) :
            RiceGenType <: FixedNoiseVectorRicianCorrector ? MMDLearning.CatScale([(-maxcorr, maxcorr)], [n]) :
            RiceGenType <: LatentVectorRicianNoiseCorrector ? MMDLearning.CatScale([(noisebounds...,)], [n]) :
            error("Unsupported corrector type: $RiceGenType")
        Flux.Chain(
            MMDLearning.MLP(ninput(RiceGenType) => noutput(RiceGenType), nhidden, hdim, Flux.relu, tanh; skip = skip)...,
            OutputScale
        ) |> to32
    end

    # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
    derived["ricegen"] = RiceGenType(models["genatr"])

    RESCNN(sz::Pair{Int,Int}, Nhid::Int, Dhid::Int, σhid = Flux.relu, σout = identity; skip = false) =
        Flux.Chain(
            x::AbstractMatrix -> reshape(x, sz[1], 1, 1, size(x,2)),
            Flux.Conv((3,1), 1=>Dhid, identity; pad = Flux.SamePad()),
            mapreduce(vcat, 1:Nhid÷2) do _
                convlayers = [Flux.Conv((3,1), Dhid=>Dhid, σhid; pad = Flux.SamePad()) for _ in 1:2]
                skip ? [Flux.SkipConnection(Flux.Chain(convlayers...), +)] : convlayers
            end...,
            Flux.Conv((1,1), Dhid=>1, identity; pad = Flux.SamePad()),
            x::AbstractArray{<:Any,4} -> reshape(x, sz[1], size(x,4)),
            Flux.Dense(sz[1], sz[2], σout),
        )

    # Encoders
    models["enc1"] = let
        hdim = settings["arch"]["enc1"]["hdim"]::Int
        nhidden = settings["arch"]["enc1"]["nhidden"]::Int
        skip = settings["arch"]["enc1"]["skip"]::Bool
        MMDLearning.MLP(n => 2*nz, nhidden, hdim, Flux.relu, identity; skip = skip) |> to32
        # RESCNN(n => 2*nz, nhidden, hdim, Flux.relu, identity; skip = skip) |> to32
    end

    models["enc2"] = let
        hdim = settings["arch"]["enc2"]["hdim"]::Int
        nhidden = settings["arch"]["enc2"]["nhidden"]::Int
        skip = settings["arch"]["enc2"]["skip"]::Bool
        MMDLearning.MLP(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity; skip = skip) |> to32
        # RESCNN(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity; skip = skip) |> to32
    end

    # Decoder
    models["dec"] = let
        hdim = settings["arch"]["dec"]["hdim"]::Int
        nhidden = settings["arch"]["dec"]["nhidden"]::Int
        skip = settings["arch"]["dec"]["skip"]::Bool
        Flux.Chain(
            MMDLearning.MLP(n + nz => 2*(nθ + k), nhidden, hdim, Flux.relu, identity; skip = skip)...,
            # RESCNN(n + nz => 2*(nθ + k), nhidden, hdim, Flux.relu, identity; skip = skip)...,
            MMDLearning.CatScale(eltype(θbd)[θbd; (-1, 1)], [ones(Int, nθ); k + nθ + k]),
        ) |> to32
    end

    # Discriminator
    models["discrim"] = let
        hdim = settings["arch"]["discrim"]["hdim"]::Int
        nhidden = settings["arch"]["discrim"]["nhidden"]::Int
        skip = settings["arch"]["discrim"]["skip"]::Bool
        dropout = settings["arch"]["discrim"]["dropout"]::Float64 |> p -> p > 0 ? eltype(phys)(p) : nothing
        encoderspace = settings["train"]["augment"]["encoderspace"]::Bool
        residuals = settings["train"]["augment"]["residuals"]::Bool
        Dchunk = settings["train"]["augment"]["Dchunk"]::Int
        nin = ifelse(Dchunk > 0, Dchunk, n) * ifelse(residuals, 2, 1) + ifelse(encoderspace, nz, 0)
        MMDLearning.MLP(nin => 1, nhidden, hdim, Flux.relu, Flux.sigmoid; skip = skip, dropout = dropout) |> to32
        # RESCNN(n => 1, nhidden, hdim, Flux.relu, Flux.sigmoid; skip = skip) |> to32
    end

    # MMD kernel bandwidths
    models["logsigma"] = let
        bwbounds = settings["arch"]["kernel"]["bwbounds"]::Vector{Float64}
        nbandwidth = settings["arch"]["kernel"]["nbandwidth"]::Int
        channelwise = settings["arch"]["kernel"]["channelwise"]::Bool
        range(bwbounds...; length = nbandwidth+2)[2:end-1] |> logσ -> (channelwise ? repeat(logσ, 1, n) : logσ) |> to32
    end

    # MMD kernel wrapper
    derived["kernel"] = MMDLearning.ExponentialKernel(models["logsigma"])

    return models, derived
end

const phys = initialize!(
    ToyModel{Float32,true}();
    ntrain = settings["data"]["ntrain"]::Int,
    ntest = settings["data"]["ntest"]::Int,
    nval = settings["data"]["nval"]::Int,
)
const models, derived = make_models(phys)
# const models = deepcopy(BSON.load("/home/jdoucette/Documents/code/BlochTorreyExperiments-master/LearningCorrections/output/ignite-cvae-2020-08-05-T-12-06-47-294/current-models.bson")["models"]) |> d -> MMDLearning.map_dict(todevice, d) #TODO
const optimizers = Dict{String,Any}(
    "cvae"    => Flux.ADAM(settings["opt"]["cvae"]["lr"]),
    "genatr"  => Flux.ADAM(settings["opt"]["genatr"]["lr"]),
    "discrim" => Flux.ADAM(settings["opt"]["discrim"]["lr"]),
    "mmd"     => Flux.ADAM(settings["opt"]["mmd"]["lr"]),
)
MMDLearning.model_summary(models, joinpath(settings["data"]["out"], "model-summary.txt"))

# Helpers
@inline flatten_apply(f, x::AbstractTensor3D) = (y = f(reshape(x, size(x,1), :)); reshape(y, size(y,1), size(x,2), size(x,3)))
@inline flatten_apply(f, x::AbstractMatrix) = f(x)
@inline split_theta_latent(θZ::AbstractMatrix) = size(θZ,1) == ntheta(phys) ? (θZ, similar(θZ,0,size(θZ,2))) : (θZ[1:ntheta(phys),:], θZ[ntheta(phys)+1:end,:])
@inline split_mean_std(μ::AbstractMatrix) = μ[1:end÷2, :], Flux.softplus.(μ[end÷2+1:end, :]) .+ sqrt(eps(eltype(μ))) #TODO Flux.softplus -> exp?
@inline sample_mv_normal(μ0::AbstractMatrix{T}, σ::AbstractMatrix{T}) where {T} = μ0 .+ σ .* randn_similar(σ, max.(size(μ0), size(σ)))
@inline sample_mv_normal(μ0::Matrix{T}, σ::Matrix{T}) where {T} = μ0 .+ σ .* randn_similar(σ, max.(size(μ0), size(σ)))
@inline sample_mv_normal(μ::Tuple) = sample_mv_normal(μ...)
@inline pow2(x) = x*x
const theta_lower_bounds = θlower(phys) |> todevice
const theta_upper_bounds = θupper(phys) |> todevice
sampleθprior_similar(Y, n = size(Y,2)) = rand_similar(Y, ntheta(phys), n) .* (theta_upper_bounds .- theta_lower_bounds) .+ theta_lower_bounds

# Misc. useful operators
derived["gradient"] = MMDLearning.CentralDifference() |> to32
derived["laplacian"] = MMDLearning.Laplacian() |> to32
derived["encoderspace"] = NotTrainable(Flux.Chain(models["enc1"]..., split_mean_std, sample_mv_normal)) # non-trainable sampling of encoder signal representations

# Augmentations
function augmentations(X::AbstractMatrix, X̂::Union{Nothing,<:AbstractMatrix})
    Dchunk = settings["train"]["augment"]["Dchunk"]::Int
    flipsignals = settings["train"]["augment"]["flipsignals"]::Bool
    scaleandshift = settings["train"]["augment"]["scaleandshift"]::Bool
    encoderspace = settings["train"]["augment"]["encoderspace"]::Bool
    residuals = settings["train"]["augment"]["residuals"]::Bool

    function transformedchunk(x::AbstractMatrix)
        i = ifelse(Dchunk <= 0, firstindex(X,1):lastindex(X,1), rand(firstindex(X,1):lastindex(X,1)-Dchunk+1) .+ (0:Dchunk-1))
        i = ifelse(!flipsignals, i, ifelse(rand(Bool), i, reverse(i)))
        y = x[i,:] #TODO view?
        if scaleandshift
            y = (y .+ randn_similar(y, 1, size(y, 2))) .* sign.(randn_similar(y, 1, size(y, 2))) # shift then flip sign
            # y = y .* randn_similar(y, 1, size(y, 2)) .+ randn_similar(y, 1, size(y, 2)) # scale then shift
        end
        return y
    end

    Xaug = transformedchunk(X)

    if residuals && !isnothing(X̂)
        Xaug = vcat(Xaug, transformedchunk(X .- X̂))
    end

    if encoderspace
        Xaug = vcat(Xaug, flatten_apply(derived["encoderspace"], X)) # include encoder-space representation
    end

    return Xaug
end
augmentations(X::AbstractTensor3D, X̂::Union{Nothing,<:AbstractMatrix}) = flatten_apply(x -> augmentations(x, X̂), X)

function X̂_augmentations(X,Z,X̂)
    #= TODO
        Gsamples = settings["train"]["augment"]["Gsamples"]::Int
        ν, ϵ = rician_params(derived["ricegen"], X, Z)
        X̂samples = add_noise_instance(derived["ricegen"], ν, ϵ, Gsamples)
    =#
    X̂samples = corrected_signal_instance(derived["ricegen"], X, Z)
    return augmentations(X̂samples, X̂)
end
Y_augmentations(Y,X̂) = Zygote.@ignore augmentations(Y,X̂) # don't differentiate through Y augmentations

# KL-divergence contribution to cross-entropy (Note: dropped constant -zdim/2 term)
KLDivergence(μq0, σq, μr0, σr) = (ϵ = sqrt(eps(eltype(μq0))); sum(@. pow2(σq / σr) + pow2((μr0 - μq0) / σr) - 2 * log(σq / σr + ϵ)) / 2)

# Negative log-likelihood/ELBO contribution to cross-entropy (Note: dropped constant +zdim*log(2π)/2 term)
EvidenceLowerBound(x, μx0, σx) = (ϵ = sqrt(eps(eltype(μx0))); sum(@. pow2((x - μx0) / σx) + 2 * log(σx + ϵ)) / 2)

D_G_X_prob(X,Z,X̂) = flatten_apply(models["discrim"], X̂_augmentations(X,Z,X̂)) # discrim on genatr data
D_Y_prob(Y,X̂) = flatten_apply(models["discrim"], Y_augmentations(Y,X̂)) # discrim on real data
Dloss(X,Y,Z,X̂) = (ϵ = sqrt(eps(eltype(X))); -mean(log.(D_Y_prob(Y,X̂) .+ ϵ) .+ log.(1 .- D_G_X_prob(X,Z,X̂) .+ ϵ)))
Gloss(X,Z,X̂) = (ϵ = sqrt(eps(eltype(X))); mean(log.(1 .- D_G_X_prob(X,Z,X̂) .+ ϵ)))

# Maximum mean discrepency (m*MMD^2) loss
MMDloss(X̂,Y) = size(Y,2) * mmd(derived["kernel"], X̂, Y)
function MMDaug(X,Y,Z)
    X̂ = corrected_signal_instance(derived["ricegen"], X, Z)
    mmds = (MMDloss(X̂, Y),)

    if settings["train"]["augment"]["fftcat"]::Bool
        # MMD of concatenated real/imag fourier components
        realfft(x) = vcat(reim(rfft(x,1))...)
        FX̂ = realfft(X̂)
        FY = Zygote.@ignore realfft(Y)
        mmds = (mmds..., MMDloss(FX̂, FY))
    elseif settings["train"]["augment"]["fftsplit"]::Bool
        # MMD of both real/imag fourier components
        rFX̂, iFX̂ = reim(rfft(X̂,1))
        rFY, iFY = Zygote.@ignore reim(rfft(Y,1))
        mmds = (mmds..., MMDloss(rFX̂, rFY), MMDloss(iFX̂, iFY))
    end

    if settings["train"]["augment"]["gradient"]::Bool
        # MMD of signal gradient
        ∇X̂ = derived["gradient"](X̂)
        ∇Y = Zygote.@ignore derived["gradient"](Y)
        mmds = (mmds..., MMDloss(∇X̂, ∇Y))
    end

    if settings["train"]["augment"]["laplacian"]::Bool
        # MMD of signal laplacian
        ∇²X̂ = derived["laplacian"](X̂)
        ∇²Y = Zygote.@ignore derived["laplacian"](Y)
        mmds = (mmds..., MMDloss(∇²X̂, ∇²Y))
    end

    if settings["train"]["augment"]["encoderspace"]::Bool
        # MMD of encoder-space signal
        X̂enc = derived["encoderspace"](X̂)
        Yenc = Zygote.@ignore derived["encoderspace"](Y)
        mmds = (mmds..., MMDloss(X̂enc, Yenc))
    end

    return mmds
end

function InvertY(Y)
    μr = models["enc1"](Y)
    μr0, σr = split_mean_std(μr)
    zr = sample_mv_normal(μr0, σr)

    μx = models["dec"](vcat(Y,zr))
    μx0, σx = split_mean_std(μx)
    x = sample_mv_normal(μx0, σx)

    θ, Z = split_theta_latent(x)
    θ = clamp.(θ, theta_lower_bounds, theta_upper_bounds)
    return θ, Z
end

function sampleθZ(Y; recover_θ = true, recover_Z = true)
    nθ, nZ = ntheta(phys)::Int, settings["arch"]["nlatent"]::Int
    if recover_θ || recover_Z
        θ, Z = InvertY(Y)
        if !recover_θ
            θ = sampleθprior_similar(Y, size(Y,2))
        end
        if !recover_Z
            Z = randn_similar(Y, nZ, size(Y,2))
        end
        return θ, Z
    else
        θ = sampleθprior_similar(Y, size(Y,2))
        Z = randn_similar(Y, nZ, size(Y,2))
        return θ, Z
    end
end

function sampleXθZ(Y; kwargs...)
    @timeit "sampleθZ"     CUDA.@sync θ, Z = sampleθZ(Y; kwargs...)
    @timeit "signal_model" CUDA.@sync X = signal_model(phys, θ)
    return X, θ, Z
end

function sampleX̂θZ(Y; kwargs...)
    @timeit "sampleXθZ" CUDA.@sync X, θ, Z = sampleXθZ(Y; kwargs...)
    @timeit "sampleX̂"   CUDA.@sync X̂ = corrected_signal_instance(derived["ricegen"], X, Z)
    return X̂, θ, Z
end

sampleX̂(Y; kwargs...) = sampleX̂θZ(Y; kwargs...)[1]

function DataConsistency(Y, μG0, σG)
    ϵ = sqrt(eps(eltype(Y)))
    # YlogL = -sum(@. MMDLearning._rician_logpdf(Y, μG0, σG)) # Rician negative log likelihood
    YlogL = sum(@. 2 * log(σG + ϵ) + pow2((Y - μG0) / σG)) / 2 # Gaussian negative likelihood for testing
    # YlogL += 1000 * sum(abs2, Y .- add_noise_instance(derived["ricegen"], μG0, σG)) / 2 # L2 norm for testing/pretraining
    # YlogL = 10 * sum(abs, Y .- add_noise_instance(derived["ricegen"], μG0, σG)) # L1 norm for testing/pretraining
    return YlogL
end

function CVAEloss(Y, θ, Z; recover_Z = true)
    # Cross-entropy loss function
    μr0, σr = split_mean_std(models["enc1"](Y))
    μq0, σq = split_mean_std(models["enc2"](vcat(Y,θ,Z)))
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_std(models["dec"](vcat(Y,zq)))

    KLdiv = KLDivergence(μq0, σq, μr0, σr)
    ELBO = if recover_Z
        EvidenceLowerBound(vcat(θ,Z), μx0, σx)
    else
        μθ0 = split_theta_latent(μx0)[1]
        σθ  = split_theta_latent(σx)[1]
        EvidenceLowerBound(θ, μθ0, σθ)
    end

    Nbatch = size(Y,2)
    Hloss = (ELBO + KLdiv) / Nbatch

    return Hloss
end

# Self-supervised CVAE loss
function SelfCVAEloss(Y; recover_Z = true)
    # Invert Y
    Nbatch = size(Y,2)
    θ, Z = InvertY(Y)

    # Limit information capacity of Z with ℓ2 regularization
    #   - Equivalently, as 1/2||Z||^2 is the negative log likelihood of Z ~ N(0,1) (dropping normalization factor)
    Zreg = recover_Z ? sum(abs2, Z) / (2*Nbatch) : zero(eltype(Z))

    # Corrected X̂ instance
    X = signal_model(phys, θ) # differentiate through physics model
    μG0, σG = rician_params(derived["ricegen"], X, Z) # Rician negative log likelihood
    X̂ = add_noise_instance(derived["ricegen"], μG0, σG)

    # Data consistency penalty
    # YlogL = DataConsistency(Y, μG0, σG) / Nbatch

    # Add MMD loss contribution
    MMDsq = MMDloss(X̂, Y)

    # Drop gradients for θ, Z, and X̂
    θ = Zygote.dropgrad(θ)
    Z = Zygote.dropgrad(Z)
    X̂ = Zygote.dropgrad(X̂)
    Hloss = CVAEloss(X̂, θ, Z; recover_Z = recover_Z) #TODO X̂ or Y?

    ℓ = Zreg + Hloss + MMDsq
    # ℓ = Zreg + YlogL + Hloss + MMDsq

    return ℓ
end

# Regularize generator outputs
function RegularizeX̂(Y; recover_Z = true)
    # Invert Y
    Nbatch = size(Y,2)
    θhat, Zhat = InvertY(Y)

    # X = signal_model(phys, θhat) # differentiate through physics model
    # μG0, σG = rician_params(derived["ricegen"], X, Zhat)
    # YlogL = DataConsistency(Y, μG0, σG) / Nbatch

    # Limit distribution of X̂ ∼ G(X) with MMD
    # X = Zygote.dropgrad(X)
    θ = Zygote.dropgrad(θhat)
    X = Zygote.dropgrad(signal_model(phys, θ))
    Z = (recover_Z ? randn : zeros)(eltype(Zhat), size(Zhat)...)
    μG0, σG = rician_params(derived["ricegen"], X, Z)
    X̂ = add_noise_instance(derived["ricegen"], μG0, σG)
    MMDsq = MMDloss(X̂, Y)

    # Return total loss
    ℓ = MMDsq
    # ℓ = YlogL + MMDsq

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
    :loss       => Union{eltype(phys), Missing}[],
    :Zreg       => Union{eltype(phys), Missing}[],
    :KLdiv      => Union{eltype(phys), Missing}[],
    :ELBO       => Union{eltype(phys), Missing}[],
    :MMDsq      => Union{eltype(phys), Missing}[],
    :Gloss      => Union{eltype(phys), Missing}[],
    :Dloss      => Union{eltype(phys), Missing}[],
    :D_Y        => Union{eltype(phys), Missing}[],
    :D_G_X      => Union{eltype(phys), Missing}[],
    :rmse       => Union{eltype(phys), Missing}[],
    :theta_err  => Union{Vector{eltype(phys)}, Missing}[],
    :Z_err      => Union{Vector{eltype(phys)}, Missing}[],
    :Yhat_logL  => Union{eltype(phys), Missing}[],
    :Yhat_rmse  => Union{eltype(phys), Missing}[],
    :Xhat_logL  => Union{eltype(phys), Missing}[],
    :Xhat_rmse  => Union{eltype(phys), Missing}[],
)

make_data_tuples(dataset) = tuple.(copy.(eachcol(sampleY(phys, :all; dataset = dataset))))
train_loader = torch.utils.data.DataLoader(make_data_tuples(:train); batch_size = settings["train"]["batchsize"], shuffle = true, drop_last = true)
val_loader = torch.utils.data.DataLoader(make_data_tuples(:val); batch_size = settings["data"]["nval"], shuffle = false, drop_last = false)

function train_step(engine, batch)
    Ytrain_cpu, = Ignite.array.(batch)
    Ytrain = Ytrain_cpu |> todevice
    metrics = Dict{Any,Any}()

    @timeit "train batch" CUDA.@sync begin
        # # CVAE and GAN take turns training for `GANcycle` consecutive epochs
        # GANcycle = settings["train"]["GANcycle"]::Int
        # train_CVAE = (GANcycle == 0) || iseven(div(engine.state.epoch-1, GANcycle))
        # train_GAN  = (GANcycle == 0) || !train_CVAE
        # train_discrim = true
        # train_genatr = true

        # # Train CVAE every iteration, GAN every `GANrate` iterations
        # train_CVAE = true
        # train_GAN  = mod(engine.state.iteration-1, settings["train"]["GANrate"]::Int) == 0
        # train_discrim = true
        # train_genatr = true
        # # train_genatr = engine.state.epoch >= settings["train"]["Dheadstart"]::Int

        # Train CVAE every iteration, MMD every `MMDrate` iterations
        train_CVAE = true
        train_MMD  = mod(engine.state.iteration-1, settings["train"]["MMDrate"]::Int) == 0

        # # `Dcycle` epochs of discrim only, followed by `Dcycle` epochs of CVAE and GAN together
        # Dcycle = settings["train"]["Dcycle"]::Int
        # if Dcycle != 0 && iseven(div(engine.state.epoch-1, Dcycle))
        #     train_CVAE = false
        #     train_GAN  = true
        #     train_discrim = true
        #     train_genatr = false
        # else
        #     # Train CVAE every iteration, GAN every `GANrate` iterations
        #     train_CVAE = true
        #     train_GAN  = mod(engine.state.iteration-1, settings["train"]["GANrate"]::Int) == 0
        #     train_discrim = true
        #     # train_genatr = true
        #     train_genatr = engine.state.epoch >= settings["train"]["Dheadstart"]::Int
        # end

        # Train CVAE loss
        train_CVAE && @timeit "cvae" CUDA.@sync let
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"])
            @timeit "sampleX̂θZ" CUDA.@sync X̂train, θtrain, Ztrain = sampleX̂θZ(Ytrain; recover_θ = false, recover_Z = false) .|> todevice
            @timeit "forward"   CUDA.@sync ℓ, back = Zygote.pullback(() -> CVAEloss(X̂train, θtrain, Ztrain; recover_Z = true), ps)
            @timeit "reverse"   CUDA.@sync gs = back(one(eltype(phys)))
            @timeit "update!"   CUDA.@sync Flux.Optimise.update!(optimizers["cvae"], ps, gs)
            metrics["CVAEloss"] = ℓ
        end

        #= Train GAN loss
        train_GAN && @timeit "gan" CUDA.@sync let
            @timeit "sampleXθZ" CUDA.@sync Xtrain, θtrain, Ztrain = sampleXθZ(Ytrain; recover_θ = true, recover_Z = false) .|> todevice
            @timeit "sampleX̂"   CUDA.@sync X̂train = sampleX̂(Ytrain; recover_θ = true, recover_Z = true) |> todevice #TODO recover_Z?
            train_discrim && @timeit "discrim" CUDA.@sync let
                ps = Flux.params(models["discrim"])
                for _ in 1:settings["train"]["Dsteps"]
                    @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> Dloss(Xtrain, Ytrain, Ztrain, X̂train), ps)
                    @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                    @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["discrim"], ps, gs)
                    metrics["Dloss"] = ℓ
                end
            end
            train_genatr && @timeit "genatr" CUDA.@sync let
                ps = Flux.params(models["genatr"])
                @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> Gloss(Xtrain, Ztrain, X̂train), ps)
                @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["genatr"], ps, gs)
                metrics["Gloss"] = ℓ
            end
        end
        =#

        # Train MMD loss
        train_MMD && @timeit "mmd" CUDA.@sync let
            @timeit "sampleXθZ" CUDA.@sync Xtrain, θtrain, Ztrain = sampleXθZ(Ytrain; recover_θ = true, recover_Z = true) .|> todevice #TODO recover_Z?
            @timeit "genatr" CUDA.@sync let
                ps = Flux.params(models["genatr"])
                @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(ps) do
                    sum(MMDaug(Xtrain, Ytrain, Ztrain))
                end
                @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["mmd"], ps, gs)
                metrics["MMD"] = ℓ
            end
        end

        #= Regularize X̂ via MMD
        if mod(engine.state.iteration-1, settings["train"]["kernelrate"]) == 0
            @timeit "mmd kernel" let
                if haskey(cb_state, "learncorrections") && cb_state["learncorrections"]
                    @timeit "regularize X̂" let
                        ps = Flux.params(models["enc1"], models["dec"], models["genatr"])
                        @timeit "forward" ℓ, back = Zygote.pullback(() -> RegularizeX̂(Ytrain; recover_Z = true), ps)
                        @timeit "reverse" gs = back(one(eltype(phys)))
                        @timeit "update!" Flux.Optimise.update!(optimizers["mmd"], ps, gs)
                    end
                end
                #=
                    X̂train = sampleX̂(Ytrain)
                    ps = models["logsigma"]
                    for _ in 1:settings["train"]["kernelsteps"]
                        success = train_kernel_bandwidth_flux!(ps, X̂train, Ytrain;
                            kernelloss = settings["opt"]["kernel"]["loss"],
                            kernellr = settings["opt"]["kernel"]["lr"],
                            bwbounds = settings["arch"]["kernel"]["bwbounds"]) # timed internally
                        !success && break
                    end
                =#
            end
        end
        =#

        #= Train MMD kernel bandwidths
            if mod(engine.state.iteration-1, settings["train"]["kernelrate"]) == 0
                @timeit "MMD kernel" let
                    @timeit "sample G(X)" X̂train = sampleX̂(Ytrain)
                    for _ in 1:settings["train"]["kernelsteps"]
                        success = train_kernel_bandwidth_flux!(
                            models["logsigma"], X̂train, Ytrain;
                            kernelloss = settings["opt"]["kernel"]["loss"],
                            kernellr = settings["opt"]["kernel"]["lr"],
                            bwbounds = settings["arch"]["kernel"]["bwbounds"]) # timed internally
                        !success && break
                    end
                end
            end
        =#

        #= Train self CVAE loss
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"], models["genatr"])
            @timeit "forward" ℓ, back = Zygote.pullback(() -> SelfCVAEloss(Ytrain; recover_Z = true), ps)
            @timeit "reverse" gs = back(one(eltype(phys)))
            @timeit "update!" Flux.Optimise.update!(optimizers["cvae"], ps, gs)
        =#
    end

    return deepcopy(metrics)
end

function compute_metrics(engine, batch; dataset)
    @timeit "compute metrics" CUDA.@sync begin
        # Update callback state
        cb_state["last_time"] = get!(cb_state, "curr_time", time())
        cb_state["curr_time"] = time()
        cb_state["metrics"] = Dict{String,Any}()

        # Invert Y and make Xs
        Y, = Ignite.array.(batch) .|> todevice
        Nbatch = size(Y,2)
        θ, Z = InvertY(Y)
        X = signal_model(phys, θ)
        δG0, σG = correction_and_noiselevel(derived["ricegen"], X, Z)
        μG0 = add_correction(derived["ricegen"], X, δG0)
        X̂ = add_noise_instance(derived["ricegen"], μG0, σG)

        # Cross-entropy loss function
        μr0, σr = split_mean_std(models["enc1"](X̂)) #TODO X̂ or Y?
        μq0, σq = split_mean_std(models["enc2"](vcat(X̂,θ,Z))) #TODO X̂ or Y?
        zq = sample_mv_normal(μq0, σq)
        μx0, σx = split_mean_std(models["dec"](vcat(X̂,zq))) #TODO X̂ or Y?

        let
            Zreg = sum(abs2, Z) / (2*Nbatch)
            # YlogL = DataConsistency(Y, μG0, σG) / Nbatch
            KLdiv = KLDivergence(μq0, σq, μr0, σr) / Nbatch
            ELBO = EvidenceLowerBound(vcat(θ,Z), μx0, σx) / Nbatch
            mmd_aug = min(size(X̂,2), settings["train"]["batchsize"]) |> m -> MMDaug(X̂[:,1:m], Y[:,1:m], Z[:,1:m])
            MMDsq = sum(mmd_aug) #TODO missing
            loss = KLdiv + ELBO #TODO Zreg, MMDsq, YlogL

            ϵ = sqrt(eps(eltype(X)))
            X̂new = sampleX̂(Y; recover_θ = true, recover_Z = true) |> todevice #TODO recover_Z?
            d_y = D_Y_prob(Y, X̂new)
            d_g_x = D_G_X_prob(X, Z, X̂new)
            # Dloss = -mean(log.(d_y .+ ϵ) .+ log.(1 .- d_g_x .+ ϵ))
            # Gloss = mean(log.(1 .- d_g_x .+ ϵ))
            # D_Y   = mean(d_y)
            # D_G_X = mean(d_g_x)
            Dloss = length(mmd_aug) >= 1 ? mmd_aug[1] : zero(MMDsq) #TODO
            Gloss = length(mmd_aug) >= 2 ? mmd_aug[2] : zero(MMDsq) #TODO
            D_Y   = length(mmd_aug) >= 3 ? mmd_aug[3] : zero(MMDsq) #TODO
            D_G_X = length(mmd_aug) >= 4 ? mmd_aug[4] : zero(MMDsq) #TODO

            @pack! cb_state["metrics"] = Zreg, KLdiv, ELBO, loss, MMDsq, Gloss, Dloss, D_Y, D_G_X
        end

        # Cache cb state variables using naming convention
        function cache_cb_state!(Y, θ, Z, Xθ, δθ, ϵθ, Xθδ, Xθhat, Yθ, Yθhat; suf::String)
            cb_state["Y"     * suf] = Y     |> Flux.cpu
            cb_state["θ"     * suf] = θ     |> Flux.cpu
            cb_state["Z"     * suf] = Z     |> Flux.cpu
            cb_state["Xθ"    * suf] = Xθ    |> Flux.cpu
            cb_state["δθ"    * suf] = δθ    |> Flux.cpu
            cb_state["ϵθ"    * suf] = ϵθ    |> Flux.cpu
            cb_state["Xθδ"   * suf] = Xθδ   |> Flux.cpu
            cb_state["Xθhat" * suf] = Xθhat |> Flux.cpu
            cb_state["Yθ"    * suf] = Yθ    |> Flux.cpu
            cb_state["Yθhat" * suf] = Yθhat |> Flux.cpu
            return cb_state
        end

        # Cache values for evaluating VAE performance for recovering Y
        let
            Yθ = hasclosedform(phys) ? signal_model(ClosedForm(phys), θ) : missing
            Yθhat = hasclosedform(phys) ? signal_model(ClosedForm(phys), θ, noiselevel(ClosedForm(phys))) : missing
            cache_cb_state!(Y, θ, Z, X, δG0, σG, μG0, X̂, Yθ, Yθhat; suf = "")

            all_Yhat_rmse = sqrt.(mean(abs2, Y .- X̂; dims = 1)) |> Flux.cpu |> vec
            all_Yhat_logL = -sum(@. MMDLearning._rician_logpdf(Flux.cpu.((Y, μG0, σG))...); dims = 1) |> vec
            Yhat_rmse = mean(all_Yhat_rmse)
            Yhat_logL = mean(all_Yhat_logL)
            @pack! cb_state["metrics"] = all_Yhat_rmse, all_Yhat_logL, Yhat_rmse, Yhat_logL
        end

        # Cache values for evaluating CVAE performance for estimating parameters of Y
        let
            θfit, Zfit = split_theta_latent(sample_mv_normal(μx0, σx))
            θfit .= clamp.(θfit, theta_lower_bounds, theta_upper_bounds)
            Xθfit = signal_model(phys, θfit)
            δθfit, ϵθfit = correction_and_noiselevel(derived["ricegen"], Xθfit, Zfit)
            Xθδfit = add_correction(derived["ricegen"], Xθfit, δθfit)
            Xθhatfit = add_noise_instance(derived["ricegen"], Xθδfit, ϵθfit)
            Yθfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit) : missing
            Yθhatfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit, noiselevel(ClosedForm(phys))) : missing
            cache_cb_state!(X̂, θfit, Zfit, Xθfit, δθfit, ϵθfit, Xθδfit, Xθhatfit, Yθfit, Yθhatfit; suf = "fit") #TODO X̂ or Y?

            rmse = hasclosedform(phys) ? sqrt(mean(abs2, Yθfit - Xθδfit)) : missing
            all_Xhat_rmse = sqrt.(mean(abs2, X̂ .- Xθhatfit; dims = 1)) |> Flux.cpu |> vec #TODO X̂ or Y?
            all_Xhat_logL = -sum(@. MMDLearning._rician_logpdf(Flux.cpu.((X̂, Xθδfit, ϵθfit))...); dims = 1) |> vec #TODO X̂ or Y?
            Xhat_rmse = mean(all_Xhat_rmse)
            Xhat_logL = mean(all_Xhat_logL)
            θ_err = 100 .* mean(abs, (θ .- θfit) ./ (theta_upper_bounds .- theta_lower_bounds); dims = 2) |> Flux.cpu |> vec |> copy
            Z_err = mean(abs, Z .- Zfit; dims = 2) |> Flux.cpu |> vec |> copy
            @pack! cb_state["metrics"] = Xhat_rmse, Xhat_logL, θ_err, Z_err, rmse, all_Xhat_rmse, all_Xhat_logL
        end

        # Initialize output metrics dictionary
        metrics = Dict{Any,Any}()
        metrics[:epoch]   = trainer.state.epoch
        metrics[:iter]    = trainer.state.iteration
        metrics[:dataset] = dataset
        metrics[:time]    = cb_state["curr_time"] - cb_state["last_time"]

        # Metrics computed in update_callback!
        metrics[:loss]  = cb_state["metrics"]["loss"]
        metrics[:Zreg]  = cb_state["metrics"]["Zreg"]
        # metrics[:YlogL] = cb_state["metrics"]["YlogL"]
        metrics[:KLdiv] = cb_state["metrics"]["KLdiv"]
        metrics[:MMDsq] = cb_state["metrics"]["MMDsq"]
        metrics[:ELBO]  = cb_state["metrics"]["ELBO"]
        metrics[:Gloss] = cb_state["metrics"]["Gloss"]
        metrics[:Dloss] = cb_state["metrics"]["Dloss"]
        metrics[:D_Y]   = cb_state["metrics"]["D_Y"]
        metrics[:D_G_X] = cb_state["metrics"]["D_G_X"]
        metrics[:rmse]  = cb_state["metrics"]["rmse"]
        metrics[:theta_err] = cb_state["metrics"]["θ_err"]
        metrics[:Z_err]     = cb_state["metrics"]["Z_err"]
        metrics[:Yhat_logL] = cb_state["metrics"]["Yhat_logL"]
        metrics[:Yhat_rmse] = cb_state["metrics"]["Yhat_rmse"]
        metrics[:Xhat_logL] = cb_state["metrics"]["Xhat_logL"]
        metrics[:Xhat_rmse] = cb_state["metrics"]["Xhat_rmse"]

        # Update logger dataframe
        is_consecutive = !isempty(logger) && (logger.epoch[end] == metrics[:epoch] && logger.iter[end] == metrics[:iter] && logger.dataset[end] === metrics[:dataset])
        if !is_consecutive
            push!(logger, metrics; cols = :subset)
            if dataset === :train # val should always be one batch only
                nbatches = div(settings["data"]["ntrain"]::Int, settings["train"]["batchsize"]::Int)
                for (k,v) in metrics
                    (k ∉ [:epoch, :iter, :dataset, :time]) && (logger[end, k] /= nbatches)
                end
            end
        else
            @assert dataset === :train # val should always be one batch only
            logger.time[end] += metrics[:time]
            nbatches = div(settings["data"]["ntrain"]::Int, settings["train"]["batchsize"]::Int)
            for (k,v) in metrics
                (k ∉ [:epoch, :iter, :dataset, :time]) && (logger[end, k] += v / nbatches)
            end
        end

        return Dict{Any,Any}(
            "CVAEloss"  => cb_state["metrics"]["loss"],
            "Zreg"      => cb_state["metrics"]["Zreg"],
            "KLdiv"     => cb_state["metrics"]["KLdiv"],
            "ELBO"      => cb_state["metrics"]["ELBO"],
            "Gloss"     => cb_state["metrics"]["Gloss"],
            "Dloss"     => cb_state["metrics"]["Dloss"],
            "D_Y"       => cb_state["metrics"]["D_Y"],
            "D_G_X"     => cb_state["metrics"]["D_G_X"],
            "rmse"      => cb_state["metrics"]["rmse"],
            "theta_err" => cb_state["metrics"]["θ_err"],
            "Z_err"     => cb_state["metrics"]["Z_err"],
            "Yhat_logL" => cb_state["metrics"]["Yhat_logL"],
            "Yhat_rmse" => cb_state["metrics"]["Yhat_rmse"],
            "Xhat_logL" => cb_state["metrics"]["Xhat_logL"],
            "Xhat_rmse" => cb_state["metrics"]["Xhat_rmse"],
        )
    end
end

function makeplots(;showplot = false)
    try
        Dict{Symbol, Any}(
            :ricemodel  => MMDLearning.plot_rician_model(logger, cb_state, phys; showplot = showplot, bandwidths = haskey(models, "logsigma") ? (permutedims(models["logsigma"]) |> Flux.cpu) : nothing),
            :signals    => MMDLearning.plot_rician_signals(logger, cb_state, phys; showplot = showplot),
            :vaesignals => MMDLearning.plot_vae_rician_signals(logger, cb_state, phys; showplot = showplot),
            :infer      => MMDLearning.plot_rician_inference(logger, cb_state, phys; showplot = showplot),
            :ganloss    => MMDLearning.plot_gan_loss(logger, cb_state, phys; showplot = showplot, lrdroprate = settings["opt"]["lrrate"], lrdrop = settings["opt"]["lrdrop"]),
            :losses     => MMDLearning.plot_selfcvae_losses(logger, cb_state, phys; showplot = showplot),
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
    @j2p function (engine)
        if isfile(joinpath(settings["data"]["out"], "stop.txt"))
            @info "Exiting: found file $(joinpath(settings["data"]["out"], "stop.txt"))"
            engine.terminate()
        end
    end
)

# Timeout
trainer.add_event_handler(
    Events.EPOCH_COMPLETED(event_filter = @j2p run_timeout(settings["train"]["timeout"])),
    @j2p function (engine)
        @info "Exiting: training time exceeded $(DECAES.pretty_time(settings["train"]["timeout"]))"
        engine.terminate()
    end
)

# Compute callback metrics
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p event_throttler(settings["eval"]["valevalperiod"])),
    @j2p (engine) -> val_evaluator.run(val_loader)
)
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p event_throttler(settings["eval"]["trainevalperiod"])),
    @j2p (engine) -> train_evaluator.run(train_loader)
)

# Checkpoint current model + logger + make plots
trainer.add_event_handler(
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p event_throttler(settings["eval"]["saveperiod"])),
    @j2p function (engine)
        @timeit "checkpoint" let models = MMDLearning.map_dict(Flux.cpu, models)
            @timeit "save current model" saveprogress(@dict(models, logger); savefolder = settings["data"]["out"], prefix = "current-")
            @timeit "make current plots" plothandles = makeplots()
            @timeit "save current plots" saveplots(plothandles; savefolder = settings["data"]["out"], prefix = "current-")
        end
    end
)

# Check for + save best model + logger + make plots
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p event_throttler(settings["eval"]["saveperiod"])),
    @j2p function (engine)
        losses = logger.Yhat_logL[logger.dataset .=== :val] |> skipmissing |> collect
        if !isempty(losses) && (length(losses) == 1 || losses[end] < minimum(losses[1:end-1]))
            @timeit "save best progress" let models = MMDLearning.map_dict(Flux.cpu, models)
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
    @j2p function (engine)
        @unpack lrrate, lrdrop, lrthresh = settings["opt"]
        epoch = engine.state.epoch
        if epoch > 1 && mod(epoch-1, lrrate) == 0
            for optname in ["genatr", "discrim", "cvae", "mmd"]
                if optname ∉ keys(optimizers)
                    @warn "Optimizer \"$optname\" not found; skipping dropping of lr"
                    continue
                end
                opt = optimizers[optname]
                new_eta = max(opt.eta / lrdrop, lrthresh)
                if new_eta > lrthresh
                    @info "$epoch: Dropping $optname optimizer learning rate to $new_eta"
                else
                    @info "$epoch: Learning rate reached minimum value $lrthresh for $optname optimizer"
                end
                opt.eta = new_eta
            end
        end
    end
)

# Print TimerOutputs timings
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p event_throttler(settings["eval"]["printperiod"])),
    @j2p function (engine)
        show(stdout, TimerOutputs.get_defaulttimer()); println("\n")
        show(stdout, last(logger, 10)); println("\n")
        (engine.state.epoch == 1) && TimerOutputs.reset_timer!() # throw out compilation timings
    end
)

####
#### Weights & biases logger
####

# Attach training/validation output handlers
if !isnothing(wandb_logger)
    for (tag, engine, event) in [
            ("step",  trainer,         Events.EPOCH_COMPLETED(event_filter = @j2p run_timeout(settings["eval"]["trainevalperiod"]))), # computed each iteration; throttle recording
            ("train", train_evaluator, Events.EPOCH_COMPLETED), # throttled above; record every epoch
            ("val",   val_evaluator,   Events.EPOCH_COMPLETED), # throttled above; record every epoch
        ]
        wandb_logger.attach_output_handler(engine;
            tag = tag,
            event_name = event,
            output_transform = @j2p(metrics -> metrics),
            global_step_transform = @j2p((args...;kwargs...) -> trainer.state.epoch),
        )
    end
end

####
#### Run trainer
####

TimerOutputs.reset_timer!()
trainer.run(train_loader, settings["train"]["epochs"])
