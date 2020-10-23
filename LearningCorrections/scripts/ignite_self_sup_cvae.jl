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

using MMDLearning: map_dict, sum_dict, sample_mv_normal, split_mean_softplus_std, apply_dim1, pow2, log2π
const Events = ignite.engine.Events

# Parse command line arguments into default settings
const settings = TOML.parse("""
    [data]
        out    = "./output/ignite-cvae-$(MMDLearning.getnow())"
        ntrain = 102_400
        ntest  = 10_240
        nval   = 10_240

    [train]
        timeout     = 1e9 #TODO 10800.0
        epochs      = 1000_000
        batchsize   = 1024 #256 #512 #4096 #2048
        MMDrate     = 1   # Train MMD loss every `MMDrate` epochs
        GANrate     = 0   # Train GAN losses every `GANrate` iterations
        Dsteps      = 5   # Train GAN losses with `Dsteps` discrim updates per genatr update
        # GANcycle    = 1   # CVAE and GAN take turns training for `GANcycle` consecutive epochs (0 trains both each iteration)
        # Dcycle      = 0   # Train for `Dcycle` epochs of discrim only, followed by `Dcycle` epochs of CVAE and GAN together
        # Dheadstart  = 0   # Train discriminator for `Dheadstart` epochs before training generator
        # kernelrate  = 10  # Train kernel every `kernelrate` iterations
        # kernelsteps = 1   # Gradient updates per kernel train
        [train.augment]
            gradient      = true  # Gradient of input signal (1D central difference)
            laplacian     = false # Laplacian of input signal (1D second order)
            encoderspace  = false # Discriminate encoder space representations
            residuals     = false # Discriminate residual vectors
            fftcat        = false # Fourier transform of input signal, concatenating real/imag
            fftsplit      = false # Fourier transform of input signal, treating real/imag separately
        [train.transform]
            flipsignals   = true  # Randomly reverse signals
            Dchunk        = 112   # Discriminator looks at random chunks of size `Dchunk` (0 uses whole signal)
            # Gsamples      = 1   # Discriminator averages over `Gsamples` instances of corrected signals

    [eval]
        valevalperiod   = 120.0 #TODO 300.0 60.0
        trainevalperiod = 120.0 #TODO 300.0 60.0
        saveperiod      = 300.0 #TODO
        printperiod     = 120.0 #TODO 300.0 60.0

    [opt]
        lrrel    = 0.1 #0.03 # Learning rate relative to batch size, i.e. lr = lrrel / batchsize
        lrthresh = 0.0 #1e-5 # Absolute minimum learning rate
        lrdrop   = 10.0 #3.17 #1.0 # Drop learning rate by factor `lrdrop` every `lrrate` epochs
        lrrate   = 1000_000 # Drop learning rate by factor `lrdrop` every `lrrate` epochs
        [opt.cvae]
            lrrel = "%PARENT%"
        [opt.genatr]
            lrrel = "%PARENT%"
        [opt.discrim]
            lrrel = "%PARENT%"
        [opt.mmd]
            lrrel = "%PARENT%"
            lambda_eps     = 10.0 # regularize noise amplitude epsilon
            lambda_deps_dz = 1.0  # regularize gradient of epsilon w.r.t. latent variables

    [arch]
        physics = "$(get(ENV, "JL_PHYS_MODEL", "toy"))" # "toy" or "mri"
        nlatent = 1 # number of latent variables Z
        zdim    = 8 # embedding dimension of z
        hdim    = 256 # size of hidden layers
        nhidden = 4 # number of hidden layers
        skip    = false # skip connection
        [arch.enc1]
            psize   = 32
            head    = 4
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
            skip    = "%PARENT%"
        [arch.enc2]
            psize   = 32
            head    = 4
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
            skip    = "%PARENT%"
        [arch.dec]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
            skip    = "%PARENT%"
        [arch.genatr]
            hdim        = 32    #TODO "%PARENT%"
            nhidden     = 4     #TODO "%PARENT%"
            skip        = false #TODO "%PARENT%"
            maxcorr     = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? 0.1 : 0.025) # correction amplitude
            noisebounds = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? [-8.0, -2.0] : [-6.0, -3.0]) # noise amplitude
        [arch.discrim]
            dropout = 0.1
            hdim    = 1         #TODO "%PARENT%"
            nhidden = 0         #TODO "%PARENT%"
            skip    = "%PARENT%"
        [arch.kernel]
            nbandwidth  = 32
            channelwise = false
            bwbounds    = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? [-8.0, 4.0] : [-10.0, 4.0]) # bounds for kernel bandwidths (logsigma)
""")
Ignite.parse_command_line!(settings)

# Init wandb_logger and save settings
wandb_logger = Ignite.init_wandb_logger(settings)
!isnothing(wandb_logger) && (settings["data"]["out"] = wandb.run.dir)
Ignite.save_and_print(settings; outpath = settings["data"]["out"], filename = "settings.toml")

####
#### Models
####

# Initialize generator + discriminator + kernel
function make_models(phys::PhysicsModel{Float32}, models = Dict{String, Any}(), derived = Dict{String, Any}())
    n   = nsignal(phys) # input signal length
    nθ  = ntheta(phys) # number of physics variables
    θbd = θbounds(phys)
    k   = settings["arch"]["nlatent"]::Int # number of latent variables Z
    nz  = settings["arch"]["zdim"]::Int # embedding dimension

    # RiceGenType = LatentVectorRicianCorrector{n,k}
    RiceGenType = LatentVectorRicianNoiseCorrector{n,k}
    # RiceGenType = VectorRicianCorrector{n,k}

    function RESCNN(sz::Pair{Int,Int}, Nhid::Int, Dhid::Int, σhid = Flux.relu, σout = identity; skip = false)
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
    end

    function TransformerEncoder(; n = 48, psize = 16, head = 8, hdim = 256, nhidden = 2)
        t = Transformers.Stack(
            Transformers.@nntopo(
                X : # Input (n × b)
                X => X : # Reshape (1 × n × b)
                X => pe : # Positional embedding pe (psize × n)
                (X, pe) => E : # Add positional embedding (psize × n × b)
                E => H : # Transformer encoder (psize × n × b)
                H => H # Flatten output (psize*n × b)
            ),
            X -> reshape(X, 1, size(X)...),
            Transformers.Basic.PositionEmbedding(psize, n; trainable = true),
            (X, pe) -> X .+ pe,
            Flux.Chain([Transformers.Basic.Transformer(psize, head, hdim; future = true, act = Flux.relu, pdrop = 0.0) for i = 1:nhidden]...),
            Flux.flatten,
        )
        Flux.fmap(Flux.testmode!, t) # Force dropout layers inactive
    end

    # Rician generator. First `n` elements for `δX` scaled to (-δ, δ), second `n` elements for `logϵ` scaled to (noisebounds[1], noisebounds[2])
    get!(models, "genatr") do
        hdim = settings["arch"]["genatr"]["hdim"]::Int
        nhidden = settings["arch"]["genatr"]["nhidden"]::Int
        skip = settings["arch"]["genatr"]["skip"]::Bool
        maxcorr = settings["arch"]["genatr"]["maxcorr"]::Float64
        noisebounds = settings["arch"]["genatr"]["noisebounds"]::Vector{Float64}
        nin, nout = ninput(RiceGenType), noutput(RiceGenType)
        OutputScale =
            RiceGenType <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? MMDLearning.CatScale([(-maxcorr, maxcorr), (noisebounds...,)], [n,n]) :
            RiceGenType <: FixedNoiseVectorRicianCorrector ? MMDLearning.CatScale([(-maxcorr, maxcorr)], [n]) :
            RiceGenType <: LatentVectorRicianNoiseCorrector ? MMDLearning.CatScale([(noisebounds...,)], [n]) :
            error("Unsupported corrector type: $RiceGenType")
        Flux.Chain(
            MMDLearning.MLP(nin => nout, nhidden, hdim, Flux.relu, tanh; skip = skip)...,
            OutputScale,
        ) |> to32
        #=
            Flux.Chain(
                Flux.Dense(nin, nout, tanh),
                OutputScale,
            ) |> to32
        =#
    end

    # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
    get!(derived, "ricegen") do; RiceGenType(models["genatr"]) end

    # Encoders
    get!(models, "enc1") do
        hdim = settings["arch"]["enc1"]["hdim"]::Int
        nhidden = settings["arch"]["enc1"]["nhidden"]::Int
        skip = settings["arch"]["enc1"]["skip"]::Bool
        psize = settings["arch"]["enc1"]["psize"]::Int
        head = settings["arch"]["enc1"]["head"]::Int
        MMDLearning.MLP(n => 2*nz, nhidden, hdim, Flux.relu, identity; skip = skip) |> to32
        # RESCNN(n => 2*nz, nhidden, hdim, Flux.relu, identity; skip = skip) |> to32
        # Transformers.Stack(
        #     Transformers.@nntopo( X : X => H : H => μr ),
        #     TransformerEncoder(; n, psize, head, hdim, nhidden),
        #     MMDLearning.MLP(psize*n => 2*nz, 0, hdim, Flux.relu, identity),
        # ) |> to32
    end

    get!(models, "enc2") do
        hdim = settings["arch"]["enc2"]["hdim"]::Int
        nhidden = settings["arch"]["enc2"]["nhidden"]::Int
        skip = settings["arch"]["enc2"]["skip"]::Bool
        psize = settings["arch"]["enc2"]["psize"]::Int
        head = settings["arch"]["enc2"]["head"]::Int
        MMDLearning.MLP(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity; skip = skip) |> to32
        # RESCNN(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity; skip = skip) |> to32
        # Transformers.Stack(
        #     Transformers.@nntopo( (X,θ,Z) : X => H : (H,θ,Z) => HθZ : HθZ => μq ),
        #     TransformerEncoder(; n, psize, head, hdim, nhidden),
        #     vcat,
        #     MMDLearning.MLP(psize*n + nθ + k => 2*nz, 0, hdim, Flux.relu, identity),
        # ) |> to32
    end

    # Decoder
    get!(models, "dec") do
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
    get!(models, "discrim") do
        hdim = settings["arch"]["discrim"]["hdim"]::Int
        nhidden = settings["arch"]["discrim"]["nhidden"]::Int
        skip = settings["arch"]["discrim"]["skip"]::Bool
        dropout = settings["arch"]["discrim"]["dropout"]::Float64 |> p -> p > 0 ? eltype(phys)(p) : nothing
        Dchunk = settings["train"]["transform"]["Dchunk"]::Int
        augsizes = Dict{String,Int}(["gradient" => n-2, "laplacian" => n-2, "encoderspace" => nz, "residuals" => n, "fftcat" => 2*(n÷2 + 1), "fftsplit" => 2*(n÷2 + 1)])
        nin = min(n, Dchunk) + sum((s -> ifelse(settings["train"]["augment"][s]::Bool, min(augsizes[s], Dchunk), 0)).(keys(augsizes)))
        MMDLearning.MLP(nin => 1, nhidden, hdim, Flux.relu, Flux.sigmoid; skip = skip, dropout = dropout) |> to32
        # RESCNN(n => 1, nhidden, hdim, Flux.relu, Flux.sigmoid; skip = skip) |> to32
    end

    # MMD kernel bandwidths
    get!(models, "logsigma") do
        bwbounds = settings["arch"]["kernel"]["bwbounds"]::Vector{Float64}
        nbandwidth = settings["arch"]["kernel"]["nbandwidth"]::Int
        channelwise = settings["arch"]["kernel"]["channelwise"]::Bool
        range(bwbounds...; length = nbandwidth+2)[2:end-1] |> logσ -> (channelwise ? repeat(logσ, 1, n) : logσ) |> to32
    end

    # MMD kernel wrapper
    get!(derived, "kernel") do; MMDLearning.ExponentialKernel(models["logsigma"]) end

    # Misc. useful operators
    get!(derived, "gradient") do; MMDLearning.CentralDifference() |> to32 end
    get!(derived, "laplacian") do; MMDLearning.Laplacian() |> to32 end
    get!(derived, "encoderspace") do; NotTrainable(MMDLearning.flattenchain(Flux.Chain(models["enc1"], split_mean_softplus_std, sample_mv_normal))) end # non-trainable sampling of encoder signal representations

    return models, derived
end

const phys = initialize!(
    MMDLearning.ToyCosineModel{Float32,true}();
    ntrain = settings["data"]["ntrain"]::Int,
    ntest = settings["data"]["ntest"]::Int,
    nval = settings["data"]["nval"]::Int,
)
const models, derived = make_models(phys)
# const models, derived = make_models(phys, map_dict(to32, deepcopy(BSON.load("/home/jdoucette/Documents/code/wandb/tmp/output/ignite-cvae-2020-10-23-T-00-43-55-948/current-models.bson")["models"])))
const optimizers = Dict{String,Any}(
    "cvae"    => Flux.ADAM(settings["opt"]["cvae"]["lrrel"]    / settings["train"]["batchsize"]),
    "genatr"  => Flux.ADAM(settings["opt"]["genatr"]["lrrel"]  / settings["train"]["batchsize"]),
    "discrim" => Flux.ADAM(settings["opt"]["discrim"]["lrrel"] / settings["train"]["batchsize"]),
    "mmd"     => Flux.ADAM(settings["opt"]["mmd"]["lrrel"]     / settings["train"]["batchsize"]),
)
MMDLearning.model_summary(models, joinpath(settings["data"]["out"], "model-summary.txt"))

####
#### Sampling
####

@inline split_theta_latent(θZ::AbstractMatrix) = size(θZ,1) == ntheta(phys) ? (θZ, similar(θZ,0,size(θZ,2))) : (θZ[1:ntheta(phys),:], θZ[ntheta(phys)+1:end,:])
@inline sampleθprior_similar(Y, n = size(Y,2)) = rand_similar(Y, ntheta(phys), n) .* (todevice(θupper(phys)) .- todevice(θlower(phys))) .+ todevice(θlower(phys)) #TODO
@inline global_temperature(epoch) = zero(eltype(phys)) #TODO 1.0 - 1e-3 ^ max(1 - epoch/1000, 0) |> eltype(phys)

function InvertY(Y)
    μr = models["enc1"](Y)
    μr0, σr = split_mean_softplus_std(μr)
    zr = sample_mv_normal(μr0, σr)

    μx = models["dec"](vcat(Y,zr))
    μx0, σx = split_mean_softplus_std(μx)
    x = sample_mv_normal(μx0, σx)

    θ, Z = split_theta_latent(x)
    θ = clamp.(θ, todevice(θlower(phys)), todevice(θupper(phys)))
    return θ, Z
end

function sampleθZ(Y; recover_θ = true, recover_Z = true, temperature)
    nθ, nz = ntheta(phys)::Int, settings["arch"]["nlatent"]::Int
    θprior() = sampleθprior_similar(Y, size(Y,2))
    Zprior() = randn_similar(Y, nz, size(Y,2))
    mask(θhat) = rand!(similar(θhat, 1, size(θhat,2))) .|> m -> ifelse(m > temperature, one(m), zero(m))
    mix(θhat) = mask(θhat) |> m -> m .* θhat .+ (1 .- m) .* θprior()
    if recover_θ || recover_Z
        θhat, Zhat = InvertY(Y)
        θ = recover_θ ? (isnothing(temperature) ? θhat : mix(θhat)) : θprior()
        Z = recover_Z ? Zhat : Zprior()
        θ, Z
    else
        θprior(), Zprior()
    end
end

function sampleXθZ(Y; kwargs...)
    @timeit "sampleθZ"     θ, Z = sampleθZ(Y; kwargs...)
    @timeit "signal_model" X = signal_model(phys, θ)
    return X, θ, Z
end

function sampleX̂θZ(Y; temperature, kwargs...)
    @timeit "sampleXθZ" X, θ, Z = sampleXθZ(Y; temperature, kwargs...)
    @timeit "sampleX̂"   X̂ = sampleX̂(X, Z; temperature)
    return X̂, θ, Z
end

sampleX̂(Y; kwargs...) = sampleX̂θZ(Y; kwargs...)[1]
function sampleX̂(X, Z; temperature)
    if isnothing(temperature)
        corrected_signal_instance(derived["ricegen"], X, Z)
    else
        ν, σ = rician_params(derived["ricegen"], X, Z)
        add_noise_instance(derived["ricegen"], ν, (1 - temperature) .* σ)
    end
end

####
#### Augmentations
####

function augmentations(X::AbstractMatrix, X̄::Union{Nothing,<:AbstractMatrix})
    ∇X   = settings["train"]["augment"]["gradient"]::Bool ? derived["gradient"](X) : nothing # Signal gradient
    ∇²X  = settings["train"]["augment"]["laplacian"]::Bool ? derived["laplacian"](X) : nothing # Signal laplacian
    Xres = settings["train"]["augment"]["residuals"]::Bool && !isnothing(X̄) ? X .- X̄ : nothing # Residual relative to different sample X̄ = X(θ), θ ~ P(θ|Y)
    Xenc = settings["train"]["augment"]["encoderspace"]::Bool ? derived["encoderspace"](X) : nothing # Encoder-space signal
    Xfft = settings["train"]["augment"]["fftcat"]::Bool ? vcat(reim(rfft(X,1))...) : nothing # Concatenated real/imag fourier components
    Xrfft, Xifft = settings["train"]["augment"]["fftsplit"]::Bool ? reim(rfft(X,1)) : (nothing, nothing) # Separate real/imag fourier components

    ks = [:signal, :grad, :lap, :res, :enc, :fft, :rfft, :ifft]
    Xs = [X, ∇X, ∇²X, Xres, Xenc, Xfft, Xrfft, Xifft]
    is = (!isnothing).(Xs)
    ks = ks[is]
    Xs = Xs[is]

    return ks, Xs
end

function transformations(X::AbstractMatrix)
    Dchunk = settings["train"]["transform"]["Dchunk"]::Int
    flipsignals = settings["train"]["transform"]["flipsignals"]::Bool
    !(0 < Dchunk < size(X,1) || flipsignals) && return X
    i = ifelse(!(0 < Dchunk < size(X,1)), firstindex(X,1):lastindex(X,1), rand(firstindex(X,1):lastindex(X,1)-Dchunk+1) .+ (0:Dchunk-1))
    i = ifelse(!flipsignals, i, ifelse(rand(Bool), i, reverse(i)))
    return X[i,:]
end
transformations(X::AbstractTensor3D) = apply_dim1(transformations, X)

####
#### GANs
####

#= TODO
function GAN_augmentations(X,Z,X̄)
    Gsamples = settings["train"]["transform"]["Gsamples"]::Int
    ν, ϵ = rician_params(derived["ricegen"], X, Z)
    return add_noise_instance(derived["ricegen"], ν, ϵ, Gsamples)
end
=#
GAN_augmentations(X::AbstractMatrix, X̄::Union{Nothing,<:AbstractMatrix}) = reduce(vcat, transformations.(augmentations(X, X̄)[2]))
D_G_X_prob(X,Z,X̄) = apply_dim1(models["discrim"], GAN_augmentations(sampleX̂(X, Z; temperature = nothing), X̄)) # discrim on genatr data
D_Y_prob(Y,X̄) = apply_dim1(models["discrim"], GAN_augmentations(Y,X̄)) # discrim on real data
Dloss(X,Y,Z,X̄) = (ϵ = sqrt(eps(eltype(X))); -sum(log.(D_Y_prob(Y,X̄) .+ ϵ) .+ log.(1 .- D_G_X_prob(X,Z,X̄) .+ ϵ)) / size(Y,2))
Gloss(X,Z,X̄) = (ϵ = sqrt(eps(eltype(X))); sum(log.(1 .- D_G_X_prob(X,Z,X̄) .+ ϵ)) / size(Y,2))

####
#### MMD
####

# Maximum mean discrepency (m*MMD^2) loss
MMDloss(X̂,Y) = size(Y,2) * mmd(derived["kernel"], X̂, Y)
function MMDlosses(Y; temperature)
    # Sample θ,Z from CVAE posterior, differentiating only through generator corrections `sampleX̂`, without limiting noise on X̂ via `temperature`
    X, θ, Z = Zygote.@ignore sampleXθZ(Y; recover_θ = true, recover_Z = true, temperature) #TODO recover_Z?

    # X̂ = sampleX̂(X, Z; temperature = nothing) #TODO
    ν, ϵ = rician_params(derived["ricegen"], X, Z)
    X̂ = add_noise_instance(derived["ricegen"], ν, ϵ)

    ks, X̂s = augmentations(X̂, nothing)
    _, Ys = augmentations(Y, nothing)
    ks = Zygote.@ignore Symbol.(:MMD_, ks)

    ℓs = MMDloss.(X̂s, Ys)
    ℓ = Dict{Symbol,eltype(Y)}(tuple.(ks, ℓs))

    λ1 = eltype(Y)(settings["opt"]["mmd"]["lambda_eps"]::Float64)
    if λ1 > 0
        MMD_reg_eps = derived["laplacian"](ϵ)
        ℓ[:MMD_reg_eps] = λ1 * √(sum(abs2, MMD_reg_eps) / length(MMD_reg_eps))
    end

    λ2 = eltype(Y)(settings["opt"]["mmd"]["lambda_deps_dz"]::Float64)
    if λ2 > 0
        Δϵ = derived["gradient"](permutedims(ϵ, (2,1))) # (b-2) × n Differences
        ΔZ = derived["gradient"](permutedims(Z, (2,1))) # (b-2) × nz Differences
        ΔZ² = sum(abs2, ΔZ; dims = 2) ./ size(ΔZ, 2) # (b-2) × 1 Mean squared distance
        ΔZ0² = eltype(Y)(1e-3)
        MMD_reg_Z = @. Δϵ^2 / (ΔZ² + ΔZ0²)
        ℓ[:MMD_reg_Z] = λ2 * √(sum(MMD_reg_Z) / length(MMD_reg_Z))
    end

    return ℓ
end

####
#### CVAE
####

KLDivUnitNormal(μ, σ) = (sum(@. pow2(σ) + pow2(μ) - 2 * log(σ)) - length(μ)) / (2 * size(μ,2)) # KL-divergence between approximation posterior and N(0, 1) prior (Note: sum over dim=1, mean over dim=2)
KLDivergence(μq0, σq, μr0, σr) = (sum(@. pow2(σq / σr) + pow2((μr0 - μq0) / σr) - 2 * log(σq / σr)) - length(μq0)) / (2 * size(μq0,2)) # KL-divergence contribution to cross-entropy (Note: sum over dim=1, mean over dim=2)
EvidenceLowerBound(x, μx0, σx) = (sum(@. pow2((x - μx0) / σx) + 2 * log(σx)) + length(μx0) * log2π(eltype(μx0))) / (2 * size(μx0,2)) # Negative log-likelihood/ELBO contribution to cross-entropy (Note: sum over dim=1, mean over dim=2)
DataConsistency(Y, μG0, σG; dims = :) = -sum(@. MMDLearning._rician_logpdf(Flux.cpu.((Y, μG0, σG))...); dims) # Rician negative log likelihood

# Conditional variational autoencoder losses
function CVAElosses(Y; recover_Z = true, temperature)
    # Sample θ,Z from priors, limiting noise on X̂ via `temperature`
    X̂, θ, Z = Zygote.@ignore sampleX̂θZ(Y; recover_θ = false, recover_Z = false, temperature) # sample θ and Z priors

    # Cross-entropy loss function
    μr0, σr = split_mean_softplus_std(models["enc1"](X̂))
    μq0, σq = split_mean_softplus_std(models["enc2"](vcat(X̂,θ,Z))) #TODO X̂,θ,Z
    zq = sample_mv_normal(μq0, σq)
    μx0, σx  = split_mean_softplus_std(models["dec"](vcat(X̂,zq)))
    μθ0, μZ0 = split_theta_latent(μx0)
    σθ,  σZ  = split_theta_latent(σx)

    ℓ = Dict{Symbol,eltype(X̂)}()
    ℓ[:KLdiv] = KLDivergence(μq0, σq, μr0, σr)
    ℓ[:ELBO] = if recover_Z
        EvidenceLowerBound(vcat(θ,Z), μx0, σx)
    else
        EvidenceLowerBound(θ, μθ0, σθ)
    end

    #=
    if recover_Z
        # We have an extra degree of freedom: the generator is trained only on Z samples from the CVAE posterior,
        # therefore we must regularize the Z posterior sample means to be unit normally distributed in order to
        # ensure the generator inputs are N(0,1) and e.g. don't degenerate to a point mass or similar
        ℓ[:Zdiv] = KLDivUnitNormal(mean(μZ0; dims = 2), std(μZ0; dims = 2)) # mean/std using prediction means alone
        # ℓ[:Zdiv] = KLDivUnitNormal(mean(μZ0; dims = 2), sqrt.(mean(pow2.(σZ); dims = 2) .+ var(μZ0; dims = 2, corrected = false))) # mean/std where predictions are treated as a gaussian ensemble
    end
    =#

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
val_loader = torch.utils.data.DataLoader(make_data_tuples(:val); batch_size = settings["train"]["batchsize"], shuffle = false, drop_last = true) #TODO drop_last=true and batch_size=train_batchsize for MMD (else, batch_size = settings["data"]["nval"] is fine)

function train_step(engine, batch)
    Ytrain_cpu, = Ignite.array.(batch)
    Ytrain = Ytrain_cpu |> todevice
    outputs = Dict{Any,Any}()

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

        # # Train MMD every `MMDrate` epochs, and CVAE every other epoch
        # train_MMD  = mod(engine.state.epoch-1, settings["train"]["MMDrate"]::Int) == 0
        # train_CVAE = !train_MMD

        # # Train CVAE every iteration, MMD every `MMDrate` iterations
        # train_CVAE = true
        # train_MMD  = mod(engine.state.iteration-1, settings["train"]["MMDrate"]::Int) == 0

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

        every(rate) = rate <= 0 ? false : mod(engine.state.iteration-1, rate) == 0
        train_CVAE = true #TODO
        train_MMD = every(settings["train"]["MMDrate"]::Int) #TODO
        train_GAN = train_discrim = train_genatr = every(settings["train"]["GANrate"]::Int) #TODO
        temperature = global_temperature(engine.state.epoch) #TODO

        # Train CVAE loss
        train_CVAE && @timeit "cvae" CUDA.@sync let
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"])
            @timeit "forward"   CUDA.@sync ℓ, back = Zygote.pullback(() -> sum_dict(CVAElosses(Ytrain; recover_Z = true, temperature)), ps) #TODO recover_Z?
            @timeit "reverse"   CUDA.@sync gs = back(one(eltype(phys)))
            @timeit "update!"   CUDA.@sync Flux.Optimise.update!(optimizers["cvae"], ps, gs)
            outputs["CVAEloss"] = ℓ
        end

        # Train MMD loss
        train_MMD && @timeit "mmd" CUDA.@sync let
            @timeit "genatr" CUDA.@sync let
                ps = Flux.params(models["genatr"])
                @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> sum_dict(MMDlosses(Ytrain; temperature)), ps)
                @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["mmd"], ps, gs)
                outputs["MMD"] = ℓ
            end
        end

        # Train GAN loss
        train_GAN && @timeit "gan" CUDA.@sync let
            @timeit "sampleXθZ" CUDA.@sync Xtrain, θtrain, Ztrain = sampleXθZ(Ytrain; recover_θ = true, recover_Z = true, temperature) #TODO recover_Z?
            @timeit "sampleX̂"   CUDA.@sync X̂train = nothing #TODO sampleX̂(Ytrain; recover_θ = true, recover_Z = true, temperature) #TODO recover_Z?
            train_discrim && @timeit "discrim" CUDA.@sync let
                ps = Flux.params(models["discrim"])
                for _ in 1:settings["train"]["Dsteps"]
                    @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> Dloss(Xtrain, Ytrain, Ztrain, X̂train), ps)
                    @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                    @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["discrim"], ps, gs)
                    outputs["Dloss"] = ℓ
                end
            end
            train_genatr && @timeit "genatr" CUDA.@sync let
                ps = Flux.params(models["genatr"])
                @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> Gloss(Xtrain, Ztrain, X̂train), ps)
                @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["genatr"], ps, gs)
                outputs["Gloss"] = ℓ
            end
        end

        #= Train MMD kernel bandwidths
            if mod(engine.state.iteration-1, settings["train"]["kernelrate"]) == 0
                @timeit "MMD kernel" let
                    @timeit "sample G(X)" X̂train = sampleX̂(Ytrain; recover_θ = true, recover_Z = true, temperature) #TODO recover_Z?
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
    end

    return deepcopy(outputs)
end

function compute_metrics(engine, batch; dataset)
    @timeit "compute metrics" CUDA.@sync begin
        # Update callback state
        cb_state["last_time"] = get!(cb_state, "curr_time", time())
        cb_state["curr_time"] = time()
        cb_state["metrics"] = Dict{String,Any}()

        # Initialize output metrics dictionary
        log_metrics = Dict{Symbol,Any}()
        log_metrics[:epoch]   = trainer.state.epoch
        log_metrics[:iter]    = trainer.state.iteration
        log_metrics[:dataset] = dataset
        log_metrics[:time]    = cb_state["curr_time"] - cb_state["last_time"]

        # Invert Y and make Xs
        Y, = Ignite.array.(batch) .|> todevice
        Nbatch = size(Y,2)
        θ, Z = InvertY(Y)
        X = signal_model(phys, θ)
        δG0, σG = correction_and_noiselevel(derived["ricegen"], X, Z)
        μG0 = add_correction(derived["ricegen"], X, δG0)
        X̂ = add_noise_instance(derived["ricegen"], μG0, σG)

        let
            temperature = global_temperature(trainer.state.epoch)

            ℓ_CVAE = CVAElosses(Y; recover_Z = true, temperature) #TODO recover_Z?
            ℓ_CVAE[:CVAEloss] = sum_dict(ℓ_CVAE)
            merge!(log_metrics, ℓ_CVAE)

            m = min(size(X̂,2), settings["train"]["batchsize"])
            ℓ_MMD = MMDlosses(Y[:,1:m]; temperature)
            ℓ_MMD[:MMDloss] = sum_dict(ℓ_MMD)
            merge!(log_metrics, ℓ_MMD)

            loss = ℓ_CVAE[:CVAEloss] + ℓ_MMD[:MMDloss]
            Zreg = sum(abs2, Z) / (2*Nbatch)
            @pack! log_metrics = loss, Zreg

            if settings["train"]["GANrate"]::Int > 0
                ϵ = sqrt(eps(eltype(X)))
                X̄new = nothing #TODO sampleX̂(Y; recover_θ = true, recover_Z = true, temperature = nothing) #TODO recover_Z?
                d_y = D_Y_prob(Y, X̄new)
                d_g_x = D_G_X_prob(X, Z, X̄new)
                Dloss = -mean(log.(d_y .+ ϵ) .+ log.(1 .- d_g_x .+ ϵ))
                Gloss = mean(log.(1 .- d_g_x .+ ϵ))
                D_Y   = mean(d_y)
                D_G_X = mean(d_g_x)
            else
                Gloss = Dloss = D_Y = D_G_X = missing #TODO
            end
            @pack! log_metrics = Gloss, Dloss, D_Y, D_G_X
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
            Yθhat = hasclosedform(phys) ? signal_model(ClosedForm(phys), θ, noiselevel(ClosedForm(phys), θ, Z)) : missing
            cache_cb_state!(Y, θ, Z, X, δG0, σG, μG0, X̂, Yθ, Yθhat; suf = "")

            all_Yhat_rmse = sqrt.(mean(abs2, Y .- X̂; dims = 1)) |> Flux.cpu |> vec
            all_Yhat_logL = DataConsistency(Y, μG0, σG; dims = 1) |> vec
            Yhat_rmse = mean(all_Yhat_rmse)
            Yhat_logL = mean(all_Yhat_logL)
            @pack! cb_state["metrics"] = all_Yhat_rmse, all_Yhat_logL
            @pack! log_metrics = Yhat_rmse, Yhat_logL
        end

        # Cache values for evaluating CVAE performance for estimating parameters of Y
        let
            θfit, Zfit = sampleθZ(X̂; recover_θ = true, recover_Z = true, temperature = nothing) #TODO recover_Z?
            θfit .= clamp.(θfit, todevice(θlower(phys)), todevice(θupper(phys)))
            Xθfit = signal_model(phys, θfit)
            δθfit, ϵθfit = correction_and_noiselevel(derived["ricegen"], Xθfit, Zfit)
            Xθδfit = add_correction(derived["ricegen"], Xθfit, δθfit)
            Xθhatfit = add_noise_instance(derived["ricegen"], Xθδfit, ϵθfit)
            Yθfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit) : missing
            Yθhatfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit, noiselevel(ClosedForm(phys), θfit, Zfit)) : missing
            cache_cb_state!(X̂, θfit, Zfit, Xθfit, δθfit, ϵθfit, Xθδfit, Xθhatfit, Yθfit, Yθhatfit; suf = "fit") #TODO X̂ or Y?

            rmse = hasclosedform(phys) ? sqrt(mean(abs2, Yθfit - Xθδfit)) : missing
            all_Xhat_rmse = sqrt.(mean(abs2, X̂ .- Xθhatfit; dims = 1)) |> Flux.cpu |> vec #TODO X̂ or Y?
            all_Xhat_logL = DataConsistency(X̂, Xθδfit, ϵθfit; dims = 1) |> vec #TODO X̂ or Y?
            Xhat_rmse = mean(all_Xhat_rmse)
            Xhat_logL = mean(all_Xhat_logL)
            theta_err = 100 .* mean(abs, (θ .- θfit) ./ (todevice(θupper(phys)) .- todevice(θlower(phys))); dims = 2) |> Flux.cpu |> vec |> copy
            Z_err = mean(abs, Z .- Zfit; dims = 2) |> Flux.cpu |> vec |> copy
            @pack! cb_state["metrics"] = all_Xhat_rmse, all_Xhat_logL
            @pack! log_metrics = Xhat_rmse, Xhat_logL, theta_err, Z_err, rmse
        end

        # Update logger dataframe
        is_consecutive = !isempty(logger) && (logger.epoch[end] == log_metrics[:epoch] && logger.iter[end] == log_metrics[:iter] && logger.dataset[end] === log_metrics[:dataset])
        nbatches = div(settings["data"]["n$dataset"]::Int, settings["train"]["batchsize"]::Int)
        !is_consecutive ? push!(logger, log_metrics; cols = :union) : (logger.time[end] += log_metrics[:time])
        for (k,v) in log_metrics
            if k ∉ [:epoch, :iter, :dataset, :time]
                !is_consecutive ? (logger[end, k] /= nbatches) : (logger[end, k] += v / nbatches)
            end
        end

        # Return metrics for logging
        output_metrics = Dict{Any,Any}(string(k) => deepcopy(v) for (k,v) in log_metrics if k ∉ [:epoch, :iter, :dataset, :time]) # output non-housekeeping metrics
        merge!(cb_state["metrics"], output_metrics) # merge all log metrics into cb_state
        filter!(((k,v),) -> !ismissing(v), output_metrics) # return non-missing metrics (wandb cannot handle missing)

        return output_metrics
    end
end

function makeplots(;showplot = false)
    function plot_epsilon(; knots = (-5.0, 5.0), seriestype = :line, showplot = false)
        function plot_epsilon_inner(; start, stop, zlen = 256, levels = 50)
            n, nθ, nz = nsignal(phys)::Int, ntheta(phys)::Int, settings["arch"]["nlatent"]::Int
            Y = sampleY(phys, zlen * nz; dataset = :train) |> to32
            θ = sampleθprior_similar(Y, zlen * nz)
            X = signal_model(phys, θ)
            Z = repeat(randn_similar(Y, nz, 1), 1, zlen, nz)
            for i in 1:nz
                Z[i,:,i] .= range(start, stop; length = zlen)
            end
            ϵ = rician_params(derived["ricegen"], X, reshape(Z, nz, :))[2] |> ϵ -> reshape(ϵ, :, zlen, nz)
            Y, θ, X, Z, ϵ = Flux.cpu.((Y, θ, X, Z, ϵ))
            ps = map(1:nz) do i
                zlabs = nz == 1 ? "" : latexstring(" (" * join(map(j -> L"$Z_%$(j)$ = %$(round(Z[1,1,j]; digits = 2))", setdiff(1:nz, i)), ", ") * ")")
                kwcommon = (; leg = :none, colorbar = :right, color = cgrad(:cividis), xlabel = L"$t$", title = L"$\epsilon$ vs. $t$ and $Z_{%$(i)}$%$(zlabs)")
                if seriestype === :surface
                    surface(reshape(1:n,n,1), Z[i,:,i]', ϵ[:,:,i]; ylabel = L"$Z_{%$(i)}$", fill_z = ϵ[:,:,i], camera = (60.0, 30.0), kwcommon...)
                elseif seriestype === :contour
                    contourf(repeat(1:n,1,zlen), repeat(Z[i,:,i]',n,1), ϵ[:,:,i]; ylabel = L"$Z_{%$(i)}$", levels, kwcommon...)
                else
                    plot(ϵ[:,:,i]; line_z = Z[i,:,i]', ylabel = L"$\epsilon$", alpha = 0.3, kwcommon...)
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
    # plot_epsilon(; seriestype = :line) #TODO
    # plot_epsilon(; seriestype = :contour) #TODO
    # plot(plot_epsilon(; seriestype = :line), plot_epsilon(; seriestype = :contour)) #TODO

    try
        Dict{Symbol, Any}(
            :ricemodel   => MMDLearning.plot_rician_model(logger, cb_state, phys; showplot = showplot, bandwidths = haskey(models, "logsigma") ? (permutedims(models["logsigma"]) |> Flux.cpu) : nothing),
            :signals     => MMDLearning.plot_rician_signals(logger, cb_state, phys; showplot = showplot),
            :vaesignals  => MMDLearning.plot_vae_rician_signals(logger, cb_state, phys; showplot = showplot),
            :infer       => MMDLearning.plot_rician_inference(logger, cb_state, phys; showplot = showplot),
            :ganloss     => MMDLearning.plot_gan_loss(logger, cb_state, phys; showplot = showplot, lrdroprate = settings["opt"]["lrrate"], lrdrop = settings["opt"]["lrdrop"]),
            :vallosses   => MMDLearning.plot_all_logger_losses(logger, cb_state, phys; dataset = :val, showplot = showplot),
            :trainlosses => MMDLearning.plot_all_logger_losses(logger, cb_state, phys; dataset = :train, showplot = showplot),
            :epsline     => plot_epsilon(; showplot = showplot, seriestype = :line), #TODO
            :epscontour  => plot_epsilon(; showplot = showplot, seriestype = :contour), #TODO
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
