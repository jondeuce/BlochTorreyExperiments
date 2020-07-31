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

const torch = pyimport("torch")
const wandb = pyimport("wandb")
const ignite = pyimport("ignite")
const logging = pyimport("logging")
py"""
from ignite.contrib.handlers.wandb_logger import *
"""

const Events = ignite.engine.Events
const WandBLogger = ignite.contrib.handlers.wandb_logger.WandBLogger
const wandb_logger = haskey(ENV, "JL_WANDB_LOGGER") ? WandBLogger() : nothing

const settings = TOML.parse("""
    [data]
        out    = "$(!isnothing(wandb_logger) ? wandb.run.dir : "./output/ignite-cvae-" * MMDLearning.getnow())"
        ntrain = 102_400
        ntest  = 10_240
        nval   = 10_240

    [train]
        timeout   = 10800.0 #TODO 1e9
        epochs    = 999_999
        batchsize = 1024 #TODO 256 2048
        kernelrate  = 10 # Train kernel every `kernelrate` iterations
        kernelsteps = 1 # Gradient updates per kernel train
        GANrate     = 1 # Train GAN losses every `GANrate` iterations
        Dsteps      = 1 # Train GAN losses with `Dsteps` discrim updates per genatr update

    [eval]
        metricperiod = 30.0 #TODO
        printperiod  = 60.0 #TODO
        saveperiod   = 300.0 #TODO
        showrate     = 1 #TODO

    [opt]
        lr       = 1e-4 #TODO
        lrdrop   = 1.0
        lrthresh = 1e-5
        lrrate   = 1000
        [opt.cvae]
            lr = "%PARENT%"
        [opt.genatr]
            lr = "%PARENT%"
        [opt.discrim]
            lr = "%PARENT%"
        [opt.mmd]
            lr = "%PARENT%"
        [opt.kernel]
            loss = "mmd" #"tstatistic"
            lr = 1e-2

    [arch]
        physics = "$(get(ENV, "JL_PHYS_MODEL", "toy"))" # "toy" or "mri"
        nlatent = 1 # number of latent variables Z
        zdim    = 6 # embedding dimension of z
        hdim    = 256 # default for models below
        nhidden = 4 # default for models below
        [arch.enc1]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
        [arch.enc2]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
        [arch.dec]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
        [arch.genatr]
            hdim        = 32
            nhidden     = 2
            maxcorr     = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? 0.1 : 0.025) # correction amplitude
            noisebounds = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? [-8.0, -2.0] : [-6.0, -3.0]) # noise amplitude
        [arch.discrim]
            hdim    = "%PARENT%"
            nhidden = "%PARENT%"
        [arch.kernel]
            nbandwidth = 8
            bwbounds   = $(get(ENV, "JL_PHYS_MODEL", "toy") == "toy" ? [-8.0, 4.0] : [-10.0, 4.0]) # bounds for kernel bandwidths (logsigma)
""")
Ignite.parse_command_line!(settings)
Ignite.save_and_print(settings; outpath = settings["data"]["out"], filename = "settings.toml")

# Initialize generator + discriminator + kernel
function make_models(phys, ::Type{Gtype}) where {Gtype<:RicianCorrector}
    models = Dict{String, Any}()
    n   = nsignal(phys) # input signal length
    nθ  = ntheta(phys) # number of physics variables
    θbd = θbounds(phys)
    k   = settings["arch"]["nlatent"]::Int # number of latent variables Z
    nz  = settings["arch"]["zdim"]::Int # embedding dimension
    toT(m) = Flux.paramtype(eltype(phys), m) |> todevice

    # Rician generator. First `n` elements for `δX` scaled to (-δ, δ), second `n` elements for `logϵ` scaled to (noisebounds[1], noisebounds[2])
    models["genatr"] = let
        hdim = settings["arch"]["genatr"]["hdim"]::Int
        nhidden = settings["arch"]["genatr"]["nhidden"]::Int
        maxcorr = settings["arch"]["genatr"]["maxcorr"]::Float64
        noisebounds = settings["arch"]["genatr"]["noisebounds"]::Vector{Float64}
        nin  = Gtype <: Union{<:VectorRicianCorrector, <:FixedNoiseVectorRicianCorrector} ? n + k :
               Gtype <: Union{<:LatentVectorRicianCorrector, <:LatentVectorRicianNoiseCorrector} ? k :
               error("Unsupported corrector type: $Gtype")
        nout = Gtype <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? 2n :
               Gtype <: Union{<:FixedNoiseVectorRicianCorrector, <:LatentVectorRicianNoiseCorrector} ? n :
               error("Unsupported corrector type: $Gtype")
        OutputScale =
            Gtype <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? MMDLearning.CatScale([(-maxcorr, maxcorr), (noisebounds...,)], [n,n]) :
            Gtype <: FixedNoiseVectorRicianCorrector ? MMDLearning.CatScale([(-maxcorr, maxcorr)], [n]) :
            Gtype <: LatentVectorRicianNoiseCorrector ? MMDLearning.CatScale([(noisebounds...,)], [n]) :
            error("Unsupported corrector type: $Gtype")
        Flux.Chain(
            MMDLearning.MLP(nin => nout, nhidden, hdim, Flux.relu, tanh)...,
            OutputScale
        ) |> toT
    end

    # Encoders
    models["enc1"] = let
        hdim = settings["arch"]["enc1"]["hdim"]::Int
        nhidden = settings["arch"]["enc1"]["nhidden"]::Int
        MMDLearning.MLP(n => 2*nz, nhidden, hdim, Flux.relu, identity) |> toT
    end

    models["enc2"] = let
        hdim = settings["arch"]["enc2"]["hdim"]::Int
        nhidden = settings["arch"]["enc2"]["nhidden"]::Int
        MMDLearning.MLP(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity) |> toT
    end

    # Decoder
    models["dec"] = let
        hdim = settings["arch"]["dec"]["hdim"]::Int
        nhidden = settings["arch"]["dec"]["nhidden"]::Int
        Flux.Chain(
            MMDLearning.MLP(n + nz => 2*(nθ + k), nhidden, hdim, Flux.relu, identity)...,
            MMDLearning.CatScale(eltype(θbd)[θbd; (-1, 1)], [ones(Int, nθ); k + nθ + k]),
        ) |> toT
    end

    # Discriminator
    models["discrim"] = let
        hdim = settings["arch"]["discrim"]["hdim"]::Int
        nhidden = settings["arch"]["discrim"]["nhidden"]::Int
        MMDLearning.MLP(n => 1, nhidden, hdim, Flux.relu, Flux.sigmoid) |> toT
    end

    # MMD kernel bandwidths
    models["logsigma"] = let
        bwbounds = settings["arch"]["kernel"]["bwbounds"]::Vector{Float64}
        nbandwidth = settings["arch"]["kernel"]["nbandwidth"]::Int
        repeat(range(bwbounds...; length = nbandwidth+2)[2:end-1], 1, n) |> toT
    end

    return models
end

const phys = initialize!(
    ToyModel{Float32,true}();
    ntrain = settings["data"]["ntrain"]::Int,
    ntest = settings["data"]["ntest"]::Int,
    nval = settings["data"]["nval"]::Int,
)
RiceGenType = LatentVectorRicianCorrector #LatentVectorRicianNoiseCorrector #VectorRicianCorrector
const models = make_models(phys, RiceGenType)
# const models = deepcopy(BSON.load("/home/jdoucette/Documents/code/BlochTorreyExperiments-master/LearningCorrections/output/ignite-cvae-2020-07-29-T-10-42-28-529/current-models.bson")["models"]) |> d -> MMDLearning.map_dict(todevice, d) #TODO
const ricegen = Dict{String,Any}(
    "genatr" => RiceGenType(models["genatr"]), # Generator produces 𝐑^2n outputs parameterizing n Rician distributions
)
const optimizers = Dict{String,Any}(
    "genatr"  => Flux.ADAM(settings["opt"]["genatr"]["lr"]),
    "discrim" => Flux.ADAM(settings["opt"]["discrim"]["lr"]),
    "mmd"     => Flux.ADAM(settings["opt"]["mmd"]["lr"]),
    "cvae"    => Flux.ADAM(settings["opt"]["cvae"]["lr"]),
)
MMDLearning.model_summary(models, joinpath(settings["data"]["out"], "model-summary.txt"))

# Helpers
@inline split_theta_latent(θZ::AbstractMatrix) = size(θZ,1) == ntheta(phys) ? (θZ, similar(θZ,0,size(θZ,2))) : (θZ[1:ntheta(phys),:], θZ[ntheta(phys)+1:end,:])
@inline split_mean_std(μ::AbstractMatrix) = μ[1:end÷2, :], Flux.softplus.(μ[end÷2+1:end, :]) .+ sqrt(eps(eltype(μ))) #TODO Flux.softplus -> exp?
@inline sample_mv_normal(μ0::AbstractMatrix{T}, σ::AbstractMatrix{T}) where {T} = μ0 .+ σ .* randn_similar(σ, max.(size(μ0), size(σ)))
@inline sample_mv_normal(μ0::Matrix{T}, σ::Matrix{T}) where {T} = μ0 .+ σ .* randn_similar(σ, max.(size(μ0), size(σ)))
@inline pow2(x) = x*x
const theta_lower_bounds = θlower(phys) |> todevice
const theta_upper_bounds = θupper(phys) |> todevice
sampleθprior_similar(Y, n = size(Y,2)) = rand_similar(Y, ntheta(phys), n) .* (theta_upper_bounds .- theta_lower_bounds) .+ theta_lower_bounds

# KL-divergence contribution to cross-entropy (Note: dropped constant -zdim/2 term)
KLDivergence(μq0, σq, μr0, σr) = sum(@. pow2(σq / σr) + pow2((μr0 - μq0) / σr) - 2 * log(σq / σr)) / 2

# Negative log-likelihood/ELBO contribution to cross-entropy (Note: dropped constant +zdim*log(2π)/2 term)
EvidenceLowerBound(x, μx0, σx) = sum(@. pow2((x - μx0) / σx) + 2 * log(σx)) / 2

# GAN losses
D_Y_loss(Y) = models["discrim"](Y) # discrim on real data
D_G_X_loss(X,Z) = models["discrim"](corrected_signal_instance(ricegen["genatr"], X, Z)) # discrim on genatr data
Dloss(X,Y,Z) = -mean(log.(D_Y_loss(Y)) .+ log.(1 .- D_G_X_loss(X,Z)))
Gloss(X,Z) = mean(log.(1 .- D_G_X_loss(X,Z)))

# Maximum mean discrepency (m*MMD^2) loss
MMDloss(X̂,Y) = size(Y,2) * mmd_flux(models["logsigma"], X̂, Y)

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
    @timeit "sampleX̂"   CUDA.@sync X̂ = corrected_signal_instance(ricegen["genatr"], X, Z)
    return X̂, θ, Z
end

sampleX̂(Y; kwargs...) = sampleX̂θZ(Y; kwargs...)[1]

function DataConsistency(Y, μG0, σG)
    # YlogL = -sum(@. MMDLearning._rician_logpdf(Y, μG0, σG)) # Rician negative log likelihood
    YlogL = sum(@. 2 * log(σG) + pow2((Y - μG0) / σG)) / 2 # Gaussian negative likelihood for testing
    # YlogL += 1000 * sum(abs2, Y .- add_noise_instance(ricegen["genatr"], μG0, σG)) / 2 # L2 norm for testing/pretraining
    # YlogL = 10 * sum(abs, Y .- add_noise_instance(ricegen["genatr"], μG0, σG)) # L1 norm for testing/pretraining
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
    μG0, σG = rician_params(ricegen["genatr"], X, Z) # Rician negative log likelihood
    X̂ = add_noise_instance(ricegen["genatr"], μG0, σG)

    # Data consistency penalty
    # YlogL = DataConsistency(Y, μG0, σG) / Nbatch #TODO

    # Add MMD loss contribution
    MMDsq = MMDloss(X̂, Y) #TODO

    # Drop gradients for θ, Z, and X̂
    θ = Zygote.dropgrad(θ)
    Z = Zygote.dropgrad(Z)
    X̂ = Zygote.dropgrad(X̂)
    Hloss = CVAEloss(X̂, θ, Z; recover_Z = recover_Z) #TODO X̂ or Y?

    ℓ = Zreg + Hloss + MMDsq #TODO
    # ℓ = Zreg + YlogL + Hloss + MMDsq

    return ℓ
end

# Regularize generator outputs
function RegularizeX̂(Y; recover_Z = true)
    # Invert Y
    Nbatch = size(Y,2)
    θhat, Zhat = InvertY(Y)

    # X = signal_model(phys, θhat) # differentiate through physics model
    # μG0, σG = rician_params(ricegen["genatr"], X, Zhat)
    # YlogL = DataConsistency(Y, μG0, σG) / Nbatch

    # Limit distribution of X̂ ∼ G(X) with MMD
    # X = Zygote.dropgrad(X)
    θ = Zygote.dropgrad(θhat)
    X = Zygote.dropgrad(signal_model(phys, θ))
    Z = (recover_Z ? randn : zeros)(eltype(Zhat), size(Zhat)...)
    μG0, σG = rician_params(ricegen["genatr"], X, Z)
    X̂ = add_noise_instance(ricegen["genatr"], μG0, σG)
    MMDsq = MMDloss(X̂, Y)

    # Return total loss
    ℓ = MMDsq #TODO
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

        if mod(engine.state.iteration-1, settings["train"]["GANrate"]) == 0
            @timeit "gan" CUDA.@sync let
                @timeit "sampleXθZ" CUDA.@sync Xtrain, θtrain, Ztrain = sampleXθZ(Ytrain; recover_θ = true, recover_Z = false) .|> todevice
                @timeit "discrim" CUDA.@sync let
                    ps = Flux.params(models["discrim"])
                    for _ in 1:settings["train"]["Dsteps"]
                        @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> Dloss(Xtrain, Ytrain, Ztrain), ps)
                        @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                        @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["discrim"], ps, gs)
                        metrics["Dloss"] = ℓ
                    end
                end
                @timeit "genatr" CUDA.@sync let
                    ps = Flux.params(models["genatr"])
                    @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> Gloss(Xtrain, Ztrain), ps)
                    @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
                    @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["genatr"], ps, gs)
                    metrics["Gloss"] = ℓ
                end
            end
        end

        # Train CVAE loss
        @timeit "cvae" CUDA.@sync let
            @timeit "sampleX̂θZ" CUDA.@sync X̂train, θtrain, Ztrain = sampleX̂θZ(Ytrain; recover_θ = false, recover_Z = false) .|> todevice
            ps = Flux.params(models["enc1"], models["enc2"], models["dec"])
            @timeit "forward" CUDA.@sync ℓ, back = Zygote.pullback(() -> CVAEloss(X̂train, θtrain, Ztrain; recover_Z = true), ps)
            @timeit "reverse" CUDA.@sync gs = back(one(eltype(phys)))
            @timeit "update!" CUDA.@sync Flux.Optimise.update!(optimizers["cvae"], ps, gs)
            metrics["CVAEloss"] = ℓ
        end

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

function val_metrics(engine, batch)
    @timeit "val batch" CUDA.@sync begin
        # Update callback state
        cb_state["last_time"] = get!(cb_state, "curr_time", time())
        cb_state["curr_time"] = time()
        cb_state["metrics"] = Dict{String,Any}()

        # Invert Y and make Xs
        Y, = Ignite.array.(batch) .|> todevice
        Nbatch = size(Y,2)
        θ, Z = InvertY(Y)
        X = signal_model(phys, θ)
        δG0, σG = correction_and_noiselevel(ricegen["genatr"], X, Z)
        μG0 = add_correction(ricegen["genatr"], X, δG0)
        X̂ = add_noise_instance(ricegen["genatr"], μG0, σG)

        # Cross-entropy loss function
        μr0, σr = split_mean_std(models["enc1"](X̂)) #TODO X̂ or Y?
        μq0, σq = split_mean_std(models["enc2"](vcat(X̂,θ,Z))) #TODO X̂ or Y?
        zq = sample_mv_normal(μq0, σq)
        μx0, σx = split_mean_std(models["dec"](vcat(X̂,zq))) #TODO X̂ or Y?

        let
            Zreg = sum(abs2, Z) / (2*Nbatch)
            # YlogL = DataConsistency(Y, μG0, σG) / Nbatch #TODO
            KLdiv = KLDivergence(μq0, σq, μr0, σr) / Nbatch
            ELBO = EvidenceLowerBound(vcat(θ,Z), μx0, σx) / Nbatch
            # MMDsq = let m = settings["train"]["batchsize"]
            #     MMDloss(X̂[:,1:min(end,m)], Y[:,1:min(end,m)]) #TODO
            # end
            MMDsq = missing
            loss = KLdiv + ELBO #TODO Zreg, MMDsq, YlogL

            d_y = D_Y_loss(Y)
            d_g_x = D_G_X_loss(X, Z)
            Gloss = mean(log.(1 .- d_g_x))
            Dloss = -mean(log.(d_y) .+ log.(1 .- d_g_x))
            D_Y   = mean(d_y)
            D_G_X = mean(d_g_x)

            @pack! cb_state["metrics"] = Zreg, KLdiv, ELBO, loss, MMDsq, Gloss, Dloss, D_Y, D_G_X #TODO YlogL
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
            δθfit, ϵθfit = correction_and_noiselevel(ricegen["genatr"], Xθfit, Zfit)
            Xθδfit = add_correction(ricegen["genatr"], Xθfit, δθfit)
            Xθhatfit = add_noise_instance(ricegen["genatr"], Xθδfit, ϵθfit)
            Yθfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit) : missing
            Yθhatfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit, noiselevel(ClosedForm(phys))) : missing
            cache_cb_state!(X̂, θfit, Zfit, Xθfit, δθfit, ϵθfit, Xθδfit, Xθhatfit, Yθfit, Yθhatfit; suf = "fit") #TODO X̂ or Y?

            # Condition for learning corrections #TODO
            # X̂_norm_diff = sqrt(mean(abs2, X̂ - Xθδfit))
            # X̂_norm_thresh = 3 * sqrt(mean(abs2, ϵθfit))
            # @show X̂_norm_diff
            # @show X̂_norm_thresh
            # cb_state["learncorrections"] = X̂_norm_diff <= X̂_norm_thresh

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
        metrics[:dataset] = :val
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
        push!(logger, metrics; cols = :subset)

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

evaluator = ignite.engine.Engine(@j2p val_metrics)
evaluator.logger = ignite.utils.setup_logger("evaluator")

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
    # Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(every = 1), #TODO
    Events.STARTED | Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p event_throttler(settings["eval"]["metricperiod"])),
    @j2p function (engine)
        evaluator.run(val_loader)
    end
)

# Checkpoint current model + logger + make plots
trainer.add_event_handler(
    # Events.STARTED | Events.EPOCH_COMPLETED(every = 25), #TODO
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
    # Events.EPOCH_COMPLETED(every = 10), #TODO
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
            for optname in ["cvae", "mmd"]
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
    # Events.EPOCH_COMPLETED(every = 10),
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p event_throttler(settings["eval"]["printperiod"])),
    @j2p function (engine)
        if mod(engine.state.epoch-1, settings["eval"]["showrate"]) == 0
            show(stdout, TimerOutputs.get_defaulttimer()); println("\n")
            show(stdout, last(logger, 10)); println("\n")
        end
        (engine.state.epoch == 1) && TimerOutputs.reset_timer!() # throw out compilation timings
    end
)

####
#### Weights & biases logger
####

# Attach training/validation output handlers
if !isnothing(wandb_logger)
    for (tag, engine) in [("training", trainer), ("validation", evaluator)]
        wandb_logger.attach_output_handler(
            engine;
            event_name = Events.EPOCH_COMPLETED(event_filter = @j2p run_timeout(settings["eval"]["metricperiod"])),
            tag = tag,
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
