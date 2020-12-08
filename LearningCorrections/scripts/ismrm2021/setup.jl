####
#### Setup
####

using MMDLearning
Ignite.init()
pyplot(size=(800,600))

Revise.includet(joinpath(@__DIR__, "physics.jl"))
Revise.includet(joinpath(@__DIR__, "cvae.jl"))
Revise.includet(joinpath(@__DIR__, "xformer.jl"))
Revise.includet(joinpath(@__DIR__, "eval.jl"))
Revise.includet(joinpath(@__DIR__, "plot.jl"))

####
#### Settings
####

new_savepath() = "./output/ignite-cvae-$(MMDLearning.getnow())"

function new_settings_template()
    TOML.parse(
    """
    [data]
        out    = "$(new_savepath())"
        ntrain = "auto" # 102_400
        ntest  = "auto" # 10_240
        nval   = "auto" # 10_240

    [train]
        timeout     = 1e9 #TODO 10800.0
        epochs      = 1000_000
        batchsize   = 1024  #256 #512 #1024 #2048 #3072 #4096
        nbatches    = 1000  # number of batches per epoch
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
        valevalperiod   = 300.0
        trainevalperiod = 600.0
        saveperiod      = 300.0
        printperiod     = 300.0

    [opt]
        lrrel    = 0.03    #0.1  # Learning rate relative to batch size, i.e. lr = lrrel / batchsize
        lrthresh = 0.0     #1e-6 # Absolute minimum learning rate
        lrdrop   = 3.16    # Drop learning rate by factor `lrdrop` every `lrrate` epochs
        lrrate   = 999_999 # Drop learning rate by factor `lrdrop` every `lrrate` epochs
        gclip    = 0.0
        wdecay   = 1e-6
        [opt.cvae]
            INHERIT = "%PARENT%"
            lambda_pseudo = 0.0 # Weighting of pseudo label loss
        [opt.genatr]
            INHERIT = "%PARENT%" #TODO: 0.01 train generator more slowly
        [opt.discrim]
            INHERIT = "%PARENT%"
        [opt.mmd]
            INHERIT = "%PARENT%"
            gclip = 1.0
            lambda_0        = 100.0 # MMD loss weighting relative to CVAE
            lambda_eps      = 0.0   # Regularize noise amplitude epsilon
            lambda_deps_dz  = 0.0   # Regularize gradient of epsilon w.r.t. latent variables
        [opt.kernel]
            INHERIT = "%PARENT%" # Kernel learning rate 
            loss  = "mmd"        # Kernel loss ("mmd", "tstatistic", or "mmd_diff")

    [arch]
        nlatent   = 1   # number of latent variables Z
        zdim      = 12  # embedding dimension of z
        hdim      = 256 # size of hidden layers
        nhidden   = 4   # number of hidden layers
        skip      = false # skip connection
        layernorm = false # layer normalization following dense layer
        head      = 2  # number of attention heads
        psize     = 48 # transformer embedding dimension
        chunksize = 48 # nshards  == (nsignals - chunksize) ÷ (chunksize - overlap) + 1
        overlap   = 47 # nsignals == nshards * (chunksize - overlap) + overlap
        [arch.enc1]
            INHERIT = "%PARENT%"
        [arch.enc2]
            INHERIT = "%PARENT%"
        [arch.dec]
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
end

make_default_settings() = Ignite.parse_command_line!(new_settings_template())

# Parse command line arguments into default settings
function make_settings()
    if haskey(ENV, "JL_CHECKPOINT_FOLDER")
        settings = TOML.parsefile(joinpath(ENV["JL_CHECKPOINT_FOLDER"], "settings.toml"))
        settings["data"]["out"] = new_savepath()
        return settings
    else
        make_default_settings()
    end
end

####
#### Physics
####

function make_physics(settings)
    # initialize!(
    #     ToyEPGModel{Float32,true}();
    #     ntrain = settings["data"]["ntrain"]::Int,
    #     ntest = settings["data"]["ntest"]::Int,
    #     nval = settings["data"]["nval"]::Int
    # )
    imageinfos = [
        (TE = 8e-3, refcon = 180.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Example_48echo_8msTE/data-in/ORIENTATION_B0_08_WIP_MWF_CPMG_CS_AXIAL_5_1.masked-image.nii.gz"),
        (TE = 7e-3, refcon = 180.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Example_56echo_7msTE_CPMG/data-in/MW_TRAINING_001_WIP_CPMG56_CS_half_2_1.masked-image.mat"),
    ]
    phys = EPGModel{Float32,false}(n = 64) #TODO
    initialize!(phys; imageinfos, seed = 0)
end

####
#### Models
####

# Initialize generator + discriminator + kernel
function make_models!(phys::PhysicsModel{Float32}, settings::Dict{String,Any}, models = Dict{String, Any}(), derived = Dict{String, Any}())
    n   = nsignal(phys) # input signal length
    nθ  = ntheta(phys) # number of physics variables
    nθM = nmarginalized(phys) # number of marginalized, i.e. recovered, physics variables
    θbd = θbounds(phys)
    θMbd= θmarginalized(phys, θbounds(phys))
    k   = settings["arch"]["nlatent"]::Int # number of latent variables Z
    nz  = settings["arch"]["zdim"]::Int # embedding dimension
    δ   = settings["arch"]["genatr"]["maxcorr"]::Float64
    σbd = settings["arch"]["genatr"]["noisebounds"]::Vector{Float64} |> bd -> (bd...,)::NTuple{2,Float64}

    #TODO: only works for Latent(*)Corrector family
    RiceGenType = LatentScalarRicianNoiseCorrector{n,k}
    # RiceGenType = LatentVectorRicianNoiseCorrector{n,k}
    # RiceGenType = LatentVectorRicianCorrector{n,k}
    # RiceGenType = VectorRicianCorrector{n,k}

    OutputScale = let
        RiceGenType <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? MMDLearning.CatScale([(-δ, δ), σbd], [n,n]) :
        RiceGenType <: FixedNoiseVectorRicianCorrector ? MMDLearning.CatScale([(-δ, δ)], [n]) :
        RiceGenType <: LatentVectorRicianNoiseCorrector ? MMDLearning.CatScale([σbd], [n]) :
        RiceGenType <: LatentScalarRicianNoiseCorrector ? MMDLearning.CatScale([σbd], [1]) :
        error("Unsupported corrector type: $RiceGenType")
    end

    # Priors for physics model and latent variables
    let
        hdim = settings["arch"]["genatr"]["hdim"]::Int
        ktheta = settings["arch"]["genatr"]["ktheta"]::Int
        klatent = settings["arch"]["genatr"]["klatent"]::Int
        nhidden = settings["arch"]["genatr"]["nhidden"]::Int
        leakyslope = settings["arch"]["genatr"]["leakyslope"]::Float64
        σinner = leakyslope == 0 ? Flux.relu : eltype(phys)(leakyslope) |> a -> (x -> Flux.leakyrelu(x, a))
        get!(models, "theta_prior") do
            Flux.Chain(
                MMDLearning.MLP(ktheta => nθ, nhidden, hdim, σinner, tanh)...,
                MMDLearning.CatScale(θbd, ones(Int, nθ)),
            ) |> to32
        end
        get!(models, "latent_prior") do
            Flux.Chain(
                MMDLearning.MLP(klatent => k, nhidden, hdim, σinner, tanh)...,
                deepcopy(OutputScale),
            ) |> to32
        end
    end

    # Rician generator mapping Z variables from prior space to Rician parameter space
    get!(models, "genatr") do
        if k == 1
            return Flux.Chain(identity) # Latent space outputs noise level directly
        else
            error("nlatent = $k not implemented")
        end

        # #TODO: only works for LatentVectorRicianNoiseCorrector
        # @assert nin == k == nlatent(RiceGenType) && nout == n
        # Flux.Chain(
        #     # position encoding
        #     Z -> vcat(Z, zeros_similar(Z, 1, size(Z,2))),   # [k x b] -> [(k+1) x b]
        #     Z -> repeat(Z, n, 1),                           # [(k+1) x b] -> [(k+1)*n x b]
        #     MMDLearning.NotTrainable(Flux.Diagonal(ones((k+1)*n), vec(vcat(zeros(k, n), uniform_range(n)')))),
        #     Z -> reshape(Z, k+1, :),                        # [(k+1)*n x b] -> [(k+1) x n*b]
        #     # position-wise mlp
        #     MMDLearning.MLP(k+1 => 1, nhidden, hdim, σinner, tanh)..., # [(k+1) x n*b] -> [1 x n*b]
        #     # output scaling
        #     Z -> reshape(Z, n, :),                          # [1 x n*b] -> [n x b]
        #     OutputScale,
        # ) |> to32
    end

    # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
    get!(derived, "ricegen") do
        R = RiceGenType(models["genatr"])
        normalizer = X -> maximum(X; dims = 1) #TODO: normalize by mean? sum? maximum? first echo?
        noisescale = X -> mean(X; dims = 1) #TODO: relative to mean? nothing?
        NormalizedRicianCorrector(R, normalizer, noisescale)
    end

    # Deep prior for data distribution model and for cvae training distribution
    let
        deepθprior = get!(settings["train"], "DeepThetaPrior", false)::Bool
        deepZprior = get!(settings["train"], "DeepLatentPrior", false)::Bool
        ktheta = get!(settings["arch"]["genatr"], "ktheta", 0)::Int
        klatent = get!(settings["arch"]["genatr"], "klatent", 0)::Int
        prior_mix = get!(settings["arch"]["genatr"], "prior_mix", 0.0)::Float64
        default_θprior(x) = sampleθprior(phys, typeof(x), size(x,2))
        default_Zprior(x) = ((lo,hi) = eltype(x).(σbd); return lo .+ (hi .- lo) .* rand_similar(x, k, size(x,2)))
        # default_Zprior(x) = randn_similar(x, k, size(x,2))

        # Data distribution prior
        get!(derived, "prior") do
            DeepPriorRicianPhysicsModel{Float32,ktheta,klatent}(
                phys,
                derived["ricegen"],
                !deepθprior || ktheta == 0 ? default_θprior : models["theta_prior"],
                !deepZprior || klatent == 0 ? default_Zprior : models["latent_prior"],
            )
        end

        # CVAE distribution prior; mix (possibly deep) data distribution prior with a fraction `prior_mix` of samples from the default distribution
        mixed_θprior(x) = MMDLearning.sample_union(default_θprior, derived["prior"].θprior, prior_mix, x)
        mixed_Zprior(x) = MMDLearning.sample_union(default_Zprior, derived["prior"].Zprior, prior_mix, x)
        get!(derived, "cvae_prior") do
            DeepPriorRicianPhysicsModel{Float32,ktheta,klatent}(phys, derived["ricegen"], mixed_θprior, mixed_Zprior)
        end
    end

    # Encoders
    get!(models, "enc1") do
        @unpack hdim, nhidden, psize, head, chunksize, overlap = settings["arch"]["enc1"]
        TransformerEncoder(; nsignals = n, ntheta = 0, nlatent = 0, pout = 2*nz, psize, chunksize, overlap, head, hdim, nhidden) |> to32
        #=
        MMDLearning.MLP(n => 2*nz, nhidden, hdim, Flux.relu, identity) |> to32
        =#
    end

    get!(models, "enc2") do
        @unpack hdim, nhidden, psize, head, chunksize, overlap = settings["arch"]["enc2"]
        TransformerEncoder(; nsignals = n, ntheta = nθ, nlatent = k, pout = 2*nz, psize, chunksize, overlap, head, hdim, nhidden) |> to32
        #=
        Transformers.Stack(
            Transformers.@nntopo( (X,θ,Z) : (X,θ,Z) => XθZ : XθZ => μq ),
            vcat,
            MMDLearning.MLP(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity),
        ) |> to32
        =#
    end

    # Decoder
    get!(models, "dec") do
        @unpack hdim, nhidden, psize, head, chunksize, overlap = settings["arch"]["dec"]
        MLPHead = Flux.Chain(
            MMDLearning.MLP(psize => 2*(nθM + k), 0, hdim, Flux.relu, identity),
            MMDLearning.CatScale(eltype(θMbd)[θMbd; (-1, 1)], [ones(Int, nθM); k + nθM + k]),
        )
        TransformerEncoder(MLPHead; nsignals = n, ntheta = 0, nlatent = nz, pout = 0, psize, chunksize, overlap, head, hdim, nhidden) |> to32
        #=
        Transformers.Stack(
            Transformers.@nntopo( (Y,zr) : (Y,zr) => Yzr : Yzr => μx : μx => μx ),
            vcat,
            MMDLearning.MLP(n + nz => 2*(nθM + k), nhidden, hdim, Flux.relu, identity),
            MMDLearning.CatScale(eltype(θMbd)[θMbd; (-1, 1)], [ones(Int, nθM); k + nθM + k]),
        ) |> to32
        =#
    end

    # Discriminator
    get!(models, "discrim") do
        hdim = settings["arch"]["discrim"]["hdim"]::Int
        nhidden = settings["arch"]["discrim"]["nhidden"]::Int
        dropout = settings["arch"]["discrim"]["dropout"]::Float64
        chunk = settings["train"]["transform"]["chunk"]::Int
        order = get!(settings["train"]["augment"], "fdcat", 0)::Int #TODO
        augsizes = Dict{String,Int}(["signal" => n, "gradient" => n-1, "laplacian" => n-2, "encoderspace" => nz, "residuals" => n, "fftcat" => 2*(n÷2 + 1), "fftsplit" => 2*(n÷2 + 1), "fdcat" => sum(n-i for i in 0:order)])
        nin = sum((s -> ifelse(settings["train"]["augment"][s]::Union{Int,Bool} > 0, min(augsizes[s], chunk), 0)).(keys(augsizes))) #TODO > 0 hack works for both boolean and integer flags
        MMDLearning.MLP(nin => 1, nhidden, hdim, Flux.relu, Flux.sigmoid; dropout) |> to32
    end

    # CVAE
    get!(derived, "cvae") do; CVAE{n,nθ,nθM,k,nz}(models["enc1"], models["enc2"], models["dec"]) end

    # Misc. useful operators
    get!(derived, "forwarddiff") do; MMDLearning.ForwardDifferemce() |> to32 end
    get!(derived, "laplacian") do; MMDLearning.Laplacian() |> to32 end
    get!(derived, "fdcat") do
        order = get!(settings["train"]["augment"], "fdcat", 0)::Int #TODO
        A = I(n) |> Matrix{Float64}
        FD = LinearAlgebra.diagm(n-1, n, 0 => -ones(n-1), 1 => ones(n-1))
        A = mapfoldl(vcat, 1:order; init = A) do i
            A = @views FD[1:end-i+1, 1:end-i+1] * A
        end
        MMDLearning.NotTrainable(Flux.Dense(A, [0.0])) |> to32
    end
    get!(derived, "encoderspace") do # non-trainable sampling of encoder signal representations
        MMDLearning.NotTrainable(MMDLearning.flattenchain(Flux.Chain(
            models["enc1"],
            split_mean_softplus_std,
            sample_mv_normal,
        )))
    end

    return models, derived
end

function load_checkpoint()
    if haskey(ENV, "JL_CHECKPOINT_FOLDER")
        models_checkpoint = BSON.load(joinpath(ENV["JL_CHECKPOINT_FOLDER"], "current-models.bson"))["models"] |> deepcopy
        return map_dict(to32, models_checkpoint)
    else
        return Dict{String, Any}()
    end
end

function make_optimizer(otype = Flux.ADAM; lr = 0.0, gclip = 0.0, wdecay = 0.0)
    os = Any[otype(lr)]
    (gclip > 0) && pushfirst!(os, Flux.ClipValue(gclip))
    (wdecay > 0) && push!(os, Flux.WeightDecay(wdecay))
    Flux.Optimiser(os)
end

function make_optimizers(settings)
    os = Dict{String,Any}()
    for k in ["mmd", "cvae", "genatr", "discrim"]
        os[k] = make_optimizer(
            Flux.ADAM;
            lr = settings["opt"][k]["lrrel"] / settings["train"]["batchsize"],
            gclip = settings["opt"][k]["gclip"],
            wdecay = settings["opt"][k]["wdecay"],
        )
    end
    return os
end

####
#### WandB logger
####

function make_wandb_logger!(settings)
    wandb_logger = Ignite.init_wandb_logger(settings)
    !isnothing(wandb_logger) && (settings["data"]["out"] = wandb.run.dir)
    return wandb_logger
end

####
#### Snapshot
####

function save_snapshot!(settings, models)
    savepath = mkpath(settings["data"]["out"])
    settings_filename = joinpath(savepath, "settings.toml")
    summary_filename = joinpath(savepath, "model-summary.txt")
    for file in readdir(Glob.glob"*.jl", @__DIR__)
        cp(file, joinpath(savepath, basename(file)); force = true)
    end
    MMDLearning.model_summary(models, summary_filename)
    Ignite.save_and_print(settings; filename = settings_filename)
    return nothing
end

####
#### Reloading old results
####

#=
module Loader

# Avoid tracking changes by aliasing Revise.includet == include
const Revise = (includet = include,)

include(joinpath(ENV["JL_CHECKPOINT_FOLDER"], "setup.jl"))

# Manually "dispatch" on new type based on type params
function BSON.constructtype(::Type{T}, Ts) where {T <: Main.SignalEncoder}
    Ts = map(BSON.normalize_typeparams, Ts)
    if length(Ts) == 8
        Loader.SignalEncoder{Ts...}
    elseif length(Ts) == 4
        Main.SignalEncoder{Ts...}
    else
        error("Incorrect number of parameters for type $T: $Ts")
    end
end

# Reconstruct old type -> new type
function Flux.functor(e::Loader.SignalEncoder{p,cY,mY,nX,nY}) where {p,cY,mY,nX,nY}
    (e.DX, e.EY, e.E0), function((DX, EY, E0),)
        shape = (; psize = p, nshards = mY, nfeatures = nX, chunksize = cY, overlap = cY-1)
        Main.SignalEncoder(DX, EY, E0, shape)
    end
end

end # module Loader

import .Loader
=#

nothing
