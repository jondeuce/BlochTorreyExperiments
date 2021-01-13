####
#### Settings
####

function settings_template end

make_default_settings(args...) = parse_command_line!(settings_template(), args...)

# Parse command line arguments into default settings
function load_settings(args...)
    if isempty(checkpointdir())
        make_default_settings(args...)
    else
        TOML.parsefile(checkpointdir("settings.toml"))
    end
end

# Settings parsing
function parse_command_line!(
        settings::AbstractDict{<:AbstractString, Any},
        args = isinteractive() ? String[] : ARGS,
    )

    # Fields "INHERIT" with value "%PARENT%" specify that all fields from (and only from) the immediate parent
    # should be copied into the child, unless that key is already present in the child
    for (parent, (key, leaf)) in reverse(breadth_first_iterator(settings))
        (parent !== nothing && get(leaf, "INHERIT", "") == "%PARENT%") || continue
        for (k,v) in parent
            (v isa AbstractDict) && continue
            !haskey(leaf, k) && (leaf[k] = deepcopy(parent[k]))
        end
        delete!(leaf, "INHERIT")
    end

    # Fields with value "%PARENT%" take default values from the corresponding field of their parent
    for (parent, (key, leaf)) in breadth_first_iterator(settings)
        (parent === nothing) && continue
        for (k,v) in leaf
            (v == "%PARENT%") && (leaf[k] = deepcopy(parent[k]))
        end
    end

    # Generate arg parser
    function populate_arg_table!(parser, leaf_settings, root_settings = leaf_settings)
        for (k,v) in leaf_settings
            if v isa AbstractDict
                populate_arg_table!(parser, Dict{String,Any}(k * "." * kin => deepcopy(vin) for (kin, vin) in v), root_settings)
            else
                props = Dict{Symbol,Any}(:default => deepcopy(v))
                if v isa AbstractVector
                    props[:arg_type] = eltype(v)
                    props[:nargs] = length(v)
                else
                    props[:arg_type] = typeof(v)
                end
                ArgParse.add_arg_table!(parser, "--" * k, props)
            end
        end
        return parser
    end
    parser = populate_arg_table!(ArgParse.ArgParseSettings(), settings, settings)

    # Parse and merge into settings
    for (k,v) in ArgParse.parse_args(args, parser)
        ksplit = String.(split(k, "."))
        din = foldl(getindex, ksplit[begin:end-1]; init = settings)
        din[ksplit[end]] = deepcopy(v)
    end

    return settings
end

####
#### Logging
####

const use_wandb_logger = Ref(false)

const _logdirname = Ref("")
set_logdirname!(dirname) = !use_wandb_logger[] ? (_logdirname[] = basename(dirname)) : error("Cannot set log directory with WandB logger active; `wandb.run.dir` is used automatically")
set_logdirname!() = !use_wandb_logger[] ? basename(mkpath(projectdir("log", set_logdirname!(getnow())))) : error("Cannot set log directory with WandB logger active; `wandb.run.dir` is used automatically")
get_logdirname() = !use_wandb_logger[] ? (isempty(_logdirname[]) ? set_logdirname!() : _logdirname[]) : error("Use `logdir` to access WandB logger directory")
logdir(args...) = !use_wandb_logger[] ? projectdir("log", get_logdirname(), args...) : joinpath(wandb.run.dir, args...)

const _checkpointdir = Ref("")
set_checkpointdir!(dir) = _checkpointdir[] = dir
get_checkpointdir() = _checkpointdir[]
clear_checkpointdir!() = set_checkpointdir!("")
checkpointdir(args...) = isempty(get_checkpointdir()) ? "" : joinpath(get_checkpointdir(), args...)

####
#### Physics
####

function load_toyepgmodel_physics(; ntrain::Int, ntest::Int, nval::Int)
    initialize!(ToyEPGModel{Float32,true}(); ntrain, ntest, nval)
end

function load_epgmodel_physics(; max_numechos = 64, image_infos = nothing, seed = 0)
    if (image_infos === nothing)
        image_infos = [
            (TE = 8e-3, refcon = 180.0, path = DrWatson.datadir("Example_48echo_8msTE", "data-in", "ORIENTATION_B0_08_WIP_MWF_CPMG_CS_AXIAL_5_1.masked-image.mat")),
            (TE = 7e-3, refcon = 180.0, path = DrWatson.datadir("Example_56echo_7msTE_CPMG", "data-in", "MW_TRAINING_001_WIP_CPMG56_CS_half_2_1.masked-image.mat")),
        ]
    end
    phys = EPGModel{Float32,false}(n = max_numechos)
    initialize!(phys; image_infos, seed)
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
    init_μlogσ_bias = (sz...) -> (sz2 = (sz[1]÷2, sz[2:end]...); vcat(Flux.zeros(sz2), 10 .* Flux.ones(sz2))) # initialize logσ bias >> 0 s.t. initial cvae loss does not blowup, since loss has 𝒪(1/σ²) and 𝒪(logσ) terms
    init_μxlogσx_slope = (sz...) -> catscale_slope(eltype(θMbd)[θMbd; (-1,1); (9.5,10.5)], [ones(Int, nθM); k; nθM + k]) # [1] μθ[i] : (-1,1) -> θMbd[i], [2] μZ[i] : (-1,1) -> (-1,1), and [3] logσθ, logσZ : (-1,1) -> (9.5,10.5)
    init_μxlogσx_bias = (sz...) -> catscale_slope(eltype(θMbd)[θMbd; (-1,1); (9.5,10.5)], [ones(Int, nθM); k; nθM + k])

    #TODO: only works for Latent(*)Corrector family
    RiceGenType = LatentScalarRicianNoiseCorrector{n,k}
    # RiceGenType = LatentVectorRicianNoiseCorrector{n,k}
    # RiceGenType = LatentVectorRicianCorrector{n,k}
    # RiceGenType = VectorRicianCorrector{n,k}

    OutputScale = let
        RiceGenType <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? CatScale([(-δ, δ), σbd], [n,n]) :
        RiceGenType <: FixedNoiseVectorRicianCorrector ? CatScale([(-δ, δ)], [n]) :
        RiceGenType <: LatentVectorRicianNoiseCorrector ? CatScale([σbd], [n]) :
        RiceGenType <: LatentScalarRicianNoiseCorrector ? CatScale([σbd], [1]) :
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
                MLP(ktheta => nθ, nhidden, hdim, σinner, tanh)...,
                CatScale(θbd, ones(Int, nθ)),
            ) |> to32
        end
        get!(models, "latent_prior") do
            Flux.Chain(
                MLP(klatent => k, nhidden, hdim, σinner, tanh)...,
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
        #     NotTrainable(Flux.Diagonal(ones((k+1)*n), vec(vcat(zeros(k, n), uniform_range(n)')))),
        #     Z -> reshape(Z, k+1, :),                        # [(k+1)*n x b] -> [(k+1) x n*b]
        #     # position-wise mlp
        #     MLP(k+1 => 1, nhidden, hdim, σinner, tanh)..., # [(k+1) x n*b] -> [1 x n*b]
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
        mixed_θprior(x) = sample_union(default_θprior, derived["prior"].θprior, prior_mix, x)
        mixed_Zprior(x) = sample_union(default_Zprior, derived["prior"].Zprior, prior_mix, x)
        get!(derived, "cvae_prior") do
            DeepPriorRicianPhysicsModel{Float32,ktheta,klatent}(phys, derived["ricegen"], mixed_θprior, mixed_Zprior)
        end
    end

    # Encoders
    get!(models, "enc1") do
        @unpack hdim, nhidden, psize, head, hsize, nshards, chunksize, overlap = settings["arch"]["enc1"]
        #=
        mlp = MLP(psize => 2*nz, 0, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias) |> to32
        TransformerEncoder(mlp; nsignals = n, ntheta = 0, nlatent = 0, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
        =#
        mlp = MLP(n => 2*nz, nhidden, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias) |> to32
    end

    get!(models, "enc2") do
        @unpack hdim, nhidden, psize, head, hsize, nshards, chunksize, overlap = settings["arch"]["enc2"]
        #=
        mlp = MLP(psize => 2*nz, 0, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias) |> to32
        TransformerEncoder(mlp; nsignals = n, ntheta = nθ, nlatent = k, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
        =#
        mlp = MLP(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias) |> to32
        Transformers.Stack(Transformers.@nntopo((X,θ,Z) => XθZ => μq), vcat, mlp) |> to32
    end

    # Decoder
    get!(models, "dec") do
        @unpack hdim, nhidden, psize, head, hsize, nshards, chunksize, overlap = settings["arch"]["dec"]
        #=
        mlp = Flux.Chain(
            MLP(psize => 2*(nθM + k), 0, hdim, Flux.relu, identity),
            Flux.Diagonal(2*(nθM + k); initα = init_μxlogσx_slope, initβ = init_μxlogσx_bias),
        ) |> to32
        TransformerEncoder(mlp; nsignals = n, ntheta = 0, nlatent = nz, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
        =#
        mlp = Flux.Chain(
            MLP(n + nz => 2*(nθM + k), nhidden, hdim, Flux.relu, identity)...,
            Flux.Diagonal(2*(nθM + k); initα = init_μxlogσx_slope, initβ = init_μxlogσx_bias),
        ) |> to32
        Transformers.Stack(Transformers.@nntopo((Y,zr) => Yzr => μx), vcat, mlp) |> to32
    end

    # Variational decoder regularizer
    get!(models, "vae_dec") do
        @unpack hdim, nhidden = settings["arch"]["vae_dec"]
        Flux.Chain(
            MLP(nz => n, nhidden, hdim, Flux.relu, Flux.softplus)...,
            # X -> X ./ maximum(X; dims = 1), #TODO
        ) |> to32
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
        MLP(nin => 1, nhidden, hdim, Flux.relu, Flux.sigmoid; dropout) |> to32
    end

    # CVAE
    get!(derived, "cvae") do; CVAE{n,nθ,nθM,k,nz}(models["enc1"], models["enc2"], models["dec"]) end

    # Misc. useful operators
    get!(derived, "forwarddiff") do; ForwardDifferemce() |> to32 end
    get!(derived, "laplacian") do; Laplacian() |> to32 end
    get!(derived, "fdcat") do
        order = get!(settings["train"]["augment"], "fdcat", 0)::Int #TODO
        A = LinearAlgebra.I(n) |> Matrix{Float64}
        FD = LinearAlgebra.diagm(n-1, n, 0 => -ones(n-1), 1 => ones(n-1))
        A = mapfoldl(vcat, 1:order; init = A) do i
            A = @views FD[1:end-i+1, 1:end-i+1] * A
        end
        NotTrainable(Flux.Dense(A, [0.0])) |> to32
    end
    get!(derived, "encoderspace") do # non-trainable sampling of encoder signal representations
        NotTrainable(flattenchain(Flux.Chain(
            models["enc1"],
            split_mean_softplus_std,
            sample_mv_normal,
        )))
    end

    return models, derived
end

function load_checkpoint()
    if isempty(checkpointdir())
        Dict{String, Any}()
    else
        BSON.load(checkpointdir("current-models.bson"))["models"] |> deepcopy |> to32
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
    for name in ["mmd", "cvae", "genatr", "discrim"]
        os[name] = make_optimizer(
            Flux.ADAM;
            lr = initial_learning_rate!(settings, name),
            gclip = settings["opt"][name]["gclip"],
            wdecay = settings["opt"][name]["wdecay"],
        )
    end
    return os
end

function initial_learning_rate!(settings, optname)
    lr = get!(settings["opt"][optname], "lr", 0.0)
    lrrel = get!(settings["opt"][optname], "lrrel", 0.0)
    batchsize = settings["train"]["batchsize"]
    @assert (lr > 0) ⊻ (lrrel > 0)
    return lr > 0 ? lr : lrrel / batchsize
end

####
#### Snapshot
####

function save_snapshot!(settings, models)
    savepath = mkpath(logdir())
    settings_filename = joinpath(savepath, "settings.toml")
    summary_filename = joinpath(savepath, "model-summary.txt")
    for file in readdir(Glob.glob"*.jl", @__DIR__)
        cp(file, joinpath(savepath, basename(file)); force = true)
    end
    model_summary(models, summary_filename)
    savesettings(settings; filename = settings_filename, verbose = false)
    return nothing
end

nothing
