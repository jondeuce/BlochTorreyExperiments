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

init_μlogσ_bias(phys::PhysicsModel{Float32}; kwargs...) = (sz...) -> (sz2 = (sz[1]÷2, sz[2:end]...); vcat(zeros(Float32, sz2), 10 .* ones(Float32, sz2))) # initialize logσ bias >> 0 s.t. initial cvae loss does not blowup, since loss has 𝒪(1/σ²) and 𝒪(logσ) terms
init_μxlogσx_slope(phys::PhysicsModel{Float32}; nlatent, kwargs...) = (sz...) -> (nθM = nmarginalized(phys); θMbd = θmarginalized(phys, θbounds(phys)); @assert(length(sz) == 1 && sz[1] == 2*(nθM + nlatent)); catscale_slope(NTuple{2,Float32}[θMbd; (-1,1); (9.5,10.5)], [ones(Int, nθM); nlatent; nθM + nlatent])) # [1] μθ[i] : (-1,1) -> θMbd[i], [2] μZ[i] : (-1,1) -> (-1,1), and [3] logσθ, logσZ : (-1,1) -> (9.5,10.5)
init_μxlogσx_bias(phys::PhysicsModel{Float32}; nlatent, kwargs...) = (sz...) -> (nθM = nmarginalized(phys); θMbd = θmarginalized(phys, θbounds(phys)); @assert(length(sz) == 1 && sz[1] == 2*(nθM + nlatent)); catscale_slope(NTuple{2,Float32}[θMbd; (-1,1); (9.5,10.5)], [ones(Int, nθM); nlatent; nθM + nlatent]))

function init_rician_outputscale(phys::PhysicsModel{Float32}; nlatent, maxcorr, noisebounds, RiceGenType = LatentScalarRicianNoiseCorrector, kwargs...)
    n = nsignal(phys)
    δ = Float32(maxcorr)
    σbd = Float32.((noisebounds...,))
    OutputScale =
        RiceGenType{n,nlatent} <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? CatScale([(-δ, δ), σbd], [n,n]) :
        RiceGenType{n,nlatent} <: FixedNoiseVectorRicianCorrector ? CatScale([(-δ, δ)], [n]) :
        RiceGenType{n,nlatent} <: LatentVectorRicianNoiseCorrector ? CatScale([σbd], [n]) :
        RiceGenType{n,nlatent} <: LatentScalarRicianNoiseCorrector ? CatScale([σbd], [1]) :
        error("Unsupported corrector type: $RiceGenType")
end

# models["genatr"]
function init_isotropic_rician_generator(phys::PhysicsModel{Float32}; kwargs...)
    # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
    n = nsignal(phys)
    R = LatentScalarRicianNoiseCorrector{n,1}(Flux.Chain(identity)) # Latent space outputs noise level directly
    normalizer = X -> maximum(X; dims = 1) #TODO: normalize by mean? sum? maximum? first echo?
    noisescale = X -> mean(X; dims = 1) #TODO: relative to mean? nothing?
    NormalizedRicianCorrector(R, normalizer, noisescale) # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
end

# models["genatr"]
function init_vector_rician_generator(phys::PhysicsModel{Float32}; nlatent, maxcorr, noisebounds, nhidden, hdim, kwargs...)
    n = nsignal(phys)
    k = nlatent
    RiceGenType = LatentVectorRicianNoiseCorrector
    OutputScale = init_rician_outputscale(phys; nlatent, maxcorr, noisebounds, RiceGenType)
    R = RiceGenType{n,k}(
        Flux.Chain(
            # position encoding
            Z -> vcat(Z, zeros_similar(Z, 1, size(Z,2))),   # [k x b] -> [(k+1) x b]
            Z -> repeat(Z, n, 1),                           # [(k+1) x b] -> [(k+1)*n x b]
            NotTrainable(Flux.Diagonal(ones((k+1)*n), vec(vcat(zeros(k, n), uniform_range(n)')))),
            Z -> reshape(Z, k+1, :),                        # [(k+1)*n x b] -> [(k+1) x n*b]
            # position-wise mlp
            MLP(k+1 => 1, nhidden, hdim, Flux.relu, tanh)..., # [(k+1) x n*b] -> [1 x n*b]
            # output scaling
            Z -> reshape(Z, n, :),                          # [1 x n*b] -> [n x b]
            OutputScale,
        ) |> to32
    )
    # Rician generator mapping Z variables from prior space to Rician parameter space
    normalizer = X -> maximum(X; dims = 1) #TODO: normalize by mean? sum? maximum? first echo?
    noisescale = X -> mean(X; dims = 1) #TODO: relative to mean? nothing?
    NormalizedRicianCorrector(R, normalizer, noisescale) # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
end

function init_default_theta_prior(phys::PhysicsModel{Float32}; kwargs...)
    default_θprior(z) = sampleθprior(phys, typeof(z), size(z,2))
end

function init_default_latent_prior(phys::PhysicsModel{Float32}; nlatent, noisebounds, kwargs...)
    lo, hi, k = Float32.(noisebounds)..., Int(nlatent)
    default_Zprior(z) = (T = eltype(z); T(lo) .+ (T(hi) .- T(lo)) .* rand_similar(z, k, size(z,2)))
end

# models["latent_prior"]
function init_deep_latent_prior(phys::PhysicsModel{Float32}; nlatent, klatent, maxcorr, noisebounds, nhidden, hdim, leakyslope, kwargs...)
    σinner = Float32(leakyslope) |> a -> a == 0 ? Flux.relu : (x -> Flux.leakyrelu(x, a))
    Flux.Chain(
        MLP(klatent => nlatent, nhidden, hdim, σinner, tanh)...,
        init_rician_outputscale(phys; nlatent, maxcorr, noisebounds, RiceGenType = LatentScalarRicianNoiseCorrector),
    ) |> to32
end

# models["theta_prior"]
function init_deep_theta_prior(phys::PhysicsModel{Float32}; nlatent, ktheta, maxcorr, noisebounds, nhidden, hdim, leakyslope, kwargs...)
    σinner = Float32(leakyslope) |> a -> a == 0 ? Flux.relu : (x -> Flux.leakyrelu(x, a))
    Flux.Chain(
        MLP(ktheta => ntheta(phys), nhidden, hdim, σinner, tanh)...,
        CatScale(θbounds(phys), ones(Int, ntheta(phys))),
    ) |> to32
end

# models["enc1"]
function init_mlp_cvae_enc1(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, kwargs...)
    MLP(nsignal(phys) => 2*zdim, nhidden, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias(phys)) |> to32
end

# models["enc1"]
function init_xformer_cvae_enc1(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, psize, head, hsize, nshards, chunksize, overlap, kwargs...)
    mlp = MLP(psize => 2*zdim, 0, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias(phys)) |> to32
    TransformerEncoder(mlp; nsignals = nsignal(phys), ntheta = 0, nlatent = 0, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
end

# models["enc2"]
function init_mlp_cvae_enc2(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, kwargs...)
    mlp = MLP(nsignal(phys) + ntheta(phys) + nlatent => 2*zdim, nhidden, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias(phys)) |> to32
    Stack(@nntopo((X,θ,Z) => XθZ => μq), vcat, mlp) |> to32
end

# models["enc2"]
function init_xformer_cvae_enc2(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, psize, head, hsize, nshards, chunksize, overlap, kwargs...)
    mlp = MLP(psize => 2*zdim, 0, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias(phys)) |> to32
    TransformerEncoder(mlp; nsignals = nsignal(phys), ntheta = ntheta(phys), nlatent, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
end

# models["dec"]
function init_mlp_cvae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, kwargs...)
    mlp = Flux.Chain(
        MLP(nsignal(phys) + zdim => 2*(nmarginalized(phys) + nlatent), nhidden, hdim, Flux.relu, identity)...,
        Flux.Diagonal(2*(nmarginalized(phys) + nlatent); initα = init_μxlogσx_slope(phys; nlatent), initβ = init_μxlogσx_bias(phys; nlatent)),
    ) |> to32
    Stack(@nntopo((Y,zr) => Yzr => μx), vcat, mlp) |> to32
end

# models["dec"]
function init_xformer_cvae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, psize, head, hsize, nshards, chunksize, overlap, kwargs...)
    mlp = Flux.Chain(
        MLP(psize => 2*(nmarginalized(phys) + nlatent), 0, hdim, Flux.relu, identity),
        Flux.Diagonal(2*(nmarginalized(phys) + nlatent); initα = init_μxlogσx_slope(phys; nlatent), initβ = init_μxlogσx_bias(phys; nlatent)),
    ) |> to32
    TransformerEncoder(mlp; nsignals = nsignal(phys), ntheta = 0, nlatent = zdim, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
end

# models["vae_dec"]
function init_mlp_cvae_vae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, kwargs...)
    MLP(nsignal(phys) => 2*zdim, nhidden, hdim, Flux.relu, identity; initb_last = init_μlogσ_bias(phys)) |> to32
    Flux.Chain(
        MLP(zdim => nsignal(phys), nhidden, hdim, Flux.relu, Flux.softplus)...,
        # X -> X ./ maximum(X; dims = 1), #TODO
    ) |> to32
end

# models["discrim"]
function init_mlp_discrim(phys::PhysicsModel{Float32}; ninput = nsignal(phys), hdim, nhidden, dropout, kwargs...)
    MLP(ninput => 1, nhidden, hdim, Flux.relu, identity; dropout) |> to32
end

# derived["genatr_prior"] -> derived["genatr_prior"]
function derived_genatr_prior(phys::PhysicsModel{Float32}, ricegen, θprior, Zprior; ktheta, klatent, kwargs...)
    DeepPriorRicianPhysicsModel{Float32,ktheta,klatent}(phys, ricegen, θprior, Zprior)
end

# derived["cvae_prior"]
function derived_cvae_prior(phys::PhysicsModel{Float32}, ricegen, θprior, Zprior; ktheta, klatent, nlatent, noisebounds, prior_mix, kwargs...)
    # CVAE distribution prior; mix (possibly deep) data distribution prior with a fraction `prior_mix` of samples from the default distribution
    mixed_θprior = DistributionUnion(init_default_theta_prior(phys), θprior; p = prior_mix)
    mixed_Zprior = DistributionUnion(init_default_latent_prior(phys; nlatent, noisebounds), Zprior; p = prior_mix)
    DeepPriorRicianPhysicsModel{Float32,ktheta,klatent}(phys, ricegen, mixed_θprior, mixed_Zprior)
end

# derived["cvae"]
function derived_cvae(phys::PhysicsModel{Float32}, enc1, enc2, dec; nlatent, zdim, kwargs...)
    CVAE{nsignal(phys),ntheta(phys),nmarginalized(phys),nlatent,zdim}(enc1, enc2, dec)
end

# Initialize generator + discriminator + kernel
function make_models!(phys::PhysicsModel{Float32}, settings::Dict{String,Any}, models = Dict{String, Any}(), derived = Dict{String, Any}())

    # Misc. useful operators
    get!(derived, "forwarddiff") do; ForwardDifference() |> to32; end
    get!(derived, "laplacian") do; Laplacian() |> to32; end
    get!(derived, "fdcat") do; CatDenseFiniteDiff(nsignal(phys), order) |> to32; end
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
