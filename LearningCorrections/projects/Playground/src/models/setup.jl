####
#### Settings
####

function settings_template end

# Parse command line arguments into default settings
function load_settings(args...; force_new_settings = false, override = nothing)
    settings = if isempty(checkpointdir()) || force_new_settings
        parse_args_from_template(settings_template(), args...)
    else
        TOML.parsefile(checkpointdir("settings.toml"))
    end
    (override !== nothing) && override_settings!(settings, override)
    return settings
end

# Generate arg parser
function generate_arg_parser!(parser, leaf_settings, root_settings = leaf_settings)
    for (k,v) in leaf_settings
        if v isa AbstractDict
            generate_arg_parser!(parser, Dict{String,Any}(k * "." * kin => deepcopy(vin) for (kin, vin) in v), root_settings)
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
generate_arg_parser(args...; kwargs...) = generate_arg_parser!(ArgParse.ArgParseSettings(), args...; kwargs...)

function clean_template!(template::AbstractDict{<:AbstractString})
    # Keys `"INHERIT"` with value `"%PARENT%"` specify that all fields from the immediate parent (i.e. non-recursive) should be copied into the child, unless that key is already present in the child
    for (parent, (key, leaf)) in reverse(breadth_first_iterator(template))
        (parent === nothing) && continue
        (get(leaf, "INHERIT", "") != "%PARENT%") && continue
        (parent !== nothing && get(leaf, "INHERIT", "") == "%PARENT%") || continue
        for (k,v) in parent
            (v isa AbstractDict) && continue
            !haskey(leaf, k) && (leaf[k] = deepcopy(parent[k]))
        end
        delete!(leaf, "INHERIT")
    end

    # Fields with value "%PARENT%" take default values from the corresponding field of their parent
    for (parent, (key, leaf)) in breadth_first_iterator(template)
        (parent === nothing) && continue
        for (k,v) in leaf
            (v == "%PARENT%") && (leaf[k] = deepcopy(parent[k]))
        end
    end

    return template
end
clean_template(template, args...; kwargs...) = clean_template!(deepcopy(template), args...; kwargs...)

function parse_args_into!(settings::AbstractDict{<:AbstractString}, args, parser; filter_args = false)
    # Parse and merge into settings
    for (k,v) in ArgParse.parse_args(args, parser)
        filter_args && !any(startswith("--" * k), args) && continue
        ks = String.(split(k, "."))
        d = foldl(getindex, ks[begin:end-1]; init = settings)
        @assert haskey(d, ks[end])
        d[ks[end]] = deepcopy(v)
    end
    return settings
end
parse_args_into(settings::AbstractDict{<:AbstractString}, args...; kwargs...) = parse_args_into!(deepcopy(settings), args...; kwargs...)

function override_settings!(settings, override)
    (override === nothing) && return settings
    for (parent, (key, leaf)) in breadth_first_iterator(settings), (k,v) in leaf
        haskey(override, k) && (leaf[k] = deepcopy(override[k]))
    end
    return settings
end

# Command line parsing
function parse_args_from_template(
        template::AbstractDict{<:AbstractString},
        args = isinteractive() ? String[] : ARGS;
        override = nothing
    )

    template_parser = generate_arg_parser(clean_template(template))
    template_updated = parse_args_into(template, args, template_parser; filter_args = true)
    template_updated = clean_template(template_updated)
    settings_parser = generate_arg_parser(template_updated)
    settings = parse_args_into(template_updated, args, settings_parser)
    (override !== nothing) && override_settings!(settings, override)

    return settings
end

function _parse_args_from_template_test()
    template = TOML.parse(
    """
    a = 0
    b = 0
    c = 0
    [B]
        INHERIT = "%PARENT%"
        a = 1
        [B.C]
            INHERIT = "%PARENT%"
            c = 2
        [B.D]
            INHERIT = "%PARENT%"
            b = "%PARENT%"
            c = 2
    """)

    check_keys(s, has, doesnt) = all(haskey.(Ref(s), has)) && !any(haskey.(Ref(s), doesnt))
    function check_keys(s)
        @assert check_keys(s, ["a", "b", "c"], ["INHERIT"])
        @assert check_keys(s["B"], ["a", "b", "c"], ["INHERIT"])
        @assert check_keys(s["B"]["C"], ["a", "c"], ["INHERIT", "b"])
        @assert check_keys(s["B"]["D"], ["a", "b", "c"], ["INHERIT"])
        return true
    end

    let s = parse_args_from_template(template, String[])
        @assert s == clean_template(template) # no args passed
        @assert check_keys(s)
        @assert s["a"] == 0 != s["B"]["a"] == s["B"]["C"]["a"] == s["B"]["D"]["a"] == 1
        @assert s["b"] == s["B"]["b"] == s["B"]["D"]["b"] == 0
    end

    let s = parse_args_from_template(template, ["--b=1", "--B.a=2", "--B.D.c=3"])
        @assert check_keys(s)
        @assert s["a"] == 0 != s["B"]["a"] == s["B"]["C"]["a"] == s["B"]["D"]["a"] == 2
        @assert s["b"] == s["B"]["b"] == s["B"]["D"]["b"] == 1
        @assert s["c"] == s["B"]["c"] == 0 != s["B"]["C"]["c"] == 2 != s["B"]["D"]["c"] == 3
    end

    return nothing
end

####
#### Logging
####

const use_wandb_logger = Ref(false)

const _logdirname = Ref("")
set_logdirname!(dirname) = !use_wandb_logger[] && (_logdirname[] = basename(dirname))
set_logdirname!() = !use_wandb_logger[] && basename(mkpath(projectdir("log", set_logdirname!(getnow()))))
get_logdirname() = !use_wandb_logger[] ? (isempty(_logdirname[]) ? set_logdirname!() : _logdirname[]) : basename(logdir())
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
    phys = EPGModel{Float32,false}(n = max_numechos)
    (image_infos === nothing) && (image_infos = [
        (TE = 8e-3, refcon = 180.0, path = DrWatson.datadir("Example_48echo_8msTE", "data-in", "ORIENTATION_B0_08_WIP_MWF_CPMG_CS_AXIAL_5_1.masked-image.mat")),
        (TE = 7e-3, refcon = 180.0, path = DrWatson.datadir("Example_56echo_7msTE_CPMG", "data-in", "MW_TRAINING_001_WIP_CPMG56_CS_half_2_1.masked-image.mat")),
    ])
    initialize!(phys; image_infos, seed)
end

####
#### Physics generator
####

function init_μlogσ_bias(::PhysicsModel{Float32}; kwargs...)
    function (sz...)
        sz2 = (sz[1]÷2, sz[2:end]...)
        vcat(zeros(Float32, sz2), 10 .* ones(Float32, sz2)) # initialize logσ bias >> 0 s.t. initial cvae loss does not blowup, since loss has 𝒪(1/σ²) and 𝒪(logσ) terms
    end
end

function init_rician_outputscale(phys::PhysicsModel{Float32}; nlatent, maxcorr, noisebounds, RiceGenType = LatentScalarRicianNoiseCorrector, kwargs...)
    n = nsignal(phys)
    δ = Float32(maxcorr)
    σbd = Float32.((noisebounds...,))
    OutputScale =
        RiceGenType{n,nlatent} <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? CatScale([(-1,1) => (-δ, δ), σbd], [n,n]) :
        RiceGenType{n,nlatent} <: FixedNoiseVectorRicianCorrector ? CatScale([(-1,1) => (-δ, δ)], [n]) :
        RiceGenType{n,nlatent} <: LatentVectorRicianNoiseCorrector ? CatScale([(-1,1) => σbd], [n]) :
        RiceGenType{n,nlatent} <: LatentScalarRicianNoiseCorrector ? CatScale([(-1,1) => σbd], [1]) :
        error("Unsupported corrector type: $RiceGenType")
end

# models["genatr"]
function init_isotropic_rician_generator(phys::PhysicsModel{Float32}; kwargs...)
    # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
    n = nsignal(phys)
    R = LatentScalarRicianNoiseCorrector{n,1}(Flux.Chain(identity)) # Latent space outputs noise level directly
    normalizer = ApplyOverDims(maximum; dims = 1) #TODO: normalize by mean? sum? maximum? first echo?
    noisescale = ApplyOverDims(mean; dims = 1) #TODO: relative to mean? nothing?
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
    normalizer = ApplyOverDims(maximum; dims = 1) #TODO: normalize by mean? sum? maximum? first echo?
    noisescale = ApplyOverDims(mean; dims = 1) #TODO: relative to mean? nothing?
    NormalizedRicianCorrector(R, normalizer, noisescale) # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
end

####
#### Priors
####

function init_default_theta_prior(phys::PhysicsModel{Float32}; kwargs...)
    default_θprior(A, _, n) = sampleθprior(phys, A, n)
    DeepPrior{Float32,ntheta(phys)}(identity, default_θprior)
end

function init_default_latent_prior(phys::PhysicsModel{Float32}; nlatent, noisebounds, kwargs...)
    lo, hi, k = Float32.(noisebounds)..., Int(nlatent)
    default_Zprior(A, _, n) = (T = eltype(A); T(lo) .+ (T(hi) .- T(lo)) .* rand_similar(A, k, n))
    DeepPrior{Float32,nlatent}(identity, default_Zprior)
end

# models["latent_prior"]
function init_deep_latent_prior(phys::PhysicsModel{Float32}; nlatent, klatent, maxcorr, noisebounds, nhidden, hdim, leakyslope, kwargs...)
    σinner = Float32(leakyslope) |> a -> a == 0 ? Flux.relu : (x -> Flux.leakyrelu(x, a))
    DeepPrior{Float32,klatent}(
        Flux.Chain(
            MLP(klatent => nlatent, nhidden, hdim, σinner, tanh)...,
            init_rician_outputscale(phys; nlatent, maxcorr, noisebounds, RiceGenType = LatentScalarRicianNoiseCorrector),
        ),
        randn_similar,
    ) |> to32
end

# models["theta_prior"]
function init_deep_theta_prior(phys::PhysicsModel{Float32}; ktheta, nhidden, hdim, leakyslope, kwargs...)
    σinner = Float32(leakyslope) |> a -> a == 0 ? Flux.relu : (x -> Flux.leakyrelu(x, a))
    DeepPrior{Float32,ktheta}(
        Flux.Chain(
            MLP(ktheta => ntheta(phys), nhidden, hdim, σinner, tanh)...,
            CatScale(Ref((-1,1)) .=> θbounds(phys), ones(Int, ntheta(phys))),
        ),
        randn_similar,
    ) |> to32
end

function derived_cvae_theta_prior(phys::PhysicsModel{Float32}, θprior; prior_mix, kwargs...)
    # CVAE distribution prior; mix (possibly deep) data distribution prior with a fraction `prior_mix` of samples from the default distribution
    DeepPrior{Float32,0}(DistributionUnion(init_default_theta_prior(phys), θprior; p = prior_mix), zeros_similar)
end

function derived_cvae_latent_prior(phys::PhysicsModel{Float32}, Zprior; nlatent, noisebounds, prior_mix, kwargs...)
    # CVAE distribution prior; mix (possibly deep) data distribution prior with a fraction `prior_mix` of samples from the default distribution
    DeepPrior{Float32,0}(DistributionUnion(init_default_latent_prior(phys; nlatent, noisebounds), Zprior; p = prior_mix), zeros_similar)
end

####
#### CVAE
####

# models["enc1"]
function init_mlp_cvae_enc1(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, kwargs...)
    MLP(nsignal(phys) => 2*zdim, nhidden, hdim, Flux.relu, identity) |> to32
end

# models["enc1"]
function init_xformer_cvae_enc1(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, psize, head, hsize, nshards, chunksize, overlap, kwargs...)
    mlp = MLP(psize => 2*zdim, 0, hdim, Flux.relu, identity) |> to32
    TransformerEncoder(mlp; nsignals = nsignal(phys), ntheta = 0, nlatent = 0, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
end

# models["enc2"]
function init_mlp_cvae_enc2(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, kwargs...)
    mlp = MLP(nsignal(phys) + ntheta(phys) + nlatent => 2*zdim, nhidden, hdim, Flux.relu, identity) |> to32
    Stack(@nntopo((X,θ,Z) => XθZ => μq), vcat, mlp) |> to32
end

# models["enc2"]
function init_xformer_cvae_enc2(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, psize, head, hsize, nshards, chunksize, overlap, kwargs...)
    mlp = MLP(psize => 2*zdim, 0, hdim, Flux.relu, identity) |> to32
    TransformerEncoder(mlp; nsignals = nsignal(phys), ntheta = ntheta(phys), nlatent, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
end

# models["dec"]
function init_mlp_cvae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, kwargs...)
    mlp = MLP(nsignal(phys) + zdim => 2*(nmarginalized(phys) + nlatent), nhidden, hdim, Flux.relu, identity) |> to32 #TODO initb_last = init_μlogσ_bias(phys)
    Stack(@nntopo((Y,zr) => Yzr => μx), vcat, mlp) |> to32
end

# models["dec"]
function init_xformer_cvae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, psize, head, hsize, nshards, chunksize, overlap, kwargs...)
    mlp = MLP(psize => 2*(nmarginalized(phys) + nlatent), 0, hdim, Flux.relu, identity) |> to32 #TODO initb_last = init_μlogσ_bias(phys)
    TransformerEncoder(mlp; nsignals = nsignal(phys), ntheta = 0, nlatent = zdim, psize, nshards, chunksize, overlap, head, hsize, hdim, nhidden) |> to32
end

# models["vae_dec"]
function init_mlp_cvae_vae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, regtype, kwargs...)
    # Output is either `nsignal` channel outputs directly for "L1", or `nsignal` mean/log-std pairs for "Rician", "Gaussian", etc.
    noutput = nsignal(phys) * (regtype == "L1" ? 1 : 2)
    MLP(zdim => noutput, nhidden, hdim, Flux.relu, identity) |> to32
end

# derived["cvae"]
function derived_cvae(phys::PhysicsModel{Float32}, enc1, enc2, dec; nlatent, zdim, posterior, kwargs...)
    # Flux.Diagonal(2*(nmarginalized(phys) + nlatent); initα = init_μxlogσx_slope(phys; nlatent), initβ = init_μxlogσx_bias(phys; nlatent))
    ϵ = 10 * eps(Float32)
    θbd = NTuple{2,Float32}.(θbounds(phys))
    θ̄bd = NTuple{2,Float32}.(fill((ϵ, 1-ϵ), ntheta(phys)))
    posterior_dist = posterior == "Kumaraswamy" ? Kumaraswamy : Gaussian
    CVAE{nsignal(phys),ntheta(phys),nmarginalized(phys),nlatent,zdim}(enc1, enc2, dec, θbd, θ̄bd; posterior_dist)
end

# derived["cvae"]
function load_pretrained_cvae(phys::PhysicsModel{Float32}; modelfolder, modelprefix = "best-")
    settings = TOML.parsefile(joinpath(modelfolder, "settings.toml"))
    models = FileIO.load(only(Glob.glob(modelprefix * "models.*", modelfolder)))["models"] |> deepcopy |> to32
    @unpack enc1, enc2, dec = models
    cvae = derived_cvae(phys, enc1, enc2, dec; make_kwargs(settings, "arch")...)
end

function pseudo_labels!(phys::EPGModel, cvae::CVAE; kwargs...)
    for img in phys.images
        pseudo_labels!(phys, cvae, img; kwargs...)
    end
    return phys
end

function pseudo_labels!(phys::EPGModel, cvae::CVAE, img::CPMGImage; nsamples = nothing, basesize = :)
    for (Yname, Y) in img.partitions
        get!(img.meta, :pseudolabels, Dict{Symbol,Any}())
        get!(img.meta[:pseudolabels], Yname, Dict{Symbol,Any}())
        Js = basesize === Colon() ? [Colon()] : Iterators.partition(1:size(Y,2), basesize)
        θs, Zs = map(Js) do J
            YJ = Y[:,J] |> todevice
            if nsamples === nothing
                # Use posterior mode, no explicit sampling
                μθ, μZ = sampleθZ(phys, cvae, MetaCPMGSignal(phys, img, YJ); posterior_mode = true) .|> Flux.cpu
            else
                # Average over `nsamples` posterior draws
                @assert nsamples::Int >= 1
                θZ_sampler_instance = θZ_sampler(phys, cvae, MetaCPMGSignal(phys, img, YJ))
                μθ, μZ = θZ_sampler_instance()
                for _ in 2:nsamples
                    θ, Z = θZ_sampler_instance()
                    μθ .+= θ
                    μZ .+= Z
                end
                μθ ./= nsamples
                μZ ./= nsamples
                return (μθ, μZ) .|> Flux.cpu
            end
        end |> unzip
        img.meta[:pseudolabels][Yname][:theta] = reduce(hcat, θs)
        img.meta[:pseudolabels][Yname][:latent] = reduce(hcat, Zs)
    end
    return img
end

####
#### GANs
####

# models["discrim"]
function init_mlp_discrim(phys::PhysicsModel{Float32}; ninput = nsignal(phys), hdim, nhidden, dropout, kwargs...)
    MLP(ninput => 1, nhidden, hdim, Flux.relu, identity; dropout) |> to32
end

# derived["kernel_$key"]
function init_mmd_kernels(phys::PhysicsModel{Float32}; bwsizes, bwbounds, nbandwidth, channelwise, embeddingdim, hdim, kwargs...)
    map(bwsizes) do nchannel
        # Initialize kernel bandwidths
        logσ = range((embeddingdim <= 0 ? bwbounds : (-5.0, 5.0))...; length = nbandwidth + 2)[2:end-1]
        logσ = repeat(logσ, 1, (channelwise ? nchannel : 1))

        # Optionally embed `nchannel` input into `embeddingdim`-dimensional learned embedding space
        embedding = embeddingdim <= 0 ? identity : Flux.Chain(
            MLP(nchannel => embeddingdim, 0, hdim, Flux.relu, identity)...,
            ApplyOverDims(Flux.normalise; dims = 1), # kernel bandwidths are sensitive to scale; normalize learned representations
            # z -> z .+ 0.1f0 .* randn_similar(z, size(z)...), # stochastic embedding prevents overfitting to Y data
        )

        # MMD kernel wrapper
        DeepExponentialKernel(logσ, embedding) |> to32
    end
end

function load_checkpoint(filename)
    isempty(checkpointdir()) && return Dict{String, Any}()
    try
        FileIO.load(checkpointdir(filename))["models"] |> deepcopy |> to32
    catch e
        @warn "Error loading checkpoint from $(checkpointdir())"
        @warn sprint(showerror, e, catch_backtrace())
        return Dict{String, Any}()
    end
end

function init_optimizer(otype = Flux.ADAM; lr = 0.0, gclip = 0.0, wdecay = 0.0, kwargs...)
    os = Any[otype(lr)]
    (gclip > 0) && pushfirst!(os, Flux.ClipValue(gclip))
    (wdecay > 0) && push!(os, Flux.WeightDecay(wdecay))
    Flux.Optimiser(os)
end

####
#### Snapshot
####

function save_snapshot(settings, models; savepath = nothing, savedirs = ["src", "test", "scripts"])
    # Save simulation settings and summary of model
    savepath = (savepath === nothing) ? mkpath(logdir()) : mkpath(savepath)
    save_settings(settings; filename = joinpath(savepath, "settings.toml"), verbose = false)
    model_summary(models; filename = joinpath(savepath, "model-summary.txt"), verbose = false)
    save_project_code(joinpath(savepath, "project"))
    return nothing
end

function save_project_code(
        savepath;
        saveitems = ["src", "test", "scripts", "Project.toml", "Manifest.toml"],
        newuuid = true,
    )
    # Save project code
    mkpath(savepath)
    for path in projectdir.(saveitems)
        cp(path, joinpath(savepath, basename(path)))
    end
    if newuuid
        replace_projectfile_uuid(joinpath(savepath, "Project.toml"))
    end
end

function replace_projectfile_uuid(projectfile)
    prj = TOML.parsefile(projectfile)
    prj["deps"] = sort(prj["deps"]) # sort dependency list for consistency with Pkg
    prj["uuid"] = string(UUIDs.uuid4()) # generate new uuid
    open(projectfile; write = true) do io
        TOML.print(io, prj)
    end
    return prj
end

nothing
