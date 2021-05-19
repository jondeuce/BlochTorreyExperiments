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
                props[:nargs] = '*'
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
const train_debug = Ref(false)

const _logdirname = Ref("")
set_logdirname!(dirname) = !use_wandb_logger[] && (_logdirname[] = basename(dirname))
set_logdirname!() = !use_wandb_logger[] && basename(mkpath(DrWatson.projectdir("log", set_logdirname!(getnow()))))
get_logdirname() = !use_wandb_logger[] ? (isempty(_logdirname[]) ? set_logdirname!() : _logdirname[]) : basename(logdir())
logdir(args...) = !use_wandb_logger[] ? DrWatson.projectdir("log", get_logdirname(), args...) : joinpath(wandb.run.dir, args...)

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

function load_cpmg_info(folder_path::AbstractString)
    folder_path = DrWatson.datadir("images", folder_path)
    info = TOML.parsefile(joinpath(folder_path, "image_info.toml"))
    info["folder_path"] = folder_path
    return info
end

function load_epgmodel_physics(; max_numechos = 64)
    phys = EPGModel{Float32,false}(n = max_numechos)
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
    normalizer = ApplyOverDims(maximum; dims = 1)
    noisescale = nothing
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
        ) |> gpu
    )
    # Rician generator mapping Z variables from prior space to Rician parameter space
    normalizer = ApplyOverDims(maximum; dims = 1)
    noisescale = nothing
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
    ) |> gpu
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
    ) |> gpu
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
    σact = Flux.relu # Flux.leakyrelu # Flux.swish
    mlp = Flux.Chain(MLP(nsignal(phys) => 2*zdim, nhidden, hdim, σact, identity), CVAELatentTransform(zdim)) |> flattenchain |> gpu
end

# models["enc1"]
function init_xformer_cvae_enc1(phys::PhysicsModel{Float32}; hdim, zdim, nlatent, nhidden, esize, nheads, headsize, seqlength, qseqlength, share, kwargs...)
    TransformerEncoder(CVAELatentTransform(zdim); esize, nheads, headsize, hdim, seqlength, qseqlength, share, nhidden, insizes = (nsignal(phys),), outsize = 2*zdim) |> gpu
end

# models["enc2"]
function init_mlp_cvae_enc2(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, kwargs...)
    σact = Flux.relu # Flux.leakyrelu # Flux.swish
    mlp = Flux.Chain(MLP(nsignal(phys) + ntheta(phys) + nlatent => 2*zdim, nhidden, hdim, σact, identity), CVAELatentTransform(zdim)) |> flattenchain |> gpu
    Stack(@nntopo((X,θ,Z) => XθZ => μq), vcat, mlp) |> gpu
end

# models["enc2"]
function init_xformer_cvae_enc2(phys::PhysicsModel{Float32}; hdim, zdim, nlatent, nhidden, esize, nheads, headsize, seqlength, qseqlength, share, kwargs...)
    TransformerEncoder(CVAELatentTransform(zdim); esize, nheads, headsize, hdim, seqlength, qseqlength, share, nhidden, insizes = (nsignal(phys), ntheta(phys), nlatent), outsize = 2*zdim) |> gpu
end

# models["dec"]
function init_mlp_cvae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, kwargs...)
    σact = Flux.relu # Flux.leakyrelu # Flux.swish
    mlp = MLP(nsignal(phys) + zdim => 2*(nmarginalized(phys) + nlatent), nhidden, hdim, σact, identity) |> gpu #TODO initb_last = init_μlogσ_bias(phys)
    Stack(@nntopo((Y,zr) => Yzr => μx), vcat, mlp) |> gpu
end

# models["dec"]
function init_xformer_cvae_dec(phys::PhysicsModel{Float32}; hdim, zdim, nlatent, nhidden, esize, nheads, headsize, seqlength, qseqlength, share, kwargs...)
    TransformerEncoder(; esize, nheads, headsize, hdim, seqlength, qseqlength, share, nhidden, insizes = (nsignal(phys), zdim), outsize = 2*(nmarginalized(phys) + nlatent)) |> gpu
end

# models["vae_dec"]
function init_mlp_cvae_vae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, regtype, kwargs...)
    # Output is either `nsignal` channel outputs directly for "L1", or `nsignal` mean/log-std pairs for "Rician", "Gaussian", etc.
    noutput = nsignal(phys) * (regtype == "L1" ? 1 : 2)
    σact = Flux.relu # Flux.leakyrelu # Flux.swish
    MLP(zdim => noutput, nhidden, hdim, σact, identity) |> gpu
end

# derived["cvae"]
function derived_cvae(phys::PhysicsModel{Float32}, enc1, enc2, dec; nlatent, zdim, posterior, kwargs...)
    sim(x,y) = Zygote.@ignore arr_similar(x, y)
    clamp_scale(x,α,β,lo,hi) = clamp.(sim(x,α) .* x .+ sim(x,β), sim(x,lo), sim(x,hi))

    nθ = ntheta(phys)
    θlo = θlower(phys) .|> Float32
    θhi = θupper(phys) .|> Float32
    θ̄lo = -ones(Float32, nθ)
    θ̄hi = +ones(Float32, nθ)
    α, β = catscale_slope_and_bias([(θlo[i], θhi[i]) => (θ̄lo[i], θ̄hi[i]) for i in 1:nθ], ones(Int, nθ))
    ᾱ, β̄ = catscale_slope_and_bias([(θ̄lo[i], θ̄hi[i]) => (θlo[i], θhi[i]) for i in 1:nθ], ones(Int, nθ))

    function normalize_inputs(in::Tuple)
        Y     = in[1]
        Ym    = maximum(Y; dims = 1)
        logYm = log.(Ym)
        Ȳ     = Y ./ Ym; 
        if length(in) == 1
            return (Ȳ, logYm)
        else
            θ, Z = in[2], in[3]
            θ̄ = clamp_scale(vcat(θ[1:6,..], θ[7:7,:] .- logYm), α, β, θ̄lo, θ̄hi)
            Z̄ = Z
            return (Ȳ, θ̄, Z̄, logYm)
        end
    end

    function unnormalize_outputs(in::Tuple)
        θ̄, Z̄, logYm = in
        θ = clamp_scale(vcat(θ̄[1:6,..], θ̄[7:7,:] .+ logYm), ᾱ, β̄, θlo, θhi)
        Z = Z̄
        return (θ, Z)
    end

    posterior_dist =
        posterior == "Kumaraswamy" ? Kumaraswamy :
        posterior == "TruncatedGaussian" ? TruncatedGaussian :
        Gaussian
    CVAE{nsignal(phys),ntheta(phys),nmarginalized(phys),nlatent,zdim}(enc1, enc2, dec, normalize_inputs, unnormalize_outputs; posterior_dist)
end

# derived["cvae"]
function load_pretrained_cvae(phys::PhysicsModel{Float32}; modelfolder, modelprefix = "best-")
    settings = TOML.parsefile(joinpath(modelfolder, "settings.toml"))
    models = load_model(only(Glob.glob(modelprefix * "models.*", modelfolder)), "models") |> deepcopy |> gpu
    @unpack enc1, enc2, dec = models
    cvae = derived_cvae(phys, enc1, enc2, dec; make_kwargs(settings, "arch")...)
end

function initialize_pseudo_labels!(
        phys::EPGModel{T}, cvae::Union{Nothing,<:CVAE} = nothing;
        labelset = :prior,
        npseudolabels = 100,
        force_recompute = true,
    ) where {T}

    for (i, img) in enumerate(phys.images)
        # Optionally skip cecomputing
        haskey(img.meta, :pseudo_labels) && !force_recompute && continue
        img.meta[:pseudo_labels] = Dict{Symbol,Any}()

        for dataset in [:train, :val, :test]
            if Symbol(labelset) === :mle
                @unpack theta, signalfit = img.meta[:mle_labels][dataset]
                theta = repeat(theta, 1, 1, npseudolabels)
            elseif Symbol(labelset) === :mcmc
                @unpack theta, signalfit = img.meta[:mcmc_labels][dataset]
                theta = repeat(theta[:,:,end], 1, 1, npseudolabels) # note: this is only initializing the sampler buffer, which only tracks the latest theta; we start from the last mcmc sample
            elseif Symbol(labelset) === :cvae
                @assert cvae !== nothing
                initial_guess = mle_biexp_epg_initial_guess(phys, img, cvae; data_subset = dataset, gpu_batch_size = 100_000)
                theta, signalfit = initial_guess.θ, initial_guess.X
                theta = repeat(theta, 1, 1, npseudolabels)
            elseif Symbol(labelset) === :prior
                initial_guess = mle_biexp_epg_initial_guess(phys, img, nothing; data_subset = dataset, gpu_batch_size = 100_000)
                theta, signalfit = initial_guess.θ, initial_guess.X
                theta = repeat(theta, 1, 1, npseudolabels)
            else
                error("Unknown labelset: $labelset")
            end

            # Assign outputs
            labels                    = img.meta[:pseudo_labels][dataset] = Dict{Symbol,Any}()
            Ymeta                     = MetaCPMGSignal(phys, img, img.partitions[dataset])
            neglogPXθ                 = repeat(negloglikelihood(phys, Ymeta, theta[:,:,1]) .|> T, 1, 1, npseudolabels)
            neglogPθ                  = repeat(neglogprior(phys, theta[:,:,1]) .|> T, 1, 1, npseudolabels)
            labels[:theta]            = theta .|> T
            labels[:signalfit]        = signalfit .|> T
            labels[:mh_sampler]       = OnlineMetropolisSampler{T}(;
                θ         = theta,         # pseudo mcmc samples
                neglogPXθ = neglogPXθ,     # initial negative log likelihoods
                neglogPθ  = neglogPθ,      # initial negative log priors
            )
        end
    end
end

function compute_mle_labels!(phys::EPGModel, cvae::Union{Nothing,<:CVAE} = nothing; kwargs...)
    for img in phys.images
        compute_mle_labels!(phys, img, cvae; kwargs...)
    end
    return phys
end

function compute_mle_labels!(
        phys::EPGModel{T}, img::CPMGImage{T}, cvae::Union{Nothing,<:CVAE} = nothing;
        sigma_reg = 0.5,
        force_recompute = true,
    ) where {T}

    # Optionally skip cecomputing
    haskey(img.meta, :mle_labels) && !force_recompute && return img

    # Perform MLE fit on all signals within mask
    @info img
    _, results = mle_biexp_epg(
        phys, img, cvae;
        batch_size = 2048 * Threads.nthreads(),
        verbose    = true,
        sigma_reg,
        initial_guess_args = (
            refine_init_logϵ = true,
            refine_init_logs = true,
            verbose          = false,
            data_subset      = :mask,
            gpu_batch_size   = 100_000,
        ),
    )

    # Copy results from within mask into relevant mask/train/val/test partitions
    all_labels              = img.meta[:mle_labels] = Dict{Symbol,Any}()
    mask_labels             = img.meta[:mle_labels][:mask] = Dict{Symbol,Any}()
    mask_labels[:signalfit] = results.signalfit |> cpu32
    mask_labels[:theta]     = results.theta |> cpu32

    for (dataset, _) in img.partitions
        dataset === :mask && continue
        indices             = findall_within(img.indices[:mask], img.indices[dataset])
        labels              = all_labels[dataset] = Dict{Symbol,Any}()
        labels[:theta]      = mask_labels[:theta][:,indices] .|> T
        labels[:signalfit]  = mask_labels[:signalfit][:,indices] .|> T

        # Errors w.r.t. true labels
        if haskey(img.meta, :true_labels)
            theta_true          = img.meta[:true_labels][dataset][:theta]
            theta_mle           = labels[:theta]
            labels[:theta_errs] = θ_errs_dict(phys, theta_true, theta_mle; suffix = "MLE")
        end
    end

    return img
end

function load_mcmc_labels!(
        phys::EPGModel{T};
        force_reload = true,
    ) where {T}

    for (i, img) in enumerate(phys.images)
        # Optionally skip reloading
        haskey(img.meta, :mcmc_labels) && !force_reload && continue
        img.meta[:mcmc_labels] = Dict{Symbol,Any}()

        # Load MCMC params
        labels_file = joinpath(img.meta[:info]["folder_path"], img.meta[:info]["mcmc_labels_path"])
        if !isfile(labels_file)
            @info "MCMC data does not exist (image = $i): $(img.meta[:info]["folder_path"])"
            continue
        else
            @info labels_file
            @info "Loading MCMC data (image = $i):"
        end
        @time labels_data = DECAES.MAT.matread(labels_file)

        mcmc_param_names  = ["alpha", "beta", "eta", "delta1", "delta2", "logepsilon", "logscale"]
        total_samples     = length(labels_data["iteration"])
        samples_per_chain = maximum(labels_data["iteration"])
        num_signals       = total_samples ÷ samples_per_chain
        x_index           = labels_data["image_x"][1 : samples_per_chain : end]
        y_index           = labels_data["image_y"][1 : samples_per_chain : end]
        z_index           = labels_data["image_z"][1 : samples_per_chain : end]
        labels_map        = Dict(CartesianIndex.(x_index, y_index, z_index) .=> 1:num_signals)

        for dataset in [:train, :val, :test]
            # Fetch theta for each partition
            theta = mapreduce(vcat, enumerate(mcmc_param_names)) do (i, param_name)
                indices   = (I -> labels_map[I]).(img.indices[dataset])
                θ         = reshape(labels_data[param_name], samples_per_chain, :)[:, indices] .|> T
                θ         = permutedims(reshape(θ, samples_per_chain, :, 1), (3,2,1))
            end

            # Compute epg signal model
            @time X = signal_model(phys, img, theta[:,:,end])

            # Assign outputs
            labels             = img.meta[:mcmc_labels][dataset] = Dict{Symbol,Any}()
            labels[:theta]     = theta # θ = α, β, η, δ1, δ2, logϵ, logs
            labels[:signalfit] = X .|> T

            # Errors w.r.t. true labels
            @time if haskey(img.meta, :true_labels)
                theta_true          = img.meta[:true_labels][dataset][:theta]
                theta_mcmc          = dropdims(mean(theta; dims = 3); dims = 3)
                theta_mcmc          = clamp.(theta_mcmc, θlower(phys), θupper(phys)) # `mean` can cause `theta_mcmc` to overflow outside of bounds
                mle_init            = (; Y = lib.cpu64(img.partitions[dataset]), θ = lib.cpu64(theta_mcmc))
                _, mle_res          = lib.mle_biexp_epg(phys, img; initial_guess = mle_init, batch_size = Colon(), verbose = true)
                theta_mle           = mle_res.theta
                labels[:theta_errs] = Dict{Symbol,Any}()
                θ_errs_dict!(labels[:theta_errs], phys, theta_true, theta_mcmc; suffix = "MCMC_mean")
                θ_errs_dict!(labels[:theta_errs], phys, theta_true, theta_mle; suffix = "MCMC_mle")
            end
        end
    end

    return nothing
end

function load_true_labels!(
        phys::EPGModel{T};
        force_reload = true,
    ) where {T}

    for (i, img) in enumerate(phys.images)
        # Optionally skip reloading
        haskey(img.meta, :true_labels) && !force_reload && continue

        # Load MCMC params
        labels_file =
            !haskey(img.meta[:info], "true_labels_path") ? nothing :
            joinpath(img.meta[:info]["folder_path"], img.meta[:info]["true_labels_path"])
        if labels_file === nothing || !isfile(labels_file)
            @info "True label data does not exist (image = $i): $(img.meta[:info]["folder_path"])"
            continue
        else
            @info labels_file
            @info "Loading true label data (image = $i):"
            img.meta[:true_labels] = Dict{Symbol,Any}()
        end
        @time labels_data = DECAES.MAT.matread(labels_file)

        param_names = ["alpha", "beta", "eta", "delta1", "delta2", "logepsilon", "logscale"]
        num_signals = length(labels_data["alpha"])

        for dataset in [:train, :val, :test]
            theta = mapreduce(vcat, enumerate(param_names)) do (i, param_name)
                J = findall_within(img.indices[:mask], img.indices[dataset])
                θ = labels_data[param_name][J]' .|> T
            end

            # Compute epg signal model
            @time X = signal_model(phys, img, theta)

            # Assign outputs
            labels             = img.meta[:true_labels][dataset] = Dict{Symbol,Any}()
            labels[:theta]     = theta # θ = α, β, η, δ1, δ2, logϵ, logs
            labels[:signalfit] = X .|> T
        end
    end

    return nothing
end

function θ_errs_dict!(d::Dict{Symbol}, phys, θ_true, θ_approx; suffix)
    θ_widths = θupper(phys) .- θlower(phys)
    θ_errs   = mean(abs, θ_true .- θ_approx; dims = 2)
    for (i, lab) in enumerate(θasciilabels(phys))
        d[Symbol("$(lab)_err_$(suffix)")] = 100 * θ_errs[i] / θ_widths[i]
    end
    return d
end
θ_errs_dict(phys, θ_true, θ_approx; suffix) = θ_errs_dict!(Dict{Symbol,Any}(), phys, θ_true, θ_approx; suffix)


function verify_mle_labels(phys::EPGModel)
    for (i, img) in enumerate(phys.images)
        dataset = :val
        @unpack theta, signalfit = img.meta[:mle_labels][dataset]
        Y = img.partitions[dataset]
        ℓ = negloglikelihood(phys, Y, signalfit, theta)
        @info "MLE labels negative log-likelihood (image = $i, dataset = $dataset):"
        @info StatsBase.summarystats(vec(ℓ))
    end
end

function verify_mcmc_labels(phys::EPGModel)
    for (i, img) in enumerate(phys.images)
        dataset = :val
        @unpack theta, signalfit = img.meta[:mcmc_labels][dataset]
        Y = img.partitions[dataset]
        ℓ = negloglikelihood(phys, Y, signalfit, theta[:,:,end])
        @info "MCMC labels negative log-likelihood (image = $i, dataset = $dataset):"
        @info StatsBase.summarystats(vec(ℓ[:, 1:findlast(!isnan, ℓ[1,:]), :]))
    end
end

function verify_pseudo_labels(phys::EPGModel)
    for (i, img) in enumerate(phys.images)
        dataset = :val
        @unpack theta, signalfit, mh_sampler = img.meta[:pseudo_labels][dataset]
        Y = img.partitions[dataset]
        ℓ = negloglikelihood(phys, Y, signalfit, theta[:,:,end])
        @info "Pseudo labels negative log-likelihood (image = $i, dataset = $dataset):"
        @info StatsBase.summarystats(vec(ℓ))
    end
end

function verify_true_labels(phys::EPGModel)
    for (i, img) in enumerate(phys.images)
        dataset = :val
        haskey(img.meta, :true_labels) || continue
        @unpack theta, signalfit = img.meta[:true_labels][dataset]
        Y = img.partitions[dataset]
        ℓ = negloglikelihood(phys, Y, signalfit, theta)
        @info "True labels negative log-likelihood (image = $i, dataset = $dataset):"
        @info StatsBase.summarystats(vec(ℓ))
    end
end

####
#### GANs
####

# models["discrim"]
function init_mlp_discrim(phys::PhysicsModel{Float32}; ninput = nsignal(phys), hdim, nhidden, dropout, kwargs...)
    MLP(ninput => 1, nhidden, hdim, Flux.relu, identity; dropout) |> gpu
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
        DeepExponentialKernel(logσ, embedding) |> gpu
    end
end

####
#### Optimizers
####

function init_optimizer(otype = Flux.ADAM; lr = 0.0, gclip = 0.0, wdecay = 0.0, kwargs...)
    os = Any[otype(lr)]
    (gclip > 0) && pushfirst!(os, Flux.ClipValue(gclip))
    (wdecay > 0) && push!(os, Flux.WeightDecay(wdecay))
    Flux.Optimiser(os)
end

####
#### Snapshot
####

load_model(args...; kwargs...) = fix_model_deprecations(FileIO.load(args...; kwargs...))

fix_model_deprecations(v) = Flux.fmap(v; exclude = x -> parentmodule(typeof(x)) === JLD2.ReconstructedTypes) do x
    Tname = string(nameof(typeof(x))) # `Base.typename` fails for JLD2.ReconstructedTypes types? Must use `string(nameof(typeof(x)))`
    if occursin("Flux.Dense{typeof(identity)", Tname)
        # Flux renamed :W and :b fields to :weight and :bias
        Flux.Dense(x.W, x.b) #TODO general case w/ non-identity activation; last I checked, reconstructed `x` did not have `:σ` field for some reason?
    else
        error("Cannot reconstruct type: $Tname")
    end
end
fix_model_deprecations(d::Dict) = map_dict(fix_model_deprecations, d)

function load_checkpoint(filename)
    isempty(checkpointdir()) && return Dict{String, Any}()
    try
        load_model(checkpointdir(filename), "models") |> deepcopy |> gpu
    catch e
        @warn "Error loading checkpoint from $(checkpointdir())"
        @warn sprint(showerror, e, catch_backtrace())
        return Dict{String, Any}()
    end
end

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
    for path in DrWatson.projectdir.(saveitems)
        cp(path, joinpath(savepath, basename(path)); force = true)
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

macro save_expression(filename, ex)
    quote
        local fname = $(esc(filename))
        open(fname; write = true) do io
            println(io, $(string(ex)))
        end
        $(esc(ex))
    end
end

nothing
