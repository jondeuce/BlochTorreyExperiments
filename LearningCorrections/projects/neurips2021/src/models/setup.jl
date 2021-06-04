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

####
#### Logging
####

const train_debug = Ref(false)

const _logdirname = Ref("")
set_logdirname!(dirname) = (_logdirname[] = basename(dirname))
set_logdirname!() = basename(mkpath(DrWatson.projectdir("log", set_logdirname!(getnow()))))
get_logdirname() = (isempty(_logdirname[]) ? set_logdirname!() : _logdirname[])
logdir(args...) = DrWatson.projectdir("log", get_logdirname(), args...)

const _checkpointdir = Ref("")
set_checkpointdir!(dir) = _checkpointdir[] = dir
get_checkpointdir() = _checkpointdir[]
clear_checkpointdir!() = set_checkpointdir!("")
checkpointdir(args...) = isempty(get_checkpointdir()) ? "" : joinpath(get_checkpointdir(), args...)

####
#### Physics
####

function load_cpmg_info(folder_path::AbstractString)
    folder_path = DrWatson.datadir("images", folder_path)
    info = TOML.parsefile(joinpath(folder_path, "image_info.toml"))
    info["folder_path"] = folder_path
    return info
end

function load_epgmodel_physics(; max_numechos = 64)
    phys = BiexpEPGModel{Float32}(n = max_numechos)
    return phys
end

####
#### CVAE
####

function init_mlp_cvae_enc1(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, kwargs...)
    σact = Flux.relu
    mlp = Flux.Chain(MLP(nsignal(phys) => 2*zdim, nhidden, hdim, σact, identity), CVAELatentTransform(zdim)) |> flattenchain |> gpu
end

function init_mlp_cvae_enc2(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, kwargs...)
    σact = Flux.relu
    mlp = Flux.Chain(MLP(nsignal(phys) + ntheta(phys) + nlatent => 2*zdim, nhidden, hdim, σact, identity), CVAELatentTransform(zdim)) |> flattenchain |> gpu
    Stack(@nntopo((X,θ,Z) => XθZ => μq), vcat, mlp) |> gpu
end

function init_mlp_cvae_dec(phys::PhysicsModel{Float32}; hdim, nhidden, zdim, nlatent, kwargs...)
    σact = Flux.relu
    mlp = MLP(nsignal(phys) + zdim => 2*(nmarginalized(phys) + nlatent), nhidden, hdim, σact, identity) |> gpu
    Stack(@nntopo((Y,zr) => Yzr => μx), vcat, mlp) |> gpu
end

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

    posterior_dist = posterior == "TruncatedGaussian" ? TruncatedGaussian : Gaussian
    CVAE{nsignal(phys),ntheta(phys),nmarginalized(phys),nlatent,zdim}(enc1, enc2, dec, normalize_inputs, unnormalize_outputs; posterior_dist)
end

function load_pretrained_cvae(phys::PhysicsModel{Float32}; modelfolder, modelprefix = "best-")
    settings = TOML.parsefile(joinpath(modelfolder, "settings.toml"))
    models = load_model(only(Glob.glob(modelprefix * "models.*", modelfolder)), "models") |> deepcopy |> gpu
    @unpack enc1, enc2, dec = models
    cvae = derived_cvae(phys, enc1, enc2, dec; make_kwargs(settings, "arch")...)
    return cvae
end

function initialize_pseudo_labels!(
        phys::BiexpEPGModel{T}, cvae::Union{Nothing,<:CVAE} = nothing;
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
                @unpack theta, signalfit = img.meta[:mcmc_labels_100][dataset]
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

function compute_mle_labels!(phys::BiexpEPGModel, cvae::Union{Nothing,<:CVAE} = nothing; kwargs...)
    for img in phys.images
        compute_mle_labels!(phys, img, cvae; kwargs...)
    end
    return phys
end

function compute_mle_labels!(
        phys::BiexpEPGModel{T}, img::CPMGImage{T}, cvae::Union{Nothing,<:CVAE} = nothing;
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
            labels[:theta_errs] = θ_rel_errs_dict(phys, theta_true .- theta_mle; suffix = "MLE")
        end
    end

    return img
end

function load_mcmc_labels!(
        phys::BiexpEPGModel{T};
        force_reload = true,
    ) where {T}

    for (img_idx, img) in enumerate(phys.images), mcmc_labels_path in [:mcmc_labels_100, :mcmc_labels_3000]
        # Optionally skip reloading
        haskey(img.meta, mcmc_labels_path) && !force_reload && continue
        !haskey(img.meta[:info], String(mcmc_labels_path)) && continue
        img.meta[mcmc_labels_path] = Dict{Symbol,Any}()

        # Load MCMC params
        labels_file = joinpath(img.meta[:info]["folder_path"], img.meta[:info][String(mcmc_labels_path)])
        if !isfile(labels_file)
            @info "MCMC data does not exist (image = $(img_idx)): $(img.meta[:info]["folder_path"])"
            continue
        else
            @info labels_file
            @info "Loading MCMC data (image = $(img_idx)):"
        end
        @time labels_data = DECAES.MAT.matread(labels_file)

        mcmc_param_names  = ["alpha", "beta", "eta", "delta1", "delta2", "logepsilon", "logscale"]
        total_samples     = length(labels_data["iteration"])
        samples_per_chain = maximum(labels_data["iteration"])
        num_signals       = total_samples ÷ samples_per_chain
        x_index           = labels_data["image_x"][1 : samples_per_chain : end]
        y_index           = labels_data["image_y"][1 : samples_per_chain : end]
        z_index           = labels_data["image_z"][1 : samples_per_chain : end]
        labels_indices    = CartesianIndex.(x_index, y_index, z_index)
        labels_map        = Dict(labels_indices .=> 1:num_signals)

        for dataset in [:train, :val, :test]
            # Fetch theta for each partition
            dataset_indices = intersect(labels_indices, img.indices[dataset])
            dataset_map     = Dict(img.indices[dataset] .=> 1:length(img.indices[dataset]))
            theta_columns   = (I -> labels_map[I]).(dataset_indices)
            theta = mapreduce(vcat, mcmc_param_names) do param_name
                θ = reshape(labels_data[param_name], samples_per_chain, :)[:, theta_columns] .|> T
                θ = permutedims(reshape(θ, samples_per_chain, :, 1), (3,2,1))
            end

            # Compute epg signal model
            @time X = signal_model(phys, img, theta[:,:,end])

            # Assign outputs
            labels             = img.meta[mcmc_labels_path][dataset] = Dict{Symbol,Any}()
            labels[:theta]     = theta # θ = α, β, η, δ1, δ2, logϵ, logs
            labels[:signalfit] = X .|> T
            labels[:indices] = dataset_indices
            labels[:columns] = (I -> dataset_map[I]).(labels[:indices])

            # Errors w.r.t. true labels
            @time if haskey(img.meta, :true_labels)
                theta_true          = img.meta[:true_labels][dataset][:theta]
                Y_true              = img.partitions[dataset]
                if mcmc_labels_path === :mcmc_labels_3000
                    theta_true      = theta_true[:, labels[:columns]]
                    Y_true          = Y_true[:, labels[:columns]]
                end
                theta_mcmc_mean     = dropdims(mean(theta; dims = 3); dims = 3)
                theta_mcmc_mean     = clamp.(theta_mcmc_mean, θlower(phys), θupper(phys)) # `mean` can cause `theta_mcmc_mean` to overflow outside of bounds
                theta_mcmc_med      = fast_median3(theta)
                theta_mcmc_med      = clamp.(theta_mcmc_med, θlower(phys), θupper(phys)) # shouldn't be necessary for median, but just in case
                mle_init            = (; Y = lib.cpu64(Y_true), θ = lib.cpu64(theta_mcmc_med))
                _, mle_res          = lib.mle_biexp_epg(phys, img; initial_guess = mle_init, batch_size = Colon(), verbose = true)
                theta_mle           = mle_res.theta
                labels[:theta_errs] = Dict{Symbol,Any}()
                θ_rel_errs_dict!(labels[:theta_errs], phys, theta_true .- theta_mcmc_mean; suffix = "MCMC_mean")
                θ_rel_errs_dict!(labels[:theta_errs], phys, theta_true .- theta_mcmc_med; suffix = "MCMC_med")
                θ_rel_errs_dict!(labels[:theta_errs], phys, theta_true .- theta_mle; suffix = "MCMC_mle")
            end
        end
    end

    return nothing
end

function load_true_labels!(
        phys::BiexpEPGModel{T};
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

function θ_errs_dict!(d::Dict{Symbol}, phys, θ_errs; suffix)
    θ_errs = mean(θ_errs; dims = 2)
    for (i, lab) in enumerate(θasciilabels(phys))
        d[Symbol("$(lab)_err_$(suffix)")] = θ_errs[i]
    end
    return d
end
θ_errs_dict(phys, θ_errs; suffix) = θ_errs_dict!(Dict{Symbol,Any}(), phys, θ_errs; suffix)

function θ_rel_errs_dict!(d::Dict{Symbol}, phys, θ_diffs; suffix)
    θ_widths = θupper(phys) .- θlower(phys)
    θ_errs   = 100 .* abs.(θ_diffs) ./ θ_widths
    θ_errs_dict!(d, phys, θ_errs; suffix)
    return d
end
θ_rel_errs_dict(phys, θ_diffs; suffix) = θ_rel_errs_dict!(Dict{Symbol,Any}(), phys, θ_diffs; suffix)

function verify_mle_labels(phys::BiexpEPGModel)
    for (i, img) in enumerate(phys.images)
        dataset = :val
        @unpack theta, signalfit = img.meta[:mle_labels][dataset]
        Y = img.partitions[dataset]
        ℓ = negloglikelihood(phys, Y, signalfit, theta)
        @info "MLE labels negative log-likelihood (image = $i, dataset = $dataset):"
        @info StatsBase.summarystats(vec(ℓ))
    end
end

function verify_mcmc_labels(phys::BiexpEPGModel)
    for (i, img) in enumerate(phys.images)
        dataset = :val
        @unpack theta, signalfit = img.meta[:mcmc_labels_100][dataset]
        Y = img.partitions[dataset]
        ℓ = negloglikelihood(phys, Y, signalfit, theta[:,:,end])
        @info "MCMC labels negative log-likelihood (image = $i, dataset = $dataset):"
        @info StatsBase.summarystats(vec(ℓ[:, 1:findlast(!isnan, ℓ[1,:]), :]))
    end
end

function verify_pseudo_labels(phys::BiexpEPGModel)
    for (i, img) in enumerate(phys.images)
        dataset = :val
        @unpack theta, signalfit, mh_sampler = img.meta[:pseudo_labels][dataset]
        Y = img.partitions[dataset]
        ℓ = negloglikelihood(phys, Y, signalfit, theta[:,:,end])
        @info "Pseudo labels negative log-likelihood (image = $i, dataset = $dataset):"
        @info StatsBase.summarystats(vec(ℓ))
    end
end

function verify_true_labels(phys::BiexpEPGModel)
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

load_model(args...; kwargs...) = FileIO.load(args...; kwargs...)

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
        ispath(path) || continue
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
