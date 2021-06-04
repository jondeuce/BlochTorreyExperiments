####
#### Settings
####

using DrWatson: @quickactivate
@quickactivate "NeurIPS2021"
using NeurIPS2021
lib.initenv()

lib.settings_template() = TOML.parsefile(joinpath(@__DIR__, "train_metropolis.toml"))
settings = lib.load_settings(force_new_settings = true)

lib.set_logdirname!()
lib.set_checkpointdir!(settings["checkpoint"]["folder"]::String == "" ? "" : lib.projectdir(settings["checkpoint"]["folder"]::String))

lib.@save_expression lib.logdir("build_physics.jl") function build_physics()
    lib.load_epgmodel_physics()
end

lib.@save_expression lib.logdir("build_models.jl") function build_models(
        phys,
        settings,
        models = Dict{String,Any}(),
        derived = Dict{String,Any}(),
    )
    kws(keys...) = lib.make_kwargs(settings, keys...)

    lib.initialize!(phys; seed = 0, image_folders = settings["data"]["image_folders"])
    get!(models, "enc1") do; lib.init_mlp_cvae_enc1(phys; kws("arch", "enc1")...); end
    get!(models, "enc2") do; lib.init_mlp_cvae_enc2(phys; kws("arch", "enc2")...); end
    get!(models, "dec") do; lib.init_mlp_cvae_dec(phys; kws("arch", "dec")...); end
    derived["cvae"] = lib.derived_cvae(phys, models["enc1"], models["enc2"], models["dec"]; kws("arch")...)

    # Load true labels
    lib.load_true_labels!(phys; force_reload = false)
    lib.verify_true_labels(phys)

    # Load mcmc labels
    lib.load_mcmc_labels!(phys; force_reload = false)
    lib.verify_mcmc_labels(phys)

    # Initial pseudo labels
    lib.initialize_pseudo_labels!(phys, derived["cvae"]; force_recompute = true, labelset = settings["data"]["labels"]["initialize_pseudo"])
    lib.verify_pseudo_labels(phys)

    return models, derived
end

lib.@save_expression lib.logdir("build_optimizers.jl") function build_optimizers(
        phys,
        settings,
        models,
        optimizers = Dict{String,Any}(),
    )
    kws(keys...) = lib.make_kwargs(settings, keys...)
    optimizers["cvae"] = lib.init_optimizer(Flux.ADAM; kws("opt", "cvae")...)
    return optimizers
end

phys = build_physics()
models, derived = build_models(phys, settings, lib.load_checkpoint(settings["checkpoint"]["model"]))
optimizers = build_optimizers(phys, settings, models)
lib.save_snapshot(settings, models)

# Global logger
logger = DataFrame(
    :epoch      => Int[],
    :iter       => Int[],
    :time       => Float64[],
    :dataset    => Symbol[],
    :img_idx    => Int[],
)

# Global trainer
trainer = ignite.engine.Engine(@j2p (args...) -> @timeit "train step" train_step(args...))
trainer.logger = ignite.utils.setup_logger("trainer")

####
#### CVAE
####

# Conditional variational autoencoder losses
function CVAElosses(Y, θ)
    # Cross-entropy loss function components
    Ymasked, Ymask = lib.pad_and_mask_signal(Y, lib.nsignal(derived["cvae"]); minkept = settings["train"]["augment"]["mask"]::Int, maxkept = lib.nsignal(derived["cvae"]))
    Ystate         = lib.CVAETrainingState(derived["cvae"], Ymasked, θ)
    KLDiv, ELBO    = lib.KL_and_ELBO(Ystate)
    ℓ = (; KLDiv, ELBO)
    return ℓ
end

####
#### Training
####

function sample_batch(batch; dataset::Symbol, batchsize::Int, signalfit::Bool = false)

    # Batch contains an index into the list of images
    img_idx = batch isa Int ? batch : only(lib.p2j_array(batch)) |> Int

    if img_idx <= 0
        # Indices <= 0 are used for indicating that simulated data should be generated on the fly
        if img_idx == 0
            # Draw θ from the full prior space, i.e. the cartesian product of prior spaces for each θ[i]
            img = img_cols = nothing
            θ = lib.sampleθprior(phys, Matrix{Float32}, batchsize) # signal_model is faster on cpu; move to gpu afterward
            X = lib.signal_model(phys, θ) # faster on cpu; move to gpu afterward
        else # img_idx < 0
            # Draw θ from the space of pseudo labels of the image with index `abs(img_idx)`
            img = phys.images[abs(img_idx)]
            @unpack mh_sampler = img.meta[:pseudo_labels][dataset]
            img_cols = rand(1:mh_sampler.ndata, batchsize)
            θ = mh_sampler.θ[:, lib.buffer_indices(mh_sampler, img_cols)] # sample randomly from most recently updated pseudo posterior samples
            X = lib.signal_model(phys, img, θ)
        end
        θ = θ |> gpu
        X = X |> gpu
        Y = lib.add_noise_instance(phys, X, θ)

        # Normalize Y, X, θ for CVAE
        Ymax = maximum(Y; dims = 1)
        Y ./= Ymax
        X ./= Ymax
        θ[7:7, ..] .= clamp.(θ[7:7, ..] .- log.(Ymax), lib.θlower(phys)[7], lib.θupper(phys)[7])

    else
        # Sample signals from the image
        img   = phys.images[img_idx]
        Y_cpu, img_cols = lib.sample_columns(img.partitions[dataset], batchsize; replace = false)
        Y     = Y_cpu |> gpu

        if settings["data"]["labels"]["image_labelset"]::String == "pseudo"
            # Use pseudo labels from the Metropolis-Hastings sampler
            @unpack mh_sampler = img.meta[:pseudo_labels][dataset]
            lib.update!(mh_sampler, phys, derived["cvae"], img, Y, Y_cpu; img_cols)
            θ = mh_sampler.θ[:, lib.buffer_indices(mh_sampler, img_cols)]
            X = !signalfit ? nothing : lib.signal_model(phys, θ) # faster on cpu; move to gpu afterward
            θ = θ |> gpu
            X = !signalfit ? nothing : (X |> gpu)

        elseif settings["data"]["labels"]["image_labelset"]::String == "mcmc"
            # Use labels drawn from precomputed MCMC chains
            θ = img.meta[:mcmc_labels_100][dataset][:theta][:, img_cols, rand(1:end)]
            X = !signalfit ? nothing : lib.signal_model(phys, img, θ) # faster on cpu; move to gpu afterward
            θ = θ |> gpu
            X = !signalfit ? nothing : (X |> gpu)

        elseif settings["data"]["labels"]["image_labelset"]::String == "mle"
            # Use precomputed MLE labels
            θ = img.meta[:mle_labels][dataset][:theta][:, img_cols] |> gpu
            X = !signalfit ? nothing : img.meta[:mle_labels][dataset][:signalfit][:, img_cols] |> gpu

        elseif (cvae_key = settings["data"]["labels"]["image_labelset"]::String) ∈ ("cvae", "pretrained_cvae")
            # Use pseudo labels drawn from a pretrained CVAE
            @assert haskey(derived, cvae_key)
            θ, _ = lib.sampleθZposterior(derived[cvae_key], Y)
            X = !signalfit ? nothing : lib.signal_model(phys, img, θ) # theta already on gpu

        else
            error("No labels chosen")
        end
    end

    return (; img, img_idx, img_cols, Y, X, θ)
end

function train_step(engine, batch)
    outputs = Dict{Any,Any}()
    trainer.should_terminate && return outputs

    @timeit "sample batch" begin
        @unpack img_idx, Y, θ = sample_batch(batch; dataset = :train, batchsize = settings["train"]["batchsize"]::Int)
    end

    # Model parameters
    ps = Flux.params(models["enc1"], models["enc2"], models["dec"])

    # Forward pass, logging metrics
    @timeit "forward" ℓ, back = Zygote.pullback(ps) do
        ℓs = CVAElosses(Y, θ)
        Zygote.@ignore let
            for (k,v) in pairs(ℓs)
                outputs["$(k)_$(img_idx)"] = v # tag output metric with image index
            end
            outputs["CVAE_$(img_idx)"] = sum(ℓs) # tag output metric with image index
        end
        return sum(ℓs)
    end

    # Reverse pass
    @timeit "reverse" gs = back(one(ℓ))

    # Update parameters
    @timeit "update!" Flux.Optimise.update!(optimizers["cvae"], ps, gs)

    return outputs
end

function compute_metrics(engine, batch; dataset::Symbol)
    outputs = Dict{Any,Any}()
    trainer.should_terminate && return outputs

    @timeit "sample batch" begin
        @unpack img, img_idx, img_cols, Y, X, θ = sample_batch(batch; dataset = dataset, batchsize = settings["eval"]["batchsize"]::Int)
    end

    # Initialize metrics
    metrics = Dict{Symbol,Any}()
    metrics[:epoch]   = trainer.state.epoch
    metrics[:iter]    = trainer.state.iteration
    metrics[:time]    = time()
    metrics[:dataset] = dataset
    metrics[:img_idx] = img_idx

    # helper function for cdf distances
    function mcmc_cdf_distances(θ₁, θ₂)
        dists    = lib.cdf_distance((1, 2), θ₁, θ₂) # array of tuples of ℓ₁ and ℓ₂ distances between the empirical cdf's
        dists    = mean(reinterpret(DECAES.SVector{2,eltype(eltype(dists))}, dists); dims = 2) # mean over SVector's is defined, but not over Tuple's, hence the reinterpret
        θ_widths = lib.θupper(phys) .- lib.θlower(phys)
        ℓ₁       = (x->x[1]).(dists) ./ θ_widths # scale to range [0,1]
        ℓ₂       = (x->x[2]).(dists) ./ sqrt.(θ_widths) # scale to range [0,1]
        return (L1 = ℓ₁, L2 = ℓ₂)
    end

    # Update Metropolis-Hastings pseudo labels for entire dataset
    (img_idx > 0) && @timeit "update pseudo labels" begin
        @unpack mh_sampler = img.meta[:pseudo_labels][dataset]
        niters = dataset === :train ?
            1 : # training pseudo labels are updated before training every time a training batch is sampled; update here only once for the whole training data set to ensure uniform coverage
            max(1, minimum(img.meta[:pseudo_labels][:train][:mh_sampler].i) - minimum(mh_sampler.i)) # update validation pseudo labels until they have been updated as least as many times as the training pseudo labels
        for i = 1:niters
            lib.update!(mh_sampler, phys, derived["cvae"], img; dataset, gpu_batch_size = 32768)
        end
    end

    # Inference using CVAE
    @timeit "sample cvae" begin
        θ′_sampler = lib.θZposterior_sampler(derived["cvae"], Y)
        θ′_mode, _ = θ′_sampler(mode = true)
        X′_mode    = img_idx != 0 ? lib.signal_model(phys, img, θ′_mode) : lib.signal_model(phys, θ′_mode)
        X′_mode    = lib.clamp_dim1(Y, X′_mode)
        Y′         = lib.add_noise_instance(phys, X′_mode, θ′_mode)
        θ′_samples = zeros(Float32, size(θ′_mode)..., 100)
        for i in 1:size(θ′_samples, 3)
            local θ′, _ = θ′_sampler()
            θ′_samples[:,:,i] .= lib.cpu32(θ′)
        end
        θ′_samples_q1, θ′_samples_q2, θ′_samples_q3 = lib.fast_quartiles3(θ′_samples)
    end

    # MLE using CVAE mode as initial guess
    (img_idx != 0) && @timeit "compute mle" begin
        mle_init   = (; Y = lib.cpu64(Y), θ = lib.cpu64(θ′_mode))
        _, mle_res = lib.mle_biexp_epg(phys, img; initial_guess = mle_init, batch_size = Colon(), verbose = false)
        metrics[:logL_MLE] = mean(mle_res.loss)
    end

    # CVAE losses
    @timeit "cvae losses" let
        ℓs = CVAElosses(Y, θ)
        metrics[:CVAE] = sum(ℓs)
        for (k,v) in pairs(ℓs)
            metrics[k] = v
        end
    end

    # Goodness of fit metrics, confidence intervals, etc. for CVAE posterior samples
    @timeit "cvae goodness of fit" let
        metrics[:rmse_CVAE] = sqrt(mean(abs2, Y .- X′_mode))
        metrics[:logL_CVAE] = mean(lib.negloglikelihood(phys, Y, X′_mode, θ′_mode))
        lib.θ_rel_errs_dict!(metrics, phys, θ′_samples_q3 .- θ′_samples_q1; suffix = "CVAE_iqr") # CVAE samples' interquartile range
    end

    # CVAE parameter cdf distances w.r.t. ground truth mcmc
    (img_idx > 0) && @timeit "cvae cdf distances" let
        θ_mcmc  = img.meta[:mcmc_labels_100][dataset][:theta]
        θ_dists = mcmc_cdf_distances(θ_mcmc[:, img_cols, :], θ′_samples)
        for (i, lab) in enumerate(lib.θasciilabels(phys)), (ℓ_name, ℓ) in pairs(θ_dists)
            metrics[Symbol("$(lab)_$(ℓ_name)_CVAE")] = ℓ[i]
        end

        θ_dists = mcmc_cdf_distances(mh_sampler.θ[:, img_cols, :], θ′_samples)
        for (i, lab) in enumerate(lib.θasciilabels(phys)), (ℓ_name, ℓ) in pairs(θ_dists)
            metrics[Symbol("$(lab)_$(ℓ_name)_Self")] = ℓ[i]
        end
    end

    (img_idx > 0) && @timeit "cvae 3000 cdf distances" let
        local θ_cols_ = img.meta[:mcmc_labels_3000][dataset][:columns]
        local θ_mcmc_ = img.meta[:mcmc_labels_3000][dataset][:theta][:, :, 2501:3000]
        local Y_mcmc_ = img.partitions[dataset][:, θ_cols_] |> lib.gpu
        local θ′_sampler_ = lib.θZposterior_sampler(derived["cvae"], Y_mcmc_)
        local θ′_samples_ = zeros(Float32, size(θ_mcmc_)...)
        for i in 1:size(θ′_samples_, 3)
            local θ′, _ = θ′_sampler_()
            θ′_samples_[:,:,i] .= lib.cpu32(θ′)
        end
        local θ_dists_ = mcmc_cdf_distances(θ_mcmc_, θ′_samples_)
        for (i, lab) in enumerate(lib.θasciilabels(phys)), (ℓ_name, ℓ) in pairs(θ_dists_)
            metrics[Symbol("$(lab)_$(ℓ_name)_CVAE_3000")] = ℓ[i]
        end
    end

    # Pseudo label metrics
    (img_idx > 0) && @timeit "pseudo label metrics" let
        # Record current negative log likelihood of pseudo labels  (note: these over datasets, not just this batch)
        buf_idx   = lib.buffer_indices(mh_sampler)
        accept    = mh_sampler.accept[:, buf_idx]
        neglogPXθ = mh_sampler.neglogPXθ[:, buf_idx]
        metrics[:accept_Pseudo] = mean(vec(accept))
        metrics[:logL_Pseudo]   = mean(filter(!isinf, vec(neglogPXθ)))

        # Record cdf distance metrics  (note: these over datasets, not just this batch)
        θ_mcmc  = img.meta[:mcmc_labels_100][dataset][:theta]
        θ_dists = mcmc_cdf_distances(θ_mcmc, mh_sampler.θ)
        for (i, lab) in enumerate(lib.θasciilabels(phys)), (ℓ_name, ℓ) in pairs(θ_dists)
            metrics[Symbol("$(lab)_$(ℓ_name)_Pseudo")] = ℓ[i]
        end

        # Sample statistics
        lib.θ_rel_errs_dict!(metrics, phys, lib.fast_iqr3(mh_sampler.θ[:, img_cols, :]); suffix = "Pseudo_iqr") # CVAE samples' interquartile range
    end

    # Error metrics w.r.t true labels
    (img_idx <= 0 || haskey(img.meta, :true_labels)) && @timeit "true label metrics" let
        θ_true = img_idx <= 0 ? lib.cpu32(θ) : img.meta[:true_labels][dataset][:theta][:, img_cols]
        lib.θ_rel_errs_dict!(metrics, phys, θ_true .- lib.cpu32(θ′_mode); suffix = "CVAE_mode") # error of CVAE posterior mode
        lib.θ_rel_errs_dict!(metrics, phys, θ_true .- dropdims(mean(θ′_samples; dims = 3); dims = 3); suffix = "CVAE_mean") # error of CVAE posterior mean
        lib.θ_rel_errs_dict!(metrics, phys, θ_true .- θ′_samples_q2; suffix = "CVAE_med") # error of CVAE posterior median
        lib.θ_errs_dict!(metrics, phys, θ′_samples_q1 .< θ_true .< θ′_samples_q3; suffix = "CVAE_iqf") # fraction of IQR's which contain the true label
        if img_idx != 0
            lib.θ_rel_errs_dict!(metrics, phys, θ_true .- (mle_res.theta .|> Float32); suffix = "CVAE_mle") # error of MLE initialized with CVAE posterior sample
        end
        if img_idx > 0
            let
                θ′ = mh_sampler.θ[:, img_cols, :]
                θ′_q1, θ′_q2, θ′_q3 = lib.fast_quartiles3(θ′)
                lib.θ_rel_errs_dict!(metrics, phys, θ_true .- dropdims(mean(θ′; dims = 3); dims = 3); suffix = "Pseudo_mean") # error of mean pseudo labels
                lib.θ_rel_errs_dict!(metrics, phys, θ_true .- θ′_q2; suffix = "Pseudo_med") # error of median pseudo labels
                lib.θ_errs_dict!(metrics, phys, θ′_q1 .< θ_true .< θ′_q3; suffix = "Pseudo_iqf") # fraction of IQR's which contain the true label
            end
        end
    end

    # Update logger dataframe
    push!(logger, metrics; cols = :union)

    # Return metrics for W&B logging
    for (k,v) in metrics
        k ∈ [:epoch, :iter, :time, :dataset] && continue # output non-housekeeping metrics
        outputs["$(k)_$(img_idx)"] = v # tag output metric with image index
    end

    return outputs
end

function make_plots(; showplot::Bool = false)
    trainer.should_terminate && return Dict{Symbol,Any}()
    plots  = Dict{String, Any}()
    groups = DataFrames.groupby(logger, [:dataset, :img_idx]) |> pairs
    try
        for (group_key, group) in groups
            ps = Any[]
            cols_iterator = group[!, DataFrames.Not([:epoch, :iter, :time, :dataset, :img_idx])] |> eachcol |> pairs |> collect |> iter -> sort(iter; by = first)
            for (colname, col) in cols_iterator
                I = (x -> !ismissing(x) && !isnan(x) && !isinf(x)).(col)
                I .&= max(first(group.epoch), min(100, last(group.epoch)-100)) .<= group.epoch
                !any(I) && continue
                epochs, col = group.epoch[I], col[I]
                p = Plots.plot(
                    epochs, col;
                    label = :none, title = "$colname",
                    titlefontsize = 10, xtickfontsize = 8, ytickfontsize = 8,
                    xlims = (min(first(epochs), 100), last(epochs)),
                    xscale = last(epochs) < 1000 ? :identity : :log10,
                    xformatter = x -> round(Int, x),
                )
                if group_key.img_idx > 0 && contains(String(colname), "_err_")
                    img              = phys.images[group_key.img_idx]
                    mcmc_errs_dict   = lib.recursive_try_get(img.meta, [:mcmc_labels_100, group_key.dataset, :theta_errs])
                    lab, _           = split(String(colname), "_err_")
                    mcmc_errs_dict !== missing && Plots.hline!(p, [mcmc_errs_dict[Symbol(lab * "_err_MCMC_mean")]]; colour = :red, label = :none)
                    mcmc_errs_dict !== missing && Plots.hline!(p, [mcmc_errs_dict[Symbol(lab * "_err_MCMC_med")]]; colour = :orange, label = :none)
                    mcmc_errs_dict !== missing && Plots.hline!(p, [mcmc_errs_dict[Symbol(lab * "_err_MCMC_mle")]]; colour = :green, label = :none)
                end
                push!(ps, p)
            end
            p = Plots.plot(ps...; size = max.((800,600), ceil(sqrt(length(ps))) .* (240,180)))
            showplot && display(p)
            plots["$(group_key.dataset)-losses-$(group_key.img_idx)"] = p
        end
        return plots
    catch e
        lib.handle_interrupt(e; msg = "Error plotting")
    end
end

####
#### Data loaders, callbacks, and events
####

function data_loader(; loader_type::Symbol)
    # Batch sampling is done on the julia side, but the image which is sampled from is chosen by Ignite
    if loader_type === :train
        train_nbatches  = settings["train"]["nbatches"]::Int
        train_indices   = settings["data"]["labels"]["train_indices"]::Vector{Int}
        train_fractions = settings["data"]["labels"]["train_fractions"]::Vector{Float64}
        train_fractions = (length(train_fractions) == 1 ? ones(length(train_indices)) : train_fractions) |> x -> x ./ sum(x)
        sampler         = torch.utils.data.WeightedRandomSampler(train_fractions, replacement = true, num_samples = train_nbatches)
        torch.utils.data.DataLoader(train_indices, sampler = sampler)
    elseif loader_type === :eval
        eval_indices    = settings["data"]["labels"]["eval_indices"]::Vector{Int}
        sampler         = torch.utils.data.SequentialSampler(collect(0:length(eval_indices)-1))
        torch.utils.data.DataLoader(eval_indices, sampler = sampler)
    else
        error("loader_type must be :train or :eval, got: $(repr(loader_type))")
    end
end

train_loader = data_loader(; loader_type = :train)
val_eval_loader = data_loader(; loader_type = :eval)
train_eval_loader = data_loader(; loader_type = :eval)

val_evaluator = ignite.engine.Engine(@j2p (args...) -> @timeit "val metrics" compute_metrics(args...; dataset = :val))
val_evaluator.logger = ignite.utils.setup_logger("val_evaluator")

train_evaluator = ignite.engine.Engine(@j2p (args...) -> @timeit "train metrics" compute_metrics(args...; dataset = :train))
train_evaluator.logger = ignite.utils.setup_logger("train_evaluator")

Events = ignite.engine.Events

# Force terminate
trainer.add_event_handler(
    Events.STARTED | Events.ITERATION_STARTED | Events.ITERATION_COMPLETED,
    @j2p lib.terminate_file_event(file = lib.logdir("stop"))
)

# Timeout
trainer.add_event_handler(
    Events.EPOCH_COMPLETED(event_filter = @j2p lib.timeout_event_filter(settings["train"]["timeout"])),
    @j2p function (engine)
        @info "Exiting: training time exceeded $(DECAES.pretty_time(settings["train"]["timeout"]))"
        engine.terminate()
    end
)

# Update data global state (optimizer learning rates, etc.)
trainer.add_event_handler(
    Events.STARTED | Events.EPOCH_COMPLETED,
    @j2p function (engine)
        num_periods(epoch, xrate) = floor(Int, max(epoch, 0)/xrate) # period 0 -> epoch [0,1,...xrate-1], period 1 -> epoch [xrate,...,2*xrate-1], etc.
        periodic_arithmetic_drop(epoch, xinit, xdrop, xrate, xmin) = max(xinit - xdrop * num_periods(epoch, xrate), xmin) # starting from `xinit`, drop by `xdrop` until `xmin` every `xrate` epoch
        periodic_geometric_drop(epoch, xinit, xdrop, xrate, xmin) = clamp(exp(periodic_arithmetic_drop(epoch, log(xinit), log(xdrop), xrate, log(xmin))), xmin, xinit) # starting from `xinit`, drop by `xdrop` until `xmin` every `xrate` epoch

        # Update learning rates
        lib.update_optimizers!(optimizers; field = :eta) do opt, opt_name
            @unpack lr, lrdrop, lrrate, lrthresh = settings["opt"]
            opt.eta = periodic_geometric_drop(trainer.state.epoch, lr, lrdrop, lrrate, lrthresh)
        end
    end
)

# Evaluator events
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["evalmetricsperiod"])),
    @j2p (engine) -> val_evaluator.run(val_eval_loader)
)

trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["evalmetricsperiod"])),
    @j2p (engine) -> train_evaluator.run(train_eval_loader)
)

# Plot current losses periodically
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["checkpointperiod"])),
    @j2p function (engine)
        @timeit "checkpoint" let
            @timeit "make checkpoint plots" plothandles = make_plots()
            @timeit "save checkpoint plots" lib.save_plots(plothandles; savefolder = lib.logdir(), prefix = "checkpoint-")
        end
    end
)

# Checkpoint current model and logger periodically
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["checkpointperiod"])),
    @j2p function (engine)
        @timeit "checkpoint" let models = lib.cpu32(models)
            @timeit "save current model" lib.save_progress(@dict(models, logger); savefolder = lib.logdir(), prefix = "current-", ext = ".jld2")
        end
    end
)

# Check for a new best model every epoch; save the coresponding model + logger if true
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED,
    @j2p function (engine)
        is_best = false
        for (group_key, group) in pairs(DataFrames.groupby(logger[!, [:dataset, :img_idx, :CVAE]], [:dataset, :img_idx]))
            group_key.dataset === :val || continue
            is_best |= length(group.CVAE) >= 2 && group.CVAE[end] < minimum(group.CVAE[1:end-1])
        end
        is_best && @timeit "save best progress" let models = lib.cpu32(models)
            @timeit "save best model" lib.save_progress(@dict(models, logger); savefolder = lib.logdir(), prefix = "best-", ext = ".jld2")
        end
    end
)

# Drop learning rate
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p lib.droplr_file_event(optimizers;
        file     = lib.logdir("droplr"),
        lrdrop   = settings["opt"]["lrdrop"]::Float64,
        lrthresh = settings["opt"]["lrthresh"]::Float64,
    )
)

# User input
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p(lib.file_event(lib.user_input_event(); file = lib.logdir("user"))),
)

# Print TimerOutputs timings
trainer.add_event_handler(
    Events.TERMINATE | Events.EPOCH_COMPLETED(event_filter = @j2p lib.throttler_event_filter(settings["eval"]["evalmetricsperiod"])),
    @j2p function (engine)
        @info "Log folder: $(lib.logdir())"
        show(TimerOutputs.get_defaulttimer()); println("")
        show_all_cols = false
        start_cols    = [:epoch, :iter, :time, :dataset, :img_idx, :CVAE, :KLDiv, :ELBO, :LatentReg, :VAE, :logL_CVAE, :logL_MLE, :logL_Pseudo]
        start_cols    = filter(c -> String(c) ∈ names(logger), start_cols)
        show_order    = [start_cols; sort([Symbol(col) for col in names(logger) if Symbol(col) ∉ start_cols])]
        display(last(logger[!, (show_all_cols ? show_order : start_cols)], 12)); println("")
        (engine.state.epoch == 1) && TimerOutputs.reset_timer!() # throw out compilation timings
    end
)

# Reset logging/timer/etc.
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    @j2p lib.file_event(file = lib.logdir("reset")) do engine
        TimerOutputs.reset_timer!()
        empty!(cb_state)
        empty!(logger)
    end
)

####
#### Run trainer
####

TimerOutputs.reset_timer!()
trainer.run(train_loader, settings["train"]["epochs"])
