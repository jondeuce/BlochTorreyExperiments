####
#### Setup
####

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MMDLearning
pyplot(size=(800,600))
const settings = load_settings(joinpath(@__DIR__, "..", "settings", "ignite_settings.toml"))

# Initialize generator + discriminator + kernel
function make_models(phys)
    models = Dict{String, Any}()
    n = nsignal(phys)
    toT(m) = Flux.paramtype(eltype(phys), m)

    # Rician generator. First `n` elements for `δX` scaled to (-δ, δ), second `n` elements for `logϵ` scaled to (logϵ_bw[1], logϵ_bw[2])
    models["G"] = let
        hdim = settings["arch"]["genatr"]["hdim"]::Int
        nhidden = settings["arch"]["genatr"]["nhidden"]::Int
        maxcorr = settings["arch"]["genatr"]["maxcorr"]::Float64
        noisebounds = settings["arch"]["genatr"]["noisebounds"]::Vector{Float64}
        Flux.Chain(
            MMDLearning.MLP(n => 2n, nhidden, hdim, Flux.relu, tanh)...,
            MMDLearning.CatScale([(-maxcorr, maxcorr), (noisebounds...,)], [n,n]),
        ) |> toT
    end

    # Discriminator
    if settings["arch"]["type"] ∈ ["gan", "hyb"]
        models["D"] = let
            hdim = settings["arch"]["discrim"]["hdim"]::Int
            nhidden = settings["arch"]["discrim"]["nhidden"]::Int
            MMDLearning.MLP(n => 1, nhidden, hdim, Flux.relu, Flux.sigmoid) |> toT
        end
    end

    # Initialize `nbandwidth` linearly spaced kernel bandwidths `logσ` for each `n` channels strictly within the range (bwbounds[1], bwbounds[2])
    if settings["arch"]["type"] ∈ ["mmd", "hyb"]
        models["logsigma"] = let
            bwbounds = settings["arch"]["kernel"]["bwbounds"]::Vector{Float64}
            nbandwidth = settings["arch"]["kernel"]["nbandwidth"]::Int
            repeat(range(bwbounds...; length = nbandwidth+2)[2:end-1], 1, n) |> toT
        end
    end

    return models
end

const phys = initialize!(
    ToyModel{Float32,true}();
    ntrain = settings["data"]["ntrain"]::Int,
    ntest = settings["data"]["ntest"]::Int,
    nval = settings["data"]["nval"]::Int,
)
const models = make_models(phys)
const ricegen = VectorRicianCorrector(models["G"]) # Generator produces 𝐑^2n outputs parameterizing n Rician distributions

# Generator and discriminator losses
D_Y_loss(Y) = models["D"](Y) # discrim on real data
D_G_X_loss(X) = models["D"](corrected_signal_instance(ricegen, X)) # discrim on genatr data
Dloss(X,Y) = -mean(log.(D_Y_loss(Y)) .+ log.(1 .- D_G_X_loss(X)))
Gloss(X) = mean(log.(1 .- D_G_X_loss(X)))
MMDloss(X,Y) = size(Y,2) * mmd_flux(models["logsigma"], corrected_signal_instance(ricegen, X), Y) # m*MMD^2 on genatr data

# Global state
const logger = DataFrame(
    :epoch   => Int[], # mandatory field
    :dataset => Symbol[], # mandatory field
    :time    => Union{Float64, Missing}[],
    :Gloss   => Union{eltype(phys), Missing}[],
    :Dloss   => Union{eltype(phys), Missing}[],
    :D_Y     => Union{eltype(phys), Missing}[],
    :D_G_X   => Union{eltype(phys), Missing}[],
    :MMDsq   => Union{eltype(phys), Missing}[],
    :MMDvar  => Union{eltype(phys), Missing}[],
    :tstat   => Union{eltype(phys), Missing}[],
    :c_alpha => Union{eltype(phys), Missing}[],
    :P_alpha => Union{eltype(phys), Missing}[],
    :rmse    => Union{eltype(phys), Missing}[],
    :theta_fit_err => Union{Vector{eltype(phys)}, Missing}[],
    :signal_fit_logL => Union{eltype(phys), Missing}[],
    :signal_fit_rmse => Union{eltype(phys), Missing}[],
)
const optimizers = Dict{String,Any}(
    "G"   => Flux.ADAM(settings["opt"]["G"]["lr"]),
    "D"   => Flux.ADAM(settings["opt"]["D"]["lr"]),
    "mmd" => Flux.ADAM(settings["opt"]["mmd"]["lr"]),
)
const cb_state = Dict{String,Any}()

####
#### Training
####

using PyCall
using MMDLearning.Ignite

const torch = pyimport("torch")
const logging = pyimport("logging")
const ignite = pyimport("ignite")
# const torchvision  pyimport("torchvision")
# const transforms = pyimport("torchvision.transforms")
# const datasets = pyimport("torchvision.datasets")
# const optim = pyimport("torch.optim")
# const nn = pyimport("torch.nn")
# const F = pyimport("torch.nn.functional")
# const ProgressBar = ignite.contrib.handlers.ProgressBar
# const Timer = ignite.handlers.Timer
# const RunningAverage = ignite.metrics.RunningAverage

function train_step(engine, batch)
    @unpack kernelrate, kernelsteps, GANrate, GANsucc, Dsteps = settings["train"]
    _, Xtrain, Ytrain = array.(batch)

    @timeit "train batch" begin
        if settings["arch"]["type"] ∈ ["mmd", "hyb"]
            # Train MMD kernel bandwidths
            if mod(engine.state.iteration-1, kernelrate) == 0
                @timeit "MMD kernel" begin
                    @timeit "sample G(X)" X̂train = corrected_signal_instance(ricegen, Xtrain)
                    for _ in 1:kernelsteps
                        success = train_kernel_bandwidth_flux!(models["logsigma"], X̂train, Ytrain;
                            kernelloss = settings["opt"]["k"]["loss"], kernellr = settings["opt"]["k"]["lr"], bwbounds = settings["arch"]["kernel"]["bwbounds"]) # timed internally
                        !success && break
                    end
                end
            end
            # Train generator on MMD loss
            @timeit "MMD generator" begin
                @timeit "forward" _, back = Zygote.pullback(() -> MMDloss(Xtrain, Ytrain), Flux.params(models["G"]))
                @timeit "reverse" gs = back(one(eltype(phys)))
                @timeit "update!" Flux.Optimise.update!(optimizers["mmd"], Flux.params(models["G"]), gs)
            end
        end

        if settings["arch"]["type"] ∈ ["gan", "hyb"]
            if settings["arch"]["type"] == "gan" || mod((engine.state.iteration-1) ÷ GANsucc, GANrate) == 0
                @timeit "GAN discriminator" for _ in 1:Dsteps
                    @timeit "forward" _, back = Zygote.pullback(() -> Dloss(Xtrain, Ytrain), Flux.params(models["D"]))
                    @timeit "reverse" gs = back(one(eltype(phys)))
                    @timeit "update!" Flux.Optimise.update!(optimizers["D"], Flux.params(models["D"]), gs)
                end
                @timeit "GAN generator" begin
                    @timeit "forward" _, back = Zygote.pullback(() -> Gloss(Xtrain), Flux.params(models["G"]))
                    @timeit "reverse" gs = back(one(eltype(phys)))
                    @timeit "update!" Flux.Optimise.update!(optimizers["G"], Flux.params(models["G"]), gs)
                end
            end
        end
    end

    return nothing
end

function eval_metrics(engine, batch)
    @timeit "eval batch" begin
        # Update callback state
        @timeit "update cb state" let
            cb_state["θ"], cb_state["Xθ"], cb_state["Y"] = array.(batch)
            if hasclosedform(phys)
                cb_state["Yθ"] = signal_model(ClosedForm(phys), cb_state["θ"])
                cb_state["Yθhat"] = signal_model(ClosedForm(phys), cb_state["θ"], epsilon(ClosedForm(phys)))
            end
            update_callback!(cb_state, phys, ricegen; ninfer = settings["eval"]["ninfer"], inferperiod = settings["eval"]["inferperiod"])
        end

        # Initialize metrics dictionary
        metrics = Dict{Any,Any}()
        metrics[:epoch]   = :val ∉ logger.dataset ? 0 : logger.epoch[findlast(d -> d === :val, logger.dataset)] + 1
        metrics[:dataset] = :val
        metrics[:time]    = cb_state["curr_time"] - cb_state["last_time"]

        # Metrics computed in update_callback!
        metrics[:rmse] = cb_state["metrics"]["rmse"]
        metrics[:theta_fit_err]   = cb_state["metrics"]["θ_fit_err"]
        metrics[:signal_fit_logL] = cb_state["metrics"]["signal_fit_logL"]
        metrics[:signal_fit_rmse] = cb_state["metrics"]["signal_fit_rmse"]

        # Perform permutation test
        if settings["arch"]["type"] ∈ ["mmd", "hyb"]
            @timeit "perm test" let
                m = settings["train"]["batchsize"] # training batch size, not size of val set
                cb_state["permtest"] = mmd_perm_test_power(models["logsigma"], MMDLearning.sample_columns(cb_state["Xθhat"], m), MMDLearning.sample_columns(cb_state["Y"], m); nperms = settings["eval"]["nperms"])
                metrics[:MMDsq]   = m * cb_state["permtest"].MMDsq
                metrics[:MMDvar]  = m^2 * cb_state["permtest"].MMDvar
                metrics[:tstat]   = cb_state["permtest"].MMDsq / cb_state["permtest"].MMDσ
                metrics[:c_alpha] = cb_state["permtest"].c_alpha
                metrics[:P_alpha] = cb_state["permtest"].P_alpha_approx
            end
        end

        # Compute GAN losses
        if settings["arch"]["type"] ∈ ["gan", "hyb"]
            @timeit "gan losses" let
                d_y = D_Y_loss(cb_state["Y"])
                d_g_x = D_G_X_loss(cb_state["Xθhat"])
                metrics[:Gloss] = mean(log.(1 .- d_g_x))
                metrics[:Dloss] = -mean(log.(d_y) .+ log.(1 .- d_g_x))
                metrics[:D_Y]   = mean(d_y)
                metrics[:D_G_X] = mean(d_g_x)
            end
        end

        # Update logger dataframe
        push!(logger, metrics; cols = :subset)
    end

    return deepcopy(metrics) #TODO convert to PyDict?
end

function makeplots(;showplot = false)
    try
        Dict{Symbol, Any}(
            :gan       => settings["arch"]["type"] ∈ ["gan", "hyb"] ? MMDLearning.plot_gan_loss(logger, cb_state, phys; showplot = showplot, lrdroprate = settings["opt"]["lrrate"], lrdrop = settings["opt"]["lrdrop"]) : nothing,
            :ricemodel => MMDLearning.plot_rician_model(logger, cb_state, phys; showplot = showplot, bandwidths = haskey(models, "logsigma") ? permutedims(models["logsigma"]) : nothing),
            :signals   => MMDLearning.plot_rician_signals(logger, cb_state, phys; showplot = showplot),
            :mmd       => settings["arch"]["type"] ∈ ["mmd", "hyb"] ? MMDLearning.plot_mmd_losses(logger, cb_state, phys; showplot = showplot, lrdroprate = settings["opt"]["lrrate"], lrdrop = settings["opt"]["lrdrop"]) : nothing,
            :infer     => MMDLearning.plot_rician_inference(logger, cb_state, phys; showplot = showplot),
            :witness   => nothing, #mmd_witness(Xϵ, Y, sigma)
            :heat      => nothing, #mmd_heatmap(Xϵ, Y, sigma)
            :perm      => settings["arch"]["type"] ∈ ["mmd", "hyb"] ? mmd_perm_test_power_plot(cb_state["permtest"]; showplot = showplot) : nothing,
        )
    catch e
        handleinterrupt(e; msg = "Error plotting")
    end
end

function make_data_tuples(dataset)
    θ = sampleθ(phys, :all; dataset = dataset)
    X = sampleX(phys, :all; dataset = dataset)
    Y = sampleY(phys, :all; dataset = dataset)
    return [(θ[:,j], X[:,j], Y[:,j]) for j in 1:size(Y,2)]
end
train_loader = torch.utils.data.DataLoader(make_data_tuples(:train); batch_size = settings["train"]["batchsize"], shuffle = true, drop_last = true)
val_loader = torch.utils.data.DataLoader(make_data_tuples(:val); batch_size = settings["data"]["nval"], shuffle = false, drop_last = false)

trainer = ignite.engine.Engine(@j2p train_step)
evaluator = ignite.engine.Engine(@j2p eval_metrics)

# Compute callback metrics
trainer.add_event_handler(
    ignite.engine.Events.STARTED | ignite.engine.Events.EPOCH_COMPLETED,
    @j2p function (engine)
        evaluator.run(val_loader)
    end
)

# Checkpoint current model + logger + make plots
trainer.add_event_handler(
    ignite.engine.Events.EPOCH_COMPLETED(event_filter = @j2p event_throttler(settings["eval"]["saveperiod"])),
    @j2p function (engine)
        @timeit "checkpoint" begin
            @timeit "save current model" saveprogress(@dict(models, optimizers, logger); savefolder = settings["data"]["out"], prefix = "current-")
            @timeit "make current plots" plothandles = makeplots()
            @timeit "save current plots" saveplots(plothandles; savefolder = settings["data"]["out"], prefix = "current-")
        end
    end
)

# Check for + save best model + logger + make plots
trainer.add_event_handler(
    ignite.engine.Events.EPOCH_COMPLETED,
    @j2p function (engine)
        losses = logger.signal_fit_logL[logger.dataset .=== :val] |> skipmissing |> collect
        if !isempty(losses) && (length(losses) == 1 || losses[end] < minimum(losses[1:end-1]))
            @timeit "save best progress" begin
                @timeit "save best model" saveprogress(@dict(models, optimizers, logger); savefolder = settings["data"]["out"], prefix = "best-")
                @timeit "make best plots" plothandles = makeplots()
                @timeit "save best plots" saveplots(plothandles; savefolder = settings["data"]["out"], prefix = "best-")
            end
        end
    end
)

# Drop learning rate
trainer.add_event_handler(
    ignite.engine.Events.EPOCH_COMPLETED,
    @j2p function (engine)
        @unpack lrrate, lrdrop, lrthresh = settings["opt"]
        epoch = engine.state.epoch
        if epoch > 1 && mod(epoch-1, lrrate) == 0
            for (optname, opt) in optimizers
                new_eta = opt.eta / lrdrop
                if new_eta >= lrthresh
                    @info "$epoch: Dropping $optname optimizer learning rate to $new_eta"
                    opt.eta = new_eta
                else
                    @info "$epoch: Learning rate dropped below $lrthresh for $optname optimizer, exiting..."
                    throw(InterruptException())
                end
            end
        end
    end
)

# Print TimerOutputs timings
trainer.add_event_handler(
    ignite.engine.Events.EPOCH_COMPLETED,
    @j2p function (engine)
        if mod(engine.state.epoch-1, settings["eval"]["showrate"]) == 0
            show(stdout, TimerOutputs.get_defaulttimer()); println("\n")
            show(stdout, last(logger[:, Not(:theta_fit_err)], 10)); println("\n")
        end
        (engine.state.epoch == 1) && TimerOutputs.reset_timer!() # throw out compilation timings
    end
)

# Timeout
trainer.add_event_handler(
    ignite.engine.Events.EPOCH_COMPLETED(event_filter = @j2p run_timeout(settings["train"]["timeout"])),
    @j2p function (engine)
        @info "Exiting: training time exceeded $(DECAES.pretty_time(settings["train"]["timeout"]))"
        engine.terminate()
    end
)

# Setup loggers
trainer.logger = ignite.utils.setup_logger("trainer")
evaluator.logger = ignite.utils.setup_logger("evaluator")

# Run trainer
trainer.run(train_loader, max_epochs = settings["train"]["epochs"])
