# Load files
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MMDLearning
pyplot(size=(800,600))
const settings = load_settings(joinpath(@__DIR__, "..", "settings", "default_settings.toml"))

# Initialize generator + discriminator + kernel
function make_models(phys)
    models = Dict{String, Any}()
    n = nsignal(phys)

    # Rician generator. First `n` elements for `δX` scaled to (-δ, δ), second `n` elements for `logϵ` scaled to (logϵ_bw[1], logϵ_bw[2])
    models["G"] = let
        hdim = settings["genatr"]["hdim"]::Int
        nhidden = settings["genatr"]["nhidden"]::Int
        maxcorr = settings["genatr"]["maxcorr"]::Float64
        noisebounds = settings["genatr"]["noisebounds"]::Vector{Float64}
        Flux.Chain(
            MMDLearning.MLP(n => 2n, nhidden, hdim, Flux.relu, tanh)...,
            MMDLearning.CatScale([(-maxcorr, maxcorr), (noisebounds...,)], [n,n]),
        ) |> Flux.f64
    end

    # Discriminator
    models["D"] = let
        hdim = settings["discrim"]["hdim"]::Int
        nhidden = settings["discrim"]["nhidden"]::Int
        MMDLearning.MLP(n => 1, nhidden, hdim, Flux.relu, Flux.sigmoid) |> Flux.f64
    end

    # Initialize `nbandwidth` linearly spaced kernel bandwidths `logσ` for each `n` channels strictly within the range (bwbounds[1], bwbounds[2])
    models["logsigma"] = let
        bwbounds = settings["kernel"]["bwbounds"]::Vector{Float64}
        nbandwidth = settings["kernel"]["nbandwidth"]::Int
        repeat(range(bwbounds...; length = nbandwidth+2)[2:end-1], 1, n) |> Matrix{Float64}
    end

    return models
end

const phys = initialize!(
    ToyModel{Float64}();
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
    :Gloss   => Union{Float64, Missing}[],
    :Dloss   => Union{Float64, Missing}[],
    :D_Y     => Union{Float64, Missing}[],
    :D_G_X   => Union{Float64, Missing}[],
    :MMDsq   => Union{Float64, Missing}[],
    :MMDvar  => Union{Float64, Missing}[],
    :tstat   => Union{Float64, Missing}[],
    :c_alpha => Union{Float64, Missing}[],
    :P_alpha => Union{Float64, Missing}[],
    :rmse    => Union{Float64, Missing}[],
    :theta_err => Union{Vector{Float64}, Missing}[],
    :Xhat_logL => Union{Float64, Missing}[],
    :Xhat_rmse => Union{Float64, Missing}[],
)
const optimizers = Dict{String,Any}(
    "G"   => Flux.ADAM(settings["opt"]["G"]["lr"]),
    "D"   => Flux.ADAM(settings["opt"]["D"]["lr"]),
    "mmd" => Flux.ADAM(settings["opt"]["mmd"]["lr"]),
)
const cb_state = initialize_callback(phys; nsamples = settings["train"]["batchsize"]) #TODO

function callback(epoch;
        m           :: Int     = settings["train"]["batchsize"],
        saveperiod  :: Float64 = settings["eval"]["saveperiod"],
        inferperiod :: Float64 = settings["eval"]["inferperiod"],
        ninfer      :: Int     = settings["eval"]["ninfer"],
        nperms      :: Int     = settings["eval"]["nperms"],
        lrthresh    :: Float64 = settings["opt"]["lrthresh"],
        lrdrop      :: Float64 = settings["opt"]["lrdrop"],
        lrdroprate  :: Int     = settings["opt"]["lrrate"],
        outfolder   :: String  = settings["data"]["out"],
    )

    # Update callback state
    @timeit "update cb state" begin
        update_callback!(cb_state, phys, ricegen; ninfer = ninfer, inferperiod = inferperiod)
    end

    # Initialize metrics dictionary
    metrics = Dict{Any,Any}()
    metrics[:epoch]   = epoch
    metrics[:dataset] = :val
    metrics[:time]    = cb_state["curr_time"] - cb_state["last_time"]

    # Perform permutation test
    @timeit "perm test" begin
        permtest = mmd_perm_test_power(models["logsigma"], MMDLearning.sample_columns(cb_state["Xθhat"], m), MMDLearning.sample_columns(cb_state["Y"], m); nperms = nperms)
        metrics[:MMDsq]   = m * permtest.MMDsq
        metrics[:MMDvar]  = m^2 * permtest.MMDvar
        metrics[:tstat]   = permtest.MMDsq / permtest.MMDσ
        metrics[:c_alpha] = permtest.c_alpha
        metrics[:P_alpha] = permtest.P_alpha_approx
    end

    # Compute GAN losses
    d_y = D_Y_loss(cb_state["Y"])
    d_g_x = D_G_X_loss(cb_state["Xθhat"])
    metrics[:Gloss] = mean(log.(1 .- d_g_x))
    metrics[:Dloss] = -mean(log.(d_y) .+ log.(1 .- d_g_x))
    metrics[:D_Y]   = mean(d_y)
    metrics[:D_G_X] = mean(d_g_x)

    # Metrics computed in update_callback!
    metrics[:rmse] = cb_state["metrics"]["rmse"]
    metrics[:theta_err] = cb_state["metrics"]["theta_err"]
    metrics[:Xhat_logL] = cb_state["metrics"]["Xhat_logL"]
    metrics[:Xhat_rmse] = cb_state["metrics"]["Xhat_rmse"]

    # Update logger dataframe
    push!(logger, metrics; cols = :setequal)

    function makeplots(;showplot = false)
        try
            Dict{Symbol, Any}(
                :gan       => MMDLearning.plot_gan_loss(logger, cb_state, phys; showplot = showplot, lrdroprate = lrdroprate, lrdrop = lrdrop),
                :ricemodel => MMDLearning.plot_rician_model(logger, cb_state, phys; showplot = showplot, bandwidths = permutedims(models["logsigma"])),
                :signals   => MMDLearning.plot_rician_signals(logger, cb_state, phys; showplot = showplot),
                :mmd       => MMDLearning.plot_mmd_losses(logger, cb_state, phys; showplot = showplot, lrdroprate = lrdroprate, lrdrop = lrdrop),
                :infer     => MMDLearning.plot_rician_inference(logger, cb_state, phys; showplot = showplot),
                :witness   => nothing, #mmd_witness(Xϵ, Y, sigma)
                :heat      => nothing, #mmd_heatmap(Xϵ, Y, sigma)
                :perm      => mmd_perm_test_power_plot(permtest; showplot = showplot),
            )
        catch e
            handleinterrupt(e; msg = "Error plotting")
        end
    end

    # Save current model + logger every `saveperiod` seconds
    if time() - cb_state["last_curr_checkpoint"] >= saveperiod
        cb_state["last_curr_checkpoint"] = time()
        @timeit "save current model" saveprogress(@dict(models, optimizers, logger); savefolder = outfolder, prefix = "current-")
        @timeit "make current plots" plothandles = makeplots()
        @timeit "save current plots" saveplots(plothandles; savefolder = outfolder, prefix = "current-")
    end

    # Check for and save best model + make best model plots every `saveperiod` seconds
    is_best_model = collect(skipmissing(logger.Xhat_logL)) |> x -> !isempty(x) && (x[end] <= minimum(x))
    if is_best_model
        @timeit "save best model" saveprogress(@dict(models, optimizers, logger); savefolder = outfolder, prefix = "best-")
        if time() - cb_state["last_best_checkpoint"] >= saveperiod
            cb_state["last_best_checkpoint"] = time()
            @timeit "make best plots" plothandles = makeplots()
            @timeit "save best plots" saveplots(plothandles; savefolder = outfolder, prefix = "best-")
        end
    end

    return deepcopy(metrics)
end

function train_hybrid_gan_model(;
        timeout    :: Float64 = settings["train"]["timeout"],
        epochs     :: Int     = settings["train"]["epochs"],
        m          :: Int     = settings["train"]["batchsize"],
        nbatches   :: Int     = settings["train"]["nbatches"],
        GANrate    :: Int     = settings["train"]["GANrate"],
        Dsteps     :: Int     = settings["train"]["Dsteps"],
        kernelsteps:: Int     = settings["train"]["kernelsteps"],
        kernelrate :: Int     = settings["train"]["kernelrate"],
        showrate   :: Int     = settings["eval"]["showrate"],
        lrdrop     :: Float64 = settings["opt"]["lrdrop"],
        lrthresh   :: Float64 = settings["opt"]["lrthresh"],
        lrdroprate :: Int     = settings["opt"]["lrrate"],
        kernelloss :: String  = settings["opt"]["k"]["loss"],
        kernellr   :: Float64 = settings["opt"]["k"]["lr"],
        bwbounds   :: Vector{Float64} = settings["kernel"]["bwbounds"],
    )
    TimerOutputs.reset_timer!()
    tstart = Dates.now()
    epoch0 = isempty(logger) ? 0 : logger.epoch[end]+1
    @timeit "initial callback" callback(epoch0)

    for loop_epoch in 1:epochs
        try
            epoch = epoch0 + loop_epoch
            @timeit "epoch" begin
                if mod(loop_epoch-1, kernelrate) == 0
                    @timeit "MMD kernel" while true
                        @timeit "sampleX" Xtrain = sampleX(phys, m; dataset = :train)
                        @timeit "sampleX̂" X̂train = corrected_signal_instance(ricegen, Xtrain)
                        @timeit "sampleY" Ytrain = sampleY(phys, m; dataset = :train)
                        atleastonestep = false
                        for _ in 1:kernelsteps
                            success = train_kernel_bandwidth_flux!(models["logsigma"], X̂train, Ytrain; kernelloss = kernelloss, kernellr = kernellr, bwbounds = bwbounds) # timed internally
                            !success && break
                            atleastonestep = true
                        end
                        atleastonestep && break
                    end
                end
                @timeit "batch loop" for _ in 1:nbatches
                    @timeit "MMD generator" begin
                        @timeit "sampleX" Xtrain = sampleX(phys, m; dataset = :train)
                        @timeit "sampleY" Ytrain = sampleY(phys, m; dataset = :train)
                        @timeit "forward" _, back = Zygote.pullback(() -> MMDloss(Xtrain, Ytrain), Flux.params(models["G"]))
                        @timeit "reverse" gs = back(1)
                        @timeit "update!" Flux.Optimise.update!(optimizers["mmd"], Flux.params(models["G"]), gs)
                    end
                    if mod(loop_epoch-1, GANrate) == 0
                        @timeit "GAN discriminator" for _ in 1:Dsteps
                            @timeit "sampleX" Xtrain = sampleX(phys, m; dataset = :train)
                            @timeit "sampleY" Ytrain = sampleY(phys, m; dataset = :train)
                            @timeit "forward" _, back = Zygote.pullback(() -> Dloss(Xtrain, Ytrain), Flux.params(models["D"]))
                            @timeit "reverse" gs = back(1)
                            @timeit "update!" Flux.Optimise.update!(optimizers["D"], Flux.params(models["D"]), gs)
                        end
                        @timeit "GAN generator" begin
                            @timeit "sampleX" Xtrain = sampleX(phys, m; dataset = :train)
                            @timeit "forward" _, back = Zygote.pullback(() -> Gloss(Xtrain), Flux.params(models["G"]))
                            @timeit "reverse" gs = back(1)
                            @timeit "update!" Flux.Optimise.update!(optimizers["G"], Flux.params(models["G"]), gs)
                        end
                    end
                end
                @timeit "callback" callback(epoch)
            end

            # Print timer and logger
            if mod(loop_epoch-1, showrate) == 0
                show(stdout, TimerOutputs.get_defaulttimer()); println("\n")
                show(stdout, last(logger[:, Not(:theta_err)], 10)); println("\n")
            end
            (loop_epoch == 1) && TimerOutputs.reset_timer!() # throw out compilation timings

            # Drop learning rate
            if loop_epoch > 1 && mod(loop_epoch-1, lrdroprate) == 0
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

            if Dates.now() - tstart >= Dates.Second(floor(Int, timeout))
                @info "Exiting: training time exceeded $(DECAES.pretty_time(timeout)) at epoch $epoch/$epochs"
                break
            end
        catch e
            if e isa InterruptException || e isa Flux.Optimise.StopException
                break
            else
                handleinterrupt(e; msg = "Error during training")
            end
        end
    end
    @info "Finished: trained for $(logger.epoch[end])/$epochs epochs"

    return nothing
end

train_hybrid_gan_model()

nothing
