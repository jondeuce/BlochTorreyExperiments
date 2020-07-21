####
#### Setup
####

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MMDLearning
using PyCall
pyplot(size=(800,600))

const settings = TOML.parse("""
    [data]
        out    = "./output/ignite-cvae-tmp"
        ntrain = 102_400
        ntest  = 10_240
        nval   = 10_240

    [train]
        timeout   = 1e9
        epochs    = 999_999
        batchsize = 1024

    [eval]
        saveperiod = 30.0 # TODO
        showrate   = 1 # TODO

    [opt]
        lrdrop   = 1.0
        lrthresh = 1e-5
        lrrate   = 1000
        [opt.cvae]
            lr = 1e-4 # default for optimizers below

    [arch]
        physics = "toy" # "toy" or "mri"
        nlatent = 1 # number of latent variables Z
        zdim    = 4 # embedding dimension of z
        hdim    = 64 # default for models below
        nhidden = 2 # default for models below
        [arch.G]
            hdim        = 0
            nhidden     = 0
            maxcorr     = 0.0 # unset by default; correction amplitude
            noisebounds = [0.0, 0.0] # unset by default; noise amplitude
        [arch.E1]
            hdim    = 0
            nhidden = 0
        [arch.E2]
            hdim    = 0
            nhidden = 0
        [arch.D]
            hdim    = 0
            nhidden = 0
""")

Ignite.parse_command_line!(settings)
Ignite.compare_and_set!(settings["arch"]["G"], "maxcorr",            0.0, settings["arch"]["physics"] == "toy" ?          0.1 :       0.025 )
Ignite.compare_and_set!(settings["arch"]["G"], "noisebounds", [0.0, 0.0], settings["arch"]["physics"] == "toy" ? [-8.0, -2.0] : [-6.0, -3.0])
Ignite.compare_and_set!.([settings["arch"][k] for k in ("G","E1","E2","D")], "hdim",    0, settings["arch"]["hdim"])
Ignite.compare_and_set!.([settings["arch"][k] for k in ("G","E1","E2","D")], "nhidden", 0, settings["arch"]["nhidden"])
Ignite.save_and_print(settings; outpath = settings["data"]["out"], filename = "settings.toml")

# Initialize generator + discriminator + kernel
function make_models(phys)
    models = Dict{String, Any}()
    n   = nsignal(phys) # input signal length
    nθ  = ntheta(phys) # number of physics variables
    θbd = θbounds(phys)
    k   = settings["arch"]["nlatent"]::Int # number of latent variables Z
    nz  = settings["arch"]["zdim"]::Int # embedding dimension
    toT(m) = Flux.paramtype(eltype(phys), m)

    # Rician generator. First `n` elements for `δX` scaled to (-δ, δ), second `n` elements for `logϵ` scaled to (noisebounds[1], noisebounds[2])
    models["G"] = let
        hdim = settings["arch"]["G"]["hdim"]::Int
        nhidden = settings["arch"]["G"]["nhidden"]::Int
        maxcorr = settings["arch"]["G"]["maxcorr"]::Float64
        noisebounds = settings["arch"]["G"]["noisebounds"]::Vector{Float64}
        Flux.Chain(
            MMDLearning.MLP(n + k => 2n, nhidden, hdim, Flux.relu, tanh)...,
            # MMDLearning.MLP(k => 2n, nhidden, hdim, Flux.relu, tanh)..., #TODO
            MMDLearning.CatScale([(-maxcorr, maxcorr), (noisebounds...,)], [n,n]),
        ) |> toT
    end

    # Encoders
    models["E1"] = let
        hdim = settings["arch"]["E1"]["hdim"]::Int
        nhidden = settings["arch"]["E1"]["nhidden"]::Int
        MMDLearning.MLP(n => 2*nz, nhidden, hdim, Flux.relu, identity) |> toT
    end

    models["E2"] = let
        hdim = settings["arch"]["E2"]["hdim"]::Int
        nhidden = settings["arch"]["E2"]["nhidden"]::Int
        MMDLearning.MLP(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity) |> toT
    end

    # Decoder
    models["D"] = let
        hdim = settings["arch"]["D"]["hdim"]::Int
        nhidden = settings["arch"]["D"]["nhidden"]::Int
        Flux.Chain(
            MMDLearning.MLP(n + nz => 2*(nθ + k), nhidden, hdim, Flux.relu, identity)...,
            MMDLearning.CatScale(eltype(θbd)[θbd; (-1, 1)], [ones(Int, nθ); k + nθ + k]),
        ) |> toT
    end

    return models
end

phys = initialize!(
    ToyModel{Float32,true}();
    ntrain = settings["data"]["ntrain"]::Int,
    ntest = settings["data"]["ntest"]::Int,
    nval = settings["data"]["nval"]::Int,
)
models = make_models(phys)
ricegen = VectorRicianCorrector(models["G"]) # Generator produces 𝐑^2n outputs parameterizing n Rician distributions
# ricegen = LatentVectorRicianCorrector(models["G"])
MMDLearning.model_summary(models)

# Helpers
split_mean_std(μ::Matrix) = (μ[1:end÷2,:], μ[end÷2+1:end,:])
split_theta_latent(μ::Matrix) = (μ[1:ntheta(phys),:], μ[ntheta(phys)+1:end,:])
sample_mv_normal(μ0::Matrix{T}, σ::Matrix{T}) where {T} = μ0 .+ σ .* randn(T, max.(size(μ0), size(σ)))
@inline square(x) = x*x

function InvertY(Y)
    μr = models["E1"](Y)
    zr = sample_mv_normal(split_mean_std(μr)...)
    μx = models["D"](vcat(Y,zr))
    x  = sample_mv_normal(split_mean_std(μx)...)
    θ, Z = split_theta_latent(x)
    return θ, Z
end

function DataConsistency(Y, μG0, σG)
    # Rician negative log likelihood
    σG2 = square.(σG)
    YlogL = -sum(@. log(Y / σG2) + MMDLearning._logbesseli0(Y * μG0 / σG2) - (Y^2 + μG0^2) / (2 * σG2))
    # YlogL = sum(@. log(σG2) + square(Y - μG0) / σG2) / 2 # Gaussian likelihood for testing
    return YlogL
end

function KLdivergence(μq0, σq, μr0, σr)
    σr2, σq2 = square.(σr), square.(σq)
    KLdiv = sum(@. (σq2 + square(μr0 - μq0)) / σr2 + log(σr2 / σq2)) / 2 # KL-divergence contribution to cross-entropy (Note: dropped constant -Zdim/2 term)
    return KLdiv
end

function EvidenceLowerBound(x, μx0, σx)
    σx2 = square.(σx)
    ELBO = sum(@. square(x - μx0) / σx2 + log(σx2)) / 2 # Negative log-likelihood/ELBO contribution to cross-entropy (Note: dropped constant +Zdim*log(2π)/2 term)
    return ELBO
end

# Self-supervised CVAE loss
function SelfCVAEloss(Y; recover_Z = false)
    # Invert Y
    θ, Z = InvertY(Y)

    # Limit information capacity of Z with ℓ2 regularization
    #   - Equivalently, as 1/2||Z||^2 is the negative log likelihood of Z ~ N(0,1) (dropping normalization factor)
    Zreg = recover_Z ?
        sum(abs2, Z) / 2 :
        zero(eltype(Z))

    # Drop gradients for θ and Z, and compute uncorrected X from physics model
    θ = Zygote.dropgrad(θ)
    Z = Zygote.dropgrad(Z)
    # X = Zygote.ignore() do
    #     signal_model(phys, θ)
    # end
    X = signal_model(phys, θ)

    # Corrected X̂ instance
    μG0, σG = rician_params(ricegen, X, Z)
    X̂ = add_noise_instance(ricegen, μG0, σG)

    # Rician negative log likelihood
    YlogL = DataConsistency(Y, μG0, σG)

    # Cross-entropy loss function
    μr0, σr = split_mean_std(models["E1"](Y)) #TODO X̂ or Y?
    μq0, σq = split_mean_std(models["E2"](vcat(X̂,θ,Z)))
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_std(models["D"](vcat(Y,zq))) #TODO X̂ or Y?

    KLdiv = KLdivergence(μq0, σq, μr0, σr)
    ELBO = recover_Z ?
        EvidenceLowerBound(vcat(θ,Z), μx0, σx) :
        EvidenceLowerBound(θ, μx0[1:ntheta(phys),:], σx[1:ntheta(phys),:])

    Nbatch = size(Y,2)
    ℓ = (Zreg + YlogL + KLdiv + ELBO) / Nbatch

    return ℓ
end

# Global state
const logger = DataFrame(
    :epoch   => Int[], # mandatory field
    :dataset => Symbol[], # mandatory field
    :time    => Union{Float64, Missing}[],
    :loss    => Union{eltype(phys), Missing}[],
    :Zreg    => Union{eltype(phys), Missing}[],
    :YlogL   => Union{eltype(phys), Missing}[],
    :KLdiv   => Union{eltype(phys), Missing}[],
    :ELBO    => Union{eltype(phys), Missing}[],
    :rmse    => Union{eltype(phys), Missing}[],
    :theta_fit_err   => Union{Vector{eltype(phys)}, Missing}[],
    :Z_fit_err       => Union{Vector{eltype(phys)}, Missing}[],
    :signal_fit_logL => Union{eltype(phys), Missing}[],
    :signal_fit_rmse => Union{eltype(phys), Missing}[],
)
const optimizers = Dict{String,Any}(
    "cvae" => Flux.ADAM(settings["opt"]["cvae"]["lr"]),
)
const cb_state = Dict{String,Any}()

####
#### Training
####

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
    Ytrain = Ignite.array(only(batch))

    @timeit "train batch" begin
        ps = Flux.params(values(models)...)
        @timeit "forward" _, back = Zygote.pullback(() -> SelfCVAEloss(Ytrain), ps)
        @timeit "reverse" gs = back(one(eltype(phys)))
        @timeit "update!" Flux.Optimise.update!(optimizers["cvae"], ps, gs)
    end

    return nothing
end

function eval_metrics(engine, batch)
    @timeit "eval batch" begin
        # Update callback state
        cb_state["last_time"] = get!(cb_state, "curr_time", time())
        cb_state["curr_time"] = time()
        cb_state["metrics"] = Dict{String,Any}()

        # Invert Y and make Xs
        Y = Ignite.array(only(batch))
        Nbatch = size(Y,2)
        θ, Z = InvertY(Y)
        X = signal_model(phys, θ)
        δG0, σG = correction_and_noiselevel(ricegen, X, Z)
        μG0 = @. abs(X + δG0)
        X̂ = add_noise_instance(ricegen, μG0, σG)

        # Cross-entropy loss function
        μr0, σr = split_mean_std(models["E1"](Y)) #TODO X̂ or Y?
        μq0, σq = split_mean_std(models["E2"](vcat(X̂,θ,Z)))
        zq = sample_mv_normal(μq0, σq)
        μx0, σx = split_mean_std(models["D"](vcat(Y,zq))) #TODO X̂ or Y?

        Zreg = sum(abs2, Z) / (2*Nbatch)
        YlogL = DataConsistency(Y, μG0, σG) / Nbatch
        KLdiv = KLdivergence(μq0, σq, μr0, σr) / Nbatch
        ELBO = EvidenceLowerBound(vcat(θ,Z), μx0, σx) / Nbatch
        loss = Zreg + YlogL + KLdiv + ELBO
        @pack! cb_state["metrics"] = Zreg, YlogL, KLdiv, ELBO, loss

        # Compute signal correction, noise instances, etc.
        cb_state["θ"], cb_state["Xθ"], cb_state["Y"] = θ, X, Y
        cb_state["Yθ"] = hasclosedform(phys) ? signal_model(ClosedForm(phys), cb_state["θ"]) : missing
        cb_state["Yθhat"] = hasclosedform(phys) ? signal_model(ClosedForm(phys), cb_state["θ"], noiselevel(ClosedForm(phys))) : missing
        let
            δθ, ϵθ, Xθδ, Xθhat = δG0, σG, μG0, X̂
            @pack! cb_state = δθ, ϵθ, Xθδ, Xθhat
        end

        # Compute signal correction, noise instances, etc.
        let
            Yfit, θfit, Zfit = Y, θ, Z
            Xθfit = signal_model(phys, θfit)
            δθfit, ϵθfit = correction_and_noiselevel(ricegen, Xθfit, Zfit)
            Xθδfit = abs.(Xθfit .+ δθfit)
            Xθhatfit = add_noise_instance(ricegen, Xθδfit, ϵθfit)
            Yθfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θ) : missing
            Yθhatfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θ, noiselevel(ClosedForm(phys))) : missing
            @pack! cb_state = Yfit, θfit, Zfit, Xθfit, δθfit, ϵθfit, Xθδfit, Xθhatfit, Yθfit, Yθhatfit
        end

        # Compute error metrics
        let
            @unpack Yfit, θfit, Zfit, Yθfit, Xθhatfit, Xθδfit, ϵθfit = cb_state
            rmse = hasclosedform(phys) ? sqrt(mean(abs2, Yθfit - Xθδfit)) : missing
            all_signal_fit_rmse = sqrt.(mean(abs2, Yfit .- Xθhatfit; dims = 1)) |> vec
            all_signal_fit_logL = .-sum(logpdf.(Rician.(Xθδfit, ϵθfit), Yfit); dims = 1) |> vec
            signal_fit_rmse = mean(all_signal_fit_rmse)
            signal_fit_logL = mean(all_signal_fit_logL)
            θsamp, Zsamp = split_theta_latent(sample_mv_normal(μx0, σx))
            θ_fit_err = mean(θerror(phys, θ, θsamp); dims = 2) |> vec |> copy
            Z_fit_err = mean(abs, Z .- Zsamp; dims = 2) |> vec |> copy
            @pack! cb_state["metrics"] = rmse, all_signal_fit_rmse, all_signal_fit_logL, signal_fit_rmse, signal_fit_logL, θ_fit_err, Z_fit_err
        end

        # Initialize output metrics dictionary
        metrics = Dict{Any,Any}()
        metrics[:epoch]   = :val ∉ logger.dataset ? 0 : logger.epoch[findlast(d -> d === :val, logger.dataset)] + 1
        metrics[:dataset] = :val
        metrics[:time]    = cb_state["curr_time"] - cb_state["last_time"]

        # Metrics computed in update_callback!
        metrics[:loss]  = cb_state["metrics"]["loss"]
        metrics[:Zreg]  = cb_state["metrics"]["Zreg"]
        metrics[:YlogL] = cb_state["metrics"]["YlogL"]
        metrics[:KLdiv] = cb_state["metrics"]["KLdiv"]
        metrics[:ELBO]  = cb_state["metrics"]["ELBO"]
        metrics[:rmse]  = cb_state["metrics"]["rmse"]
        metrics[:theta_fit_err]   = cb_state["metrics"]["θ_fit_err"]
        metrics[:Z_fit_err]       = cb_state["metrics"]["Z_fit_err"]
        metrics[:signal_fit_logL] = cb_state["metrics"]["signal_fit_logL"]
        metrics[:signal_fit_rmse] = cb_state["metrics"]["signal_fit_rmse"]

        # Update logger dataframe
        push!(logger, metrics; cols = :subset)

        return deepcopy(metrics) #TODO convert to PyDict?
    end
end

function makeplots(;showplot = false)
    try
        Dict{Symbol, Any}(
            :ricemodel => MMDLearning.plot_rician_model(logger, cb_state, phys; showplot = showplot, bandwidths = haskey(models, "logsigma") ? permutedims(models["logsigma"]) : nothing),
            :signals   => MMDLearning.plot_rician_signals(logger, cb_state, phys; showplot = showplot),
            :infer     => MMDLearning.plot_rician_inference(logger, cb_state, phys; showplot = showplot),
        )
    catch e
        handleinterrupt(e; msg = "Error plotting")
    end
end

make_data_tuples(dataset) = tuple.(copy.(eachcol(sampleY(phys, :all; dataset = dataset))))
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
            show(stdout, last(logger, 10)); println("\n")
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
