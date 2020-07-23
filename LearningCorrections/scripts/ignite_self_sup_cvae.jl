####
#### Setup
####

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MMDLearning
using PyCall
pyplot(size=(800,600))

const torch = pyimport("torch")
const logging = pyimport("logging")
const ignite = pyimport("ignite")

const settings = TOML.parse("""
    [data]
        out    = "./output/ignite-cvae-$(MMDLearning.getnow())"
        ntrain = 102_400
        # ntrain = 10_240
        ntest  = 10_240
        nval   = 10_240

    [train]
        timeout   = 1e9
        epochs    = 999_999
        batchsize = 2048
        kernelrate  = 25 # Train kernel every `kernelrate` iterations
        kernelsteps = 10 # Gradient updates per kernel train

    [eval]
        saveperiod = 60.0 # TODO
        showrate   = 1 # TODO

    [opt]
        lrdrop   = 1.0
        lrthresh = 1e-5
        lrrate   = 1000
        [opt.pre]
            lr = 3e-4
        [opt.cvae]
            lr = 3e-4
        [opt.kernel]
            loss = "tstatistic" #"MMD"
            lr = 1e-2

    [arch]
        physics = "toy" # "toy" or "mri"
        nlatent = 1 # number of latent variables Z
        zdim    = 6 # embedding dimension of z
        hdim    = 32 # default for models below
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
        [arch.kernel]
            nbandwidth = 4
            bwbounds   = [0.0, 0.0] # unset by default; bounds for kernel bandwidths (logsigma)
""")

Ignite.parse_command_line!(settings)
Ignite.compare_and_set!(settings["arch"]["G"],      "maxcorr",     0.0,        settings["arch"]["physics"] == "toy" ? 0.1          : 0.025)
Ignite.compare_and_set!(settings["arch"]["G"],      "noisebounds", [0.0, 0.0], settings["arch"]["physics"] == "toy" ? [-8.0, -2.0] : [-6.0, -3.0])
Ignite.compare_and_set!(settings["arch"]["kernel"], "bwbounds",    [0.0, 0.0], settings["arch"]["physics"] == "toy" ? [-8.0, 4.0]  : [-10.0, 4.0])
Ignite.compare_and_set!.([settings["arch"][k] for k in ("G","E1","E2","D")], "hdim",    0, settings["arch"]["hdim"])
Ignite.compare_and_set!.([settings["arch"][k] for k in ("G","E1","E2","D")], "nhidden", 0, settings["arch"]["nhidden"])
Ignite.save_and_print(settings; outpath = settings["data"]["out"], filename = "settings.toml")

# Initialize generator + discriminator + kernel
function make_models(phys, Gtype::Type{<:RicianCorrector})
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
        nin  = Gtype <: Union{<:VectorRicianCorrector, <:FixedNoiseVectorRicianCorrector} ? n + k :
               Gtype <: Union{<:LatentVectorRicianCorrector, <:LatentVectorRicianNoiseCorrector} ? k :
               error("Unsupported corrector type: $Gtype")
        nout = Gtype <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? 2n :
               Gtype <: Union{<:FixedNoiseVectorRicianCorrector, <:LatentVectorRicianNoiseCorrector} ? n :
               error("Unsupported corrector type: $Gtype")
        OutputScale =
            Gtype <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? MMDLearning.CatScale([(-maxcorr, maxcorr), (noisebounds...,)], [n,n]) :
            Gtype <: FixedNoiseVectorRicianCorrector ? MMDLearning.CatScale([(-maxcorr, maxcorr)], [n]) :
            Gtype <: LatentVectorRicianNoiseCorrector ? MMDLearning.CatScale([(noisebounds...,)], [n]) :
            error("Unsupported corrector type: $Gtype")
        Flux.Chain(
            MMDLearning.MLP(nin => nout, nhidden, hdim, Flux.relu, tanh)...,
            OutputScale
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

    # MMD kernel bandwidths
    models["logsigma"] = let
        bwbounds = settings["arch"]["kernel"]["bwbounds"]::Vector{Float64}
        nbandwidth = settings["arch"]["kernel"]["nbandwidth"]::Int
        repeat(range(bwbounds...; length = nbandwidth+2)[2:end-1], 1, n) |> toT
    end

    return models
end

const phys = initialize!(
    ToyModel{Float32,true}();
    ntrain = settings["data"]["ntrain"]::Int,
    ntest = settings["data"]["ntest"]::Int,
    nval = settings["data"]["nval"]::Int,
)
const RiceGen = LatentVectorRicianCorrector #LatentVectorRicianNoiseCorrector #VectorRicianCorrector
# const models = make_models(phys, RiceGen)
const models = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/LearningCorrections/output/ignite-cvae-2020-07-23-T-14-19-58-922/current-models.bson")["models"]) #TODO
const ricegen = RiceGen(models["G"]) # Generator produces 𝐑^2n outputs parameterizing n Rician distributions
MMDLearning.model_summary(models, joinpath(settings["data"]["out"], "model-summary.txt"))

# Helpers
@inline split_theta_latent(θZ::Matrix) = size(θZ,1) == ntheta(phys) ? (θZ, similar(θZ,0,size(θZ,2))) : (θZ[1:ntheta(phys),:], θZ[ntheta(phys)+1:end,:])
@inline split_mean_std(μ::Matrix) = μ[1:end÷2, :], Flux.softplus.(μ[end÷2+1:end, :]) .+ sqrt(eps(eltype(μ))) #TODO Flux.softplus -> exp?
@inline sample_mv_normal(μ0::Matrix{T}, σ::Matrix{T}) where {T} = μ0 .+ σ .* randn(T, max.(size(μ0), size(σ)))
@inline pow2(x) = x*x

function InvertY(Y)
    μr = models["E1"](Y)
    μr0, σr = split_mean_std(μr)
    zr = sample_mv_normal(μr0, σr)

    μx = models["D"](vcat(Y,zr))
    μx0, σx = split_mean_std(μx)
    x = sample_mv_normal(μx0, σx)

    θ, Z = split_theta_latent(x)
    θ = clamp.(θ, θlower(phys), θupper(phys))
    return θ, Z
end

function sampleX̂(Y)
    θ, Z = InvertY(Y)
    X = signal_model(phys, θ)
    μG0, σG = rician_params(ricegen, X, Z)
    X̂ = add_noise_instance(ricegen, μG0, σG)
    return X̂
end

function DataConsistency(Y, μG0, σG)
    # Rician negative log likelihood
    # YlogL = -sum(@. MMDLearning._rician_logpdf(Y, μG0, σG))
    # YlogL = -sum(@. log(Y / σG2) + MMDLearning._logbesseli0(Y * μG0 / σG2) - (Y^2 + μG0^2) / (2 * σG2))
    # YlogL = sum(@. 2 * log(σG) + pow2((Y - μG0) / σG)) / 2 # Gaussian likelihood for testing
    YlogL = 100 * sum(abs2, Y .- add_noise_instance(ricegen, μG0, σG)) / 2 # L2 norm for testing/pretraining
    # YlogL = 10 * sum(abs, Y .- add_noise_instance(ricegen, μG0, σG)) # L1 norm for testing/pretraining
    return YlogL
end

# KL-divergence contribution to cross-entropy (Note: dropped constant -zdim/2 term)
@inline kl_div_kernel(μq0, σq, μr0, σr) = pow2(σq / σr) + pow2((μr0 - μq0) / σr) - 2 * log(σq / σr)
KLDivergence(μq0, σq, μr0, σr) = sum(@. kl_div_kernel(μq0, σq, μr0, σr)) / 2

# Negative log-likelihood/ELBO contribution to cross-entropy (Note: dropped constant +zdim*log(2π)/2 term)
@inline elbo_kernel(x, μx0, σx) = pow2((x - μx0) / σx) + 2 * log(σx)
EvidenceLowerBound(x, μx0, σx) = sum(@. elbo_kernel(x, μx0, σx)) / 2

# Maximum mean discrepency (m*MMD^2) loss
MMDloss(X̂,Y) = size(Y,2) * mmd_flux(models["logsigma"], X̂, Y)

function CVAEloss(Y, θ, Z; recover_Z = true)
    # Cross-entropy loss function
    μr0, σr = split_mean_std(models["E1"](Y)) #TODO X̂ or Y?
    μq0, σq = split_mean_std(models["E2"](vcat(Y,θ,Z))) #TODO X̂ or Y?
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_std(models["D"](vcat(Y,zq))) #TODO X̂ or Y?

    KLdiv = KLDivergence(μq0, σq, μr0, σr)
    ELBO = if recover_Z
        EvidenceLowerBound(vcat(θ,Z), μx0, σx)
    else
        μθ0 = split_theta_latent(μx0)[1]
        σθ  = split_theta_latent(σx)[1]
        EvidenceLowerBound(θ, μθ0, σθ)
    end
    Hloss = ELBO + KLdiv

    return Hloss
end

# Self-supervised CVAE loss
function SelfCVAEloss(Y; recover_Z = true)
    # Invert Y
    θ, Z = InvertY(Y)

    # Limit information capacity of Z with ℓ2 regularization
    #   - Equivalently, as 1/2||Z||^2 is the negative log likelihood of Z ~ N(0,1) (dropping normalization factor)
    Zreg = recover_Z ? sum(abs2, Z) / 2 : zero(eltype(Z))

    # # Drop gradients for θ, Z, and X
    # θ = Zygote.dropgrad(θ)
    # Z = Zygote.dropgrad(Z)
    # X = Zygote.dropgrad(signal_model(phys, θ))

    # Differentiate through physics model
    X = signal_model(phys, θ)

    # Rician negative log likelihood
    μG0, σG = rician_params(ricegen, X, Z)
    YlogL = DataConsistency(Y, μG0, σG) #TODO
    # YlogL = DataConsistency(Y, μG0, σG) + DataConsistency(Y, X, zero(eltype(Y))) #TODO

    # Corrected X̂ instance
    X̂ = add_noise_instance(ricegen, μG0, σG)

    # Add MMD loss contribution
    MMDsq = 100 * MMDloss(X̂, Y) #TODO

    # Drop gradients for θ, Z, and X̂
    θ = Zygote.dropgrad(θ)
    Z = Zygote.dropgrad(Z)
    X̂ = Zygote.dropgrad(X̂)
    Hloss = CVAEloss(Y, θ, Z; recover_Z = recover_Z)

    Nbatch = size(Y,2) #TODO
    ℓ = (Zreg + YlogL + Hloss) / Nbatch + MMDsq

    return ℓ
end

# Global state
const cb_state = Dict{String,Any}()
const logger = DataFrame(
    :epoch      => Int[], # mandatory field
    :dataset    => Symbol[], # mandatory field
    :time       => Union{Float64, Missing}[],
    :loss       => Union{eltype(phys), Missing}[],
    :Zreg       => Union{eltype(phys), Missing}[],
    :YlogL      => Union{eltype(phys), Missing}[],
    :KLdiv      => Union{eltype(phys), Missing}[],
    :ELBO       => Union{eltype(phys), Missing}[],
    :MMDsq      => Union{eltype(phys), Missing}[],
    :rmse       => Union{eltype(phys), Missing}[],
    :theta_err => Union{Vector{eltype(phys)}, Missing}[],
    :Z_err     => Union{Vector{eltype(phys)}, Missing}[],
    :Xhat_logL  => Union{eltype(phys), Missing}[],
    :Xhat_rmse  => Union{eltype(phys), Missing}[],
    :Yhat_logL  => Union{eltype(phys), Missing}[],
    :Yhat_rmse  => Union{eltype(phys), Missing}[],
)
const optimizers = Dict{String,Any}(
    "pre" => Flux.ADAM(settings["opt"]["cvae"]["lr"]),
    "cvae" => Flux.ADAM(settings["opt"]["cvae"]["lr"]),
)

####
#### Pre-training
####

function pre_train_step(engine, batch)
    Ytrain, θtrain, Ztrain = Ignite.array.(batch)

    @timeit "pre-train batch" begin
        ps = Flux.params(models["E1"], models["E2"], models["D"])
        @timeit "forward" ℓ, back = Zygote.pullback(() -> CVAEloss(Ytrain, θtrain, Ztrain; recover_Z = false), ps)
        @timeit "reverse" gs = back(one(eltype(phys)))
        @timeit "update!" Flux.Optimise.update!(optimizers["pre"], ps, gs)
    end

    return nothing
end

function pre_eval_metrics(engine, batch)
    @timeit "pre-eval batch" begin
        Y, θ, Z = Ignite.array.(batch)
        Nbatch = size(Y,2)

        # Cross-entropy loss function
        μr0, σr = split_mean_std(models["E1"](Y)) #TODO X̂ or Y?
        μq0, σq = split_mean_std(models["E2"](vcat(Y,θ,Z))) #TODO X̂ or Y?
        zq = sample_mv_normal(μq0, σq)
        μx0, σx = split_mean_std(models["D"](vcat(Y,zq))) #TODO X̂ or Y?

        KLdiv = KLDivergence(μq0, σq, μr0, σr) / Nbatch
        ELBO = EvidenceLowerBound(vcat(θ,Z), μx0, σx) / Nbatch
        loss = KLdiv + ELBO

        θfit, _ = InvertY(Y)
        Yfit = signal_model(phys, θfit)
        theta_err = mean(θerror(phys, θ, θfit); dims = 2) |> vec |> copy
        Y_fit_err = sqrt.(mean(abs2, Yfit .- Y; dims = 1)) |> mean

        # Initialize output metrics dictionary
        metrics = Dict{Any,Any}()
        metrics[:epoch]   = engine.state.iteration
        metrics[:dataset] = :val
        metrics[:loss]    = loss
        metrics[:KLdiv]   = KLdiv
        metrics[:ELBO]    = ELBO
        metrics[:theta_err] = theta_err
        metrics[:Y_fit_err] = Y_fit_err

        # Print progress
        if mod(engine.state.epoch-1, settings["eval"]["showrate"]) == 0
            show(stdout, TimerOutputs.get_defaulttimer()); println("\n")
            show(stdout, DataFrame(metrics)); println("\n")
        end
    end

    return deepcopy(metrics) #TODO convert to PyDict?
end

function make_pre_train_data_tuples(dataset)
    #TODO this should be done with some sort of fitting etc.
    θ = sampleθ(phys, :all; dataset = dataset)
    Y = signal_model(phys, θ, noiselevel(ClosedForm(phys)))
    Z = zeros(eltype(phys), settings["arch"]["nlatent"], size(θ,2))
    tuple.(copy.(eachcol(Y)), copy.(eachcol(θ)), copy.(eachcol(Z)))
end
pre_train_loader = torch.utils.data.DataLoader(make_pre_train_data_tuples(:train); batch_size = settings["train"]["batchsize"], shuffle = true, drop_last = true)
pre_val_loader = torch.utils.data.DataLoader(make_pre_train_data_tuples(:val); batch_size = settings["data"]["nval"], shuffle = false, drop_last = false)

pre_trainer = ignite.engine.Engine(@j2p pre_train_step)
pre_evaluator = ignite.engine.Engine(@j2p pre_eval_metrics)

# Force terminate
pre_trainer.add_event_handler(
    ignite.engine.Events.STARTED | ignite.engine.Events.ITERATION_STARTED | ignite.engine.Events.ITERATION_COMPLETED,
    @j2p function (engine)
        if isfile(joinpath(settings["data"]["out"], "stop.txt"))
            @info "Exiting: found file $(joinpath(settings["data"]["out"], "stop.txt"))"
            engine.terminate()
        end
    end
)

# Compute callback metrics
pre_trainer.add_event_handler(
    ignite.engine.Events.STARTED | ignite.engine.Events.EPOCH_COMPLETED,
    @j2p function (engine)
        pre_evaluator.run(pre_val_loader)
    end
)

# Timeout
pre_trainer.add_event_handler(
    ignite.engine.Events.EPOCH_COMPLETED(event_filter = @j2p run_timeout(settings["train"]["timeout"])),
    @j2p function (engine)
        @info "Exiting: training time exceeded $(DECAES.pretty_time(settings["train"]["timeout"]))"
        engine.terminate()
    end
)

# Setup loggers
pre_trainer.logger = ignite.utils.setup_logger("pre_trainer")
pre_evaluator.logger = ignite.utils.setup_logger("pre_evaluator")

# Run pre-trainer
#TODO pre_trainer.run(pre_train_loader, max_epochs = settings["train"]["epochs"])

#TODO drop lr?
#TODO saveprogress(@dict(models, optimizers); savefolder = settings["data"]["out"], prefix = "pretrain-")

####
#### Training
####

function train_step(engine, batch)
    Ytrain, = Ignite.array.(batch)

    @timeit "train batch" begin
        # Train MMD kernel bandwidths
        if mod(engine.state.iteration-1, settings["train"]["kernelrate"]) == 0
            @timeit "MMD kernel" let
                @timeit "sample G(X)" X̂train = sampleX̂(Ytrain)
                for _ in 1:settings["train"]["kernelsteps"]
                    success = train_kernel_bandwidth_flux!(models["logsigma"], X̂train, Ytrain;
                        kernelloss = settings["opt"]["kernel"]["loss"],
                        kernellr = settings["opt"]["kernel"]["lr"],
                        bwbounds = settings["arch"]["kernel"]["bwbounds"]) # timed internally
                    !success && break
                end
            end
        end

        ps = Flux.params(values(models)...)
        @timeit "forward" ℓ, back = Zygote.pullback(() -> SelfCVAEloss(Ytrain), ps)
        @timeit "reverse" gs = back(one(eltype(phys)))
        @timeit "update!" Flux.Optimise.update!(optimizers["cvae"], ps, gs)
    end

    return nothing
end

function val_metrics(engine, batch)
    @timeit "val batch" begin
        # Update callback state
        cb_state["last_time"] = get!(cb_state, "curr_time", time())
        cb_state["curr_time"] = time()
        cb_state["metrics"] = Dict{String,Any}()

        # Invert Y and make Xs
        Y, = Ignite.array.(batch)
        Nbatch = size(Y,2)
        θ, Z = InvertY(Y)
        X = signal_model(phys, θ)
        δG0, σG = correction_and_noiselevel(ricegen, X, Z)
        μG0 = add_correction(ricegen, X, δG0)
        X̂ = add_noise_instance(ricegen, μG0, σG)

        # Cross-entropy loss function
        μr0, σr = split_mean_std(models["E1"](Y)) #TODO X̂ or Y?
        μq0, σq = split_mean_std(models["E2"](vcat(Y,θ,Z))) #TODO X̂ or Y?
        zq = sample_mv_normal(μq0, σq)
        μx0, σx = split_mean_std(models["D"](vcat(Y,zq))) #TODO X̂ or Y?

        Zreg = sum(abs2, Z) / (2*Nbatch)
        YlogL = DataConsistency(Y, μG0, σG) / Nbatch #TODO
        KLdiv = KLDivergence(μq0, σq, μr0, σr) / Nbatch
        ELBO = EvidenceLowerBound(vcat(θ,Z), μx0, σx) / Nbatch
        MMDsq = let m = settings["train"]["batchsize"]
            100 * MMDloss(X̂[:,1:min(end,m)], Y[:,1:min(end,m)]) #TODO
        end
        loss = Zreg + YlogL + KLdiv + ELBO + MMDsq
        @pack! cb_state["metrics"] = Zreg, YlogL, KLdiv, ELBO, loss, MMDsq

        # Cache cb state variables using naming convention
        function cache_cb_state!(Y, θ, Z, Xθ, δθ, ϵθ, Xθδ, Xθhat, Yθ, Yθhat; suf::String)
            cb_state["Y"     * suf] = Y
            cb_state["θ"     * suf] = θ
            cb_state["Z"     * suf] = Z
            cb_state["Xθ"    * suf] = Xθ
            cb_state["δθ"    * suf] = δθ
            cb_state["ϵθ"    * suf] = ϵθ
            cb_state["Xθδ"   * suf] = Xθδ
            cb_state["Xθhat" * suf] = Xθhat
            cb_state["Yθ"    * suf] = Yθ
            cb_state["Yθhat" * suf] = Yθhat
            return cb_state
        end

        # Cache values for evaluating VAE performance for recovering Y
        let
            Yθ = hasclosedform(phys) ? signal_model(ClosedForm(phys), θ) : missing
            Yθhat = hasclosedform(phys) ? signal_model(ClosedForm(phys), θ, noiselevel(ClosedForm(phys))) : missing
            cache_cb_state!(Y, θ, Z, X, δG0, σG, μG0, X̂, Yθ, Yθhat; suf = "")

            all_Xhat_rmse = sqrt.(mean(abs2, Y .- X̂; dims = 1)) |> vec
            all_Xhat_logL = -sum(@. MMDLearning._rician_logpdf(Y, μG0, σG); dims = 1) |> vec
            Xhat_rmse = mean(all_Xhat_rmse)
            Xhat_logL = mean(all_Xhat_logL)
            @pack! cb_state["metrics"] = all_Xhat_rmse, all_Xhat_logL, Xhat_rmse, Xhat_logL
        end

        # Cache values for evaluating CVAE performance for estimating parameters of Y
        let
            Yfit = Y #TODO X̂ or Y?
            θfit, Zfit = split_theta_latent(sample_mv_normal(μx0, σx))
            θfit .= clamp.(θfit, θlower(phys), θupper(phys))
            Xθfit = signal_model(phys, θfit)
            δθfit, ϵθfit = correction_and_noiselevel(ricegen, Xθfit, Zfit)
            Xθδfit = add_correction(ricegen, Xθfit, δθfit)
            Xθhatfit = add_noise_instance(ricegen, Xθδfit, ϵθfit)
            Yθfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit) : missing
            Yθhatfit = hasclosedform(phys) ? signal_model(ClosedForm(phys), θfit, noiselevel(ClosedForm(phys))) : missing
            cache_cb_state!(Yfit, θfit, Zfit, Xθfit, δθfit, ϵθfit, Xθδfit, Xθhatfit, Yθfit, Yθhatfit; suf = "fit")

            rmse = hasclosedform(phys) ? sqrt(mean(abs2, Yθfit - Xθδfit)) : missing
            all_Yhat_rmse = sqrt.(mean(abs2, Yfit .- Xθhatfit; dims = 1)) |> vec
            all_Yhat_logL = -sum(@. MMDLearning._rician_logpdf(Yfit, Xθδfit, ϵθfit); dims = 1) |> vec
            Yhat_rmse = mean(all_Yhat_rmse)
            Yhat_logL = mean(all_Yhat_logL)
            θ_err = mean(θerror(phys, θ, θfit); dims = 2) |> vec |> copy
            Z_err = mean(abs, Z .- Zfit; dims = 2) |> vec |> copy
            @pack! cb_state["metrics"] = Yhat_rmse, Yhat_logL, θ_err, Z_err, rmse, all_Yhat_rmse, all_Yhat_logL
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
        metrics[:MMDsq] = cb_state["metrics"]["MMDsq"]
        metrics[:ELBO]  = cb_state["metrics"]["ELBO"]
        metrics[:rmse]  = cb_state["metrics"]["rmse"]
        metrics[:theta_err] = cb_state["metrics"]["θ_err"]
        metrics[:Z_err]     = cb_state["metrics"]["Z_err"]
        metrics[:Xhat_logL] = cb_state["metrics"]["Xhat_logL"]
        metrics[:Xhat_rmse] = cb_state["metrics"]["Xhat_rmse"]
        metrics[:Yhat_logL] = cb_state["metrics"]["Yhat_logL"]
        metrics[:Yhat_rmse] = cb_state["metrics"]["Yhat_rmse"]

        # Update logger dataframe
        push!(logger, metrics; cols = :subset)

        return deepcopy(metrics) #TODO convert to PyDict?
    end
end

function makeplots(;showplot = false)
    try
        Dict{Symbol, Any}(
            :ricemodel  => MMDLearning.plot_rician_model(logger, cb_state, phys; showplot = showplot, bandwidths = haskey(models, "logsigma") ? permutedims(models["logsigma"]) : nothing),
            :signals    => MMDLearning.plot_rician_signals(logger, cb_state, phys; showplot = showplot),
            :vaesignals => MMDLearning.plot_vae_rician_signals(logger, cb_state, phys; showplot = showplot),
            :infer      => MMDLearning.plot_rician_inference(logger, cb_state, phys; showplot = showplot),
        )
    catch e
        handleinterrupt(e; msg = "Error plotting")
    end
end

make_data_tuples(dataset) = tuple.(copy.(eachcol(sampleY(phys, :all; dataset = dataset))))
train_loader = torch.utils.data.DataLoader(make_data_tuples(:train); batch_size = settings["train"]["batchsize"], shuffle = true, drop_last = true)
val_loader = torch.utils.data.DataLoader(make_data_tuples(:val); batch_size = settings["data"]["nval"], shuffle = false, drop_last = false)

trainer = ignite.engine.Engine(@j2p train_step)
evaluator = ignite.engine.Engine(@j2p val_metrics)

# Force terminate
trainer.add_event_handler(
    ignite.engine.Events.STARTED | ignite.engine.Events.ITERATION_STARTED | ignite.engine.Events.ITERATION_COMPLETED,
    @j2p function (engine)
        if isfile(joinpath(settings["data"]["out"], "stop.txt"))
            @info "Exiting: found file $(joinpath(settings["data"]["out"], "stop.txt"))"
            engine.terminate()
        end
    end
)

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
        losses = logger.Xhat_logL[logger.dataset .=== :val] |> skipmissing |> collect
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
            for optname in ["cvae"]
                opt = optimizers[optname]
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
