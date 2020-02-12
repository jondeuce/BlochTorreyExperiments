# Load files
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
include(joinpath(@__DIR__, "src", "mmd_math.jl"))
include(joinpath(@__DIR__, "src", "mmd_utils.jl"))

####
#### VAE training
####

# sampleX, sampleY, sampleθ = make_gmm_data_samplers(image);
sampleX, sampleY, sampleθ = make_toy_samplers(;
    ntrain = settings["vae"]["batchsize"]::Int,
    epsilon = 0.001,
);

# settings = TOML.parsefile(joinpath(@__DIR__, "src/default_settings.toml")); #TODO

@unpack A, f = let 
    # Extract settings
    n   = settings["data"]["nsignal"]::Int
    Dz  = settings["vae"]["zdim"]::Int
    Dh  = settings["vae"]["hdim"]::Int
    Nh  = settings["vae"]["nhidden"]::Int
    act = Flux.relu
    hidden(nlayers) = [Flux.Dense(Dh, Dh, act) for _ in 1:nlayers]

    # Components of recognition model / "encoder" MLP
    A = Flux.Chain(
        Flux.Dense(n, Dh, act),
        hidden(Nh)...,
        Flux.Dense(Dh, 2*Dz)
    ) |> Flux.f64

    # Generative model / "decoder" MLP
    f = Flux.Chain(
        Flux.Dense(Dz, Dh, act),
        hidden(Nh)...,
        Flux.Dense(Dh, n, Flux.σ), # ensure positive signal
        # x -> x ./ sum(x; dims = 1), #TODO
        # Flux.Dense(Dh, n),
        # Flux.softmax,
    ) |> Flux.f64

    @ntuple(A, f)
end

function train_vae_model(
        A, f,
        sampleX,
        sampleY;
        gamma      = settings["vae"]["gamma"]::Float64,
        m          = settings["vae"]["batchsize"]::Int,
        nbatches   = settings["vae"]["nbatches"]::Int, #div(N, m), #TODO
        N          = m, #size(sampleY(nothing; dataset = :train), 2), #TODO
        mutations  = settings["vae"]["mutations"]::Int,
        lr         = settings["vae"]["stepsize"]::Float64,
        lrdrop     = settings["vae"]["stepdrop"]::Float64,
        lrdroprate = settings["vae"]["steprate"]::Int,
        epochs     = settings["vae"]["epochs"]::Int,
        timeout    = settings["vae"]["traintime"]::Float64,
        saveperiod = settings["vae"]["saveperiod"]::Float64,
        outfolder  = settings["data"]["out"]::String,
    )
    tstart = Dates.now()
    timer = TimerOutput()
    df = DataFrame(epoch = Int[], time = Float64[], loss = Float64[], logP_x_z = Float64[], KL_q_p = Float64[], rmse = Float64[], mae = Float64[], linf = Float64[])

    μ_logσ(Y) = (h = A(Y); (h[1:end÷2, :], h[end÷2+1:end, :]))
    z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(logσ)...)
    kl_q_p(μ, logσ) = 1//m * sum(0.5 .* (exp.(2 .* logσ) .+ μ.^2) .- 0.5 .- logσ) # KL-divergence between approximation posterior and N(0, 1) prior.
    logp_x_z(x, z) = logp_x_xhat(x, f(z)) # logp(x|z) - conditional probability of data given latents (γ = 1/2σ^2 is a hyperparameter)
    logp_x_xhat(x, xhat) = -gamma * N//m * sum(abs2, x .- xhat)
    Lhat(Y) = ((μhat, logσhat) = μ_logσ(Y); logp_x_z(Y, z(μhat, logσhat)) - kl_q_p(μhat, logσhat)) # Monte Carlo estimator of mean ELBO using m samples.
    function Lhat_mutate(Y)
        Ymut = mutate_signal(Y; meanmutations = mutations) # Randomly set signal elements to zero
        μhat, logσhat = μ_logσ(Ymut) # Learn latent space from mutated signal
        return logp_x_z(Y, z(μhat, logσhat)) - kl_q_p(μhat, logσhat) # Monte Carlo estimator of mean ELBO using m samples.
    end

    # Sample from the learned model
    encodedecode(Y) = f(z(μ_logσ(Y)...))
    sampledecoder(z) = f(z)

    # loss(Y) = -Lhat(Y)
    loss(Y) = -Lhat_mutate(Y) #TODO

    callback = let
        last_time = Ref(time())
        last_checkpoint = Ref(time())
        function(epoch, Y)
            @timeit timer "μ_logσ(Y)" μhat, logσhat = μ_logσ(Y)
            zsample = z(μhat, logσhat) # noise sample
            @timeit timer "f(zsample)" Yhat = f(zsample) # sample encode + decode
            dY = Y .- Yhat

            KL_q_p = kl_q_p(μhat, logσhat)
            logP_x_z = logp_x_xhat(Y, Yhat)
            ℓ = -(logP_x_z - KL_q_p)
            rmse = √mean(abs2, dY)
            mae = mean(abs, dY)
            ℓinf = maximum(abs, dY)

            dt, last_time[] = time() - last_time[], time()
            push!(df, [epoch, dt, ℓ, logP_x_z, KL_q_p, rmse, mae, ℓinf])

            function makeplots()
                s = x -> round(x; sigdigits = 3) # for plotting
                try
                    pencdec = let ps = []
                        for j in sample(1:size(Y,2), 4; replace = false)
                            p = plot()
                            plot!(p, Y[:, j]; line = (4, :red), label = "original")#, leg = :none)
                            plot!(p, Yhat[:, j]; line = (2, :blue), label = "encode-decode")#, leg = :none)
                            plot!(p, Y[:, j] - Yhat[:, j]; line = (2, :green), label = "difference")#, leg = :none)
                            # plot!(p, 10 * (Y[:, j] - Yhat[:, j]); line = (2, :green), label = "10X difference")#, leg = :none)
                            push!(ps, p)
                        end
                        plot(ps...)
                    end
                    display(pencdec)

                    p1 = plot(mean(zsample; dims = 2); ribbon = std(zsample; dims = 2), label = "z spectrum")
                    p2 = plot([plot(sampledecoder(randn(size(zsample,1),1)); line = (2,), lab = "decoder sample") for _ in 1:3]...; layout = (3,1))
                    psamples = plot(p1, p2)
                    display(psamples)

                    window = 100 #todo
                    dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, df)
                    # p1 = @df dfp plot(:epoch, [:loss :logP_x_z :KL_q_p]; line = (2,), title = "loss vs. epoch")
                    p1 = plot(dfp.epoch, [dfp.loss -dfp.logP_x_z dfp.KL_q_p]; lab = ["loss" "-logP_x_z" "KL_q_p"], line = (2,), title = "best loss = $(round(minimum(df.loss); sigdigits = 4))")
                    p2 = @df dfp plot(:epoch, :rmse; line = (2,), lab = "rmse", title = "best rmse = $(round(minimum(df.rmse); sigdigits = 4))")
                    p3 = @df dfp plot(:epoch, :mae;  line = (2,), lab = "mae",  title = "best mae = $(round(minimum(df.mae); sigdigits = 4))")
                    p4 = @df dfp plot(:epoch, :linf; line = (2,), lab = "linf", title = "best linf = $(round(minimum(df.linf); sigdigits = 4))")
                    foreach([p1, p2, p3, p4]) do p
                        (epoch >= lrdroprate) && vline!(p, lrdroprate:lrdroprate:epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                        plot!(p; xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                    end
                    ploss = plot(p1, p2, p3, p4)
                    display(ploss)

                    return @ntuple(pencdec, psamples, ploss)
                catch e
                    @warn "Error plotting"
                    @warn sprint(showerror, e, catch_backtrace())
                end
            end

            function saveplots(savefolder, prefix, suffix, plothandles)
                !isdir(savefolder) && mkpath(savefolder)
                @unpack pencdec, psamples, ploss = plothandles
                savefig(pencdec,  joinpath(savefolder, "$(prefix)encdec$(suffix).png"))
                savefig(psamples, joinpath(savefolder, "$(prefix)samples$(suffix).png"))
                savefig(ploss,    joinpath(savefolder, "$(prefix)loss$(suffix).png"))
            end

            function saveprogress(savefolder, prefix, suffix)
                !isdir(savefolder) && mkpath(savefolder)
                try
                    BSON.bson(joinpath(savefolder, "$(prefix)progress$(suffix).bson"), Dict("progress" => deepcopy(df)))
                    BSON.bson(joinpath(savefolder, "$(prefix)model$(suffix).bson"), Dict("A" => deepcopy(A), "f" => deepcopy(f)))
                catch e
                    @warn "Error saving progress"
                    @warn sprint(showerror, e, catch_backtrace())
                end
            end

            # Save model + progress each iteration
            @timeit timer "current progress" saveprogress(outfolder, "current-", "")

            # Check for best loss + save 
            isbest = df.loss[end] <= minimum(df.loss)
            isbest && saveprogress(outfolder, "best-", "")

            # Periodically checkpoint + plot
            if time() - last_checkpoint[] >= saveperiod
                last_checkpoint[] = time()
                estr = lpad(epoch, ndigits(epochs), "0")
                @timeit timer "checkpoint progress" saveprogress(joinpath(outfolder, "checkpoint"), "checkpoint-", ".epoch.$estr")
                @timeit timer "make plots" plothandles = makeplots()
                @timeit timer "checkpoint plots" saveplots(joinpath(outfolder, "checkpoint"), "checkpoint-", ".epoch.$estr", plothandles)
                @timeit timer "current plots" saveplots(outfolder, "current-", "", plothandles)
                # isbest && saveplots(outfolder, "best-", "", plothandles)

                # @info "$epoch: " * "time = $(round(dt; sigdigits=3))s, " * "loss = $(round(ℓ; sigdigits=6)), " * "KL = $(round(KL_q_p; sigdigits=6)), " * "-logP = $(round(-logP_x_z; sigdigits=6)), " * "rmse = $(round(rmse; sigdigits=3)), " * "mae = $(round(mae; sigdigits=3)), " * "ℓinf = $(round(ℓinf; sigdigits=3))"
                show(stdout, timer); println("\n")
                show(stdout, last(df, 6)); println("\n")
            end

            if epoch > 0 && mod(epoch, lrdroprate) == 0
                opt.eta /= lrdrop
                lrthresh = 0.9e-6
                if opt.eta >= lrthresh
                    @info "$epoch: Dropping learning rate to $(opt.eta)"
                else
                    @info "$epoch: Learning rate dropped below $lrthresh, exiting..."
                    throw(InterruptException())
                end
            end
        end
    end
    
    modelparams = Flux.params(A, f)
    opt = Flux.ADAM(lr)

    callback(0, sampleY(nothing; dataset = :test))
    for epoch in 1:epochs
        try
            @timeit timer "epoch" for _ in 1:nbatches
                @timeit timer "sampleY" Ytrain = sampleY(m; dataset = :train)
                @timeit timer "batch" Flux.train!(loss, modelparams, [(Ytrain,)], opt)
            end
            @timeit timer "callback" begin
                @timeit timer "sampleY" Ytest = sampleY(nothing; dataset = :test)
                callback(epoch, Ytest)
            end

            if Dates.now() - tstart >= Dates.Second(floor(Int, timeout))
                @info "Exiting: training time exceeded $(DECAES.pretty_time(timeout)) at epoch $epoch/$epochs"
                break
            end
        catch e
            if e isa InterruptException
                break
            else
                rethrow(e)
            end
        end
    end
    @info "Finished: trained for $(df.epoch[end])/$epochs epochs"

    return df
end

# df = train_vae_model(A, f, sampleX, sampleY; gamma = 1e3, epochs = 10)
df = train_vae_model(A, f, sampleX, sampleY)

nothing
