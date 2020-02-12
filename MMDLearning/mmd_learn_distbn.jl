# Load files
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
include(joinpath(@__DIR__, "src", "mmd_math.jl"))
include(joinpath(@__DIR__, "src", "mmd_utils.jl"))

#=
for a in [5], m in 50:50:2500
    k = Δ -> exp(-Δ)
    sampleX = () -> randn(2,m)
    sampleY = () -> ((1/√2) * [1 -1; 1 1] * [√a 0; 0 1/√a]) * randn(2,m)
    # sampleX = () -> rand(2,m)
    # sampleY = () -> [2-1/a; 1/a] .* rand(2,m)
    X, Y = sampleX(), sampleY()
    work = mmd_work(X, Y)
    # mmd!(work, k, X, Y)
    mmdvar!(work, k, X, Y)
    # @show m, a
end
=#

#=
for a in [1, 5, 20, 50], m in 50:50:500
    k = Δ -> exp(-Δ)
    sampleX = () -> randn(2,m)
    sampleY = () -> ((1/√2) * [1 -1; 1 1] * [√a 0; 0 1/√a]) * randn(2,m)
    # sampleX = () -> rand(2,m)
    # sampleY = () -> [2-1/a; 1/a] .* rand(2,m)
    work = mmd_work(sampleX(), sampleY())
    mmds = [mmd!(work, k, sampleX(), sampleY()) for _ in 1:100]
    mmdvars = [mmdvar!(work, k, sampleX(), sampleY()) for _ in 1:100]
    @show m, a, mean(mmds), √m * std(mmds), sqrt(mean(mmdvars))
end
=#

#=
let
    sampleX, sampleY = make_gmm_data_samplers(gmm, bounds, bounds_trans, f, g; noise = 0.0)
    sig = vec(std(sampleY()[45:end,:]; dims = 1))
    histogram(sig; title = "sigma = $(mean(sig))") |> display
    # sigma = mean(sig)
    sigma = 0.0025
    sampleX, sampleY = make_gmm_data_samplers(gmm, bounds, bounds_trans, f, g; noise = sigma)
    p = plot(; title = "sigma = $sigma")
    plot!(p, sampleY()[40:end, 1:10]; c = :blue, leg = :none)
    plot!(p, sampleX()[40:end, 1:10]; c = :red, leg = :none)
    display(p)
end
=#

#=
let niters = 100, logσ = -4
    for m in settings["mmd"]["batchsize"]::Int #[1000] #a in [12.0] #1:0.5:5 #4:6 #1:10
        s = x -> round(x; sigdigits = 3) # for plotting
        # a = 12.0
        # sampleX = () -> randn(2,m)
        # sampleY = () -> ((1/√2) * [1 -1; 1 1] * [√a 0; 0 1/√a]) * randn(2,m)

        # w = a .* [1.01, 1.0, 1.03, 1.04, 0.95]
        # sampleX = () -> 5 .+ 10 .* rand(length(w), m)
        # sampleY = () -> 5 .+  w .* rand(length(w), m)

        noise = 0.0025
        sampleX, sampleY = make_gmm_data_samplers(gmm, bounds, bounds_trans, f, g; noise = noise)
        mse = vec(mean(abs2, sampleX() - sampleY(); dims=1))
        histogram(mse; label = "mse", title = "min, med, max = $(s(minimum(mse))), $(s(median(mse))), $(s(maximum(mse)))") |> display

        print("before opt: ")
        sigma = √(median(mse)/2) # exp(-Δ/2σ^2) = 1/e --> σ = √(Δ/2) where Δ = median(mse)
        gamma = inv(2*sigma^2)
        k = Δ -> exp(-gamma*Δ)
        @unpack c_α, P_α, P_α_approx, MMDsq, MMDσ, c_α_samples, mmd_samples = mmd_permutation_test(k, sampleX, sampleY; niters = niters)
        @show m, c_α, m*MMDsq, MMDσ, P_α, P_α_approx

        mmd_heatmap(sampleX()[:,1:min(50,end)], sampleY()[:,1:min(50,end)], sigma; skipdiag = true) |> display
        mmd_witness(sampleX(), sampleY(), sigma; skipdiag = false) |> display

        phist = plot()
        density!(phist,    c_α_samples; l = (4, :blue), label = "before: μ = $(s(mean(   c_α_samples))), σ = $(s(std(   c_α_samples)))")
        density!(phist, m.*mmd_samples; l = (4, :red),  label = "before: μ = $(s(mean(m.*mmd_samples))), σ = $(s(std(m.*mmd_samples)))")
        display(phist)

        # # logσ_samples, tstat = mmd_bandwidth_bruteopt(sampleX, sampleY, (-1.5,1.5); nsigma = 25, nevals = 50)
        # # logσ = logσ_samples[findmax(tstat)[2]]
        # sigma = exp(logσ)
        # gamma = inv(2*sigma^2)
        # k = Δ -> exp(-gamma*Δ)

        # print("after opt: ")
        # @unpack c_α, P_α, P_α_approx, MMDsq, MMDσ, c_α_samples, mmd_samples = mmd_permutation_test(k, sampleX, sampleY; niters = niters)
        # @show m, c_α, m*MMDsq, MMDσ, P_α, P_α_approx

        # mmd_heatmap(sampleX()[:,1:min(50,end)], sampleY()[:,1:min(50,end)], sigma; skipdiag = true) |> display
        # mmd_witness(sampleX(), sampleY(), sigma; skipdiag = false) |> display

        # # psig = plot(logsigma, tstat; title = "a = $a, logσ = $(s(log(sigma))), γ = $(s(gamma))")
        # # display(psig)

        # # pscat = scatter((X->(X[1,:], X[2,:]))(sampleX())...; m = 10, xlim = (-5,5), ylim = (-5,5))
        # # scatter!((X->(X[1,:], X[2,:]))(sampleY())...; m = 10, xlim = (-5,5), ylim = (-5,5))
        # # display(pscat)

        # phist = plot()
        # density!(phist,    c_α_samples; l = (4, :cyan),   label = "after: μ = $(s(mean(   c_α_samples))), σ = $(s(std(   c_α_samples)))")
        # density!(phist, m.*mmd_samples; l = (4, :orange), label = "after: μ = $(s(mean(m.*mmd_samples))), σ = $(s(std(m.*mmd_samples)))")
        # display(phist)
    end
end
=#

#=
df = DataFrame(epoch = Int[], loss = Float64[], noise = Vector{Float64}[])
let niters = 5, m = 1000
    n = settings["data"]["nsignal"]::Int
    sampleX, sampleY = make_gmm_data_samplers(gmm, bounds, bounds_trans, f, g; input_noise = true, batchsize = m)
    mse = vec(mean(abs2, sampleX([0.0]) - sampleY(); dims=1))
    sigma = √(median(mse)/2) # exp(-Δ/2σ^2) = 1/e --> σ = √(Δ/2) where Δ = median(mse)
    gamma = inv(2*sigma^2)
    k = Δ -> exp(-gamma*Δ)

    noise_instance = (X, logϵ) -> exp.(logϵ) .* randn(size(X)) .* X[1:1,:]
    add_noise = (X, logϵ) -> sqrt.((X .+ noise_instance(X, logϵ)).^2 .+ noise_instance(X, logϵ).^2)
    loss = (X, Y, logϵ) -> mean([m * mmd(k, add_noise(X, logϵ), Y) for _ in 1:niters])
    ∇loss = (X, Y, logϵ) -> ForwardDiff.gradient(logϵ -> loss(X, Y, logϵ), logϵ)

    η = 1e-2
    opt = Flux.ADAM(η)
    logϵ = collect(range(-3.0, -7.5, length = n)) #fill(-6.0, n)

    outfolder = settings["data"]["out"]
    callback = function(epoch, X, Y, logϵ)
        ℓ = loss(X, Y, logϵ)
        push!(df, [epoch, ℓ, copy(logϵ)])
        @info epoch, extrema(logϵ), ℓ

        if mod(epoch, 25) == 0
            # Save current progress
            !isdir(outfolder) && mkpath(outfolder)
            try
                filename = "progress.epoch.$(lpad(epoch, 4, "0")).bson"
                BSON.bson(joinpath(outfolder, filename), Dict("progress" => df))
            catch e
            end

            # Plot and save plots
            try
                pnoise = plot(logϵ; title = "noise vector", label = "logϵ") |> display
                ploss = plot(df.epoch, df.loss; title = "minimum loss = $(minimum(df.loss))", label = "m * MMDsq") |> display
                psig = plot(; title = "blue: real signals - red: simulated")
                plot!(psig, Y[:,1:10]; c = :blue, leg = :none)
                plot!(psig, X[:,1:10]; c = :red, leg = :none)
                display(psig)

                savefig(pnoise, "noise.epoch.$(lpad(epoch, 4, "0")).png")
                savefig(ploss, "loss.epoch.$(lpad(epoch, 4, "0")).png")
                savefig(psig, "signals.epoch.$(lpad(epoch, 4, "0")).png")
            catch e
            end
        end
    end

    callback(0, sampleX(exp.(logϵ)), sampleY(), logϵ)
    for epoch in 1:10000 # settings["mmd"]["epochs"]::Int
        try
            X, Y = sampleX(exp.(logϵ)), sampleY()
            Flux.Optimise.update!(opt, logϵ, ∇loss(X, Y, logϵ))
            callback(epoch, X, Y, logϵ)
        catch e
            if e isa InterruptException
                break
            else
                rethrow(e)
            end
        end
    end

    nothing
end
=#

#=
let niters = 100, m = settings["mmd"]["batchsize"]::Int
    # a = 12.0
    # w = a .* [1.01, 1.0, 1.03, 1.04, 0.95]
    # sampleX = () -> 5 .+ 10 .* rand(length(w), m)
    # sampleY = () -> 5 .+  w .* rand(length(w), m)
    # noise = 0.0025
    # sampleX, sampleY = make_gmm_data_samplers(gmm, bounds, bounds_trans, f, g; noise = noise)

    loss = logσ -> mmd_bandwidth_optfun(logσ, sampleX(m), sampleY(m), niters)
    ∇loss = logσ -> ∇mmd_bandwidth_optfun(logσ, sampleX(m), sampleY(m), niters)

    # initsweep = function(logσ)
    #     t = loss(logσ)
    #     @info logσ, t
    #     return t
    # end
    # logσ_samples = range(-8.0, -4.0, length = 10)
    # t_samples = [initsweep(logσ) for logσ in logσ_samples]
    # logσ = [logσ_samples[findmax(t_samples)[2]]]
    logσ = [-6.0]

    η = 1e-2
    opt = Flux.ADAM(η)
    callback = epoch -> @info epoch, logσ[], loss(logσ)

    callback(0)
    for epoch in 1:10000 # settings["mmd"]["epochs"]::Int
        Flux.Optimise.update!(opt, logσ, -∇loss(logσ))
        callback(epoch)
    end
end
=#

sampleX, sampleY, sampleθ = make_gmm_data_samplers(image);

model = let
    n      = settings["data"]["nsignal"]::Int
    ndense = settings["mmd"]["ndense"]::Int
    zdim   = settings["mmd"]["zdim"]::Int

    H    = [n; fill(zdim, ndense-1); 2n]
    act  = x -> Flux.relu.(x)
    Dins = [[act, Flux.Dense(H[i], H[i+1])] for i in 2:length(H)-1]
    Din  = isempty(Dins) ? () : reduce(vcat, Dins)

    Flux.Chain(
        Flux.Dense(H[1], H[2]),
        Din...,
        x -> tanh.(x),
        # x -> relu.(x),
        # Flux.Diagonal(0.1*randn(H[end]), -8 .+ 0.1*randn(H[end])),
        Flux.Diagonal([fill(-0.01, H[end]÷2); ones(H[end]÷2)], [zeros(H[end]÷2); fill(-6.0, H[end]÷2)]),
        # Flux.Diagonal(H[end]),
        # x -> 0.01 .* x,
    ) |> Flux.f64
end

function train_mmd_model(
        model,
        sampleX,
        sampleY;
        n          = settings["data"]["nsignal"]::Int,
        m          = settings["mmd"]["batchsize"]::Int,
        lr         = settings["mmd"]["stepsize"]::Float64,
        nbatches   = settings["mmd"]["nbatches"]::Int,
        epochs     = settings["mmd"]["epochs"]::Int,
        outfolder  = settings["data"]["out"]::String,
        timeout    = settings["mmd"]["traintime"]::Float64,
        logsigma   = settings["mmd"]["logsigma"]::Float64,
        saveperiod = settings["mmd"]["saveperiod"]::Float64,
        nperm = 100,
    )
    tstart = Dates.now()
    df = DataFrame(epoch = Int[], time = Float64[], loss = Float64[])

    sigma = logsigma !== nothing ? exp(logsigma) : √(median(vec(mean(abs2, sampleX(m) - sampleY(m); dims=1)))/2) # exp(-Δ/2σ^2) = 1/e --> σ = √(Δ/2) where Δ = median(mse)
    gamma = inv(2*sigma^2)
    k = Δ -> exp(-gamma*Δ)

    correction_instance = X -> model(X)[1:n,:]
    noise_instance = X -> exp.(model(X)[n+1:end,:]) .* randn(size(X))
    add_noise = function(X)
        out = model(X)
        dX, ϵ = out[1:n, :], exp.(out[n+1:end, :])
        ϵ1, ϵ2 = ϵ .* randn(size(X)), ϵ .* randn(size(X))
        Xϵ = sqrt.((X .+ dX .+ ϵ1).^2 .+ ϵ2.^2)
        return Xϵ ./ sum(Xϵ; dims = 1)
    end

    loss = (X, Y) -> mean((_ -> m * mmd_flux(k, add_noise(X), Y)).(1:nbatches))
    ∇loss = (X, Y) -> Flux.gradient(() -> loss(X, Y), Flux.params(model))
    opt = Flux.ADAM(lr)

    callback = let
        last_time = Ref(time())
        last_checkpoint = Ref(time())
        function(epoch, X, Y)
            ℓ = loss(X, Y)

            dt, last_time[] = time() - last_time[], time()
            push!(df, [epoch, dt, ℓ])
            @info "$epoch: loss = $(round(ℓ;sigdigits=6)), time = $(round(dt;sigdigits=3))s"

            function makeplots()
                s = x -> round(x; sigdigits = 3) # for plotting
                try
                    ϵ = noise_instance(X)
                    dX = correction_instance(X)
                    Xϵ = add_noise(X)
                    pnoise = plot(mean(ϵ; dims = 2); yerr = std(ϵ; dims = 2), label = "noise vector");
                    plot!(pnoise, mean(dX; dims = 2); yerr = std(dX; dims = 2), label = "correction vector"); display(pnoise)
                    psig = plot(; title = "blue: real signals - red: simulated"); plot!(psig, Y[:,1:10]; c = :blue, leg = :none); plot!(psig, Xϵ[:,1:10]; c = :red, leg = :none); display(psig)
                    ploss = plot(df.epoch, df.loss; title = "minimum loss = $(minimum(df.loss))", label = "m * MMDsq"); display(ploss)
                    pwit = mmd_witness(Xϵ, Y, sigma); display(pwit)
                    pheat = mmd_heatmap(Xϵ, Y, sigma)

                    # @unpack c_α, P_α, P_α_approx, MMDsq, MMDσ, c_α_samples, mmd_samples =
                    #     mmd_permutation_test(k, () -> add_noise(sampleX(m)), () -> sampleY(m); niters = nperm)
                    # pperm = plot()
                    # density!(pperm,    c_α_samples; l = (4, :blue), label = "before: μ = $(s(mean(   c_α_samples))), σ = $(s(std(   c_α_samples)))")
                    # density!(pperm, m.*mmd_samples; l = (4, :red),  label = "before: μ = $(s(mean(m.*mmd_samples))), σ = $(s(std(m.*mmd_samples)))")
                    # display(pperm)

                    return @ntuple(pnoise, psig, ploss, pwit, pheat)#, pperm)
                catch e
                    @warn "Error plotting"
                    @warn sprint(showerror, e, catch_backtrace())
                end
            end

            function saveplots(savefolder, prefix, suffix, plothandles)
                !isdir(savefolder) && mkpath(savefolder)
                @unpack pnoise, psig, ploss, pwit, pheat = plothandles # pperm
                savefig(pnoise, joinpath(savefolder, "$(prefix)noise$(suffix).png"))
                savefig(psig,   joinpath(savefolder, "$(prefix)signals$(suffix).png"))
                savefig(ploss,  joinpath(savefolder, "$(prefix)loss$(suffix).png"))
                savefig(pwit,   joinpath(savefolder, "$(prefix)witness$(suffix).png"))
                savefig(pheat,  joinpath(savefolder, "$(prefix)heat$(suffix).png"))
                #savefig(pperm, joinpath(savefolder, "$(prefix)perm$(suffix).png"))
            end

            function saveprogress(savefolder, prefix, suffix)
                !isdir(savefolder) && mkpath(savefolder)
                try
                    BSON.bson(joinpath(savefolder, "$(prefix)progress$(suffix).bson"), Dict("progress" => deepcopy(df)))
                    BSON.bson(joinpath(savefolder, "$(prefix)model$(suffix).bson"), Dict("model" => deepcopy(model)))
                catch e
                    @warn "Error saving progress"
                    @warn sprint(showerror, e, catch_backtrace())
                end
            end

            if epoch == 0 || time() - last_checkpoint[] >= saveperiod
                last_checkpoint[] = time()
                estr = lpad(epoch, min(ndigits(epochs), 5), "0")
                isbest = df.loss[end] <= minimum(df.loss)
                saveprogress(joinpath(outfolder, "checkpoint"), "checkpoint-", ".epoch.$estr")
                saveprogress(outfolder, "current-", "")
                isbest && saveprogress(outfolder, "best-", "")

                plothandles = makeplots()
                saveplots(joinpath(outfolder, "checkpoint"), "checkpoint-", ".epoch.$estr", plothandles)
                saveplots(outfolder, "current-", "", plothandles)
                isbest && saveplots(outfolder, "best-", "", plothandles)
            end
        end
    end

    callback(0, sampleX(m), sampleY(m; dataset = :test))
    for epoch in 1:epochs
        try
            X, Ytrain, Ytest = sampleX(m), sampleY(m; dataset = :train), sampleY(m; dataset = :test)
            Flux.train!(loss, Flux.params(model), [(X,Ytrain)], opt)
            callback(epoch, X, Ytest)
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

df = train_mmd_model(model, sampleX, sampleY)

nothing
