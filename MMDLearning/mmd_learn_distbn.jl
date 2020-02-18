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
    corrected_signal = (X, logϵ) -> sqrt.((X .+ noise_instance(X, logϵ)).^2 .+ noise_instance(X, logϵ).^2)
    loss = (X, Y, logϵ) -> mean([m * mmd(k, corrected_signal(X, logϵ), Y) for _ in 1:niters])
    ∇loss = (X, Y, logϵ) -> ForwardDiff.gradient(logϵ -> loss(X, Y, logϵ), logϵ)

    lr = 1e-2
    opt = Flux.ADAM(lr)
    logϵ = collect(range(-3.0, -7.5, length = n)) #fill(-6.0, n)

    outfolder = settings["data"]["out"]::String
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
for _ in 1:25
    let y = sampleY(1)
        p = plot()
        plot!(p, reduce(hcat, [decoder(encoder(y)) for _ in 1:10]); line = (:blue,), leg = :none)
        plot!(y; line = (:red, 3))
        display(p)
    end
end
let
    p = plot()
    plot!(p, sampleY(2); lab="Y", line = (3, :blue))
    plot!(p, sampleX(2); lab="X", line = (3, :red))
    display(p)
end
=#

#=
let m = 512, nperms = 1024, nsamples = 64
    logsigma_allowed = 0.0:0.25:5.0
    best_res = Dict("P_alpha" => 0.0, "logsigma" => [])
    
    corrected_signal = function(X)
        # out = model(X)
        # dX, ϵ = out[1:end÷2, :], exp.(out[end÷2+1:end, :])
        # ϵ1, ϵ2 = ϵ .* randn(size(X)), ϵ .* randn(size(X))
        # Xϵ = @. sqrt((X + dX + ϵ1)^2 + ϵ2^2)
        # Xϵ = Flux.softmax(Xϵ)

        dX = model(encoder(X))
        Xϵ = @. Flux.σ(X + dX)
        return Xϵ
    end

    for _ in 1:10_000
        kernelargs = sort(sample(logsigma_allowed, rand(2:8); replace = false))
        # kernelargs = [1.5, 3.75, 4.25, 5.0]
        res = mmd_perm_test_power(
            kernelargs,
            m -> encoder(corrected_signal(sampleX(m))),
            m -> encoder(sampleY(m; dataset = :test));
            batchsize = m,
            nperms = nperms,
            nsamples = nsamples
        )
        if res.P_alpha_approx > best_res["P_alpha"]
            best_res["P_alpha"] = res.P_alpha_approx
            best_res["logsigma"] = copy(kernelargs)
            @show best_res
        end
        plot(
            mmd_perm_test_power_plot(res),
            plot(kernelargs; title = "$kernelargs");
            layout = (2,1),
         ) |> display
    end
end
=#

# sampleX, sampleY, sampleθ = make_gmm_data_samplers(image);
sampleX, sampleY, sampleθ = make_toy_samplers(ntrain = settings["mmd"]["batchsize"]::Int, epsilon = 1e-3, power = 4.0);

# vae_model_dict = BSON.load("/scratch/st-arausch-1/jcd1994/MMD-Learning/toyvaeopt-v1/sweep/45/best-model.bson")
# encoder = Flux.Chain(deepcopy(vae_model_dict["A"]), h -> ((μ, logσ) = (h[1:end÷2, :], h[end÷2+1:end, :]); μ .+ exp.(logσ) .* randn(size(logσ)...)))
# encoder = Flux.Chain(deepcopy(vae_model_dict["A"]), h -> h[1:end÷2, :])
# decoder = deepcopy(vae_model_dict["f"])
encoder = identity
decoder = identity

model = let
    n    = settings["data"]["nsignal"]::Int
    Dz   = settings["mmd"]["zdim"]::Int
    Dh   = settings["mmd"]["hdim"]::Int
    Nh   = settings["mmd"]["nhidden"]::Int
    act  = Flux.relu
    hidden(nlayers) = [Flux.Dense(Dh, Dh, act) for _ in 1:nlayers]

    # Slope/intercept for scaling dX to [-0.1,0.1], logσ to [-10,-2]
    α = [fill(0.1, n); fill( 4.0, n)]
    β = [fill(0.0, n); fill(-6.0, n)]

    Flux.Chain(
        (encoder == identity ? Flux.Dense(n, Dh, act) : Flux.Dense(Dz, Dh, act)),
        hidden(Nh)...,
        # Flux.Dense(Dh, n),
        # Flux.Dense(Dh, n, tanh),
        # Flux.Dense(Dh, n, Flux.σ),
        # x -> 0.1 .* x,
        Flux.Dense(Dh, 2n, tanh),
        x -> α .* x .+ β,
    ) |> Flux.f64
end

function corrected_signal(X) # Learning correction + noise
    out = model(encoder(X))
    dX  = out[1:end÷2, :]
    ϵ   = exp.(out[end÷2+1:end, :])
    ϵR  = ϵ .* randn(size(X))
    ϵI  = ϵ .* randn(size(X))
    Xϵ  = @. sqrt((X + dX + ϵR)^2 + ϵI^2)
    #Xϵ = Flux.softmax(Xϵ)
    return Xϵ
end
additive_correction(X) = model(encoder(X))[1:end÷2,:]
noise_instance(X) = exp.(model(encoder(X))[end÷2+1:end,:]) .* randn(size(X))

# function corrected_signal(X) # Learning correction w/ fixed noise
#     dX = model(encoder(X))
#     ϵR = 1e-3 .* randn(size(X))
#     ϵI = 1e-3 .* randn(size(X))
#     Xϵ = @. sqrt((X + dX + ϵR)^2 + ϵI^2)
#     return Xϵ
# end
# additive_correction(X) = model(encoder(X))
# noise_instance(X) = 1e-3 .* randn(size(X))

# corrected_signal(X) = X + model(encoder(X)) # Learning correction only
# additive_correction(X) = model(encoder(X))
# noise_instance(X) = zero(X)

sampleLatentX(m; kwargs...) = encoder(corrected_signal(sampleX(m; kwargs...)))
sampleLatentY(m; kwargs...) = encoder(sampleY(m; kwargs...))

# settings = TOML.parsefile(joinpath(@__DIR__, "src/default_settings.toml")); #TODO

function train_mmd_kernel!(
        logsigma,
        X = nothing,
        Y = nothing;
        m          =  settings["mmd"]["batchsize"]          :: Int,
        lr         =  settings["mmd"]["kernel"]["stepsize"] :: Float64,
        nbatches   =  settings["mmd"]["kernel"]["nbatches"] :: Int,
        epochs     =  settings["mmd"]["kernel"]["epochs"]   :: Int,
        kernelloss =  settings["mmd"]["kernel"]["losstype"] :: String,
        # outfolder  =  settings["data"]["out"]            :: String,
        # timeout    =  settings["mmd"]["traintime"]       :: Float64,
        # saveperiod =  settings["mmd"]["saveperiod"]      :: Float64,
    )

    loss = if kernelloss == "tstatistic"
        (logσ,X,Y) -> -mmd_flux_bandwidth_optfun(logσ, X, Y) # minimize -t = -MMDsq/MMDσ
    elseif kernelloss == "MMD"
        (logσ,X,Y) -> -m * mmd_flux(logσ, X, Y) # minimize -m*MMDsq
    else
        error("Unknown kernel loss: $kernelloss")
    end

    diffres = ForwardDiff.DiffResults.GradientResult(logsigma)
    function gradloss(logσ, X, Y)
        ForwardDiff.gradient!(diffres, _logσ -> loss(_logσ, X, Y), logσ)
        return DiffResults.value(diffres), DiffResults.gradient(diffres)
    end

    callback = function(epoch, X, Y)
        ℓ = loss(logsigma, X, Y)
        MMDsq = m * mmd_flux(logsigma, X, Y)
        MMDvar = m^2 * mmdvar_flux(logsigma, X, Y)
        MMDσ = √max(MMDvar, eps(typeof(MMDvar)))
        # @info epoch, ℓ, MMDsq/MMDσ, MMDsq, MMDσ, logsigma
    end

    opt = Flux.ADAM(lr)
    for epoch in 1:epochs
        while true
            _X = !isnothing(X) ? X : sampleLatentX(m)
            _Y = !isnothing(Y) ? Y : sampleLatentY(m; dataset = :train)

            if kernelloss == "tstatistic"
                ℓ = loss(logsigma, _X, _Y)
                if abs(ℓ) > 100
                    # @info "$epoch, loss too large: ℓ = $ℓ"
                    continue
                elseif ℓ > 0
                    # @info "$epoch, loss is positive: ℓ = $ℓ"
                    continue
                end
            end

            ℓ, ∇ℓ = gradloss(logsigma, _X, _Y)
            Flux.Optimise.update!(opt, logsigma, ∇ℓ)
            callback(epoch, _X, _Y)
            break
        end
    end
end

function train_mmd_model(;
        n           =  settings["data"]["nsignal"]        :: Int,
        m           =  settings["mmd"]["batchsize"]       :: Int,
        lr          =  settings["mmd"]["stepsize"]        :: Float64,
        lrdrop      =  settings["mmd"]["stepdrop"]        :: Float64,
        lrdroprate  =  settings["mmd"]["steprate"]        :: Int,
        # powercutoff =  settings["mmd"]["powercutoff"]     :: Float64,
        powerrate   =  settings["mmd"]["powerrate"]       :: Int,
        lrthresh    =  0.9e-7,
        nbatches    =  settings["mmd"]["nbatches"]        :: Int,
        epochs      =  settings["mmd"]["epochs"]          :: Int,
        outfolder   =  settings["data"]["out"]            :: String,
        timeout     =  settings["mmd"]["traintime"]       :: Float64,
        nbandwidth  =  settings["mmd"]["nbandwidth"]      :: Int,
        logsigma    =  collect(range(-4.0, 0.0; length = nbandwidth)) :: Vector{Float64},
        #logsigma   = (settings["mmd"]["logsigma"]|>copy) :: Vector{Float64},
        saveperiod  =  settings["mmd"]["saveperiod"]      :: Float64,
        nperms      =  settings["mmd"]["nperms"]          :: Int,
        nsamples    =  settings["mmd"]["nsamples"]        :: Int,
    )
    tstart = Dates.now()
    df = DataFrame(epoch = Int[], time = Float64[], loss = Float64[], c_alpha = Float64[], P_alpha = Float64[], t_perm = Float64[], rmse = Float64[], logsigma = Vector{Float64}[])

    # lambda = 1e3
    # Lap = diagm(1 => ones(n-1), 0 => -2*ones(n), -1 => ones(n-1))
    # loss = (X, Y) -> m * mmd_flux(logsigma, corrected_signal(X), Y) + lambda * mean(abs2, Lap * additive_correction(X))
    # loss = (X, Y) -> m * mmd_flux(logsigma, encoder(corrected_signal(X)), encoder(Y)) + lambda * mean(abs2, Lap * additive_correction(X))
    # @show lambda * mean(abs2, Lap * additive_correction(sampleX(m)))

    loss = (X, Y) -> m * mmd_flux(logsigma, corrected_signal(X), Y)

    callback = let
        last_time = Ref(time())
        last_checkpoint = Ref(time())
        function(epoch, X, Y)
            dt, last_time[] = time() - last_time[], time()

            ℓ = loss(X, Y)
            ϵ = noise_instance(X)
            dX = additive_correction(X)
            Xϵ = corrected_signal(X)

            θ = sampleθ(m)
            Yθ = toy_signal_model(θ, nothing, 2)
            Xθ = toy_signal_model(θ, nothing, 4)
            dXθ = additive_correction(Xθ)
            Xθϵ = corrected_signal(Xθ)
            rmse = sqrt(mean(abs2, Yθ - (Xθ + dXθ)))

            permtest = mmd_perm_test_power(logsigma, m -> sampleLatentX(m), m -> sampleLatentY(m; dataset = :test), batchsize = m, nperms = nperms, nsamples = nsamples)
            c_α = permtest.c_alpha
            P_α = permtest.P_alpha_approx
            t_perm = permtest.MMDsq / permtest.MMDσ

            # Update and show progress
            push!(df, [epoch, dt, ℓ, c_α, P_α, t_perm, rmse, copy(logsigma)])
            show(stdout, last(df, 6)); println("\n")

            function makeplots()
                s = x -> round(x; sigdigits = 4) # for plotting
                try
                    pnoise = plot()
                    plot!(pnoise, mean(ϵ; dims = 2); yerr = std(ϵ; dims = 2), label = "noise vector");
                    plot!(pnoise, mean(dX; dims = 2); yerr = std(dX; dims = 2), label = "correction vector");
                    # display(pnoise) #TODO

                    nθplot = 2
                    psig = plot(
                        # [plot(Y[:,j]; c = :blue, lab = "Real signal Y") for j in 1:nθplot]...,
                        [plot(hcat(Yθ[:,j], Xθϵ[:,j]); c = [:blue :red], lab = ["Goal Yθ" "Simulated Xθϵ"]) for j in 1:nθplot]...,
                        [plot(hcat(Yθ[:,j] - Xθ[:,j], dXθ[:,j]); c = [:blue :red], lab = ["Goal Yθ-Xθ" "Simulated dXθ"]) for j in 1:nθplot]...,
                        [plot(Yθ[:,j] - Xθ[:,j] - dXθ[:,j]; lab = "Yθ-(Xθ+dXθ)") for j in 1:nθplot]...;
                        layout = (3, nθplot),
                    );
                    # display(psig) #TODO

                    window = 100 #TODO
                    dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, df)
                    ploss = if !isempty(dfp)
                        plosses = [
                            plot(dfp.epoch, dfp.loss; title = "min loss = $(s(minimum(df.loss)))", label = "m * MMD^2"),
                            plot(dfp.epoch, dfp.rmse; title = "min rmse = $(s(minimum(df.rmse)))", label = "rmse"),
                            plot(dfp.epoch, dfp.t_perm; title = "median t = $(s(median(df.t_perm)))", label = "t = MMD^2/MMDσ"),
                            plot(dfp.epoch, permutedims(reduce(hcat, dfp.logsigma)); label = ["logσ" fill(nothing, 1, length(df.logsigma[1])-1)]),
                        ]
                        foreach(plosses) do p
                            (epoch >= lrdroprate) && vline!(p, lrdroprate:lrdroprate:epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                            plot!(p; xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                        end
                        plot(plosses...)
                    else
                        plot()
                    end
                    # display(ploss) #TODO

                    # pwit = mmd_witness(Xϵ, Y, sigma)
                    # pheat = mmd_heatmap(Xϵ, Y, sigma)

                    pperm = mmd_perm_test_power_plot(permtest)
                    # display(pperm) #TODO

                    return @ntuple(pnoise, psig, ploss, pperm) #pwit, pheat
                catch e
                    @warn "Error plotting"
                    @warn sprint(showerror, e, catch_backtrace())
                end
            end

            function saveplots(savefolder, prefix, suffix, plothandles)
                !isdir(savefolder) && mkpath(savefolder)
                @unpack pnoise, psig, ploss, pperm = plothandles #pwit, pheat
                savefig(pnoise, joinpath(savefolder, "$(prefix)noise$(suffix).png"))
                savefig(psig,   joinpath(savefolder, "$(prefix)signals$(suffix).png"))
                savefig(ploss,  joinpath(savefolder, "$(prefix)loss$(suffix).png"))
                # savefig(pwit,   joinpath(savefolder, "$(prefix)witness$(suffix).png"))
                # savefig(pheat,  joinpath(savefolder, "$(prefix)heat$(suffix).png"))
                savefig(pperm,  joinpath(savefolder, "$(prefix)perm$(suffix).png"))
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

            # Check for best loss + save
            # isbest = df.loss[end] <= minimum(df.loss)
            isbest = df.rmse[end] <= minimum(df.rmse)
            isbest && saveprogress(outfolder, "best-", "")

            if epoch == 0 || time() - last_checkpoint[] >= saveperiod
                last_checkpoint[] = time()
                estr = lpad(epoch, ndigits(epochs), "0")
                saveprogress(joinpath(outfolder, "checkpoint"), "checkpoint-", ".epoch.$estr")
                saveprogress(outfolder, "current-", "")

                plothandles = makeplots()
                saveplots(joinpath(outfolder, "checkpoint"), "checkpoint-", ".epoch.$estr", plothandles)
                saveplots(outfolder, "current-", "", plothandles)
                # isbest && saveplots(outfolder, "best-", "", plothandles)
            end

            if epoch > 0 && mod(epoch, lrdroprate) == 0
                opt.eta /= lrdrop
                if opt.eta >= lrthresh
                    @info "$epoch: Dropping learning rate to $(opt.eta)"
                else
                    @info "$epoch: Learning rate dropped below $lrthresh, exiting..."
                    throw(InterruptException())
                end
            end

            # Optimise kernel bandwidths
            # if df[end, :P_alpha] < powercutoff
            if epoch > 0 && mod(epoch, powerrate) == 0
                train_mmd_kernel!(logsigma)
                # @time callback(epoch, X, Ytest)
            end
        end
    end

    opt = Flux.ADAM(lr)
    callback(0, sampleX(m), sampleY(m; dataset = :test))
    for epoch in 1:epochs
        try
            # Minimize MMD^2
            X = sampleX(m)
            for _ in 1:nbatches #@time
                Ytrain = sampleY(m; dataset = :train)
                gs = Flux.gradient(() -> loss(X, Ytrain), Flux.params(model)) #@time
                Flux.Optimise.update!(opt, Flux.params(model), gs)
            end

            Ytest = sampleY(m; dataset = :test)
            callback(epoch, X, Ytest) #@time

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

df = train_mmd_model()

nothing
