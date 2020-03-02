# Load files
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))

#=
let
    for _ in 1:1
        y = sampleY(1)
        x = sampleX(1)
        p1 = plot()
        plot!(p1, reduce(hcat, [decoder(encoder(y)) for _ in 1:10]); line = (:blue,), leg = :none)
        plot!(p1, y; line = (:red, 3))
        
        p2 = plot()
        plot!(p2, sampleY(1); lab="Y", line = (3, :blue))
        plot!(p2, sampleX(1); lab="X", line = (3, :red))

        plot(p1,p2) |> display
    end
end
=#

const NOISE_LEVEL = 1e-2
sampleX, sampleY, sampleθ = make_toy_samplers(ntrain = settings["mmd"]["batchsize"]::Int, epsilon = NOISE_LEVEL, power = 4.0);
# sampleX, sampleY, sampleθ = make_gmm_data_samplers(image);

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
        Flux.Dense(Dh, 2n, tanh),
        x -> α .* x .+ β,
    ) |> Flux.f64
end
# model = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/MMDLearning/output/toymmd-v2-vector-logsigma/2020-02-26T17:00:51.433/best-model.bson")["model"]) #TODO
# model = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/MMDLearning/output/22/best-model.bson")["model"]) #TODO

split_correction_and_noise(X) = X[1:end÷2, :], exp.(X[end÷2+1:end, :])
noise_instance(X, ϵ) = ϵ .* randn(eltype(X), size(X)...)
get_correction_and_noise(X) = split_correction_and_noise(model(encoder(X))) # Learning correction + noise
# get_correction_and_noise(X) = model(encoder(X)), fill(eltype(X)(NOISE_LEVEL), size(X)...) # Learning correction w/ fixed noise
# get_correction_and_noise(X) = model(encoder(X)), zeros(eltype(X), size(X)...) # Learning correction only
get_correction(X) = get_correction_and_noise(X)[1]
get_noise_instance(X) = noise_instance(X, get_correction_and_noise(X)[2])
get_corrected_signal(X) = get_corrected_signal(X, get_correction_and_noise(X)...)
function get_corrected_signal(X, dX, ϵ)
    ϵR, ϵI = noise_instance(X, ϵ), noise_instance(X, ϵ)
    Xϵ = @. sqrt((X + dX + ϵR)^2 + ϵI^2)
    #Xϵ = Flux.softmax(Xϵ)
    return Xϵ
end

sampleLatentX(m; kwargs...) = encoder(get_corrected_signal(sampleX(m; kwargs...)))
sampleLatentY(m; kwargs...) = encoder(sampleY(m; kwargs...))

error("exiting...") #TODO
#=
cd(@__DIR__) #TODO
settings = TOML.parsefile(joinpath(@__DIR__, "src/default_settings.toml")); #TODO
let #TODO
    outpath = "./output/$(Dates.now())"
    settings["data"]["out"] = outpath
    !isdir(outpath) && mkpath(outpath)
    open(joinpath(outpath, "settings.toml"); write = true) do io
        TOML.print(io, settings)
    end
end
=#

function train_mmd_kernel!(
        logsigma        :: AbstractVecOrMat{Float64},
        X               :: Union{AbstractMatrix{Float64}, Nothing} = nothing,
        Y               :: Union{AbstractMatrix{Float64}, Nothing} = nothing;
        m               :: Int             = settings["mmd"]["batchsize"],
        lr              :: Float64         = settings["mmd"]["kernel"]["stepsize"],
        nbatches        :: Int             = settings["mmd"]["kernel"]["nbatches"],
        epochs          :: Int             = settings["mmd"]["kernel"]["epochs"],
        kernelloss      :: String          = settings["mmd"]["kernel"]["losstype"],
        bwbounds        :: Vector{Float64} = settings["mmd"]["kernel"]["bwbounds"],
        lambda          :: Float64         = settings["mmd"]["kernel"]["lambda"],
        recordprogress  :: Bool            = true,
        showprogress    :: Bool            = true,
        plotprogress    :: Bool            = false,
        method          :: Symbol          = :flux,
        showrate        :: Int             = 10,
        plotrate        :: Int             = 10,
    )

    df = DataFrame(
        epoch = Int[],
        loss = Float64[],
        tstat = Float64[],
        MMDsq = Float64[],
        MMDsigma = Float64[],
        logsigma = typeof(logsigma)[],
    )

    regularizer = make_tv_penalty(permutedims(logsigma))

    function loss(_logσ, X, Y)
        ℓ = kernelloss == "tstatistic" ?
            -mmd_flux_bandwidth_optfun(_logσ, X, Y) : # Minimize -t = -MMDsq/MMDσ
            -m * mmd_flux(_logσ, X, Y) # Minimize -m*MMDsq
        if lambda != 0
            ℓ += lambda * regularizer(permutedims(_logσ))
        end
        return ℓ
    end

    gradloss = let
        if length(logsigma) <= 16
            diffres = ForwardDiff.DiffResults.GradientResult(logsigma)
            function(logσ, X, Y)
                ForwardDiff.gradient!(diffres, _logσ -> loss(_logσ, X, Y), logσ)
                ℓ = DiffResults.value(diffres)
                back = _ -> (DiffResults.gradient(diffres),) # imitate Zygote api
                return ℓ, back
            end
        else
            function(logσ, X, Y)
                ℓ, back = Flux.Zygote.pullback(_logσ -> loss(_logσ, X, Y), logσ)
                return ℓ, back
            end
        end
    end

    function checkloss(ℓ, _logσ)
        if kernelloss == "tstatistic"
            t = ℓ - lambda * regularizer(permutedims(_logσ))
            if abs(t) > 100
                # Denominator has likely shrunk to sqrt(eps), or else we are overtraining
                @info "Loss is too large (ℓ = $ℓ)"
                return false
            end
        end
        return true
    end

    callback = function(epoch, X, Y)
        ℓ = loss(logsigma, X, Y)
        MMDsq, MMDvar = mmd_and_mmdvar_flux(logsigma, X, Y)
        MMDsq, MMDvar = m*MMDsq, m^2*MMDvar
        MMDσ = √max(MMDvar, eps(typeof(MMDvar)))
        push!(df, [epoch, ℓ, MMDsq/MMDσ, MMDsq, MMDσ, copy(logsigma)])

        if plotprogress && mod(epoch, plotrate) == 0
            plot(
                plot(permutedims(df.logsigma[end]); leg = :none, title = "logσ vs. data channel"),
                kernelloss == "tstatistic" ?
                    plot(df.epoch, df.tstat; lab = "t = MMD²/MMDσ", title = "t = MMD²/MMDσ vs. epoch", m = :circle, line = ([3,1], [:solid,:dot])) :
                    plot(df.epoch, df.MMDsq; lab = "m*MMD²", title = "m*MMD² vs. epoch", m = :circle, line = ([3,1], [:solid,:dot]))
            ) |> display
        end

        if showprogress && mod(epoch, showrate) == 0
            show(stdout, last(df[:, Not(:logsigma)], 6))
            println("\n")
        end
    end

    function sampleXY()
        function _tstat_check(_X, _Y)
            (kernelloss != "tstatistic") && return true
            MMDsq, MMDvar = mmd_and_mmdvar_flux(logsigma, _X, _Y)
            MMDsq, MMDvar = m*MMDsq, m^2*MMDvar
            (MMDsq < 0) && return false
            (MMDvar < 100*eps(typeof(MMDvar))) && return false
            return true
        end

        while true
            _X = !isnothing(X) ? X : sampleLatentX(m)
            _Y = !isnothing(Y) ? Y : sampleLatentY(m; dataset = :train)
            _tstat_check(_X, _Y) && return _X, _Y
        end
    end

    for epoch in 1:epochs
        try
            _X, _Y = sampleXY()
            if method == :flux
                opt = Flux.ADAM(lr) # new optimizer for each X, Y; loss jumps too wildly
                recordprogress && callback(epoch, _X, _Y)
                for _ in 1:nbatches
                    ℓ, back = gradloss(logsigma, _X, _Y)
                    !checkloss(ℓ, logsigma) && break
                    ∇ℓ = back(1)[1]
                    Flux.Optimise.update!(opt, logsigma, ∇ℓ)
                    clamp!(logsigma, bwbounds...)
                end
                recordprogress && callback(epoch, _X, _Y)
            else
                fg! = function (F,G,x)
                    _logσ = reshape(x, size(logsigma)...)

                    ℓ = nothing
                    if !isnothing(G)
                        ℓ, back = gradloss(_logσ, _X, _Y)
                        ∇ℓ = back(1)[1]
                        G .= reshape(∇ℓ, :)
                    elseif !isnothing(F)
                        ℓ = loss(_logσ, _X, _Y)
                    end

                    return ℓ
                end

                optim_cb = function(tr)
                    curr_ℓ = tr[end].value
                    curr_x = tr[end].metadata["x"]
                    
                    _logσ = reshape(curr_x, size(logsigma)...)
                    !checkloss(curr_ℓ, _logσ) && return true

                    logsigma .= _logσ
                    recordprogress && callback(epoch, _X, _Y)

                    return false
                end

                lower = fill(eltype(logsigma)(bwbounds[1]), length(logsigma))
                upper = fill(eltype(logsigma)(bwbounds[2]), length(logsigma))
                Optim.optimize(
                    Optim.only_fg!(fg!),
                    lower, upper, logsigma[:],
                    Optim.Fminbox(Optim.LBFGS()),
                    Optim.Options(
                        x_tol = 1e-3, # absolute tolerance on logsigma
                        f_tol = 1e-3, # relative tolerance on loss
                        callback = optim_cb,
                        f_calls_limit = nbatches,
                        g_calls_limit = nbatches,
                        outer_iterations = 1,
                        iterations = 1,
                        allow_f_increases = false,
                        store_trace = true,
                        extended_trace = true,
                        show_trace = false,
                    ),
                )
            end
        catch e
            if e isa InterruptException
                break
            else
                rethrow(e)
            end
        end
    end

    return df
end

function train_mmd_model(;
        n          :: Int     = settings["data"]["nsignal"],
        m          :: Int     = settings["mmd"]["batchsize"],
        lr         :: Float64 = settings["mmd"]["stepsize"],
        lrthresh   :: Float64 = settings["mmd"]["stepthresh"],
        lrdrop     :: Float64 = settings["mmd"]["stepdrop"],
        lrdroprate :: Int     = settings["mmd"]["steprate"],
        lambda     :: Float64 = settings["mmd"]["lambda"],
        nperms     :: Int     = settings["mmd"]["nperms"],
        nsamples   :: Int     = settings["mmd"]["nsamples"],
        epochs     :: Int     = settings["mmd"]["epochs"],
        nbatches   :: Int     = settings["mmd"]["nbatches"],
        timeout    :: Float64 = settings["mmd"]["traintime"],
        saveperiod :: Float64 = settings["mmd"]["saveperiod"],
        showrate   :: Int     = settings["mmd"]["showrate"],
        outfolder  :: String  = settings["data"]["out"],
        nbandwidth :: Int     = settings["mmd"]["kernel"]["nbandwidth"],
        kernelrate :: Int     = settings["mmd"]["kernel"]["rate"],
        bwbounds   :: Vector{Float64} = settings["mmd"]["kernel"]["bwbounds"],
        logsigma   :: AbstractVecOrMat{Float64} = nbandwidth == 1 ? fill(mean(bwbounds), 1, n) : repeat(range(bwbounds...; length = nbandwidth+2)[2:end-1], 1, n),
    )
    tstart = Dates.now()
    timer = TimerOutput()
    df = DataFrame(
        epoch    = Int[],
        time     = Float64[],
        loss     = Float64[],
        reg      = Float64[],
        MMDsq    = Float64[],
        MMDvar   = Float64[],
        tstat    = Float64[],
        c_alpha  = Float64[],
        P_alpha  = Float64[],
        rmse     = Float64[],
        logsigma = typeof(logsigma)[],
        theta_fit_err = Union{Vector{Float64}, Missing}[],
        signal_fit_logL = Union{Float64, Missing}[],
        signal_fit_rmse = Union{Float64, Missing}[],
    )

    regularizer = make_tikh_penalty(n, Float64)
    function loss(X,Y)
        dX, ϵ = get_correction_and_noise(X)
        Xϵ = get_corrected_signal(X, dX, ϵ)
        ℓ = m * mmd_flux(logsigma, encoder(Xϵ), encoder(Y))
        if lambda != 0
            ℓ += lambda * regularizer(dX)
        end
        return ℓ
    end

    callback = let
        last_time = Ref(time())
        last_checkpoint = Ref(time())
        function(epoch, X, Y)
            dt, last_time[] = time() - last_time[], time()

            # Compute signal correction, noise instances, etc.
            dX, ϵ = get_correction_and_noise(X)
            Xϵ = get_corrected_signal(X, dX, ϵ)

            θ = sampleθ(m)
            Yθ = toy_signal_model(θ, nothing, 2)
            Yθϵ = toy_signal_model(θ, NOISE_LEVEL, 2)
            Xθ = toy_signal_model(θ, nothing, 4)
            dXθ, ϵθ = get_correction_and_noise(Xθ)
            Xθϵ = get_corrected_signal(Xθ, dXθ, ϵθ)
            rmse = sqrt(mean(abs2, Yθ - (Xθ + dXθ)))

            # Perform permutation test
            #@timeit timer "perm test" permtest= mmd_perm_test_power(logsigma, m -> sampleLatentX(m), m -> sampleLatentY(m; dataset = :test), batchsize = m, nperms = nperms, nsamples = nsamples)
            @timeit timer "perm test" permtest = mmd_perm_test_power(logsigma, encoder(Xϵ), encoder(Y); batchsize = m, nperms = nperms, nsamples = 1)
            c_α = permtest.c_alpha
            P_α = permtest.P_alpha_approx
            tstat = permtest.MMDsq / permtest.MMDσ
            MMDsq = m * permtest.MMDsq
            MMDvar = m^2 * permtest.MMDvar

            # Use permutation test results to compute loss
            #@timeit timer "test loss" ℓ = loss(X, Y)
            reg = lambda * regularizer(dX)
            ℓ = MMDsq + reg

            # Update dataframe
            push!(df, [epoch, dt, ℓ, reg, MMDsq, MMDvar, tstat, c_α, P_α, rmse, copy(logsigma), missing, missing, missing])

            function makeplots()
                s = x -> round(x; sigdigits = 4) # for plotting
                try
                    z = noise_instance(X, ϵ)
                    pmodel = plot(
                        plot([mean(z; dims = 2) mean(dX; dims = 2)]; yerr = [std(z; dims = 2) std(dX; dims = 2)], label = ["noise vector" "correction vector"], title = "model outputs vs. data channel"),
                        plot(permutedims(df.logsigma[end]); leg = :none, title = "logσ vs. data channel"),
                    )
                    display(pmodel) #TODO

                    nθplot = 3
                    psignals = plot(
                        # [plot(Y[:,j]; c = :blue, lab = "Real signal Y") for j in 1:nθplot]...,
                        [plot(hcat(Yθ[:,j], Xθϵ[:,j]); c = [:blue :red], lab = ["Goal Yθ" "Simulated Xθϵ"]) for j in 1:nθplot]...,
                        [plot(hcat(Yθ[:,j] - Xθ[:,j], dXθ[:,j]); c = [:blue :red], lab = ["Goal Yθ-Xθ" "Simulated dXθ"]) for j in 1:nθplot]...,
                        [plot(Yθ[:,j] - Xθ[:,j] - dXθ[:,j]; lab = "Yθ-(Xθ+dXθ)") for j in 1:nθplot]...;
                        layout = (3, nθplot),
                    )
                    display(psignals) #TODO

                    window = 100 #TODO
                    dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, df)
                    tstat_filtered = map((_tstat, _mmdvar) -> _mmdvar > eps() ? _tstat : NaN, dfp.tstat, dfp.MMDvar)
                    tstat_median = all(isnan, tstat_filtered) ? NaN : median(filter(!isnan, tstat_filtered))
                    ploss = nothing
                    if !isempty(dfp)
                        p1 = plot(dfp.epoch, dfp.MMDsq; label = "m*MMD²", title = "median loss = $(s(median(df.loss)))")
                        (lambda != 0) && plot!(p1, dfp.epoch, dfp.reg; label = "λ*reg (λ = $lambda)")
                        p2 = plot(dfp.epoch, dfp.MMDvar; label = "m²MMDvar", title = "median m²MMDvar = $(s(median(df.MMDvar)))")
                        p3 = plot(dfp.epoch, tstat_filtered; title = "median t = $(s(tstat_median))", label = "t = MMD²/MMDσ", ylim = quantile(filter(!isnan, tstat_filtered), [0.05, 0.95]))
                        p4 = plot(dfp.epoch, dfp.P_alpha; label = "P_α", title = "median P_α = $(s(median(df.P_alpha)))")
                        foreach([p1,p2,p3,p4]) do p
                            (epoch >= lrdroprate) && vline!(p, lrdroprate:lrdroprate:epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                            plot!(p; xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                        end
                        ploss = plot(p1, p2, p3, p4)
                        display(ploss) #TODO
                    end

                    pwitness = nothing #mmd_witness(Xϵ, Y, sigma)
                    pheat = nothing #mmd_heatmap(Xϵ, Y, sigma)

                    pperm = mmd_perm_test_power_plot(permtest)
                    display(pperm) #TODO

                    nθbb = 64 # multiple of 8
                    @timeit timer "theta inference" begin
                        get_ν_and_σ = x -> ((dx, ϵ) = get_correction_and_noise(x); return (abs.(x.+dx), ϵ))
                        res = toy_theta_loglikelihood_inference(Yθϵ[:,1:nθbb], get_ν_and_σ; objective = :mle)
                        bbres, optres = (x->x[1]).(res), (x->x[2]).(res)
                        θbb = reduce(hcat, Optim.minimizer.(optres))
                        Xθbb = toy_signal_model(θbb, nothing, 4)
                        dXθbb, ϵθbb = get_correction_and_noise(Xθbb)
                        Xθϵbb = get_corrected_signal(Xθbb, dXθbb, ϵθbb)
                        mle_err = Optim.minimum.(optres)
                        rmse_err = sqrt.(mean(abs2, Yθϵ[:,1:nθbb] .- Xθϵbb; dims = 1)) |> vec

                        θidx = sortperm(mle_err)[1:7*(nθbb÷8)]
                        df[end, :theta_fit_err] = mean(toy_theta_error(θ[:,θidx], θbb[:,θidx]); dims = 2) |> vec |> copy
                        df[end, :signal_fit_logL] = mean(mle_err[θidx])
                        df[end, :signal_fit_rmse] = mean(rmse_err[θidx])
                        dfp = filter(r -> !ismissing(r.theta_fit_err) && !ismissing(r.signal_fit_rmse), df)
                        
                        pinfer = plot(
                            plot(
                                plot(hcat(Yθ[:,θidx[end÷2]], Xθϵbb[:,θidx[end÷2]]); c = [:blue :red], lab = ["Goal Yθ" "Fit X̄θϵ"]),
                                sticks(sort(rmse_err); m = (:circle,4), lab = "rmse: fits"),
                                sticks(sort(mle_err); m = (:circle,4), lab = "-logL: fits"),
                                layout = @layout([a{0.25h}; b{0.375h}; c{0.375h}]),
                            ),
                            plot(
                                plot(df.epoch, df.rmse; title = "min rmse = $(s(minimum(df.rmse)))", label = "rmse: Yθ - (Xθ + dXθ)"),
                                plot(dfp.epoch, permutedims(reduce(hcat, dfp.theta_fit_err)); title = "min max error = $(s(minimum(maximum.(dfp.theta_fit_err))))", label = "θ" .* string.(permutedims(1:size(θ,1)))),
                                plot(dfp.epoch, dfp.signal_fit_rmse; title = "min rmse = $(s(minimum(dfp.signal_fit_rmse)))", label = "rmse: Yθϵ - X̄θϵ"),
                                plot(dfp.epoch, dfp.signal_fit_logL; title = "min -logL = $(s(minimum(dfp.signal_fit_logL)))", label = "-logL: Yθϵ - X̄θϵ"),
                            ),
                            layout = @layout([a{0.25w} b{0.75w}]),
                        )
                        display(pinfer) #TODO
                    end

                    return @ntuple(pmodel, psignals, ploss, pperm, pwitness, pheat, pinfer)
                catch e
                    if e isa InterruptException
                        @warn "Plotting interrupted"
                        rethrow(e)
                    else
                        @warn "Error plotting"
                        @warn sprint(showerror, e, catch_backtrace())
                    end
                end
            end

            function saveplots(savefolder, prefix, suffix, plothandles)
                !isdir(savefolder) && mkpath(savefolder)
                try
                    for (name, p) in zip(keys(plothandles), values(plothandles))
                        !isnothing(p) && savefig(p, joinpath(savefolder, "$(prefix)$(string(name)[2:end])$(suffix).png"))
                    end
                catch e
                    if e isa InterruptException
                        @warn "Saving plots interrupted"
                        rethrow(e)
                    else
                        @warn "Error saving plots"
                        @warn sprint(showerror, e, catch_backtrace())
                    end
                end
            end

            function saveprogress(savefolder, prefix, suffix)
                !isdir(savefolder) && mkpath(savefolder)
                try
                    BSON.bson(joinpath(savefolder, "$(prefix)progress$(suffix).bson"), Dict("progress" => deepcopy(df)))
                    BSON.bson(joinpath(savefolder, "$(prefix)model$(suffix).bson"), Dict("model" => deepcopy(model)))
                catch e
                    if e isa InterruptException
                        @warn "Saving progress interrupted"
                        rethrow(e)
                    else
                        @warn "Error saving progress"
                        @warn sprint(showerror, e, catch_backtrace())
                    end
                end
            end

            # Check for best loss + save
            if df.rmse[end] <= minimum(df.rmse) #df.loss[end] <= minimum(df.loss)
                @timeit timer "best model" saveprogress(outfolder, "best-", "")
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
            if epoch > 0 && mod(epoch, kernelrate) == 0
                @timeit timer "train kernel" train_mmd_kernel!(logsigma;
                    recordprogress = false,
                    showprogress = false,
                    plotprogress = false,
                )
            end

            if epoch == 0 || time() - last_checkpoint[] >= saveperiod
                last_checkpoint[] = time()
                estr = lpad(epoch, ndigits(epochs), "0")
                @timeit timer "checkpoint model" saveprogress(joinpath(outfolder, "checkpoint"), "checkpoint-", ".epoch.$estr")
                @timeit timer "current model" saveprogress(outfolder, "current-", "")
                @timeit timer "make plots" plothandles = makeplots()
                @timeit timer "checkpoint plots" saveplots(joinpath(outfolder, "checkpoint"), "checkpoint-", ".epoch.$estr", plothandles)
                @timeit timer "current plots" saveplots(outfolder, "current-", "", plothandles)
            end
        end
    end

    opt = Flux.ADAM(lr)
    for epoch in 1:epochs
        try
            @timeit timer "epoch" begin
                @timeit timer "sampleX" X = sampleX(m)

                if epoch == 1
                    @timeit timer "sampleY"  Ytest = sampleY(m; dataset = :test)
                    @timeit timer "callback" callback(0, X, Ytest)
                end

                @timeit timer "batch loop" for _ in 1:nbatches
                    @timeit timer "sampleY" Ytrain = sampleY(m; dataset = :train)
                    @timeit timer "forward" _, back = Flux.Zygote.pullback(() -> loss(X, Ytrain), Flux.params(model))
                    @timeit timer "reverse" gs = back(1)
                    @timeit timer "update!" Flux.Optimise.update!(opt, Flux.params(model), gs)
                end

                @timeit timer "sampleY"  Ytest = sampleY(m; dataset = :test)
                @timeit timer "callback" callback(epoch, X, Ytest)
            end

            if mod(epoch, showrate) == 0
                show(stdout, timer); println("\n")
                show(stdout, last(df[:, Not([:logsigma, :theta_fit_err])], 10)); println("\n")
            end
            if epoch == 1
                TimerOutputs.reset_timer!(timer) # throw out initial loop (Zygote compilation, first plot, etc.)
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

#=
logsigma = let
    n = settings["data"]["nsignal"]::Int
    nbandwidth = settings["mmd"]["kernel"]["nbandwidth"]::Int
    # -4 .+ 0.5 .* randn(nbandwidth, n)
    repeat(range(-4, -1; length = nbandwidth), 1, n)
end
Random.seed!(0);
df_kernel = train_mmd_kernel!(logsigma;
    m = 1024,
    kernelloss = "tstatistic",
    lr = 1e-2,
    epochs = 100,
    nbatches = 10,
    bwbounds = [-5.0, 2.0],
    plotprogress = true,
    showprogress = false,
    showrate = 1,
    plotrate = 1,
    lambda = 1,
)
df = train_mmd_model(
    logsigma = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/MMDLearning/output/toymmd-v2-vector-logsigma/2020-02-25T23:48:34.905/current-progress.bson")["progress"][end,:logsigma]),
)
=#

df = train_mmd_model()

nothing
