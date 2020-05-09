# Load files
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
using MWFLearning
pyplot(size=(800,600))
Random.seed!(0);

const IS_TOY_MODEL = false
const TOY_NOISE_LEVEL = 1e-2
const models = Dict{String, Any}()
const settings = load_settings()

global sampleX, sampleY, sampleθ, fits_train, fits_test, fits_val
if IS_TOY_MODEL
    global sampleX, sampleY, sampleθ = make_toy_samplers(ntrain = settings["mmd"]["batchsize"]::Int, epsilon = TOY_NOISE_LEVEL, power = 4.0);
    global fits_train, fits_test, fits_val = nothing, nothing, nothing
else
    global sampleX, sampleY, sampleθ, fits_train, fits_test, fits_val = make_mle_data_samplers(
        settings["prior"]["data"]["image"]::String,
        settings["prior"]["data"]["thetas"]::String;
        ntheta = settings["data"]["ntheta"]::Int,
        normalizesignals = settings["data"]["normalize"]::Bool, #TODO
        plothist = false,
        padtrain = false,
        filteroutliers = false,
    );
end;

#= #TODO
let X = sampleX(nothing; dataset = :train), Y = sampleY(nothing; dataset = :train), fits = fits_train
    plot([plot(hcat(Y[:,j], X[:,j]); lab = ["data" "fit"]) for j in sample(1:size(X,2), 9; replace = false)]...) |> display
    histogram(fits.logsigma; title = "fits logsigma") |> display
    xl = quantile(fits.rmse, [0.001, 0.999])
    histogram(fits.rmse; title = "fits rmse", xlim = xl) |> display
    histogram(vec(sqrt.(sum(abs2, X.-Y; dims=1))); title = "real rmse", xlim = xl) |> display
    neglogL(j) = -sum(logpdf.(Rician.(X[:,j], exp(fits.logsigma[j])), Y[:,j]))
    xl = quantile(fits.loss, [0.001, 0.999])
    histogram(fits.loss; title = "fits -logL", xlim = xl) |> display
    histogram(neglogL.(1:size(X,2)); title = "real -logL", xlim = xl) |> display
end
=#

# vae_model_dict = BSON.load("/scratch/st-arausch-1/jcd1994/MMD-Learning/toyvaeopt-v1/sweep/45/best-model.bson")
# models["vae.E"] = Flux.Chain(deepcopy(vae_model_dict["A"]), h -> ((μ, logσ) = (h[1:end÷2, :], h[end÷2+1:end, :]); μ .+ exp.(logσ) .* randn(size(logσ)...)))
# models["vae.E"] = Flux.Chain(deepcopy(vae_model_dict["A"]), h -> h[1:end÷2, :])
# models["vae.D"] = deepcopy(vae_model_dict["f"])
models["vae.E"] = identity # encoder
models["vae.D"] = identity # decoder
models["mmd"] = let
    n    = settings["data"]["nsignal"]::Int
    Dz   = settings["vae"]["zdim"]::Int
    Dh   = settings["mmd"]["hdim"]::Int
    Nh   = settings["mmd"]["nhidden"]::Int
    dx   = settings["mmd"]["maxcorr"]::Float64
    bw   = settings["mmd"]["noisebounds"]::Vector{Float64}
    act  = Flux.relu
    hidden(nlayers) = [Flux.Dense(Dh, Dh, act) for _ in 1:nlayers]

    # Slope/intercept for scaling dX to (-dx, dx), logσ to (bw[1], bw[2])
    αbw, βbw = (bw[2]-bw[1])/2, (bw[1]+bw[2])/2
    α = [fill( dx, n); fill(αbw, n)]
    β = [fill(0.0, n); fill(βbw, n)]

    Flux.Chain(
        (models["vae.E"] == identity ? Flux.Dense(n, Dh, act) : Flux.Dense(Dz, Dh, act)),
        hidden(Nh)...,
        Flux.Dense(Dh, 2n, tanh),
        MWFLearning.Scale(α, β),
    ) |> Flux.f64
end
# models["mmd"] = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/MMDLearning/output/toymmd-v2-vector-logsigma/2020-02-26T17:00:51.433/best-model.bson")["model"]) #TODO
# models["mmd"] = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/MMDLearning/output/22/best-model.bson")["model"]) #TODO
@assert models["vae.E"] === identity && models["vae.D"] === identity "encoder/decoder is currently assumed to be identity"

# Convenience functions
split_correction_and_noise(μlogσ) = μlogσ[1:end÷2, :], exp.(μlogσ[end÷2+1:end, :])
noise_instance(X, ϵ) = ϵ .* randn(eltype(X), size(X)...)
get_correction_and_noise(X) = split_correction_and_noise(models["mmd"](models["vae.E"](X))) # Learning correction + noise
# get_correction_and_noise(X) = models["mmd"](models["vae.E"](X)), fill(eltype(X)(TOY_NOISE_LEVEL), size(X)...) # Learning correction w/ fixed noise
# get_correction_and_noise(X) = models["mmd"](models["vae.E"](X)), zeros(eltype(X), size(X)...) # Learning correction only
get_correction(X) = get_correction_and_noise(X)[1]
get_noise(X) = get_correction_and_noise(X)[2]
get_noise_instance(X) = noise_instance(X, get_noise(X))
get_corrected_signal(X) = get_corrected_signal(X, get_correction_and_noise(X)...)
get_corrected_signal(X, dX, ϵ) = get_corrected_signal(abs.(X .+ dX), ϵ)
function get_corrected_signal(X, ϵ)
    ϵR, ϵI = noise_instance(X, ϵ), noise_instance(X, ϵ)
    Xϵ = @. sqrt((X + ϵR)^2 + ϵI^2)
    # Don't use Flux.softmax(Xϵ), as then we can't interpret dX and ϵ as offset + Rician noise
    # !IS_TOY_MODEL && (Xϵ = Xϵ ./ sum(Xϵ; dims = 1)) #TODO don't need to normalize learned corrections
    return Xϵ
end

sampleLatentX(m; kwargs...) = models["vae.E"](get_corrected_signal(sampleX(m; kwargs...)))
sampleLatentY(m; kwargs...) = models["vae.E"](sampleY(m; kwargs...))

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
                ℓ, back = Zygote.pullback(_logσ -> loss(_logσ, X, Y), logσ)
                return ℓ, back
            end
        end
    end

    function checkloss(ℓ, _logσ)
        if kernelloss == "tstatistic"
            t = ℓ
            if lambda != 0
                t -= lambda * regularizer(permutedims(_logσ))
            end
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
        ntheta     :: Int     = settings["data"]["ntheta"],
        m          :: Int     = settings["mmd"]["batchsize"],
        lr         :: Float64 = settings["mmd"]["stepsize"],
        lrthresh   :: Float64 = settings["mmd"]["stepthresh"],
        lrdrop     :: Float64 = settings["mmd"]["stepdrop"],
        lrdroprate :: Int     = settings["mmd"]["steprate"],
        lambda     :: Float64 = settings["mmd"]["lambda"],
        ninfer     :: Int     = settings["mmd"]["ninfer"],
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
        rmse     = Union{Float64, Missing}[],
        logsigma = AbstractVecOrMat{Float64}[],
        theta_fit_err = Union{Vector{Float64}, Missing}[],
        signal_fit_logL = Union{Float64, Missing}[],
        signal_fit_rmse = Union{Float64, Missing}[],
    )

    regularizer = make_tikh_penalty(n, Float64)
    function loss(X,Y)
        dX, ϵ = get_correction_and_noise(X)
        Xϵ = get_corrected_signal(X, dX, ϵ)
        ℓ = m * mmd_flux(logsigma, models["vae.E"](Xϵ), models["vae.E"](Y))
        if lambda != 0
            ℓ += lambda * regularizer(dX)
        end
        return ℓ
    end

    callback = let
        cb_state = (
            last_time = Ref(time()),
            last_checkpoint = Ref(time()),
            last_best_checkpoint = Ref(time()),
            last_θbb = Union{AbstractVecOrMat{Float64}, Nothing}[nothing],
            last_global_i_fits = Union{Vector{Int}, Nothing}[nothing],
        )
        function(epoch, X, Y, θ, fits)
            dt, cb_state.last_time[] = time() - cb_state.last_time[], time()

            # Compute signal correction, noise instances, etc.
            dX, ϵ = get_correction_and_noise(X)
            Xϵ = get_corrected_signal(X, dX, ϵ)

            true_forward_model = IS_TOY_MODEL ?
                (θ, noise) -> toy_signal_model(θ, noise, 2) :
                (θ, noise) -> Y # don't have true underlying model for real data
            mock_forward_model = IS_TOY_MODEL ?
                (θ, noise) -> toy_signal_model(θ, noise, 4) :
                (θ, noise) -> signal_model(θ, noise; nTE = n, TE = settings["data"]["echotime"]::Float64, normalize = false) #TODO don't need to normalize learned corrections
            theta_error_fun = IS_TOY_MODEL ? toy_theta_error : signal_theta_error
            theta_bounds_fun = IS_TOY_MODEL ? toy_theta_bounds : () -> vec(extrema(θ; dims = 2)) #TODO () -> theta_bounds(Float64; ntheta = ntheta)

            Yθ = true_forward_model(θ, nothing)
            Yθϵ = true_forward_model(θ, TOY_NOISE_LEVEL)
            Xθ = mock_forward_model(θ, nothing)
            dXθ, ϵθ = get_correction_and_noise(Xθ)
            Xθϵ = get_corrected_signal(Xθ, dXθ, ϵθ)

            rmse, θ_fit_err, sig_fit_logL, sig_fit_rmse = missing, missing, missing, missing
            if IS_TOY_MODEL
                rmse = sqrt(mean(abs2, Yθ - (Xθ + dXθ)))
            end

            # Get corrected rician model params; input ν is a model signal
            function get_corrected_ν_and_σ(ν)
                dν, σ = get_correction_and_noise(ν)
                return abs.(ν .+ dν), σ
            end

            if !isnothing(cb_state.last_θbb[])
                # θbb results from inference on Yθϵ from a previous iteration; use this θbb as a proxy for the current "best guess" θ
                θbb = cb_state.last_θbb[]
                Xθbb = mock_forward_model(θbb, nothing)
                dXθbb, ϵθbb = get_correction_and_noise(Xθbb)
                Xθϵbb = get_corrected_signal(Xθbb, dXθbb, ϵθbb)

                global_i_fits = cb_state.last_global_i_fits[] # use same data as previous mle fits
                mle_err = [sum(.-logpdf.(Rician.(get_corrected_ν_and_σ(Xθbb[:,j])...; check_args = false), Yθϵ[:,jY])) for (j,jY) in enumerate(global_i_fits)]
                rmse_err = [sqrt(mean(abs2, Xθϵbb[:,j] .- Yθϵ[:,jY])) for (j,jY) in enumerate(global_i_fits)]

                i_sorted = sortperm(mle_err) #sortperm(mle_err)[1:7*(ninfer÷8)] #TODO
                sig_fit_logL = mean(mle_err[i_sorted])
                sig_fit_rmse = mean(rmse_err[i_sorted])

                if IS_TOY_MODEL
                    # θbb only matches θ for mock data; we don't have true θ for real data Y
                    global_i_sorted = global_i_fits[i_sorted]
                    θ_fit_err = mean(theta_error_fun(θ[:,global_i_sorted], θbb[:,i_sorted]); dims = 2) |> vec |> copy
                end
            end

            # Perform permutation test
            @timeit timer "perm test" begin
                permtest = mmd_perm_test_power(logsigma, models["vae.E"](Xϵ), models["vae.E"](Y); batchsize = m, nperms = nperms, nsamples = 1)
                c_α = permtest.c_alpha
                P_α = permtest.P_alpha_approx
                tstat = permtest.MMDsq / permtest.MMDσ
                MMDsq = m * permtest.MMDsq
                MMDvar = m^2 * permtest.MMDvar
            end

            # Use permutation test results to compute loss
            #@timeit timer "test loss" ℓ = loss(X, Y)
            reg = lambda * regularizer(dX)
            ℓ = MMDsq + reg

            # Update dataframe
            push!(df, [epoch, dt, ℓ, reg, MMDsq, MMDvar, tstat, c_α, P_α, rmse, copy(logsigma), θ_fit_err, sig_fit_logL, sig_fit_rmse])

            function makeplots()
                s = x -> x == round(x) ? round(Int, x) : round(x; sigdigits = 4) # for plotting
                window = 100 # window for plotting error metrics etc.
                nθplot = 4 # number of sets θ to draw for plotting simulated signals
                try
                    pmodel = @timeit timer "model plot" plot(
                        plot(
                            plot(mean(dX; dims = 2); yerr = std(dX; dims = 2), label = "signal correction", c = :red, title = "model outputs vs. data channel"),
                            plot(mean(ϵ; dims = 2); yerr = std(ϵ; dims = 2), label = "noise amplitude", c = :blue);
                            layout = (2,1),
                        ),
                        plot(permutedims(df.logsigma[end]); leg = :none, title = "logσ vs. data channel"),
                    )
                    # display(pmodel) #TODO

                    psignals = @timeit timer "signal plot" begin
                        θplotidx = sample(1:size(Xθ,2), nθplot; replace = false)
                        if IS_TOY_MODEL
                            plot(
                                [plot(hcat(Yθ[:,j], Xθϵ[:,j]); c = [:blue :red], lab = ["Goal Yθ" "Simulated Xθϵ"]) for j in θplotidx]...,
                                [plot(hcat(Yθ[:,j] - Xθ[:,j], dXθ[:,j]); c = [:blue :red], lab = ["Goal Yθ-Xθ" "Simulated dXθ"]) for j in θplotidx]...,
                                [plot(Yθ[:,j] - Xθ[:,j] - dXθ[:,j]; lab = "Yθ-(Xθ+dXθ)") for j in θplotidx]...;
                                layout = (3, nθplot),
                            )
                        else
                            plot(
                                [plot(hcat(Xθ[:,j], Xθϵ[:,j]); c = [:blue :red], lab = ["Uncorrected Xθ" "Corrected Xθϵ"]) for j in θplotidx]...,
                                [plot(Xθϵ[:,j] - Xθ[:,j]; c = :red, lab = "Xθϵ-Xθ") for j in θplotidx]...,
                                [plot(dXθ[:,j]; lab = "dXθ") for j in θplotidx]...;
                                layout = (3, nθplot),
                            )
                        end
                    end
                    # display(psignals) #TODO

                    ploss = nothing
                    @timeit timer "loss plot" begin
                        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, df)
                        if !isempty(dfp)
                            tstat_nan_outliers = map((_tstat, _mmdvar) -> _mmdvar > eps() ? _tstat : NaN, dfp.tstat, dfp.MMDvar)
                            tstat_drop_outliers = filter(!isnan, tstat_nan_outliers)
                            tstat_median = isempty(tstat_drop_outliers) ? NaN : median(tstat_drop_outliers)
                            tstat_ylim = isempty(tstat_drop_outliers) ? nothing : quantile(tstat_drop_outliers, [0.01, 0.99])
                            p1 = plot(dfp.epoch, dfp.MMDsq; label = "m*MMD²", title = "median loss = $(s(median(df.loss)))") # ylim = quantile(df.loss, [0.01, 0.99])
                            (lambda != 0) && plot!(p1, dfp.epoch, dfp.reg; label = "λ*reg (λ = $lambda)")
                            p2 = plot(dfp.epoch, dfp.MMDvar; label = "m²MMDvar", title = "median m²MMDvar = $(s(median(df.MMDvar)))") # ylim = quantile(df.MMDvar, [0.01, 0.99])
                            p3 = plot(dfp.epoch, tstat_nan_outliers; title = "median t = $(s(tstat_median))", label = "t = MMD²/MMDσ", ylim = tstat_ylim)
                            p4 = plot(dfp.epoch, dfp.P_alpha; label = "P_α", title = "median P_α = $(s(median(df.P_alpha)))", ylim = (0,1))
                            foreach([p1,p2,p3,p4]) do p
                                (epoch >= lrdroprate) && vline!(p, lrdroprate:lrdroprate:epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                                plot!(p; xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                            end
                            ploss = plot(p1, p2, p3, p4)
                            # display(ploss) #TODO
                        end
                    end

                    pwitness = nothing #mmd_witness(Xϵ, Y, sigma)
                    pheat = nothing #mmd_heatmap(Xϵ, Y, sigma)

                    pperm = @timeit timer "permutation plot" mmd_perm_test_power_plot(permtest)
                    # display(pperm) #TODO

                    pinfer = nothing
                    @timeit timer "theta inference plot" begin
                        @timeit timer "theta inference" begin
                            global_i_fits = sample(1:size(Yθϵ,2), ninfer; replace = false) # 1:ninfer
                            res = signal_loglikelihood_inference(
                                Yθϵ[:,global_i_fits], nothing, get_corrected_ν_and_σ, θ -> mock_forward_model(θ, nothing);
                                objective = :mle, bounds = theta_bounds_fun(),
                            )
                        end
                        bbres, optres = (x->x[1]).(res), (x->x[2]).(res)
                        θbb = reduce(hcat, Optim.minimizer.(optres))
                        cb_state.last_θbb[] = copy(θbb)
                        cb_state.last_global_i_fits[] = copy(global_i_fits)

                        # let θbb_mat = [permutedims(θbb) [fits.alpha[global_i_fits] fits.T2short[global_i_fits] fits.dT2[global_i_fits] fits.Ashort[global_i_fits] fits.Along[global_i_fits]]]
                        #     display(θbb_mat[:,[1,6,2,7,3,8,4,9,5,10]]) #TODO
                        # end

                        Xθbb = mock_forward_model(θbb, nothing)
                        dXθbb, ϵθbb = get_correction_and_noise(Xθbb)
                        Xθϵbb = get_corrected_signal(Xθbb, dXθbb, ϵθbb)
                        mle_err = Optim.minimum.(optres)
                        rmse_err = sqrt.(mean(abs2, Yθϵ[:,global_i_fits] .- Xθϵbb; dims = 1)) |> vec

                        # i_sorted = sortperm(mle_err)[1:7*(ninfer÷8)] # best 7/8 of mle fits (sorted)
                        i_sorted = sortperm(mle_err) # all mle fits (sorted)
                        global_i_sorted = global_i_fits[i_sorted]

                        if IS_TOY_MODEL
                            df[end, :theta_fit_err] = mean(theta_error_fun(θ[:,global_i_sorted], θbb[:,i_sorted]); dims = 2) |> vec |> copy
                        end
                        df[end, :signal_fit_logL] = mean(mle_err[i_sorted])
                        df[end, :signal_fit_rmse] = mean(rmse_err[i_sorted])

                        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, df)
                        df_inf = filter(dfp) do r
                            !ismissing(r.signal_fit_rmse) && !ismissing(r.signal_fit_logL) && !(IS_TOY_MODEL && ismissing(r.theta_fit_err))
                        end
                        if !isempty(dfp) && !isempty(df_inf)
                            pinfer = plot(
                                plot(
                                    IS_TOY_MODEL ?
                                        plot(hcat( Yθ[:,global_i_sorted[end÷2]], Xθϵbb[:,i_sorted[end÷2]]); c = [:blue :red], lab = ["Goal Yθ" "Fit X̄θϵ"]) :
                                        plot(hcat(Yθϵ[:,global_i_sorted[end÷2]], Xθϵbb[:,i_sorted[end÷2]]); c = [:blue :red], lab = ["Data Yθϵ" "Fit X̄θϵ"]),
                                    sticks(rmse_err[i_sorted]; m = (:circle,4), lab = "rmse: fits"),
                                    sticks(mle_err[i_sorted]; m = (:circle,4), lab = "-logL: fits"),
                                    layout = @layout([a{0.25h}; b{0.375h}; c{0.375h}]),
                                ),
                                plot(vcat(
                                    !IS_TOY_MODEL ? [] :
                                        plot(dfp.epoch, dfp.rmse; title = "min rmse = $(s(minimum(dfp.rmse)))", label = "rmse: Yθ - (Xθ + dXθ)", xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10)),
                                    !IS_TOY_MODEL ? [] :
                                        plot(df_inf.epoch, permutedims(reduce(hcat, df_inf.theta_fit_err)); title = "min max error = $(s(minimum(maximum.(df_inf.theta_fit_err))))", label = "θ" .* string.(permutedims(1:size(θ,1))), xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10)),
                                    let
                                        lab = "rmse: Yθϵ - X̄θϵ" * (IS_TOY_MODEL ? "" : "\nrmse prior: $(round(mean(fits.rmse); sigdigits = 4))")
                                        plot(df_inf.epoch, df_inf.signal_fit_rmse; title = "min rmse = $(s(minimum(df_inf.signal_fit_rmse)))", lab = lab, xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                                    end,
                                    let
                                        lab = "-logL: Yθϵ - X̄θϵ" * (IS_TOY_MODEL ? "" : "\n-logL prior: $(round(mean(fits.loss); sigdigits = 4))")
                                        plot(df_inf.epoch, df_inf.signal_fit_logL; title = "min -logL = $(s(minimum(df_inf.signal_fit_logL)))", lab = lab, xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                                    end,
                                )...),
                                layout = @layout([a{0.25w} b{0.75w}]),
                            )
                            # display(pinfer) #TODO
                        end
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
                    BSON.bson(joinpath(savefolder, "$(prefix)model$(suffix).bson"), deepcopy(models))
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

            # Check for best loss + save model/plots
            is_best_model = if IS_TOY_MODEL
                df.rmse[end] <= minimum(df.rmse) #df.loss[end] <= minimum(df.loss)
            else
                any(!ismissing, df.signal_fit_logL) && df.signal_fit_logL[findlast(!ismissing, df.signal_fit_logL)] <= minimum(skipmissing(df.signal_fit_logL))
            end

            if is_best_model
                @timeit timer "best model" saveprogress(outfolder, "best-", "")
                if time() - cb_state.last_best_checkpoint[] >= saveperiod
                    cb_state.last_best_checkpoint[] = time()
                    @timeit timer "make best plots" plothandles = makeplots()
                    @timeit timer "save best plots" saveplots(outfolder, "best-", "", plothandles)
                end
            end

            if epoch == 0 || time() - cb_state.last_checkpoint[] >= saveperiod
                cb_state.last_checkpoint[] = time()
                @timeit timer "current model" saveprogress(outfolder, "current-", "")
                @timeit timer "make current plots" plothandles = makeplots()
                @timeit timer "save current plots" saveplots(outfolder, "current-", "", plothandles)
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
                @timeit timer "train kernel" begin
                    train_mmd_kernel!(
                        logsigma;
                        recordprogress = false,
                        showprogress = false,
                        plotprogress = false,
                    )
                end
            end
        end
    end

    local Xtest, Ytest, θtest, fits
    if IS_TOY_MODEL
        # X and θ are generated on demand
        Xtest, Ytest, θtest, fits = sampleX(m), sampleY(m; dataset = :test), sampleθ(m), nothing
    else
        # Sample X and θ randomly, but choose Ys + corresponding fits consistently in order to compare models
        Xtest = sampleX(m; dataset = :test)
        θtest = sampleθ(m; dataset = :test)
        iY = if settings["data"]["normalize"]::Bool
            iY = filter(i -> fits_test.loss[i] <= -200.0 && fits_test.rmse[i] <= 0.002, 1:nrow(fits_test)) #TODO
        else
            iY = filter(i -> fits_test.loss[i] <= -125.0 && fits_test.rmse[i] <= 0.15, 1:nrow(fits_test)) #TODO
        end
        iY = sample(MersenneTwister(0), iY, m; replace = false)
        Ytest = sampleY(nothing; dataset = :test)[..,iY]
        fits = fits_test[iY,:]
    end

    opt = Flux.ADAM(lr)
    for epoch in 1:epochs
        try
            @timeit timer "epoch" begin
                (epoch == 1) && @timeit timer "callback" callback(0, Xtest, Ytest, θtest, fits)
                @timeit timer "batch loop" for _ in 1:nbatches
                    @timeit timer "sampleX" Xtrain = sampleX(m)
                    @timeit timer "sampleY" Ytrain = sampleY(m; dataset = :train)
                    @timeit timer "forward" _, back = Zygote.pullback(() -> loss(Xtrain, Ytrain), Flux.params(models["mmd"]))
                    @timeit timer "reverse" gs = back(1)
                    @timeit timer "update!" Flux.Optimise.update!(opt, Flux.params(models["mmd"]), gs)
                end
                @timeit timer "callback" callback(epoch, Xtest, Ytest, θtest, fits)
            end

            if mod(epoch, showrate) == 0
                show(stdout, timer); println("\n")
                show(stdout, last(df[:, Not([:logsigma, :theta_fit_err])], 10)); println("\n")
            end
            (epoch == 1) && TimerOutputs.reset_timer!(timer) # throw out initial loop (precompilation, first plot, etc.)

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
