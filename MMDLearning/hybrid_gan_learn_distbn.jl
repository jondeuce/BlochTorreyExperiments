# Load files
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
using MWFLearning
pyplot(size=(800,600))
Random.seed!(0);

const IS_TOY_MODEL = true
const TOY_NOISE_LEVEL = 1e-2
const models = Dict{String, Any}()
const settings = load_settings(joinpath(@__DIR__, "src", "hybrid_settings.toml"))

# Load data samplers
global sampleX, sampleY, sampleθ, fits_train, fits_test, fits_val
if IS_TOY_MODEL
    _sampleX, _sampleY, _sampleθ = make_toy_samplers(ntrain = settings["gan"]["batchsize"]::Int, epsilon = TOY_NOISE_LEVEL, power = 4.0)
    global sampleX = (m; dataset = :train) -> _sampleX(m) # samples are generated on demand; train/test not relevant
    global sampleθ = (m; dataset = :train) -> _sampleθ(m) # samples are generated on demand; train/test not relevant
    global sampleY = _sampleY
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

# Initialize generator + discriminator + kernel
let
    n  = settings["data"]["nsignal"]::Int
    Dh = settings["gan"]["hdim"]::Int
    Nh = settings["gan"]["nhidden"]::Int
    δ  = settings["gan"]["maxcorr"]::Float64
    logϵ_bw  = settings["gan"]["noisebounds"]::Vector{Float64}
    logσ_bw  = settings["gan"]["kernel"]["bwbounds"]::Vector{Float64}
    logσ_nbw = settings["gan"]["kernel"]["nbandwidth"]::Int
    Gact = Flux.relu
    Dact = Flux.relu

    # Helper for creating `nlayers` dense hidden layers
    hidden(nlayers, act) = [Flux.Dense(Dh, Dh, act) for _ in 1:nlayers]

    # Slope/intercept for scaling dX to (-δ, δ) and logϵ to (logϵ_bw[1], logϵ_bw[2])
    logϵ_α, logϵ_β = (logϵ_bw[2] - logϵ_bw[1])/2, (logϵ_bw[1] + logϵ_bw[2])/2
    α = [fill(  δ, n); fill(logϵ_α, n)]
    β = [fill(0.0, n); fill(logϵ_β, n)]

    models["G"] = Flux.Chain(
        Flux.Dense(n, Dh, Gact),
        hidden(Nh, Gact)...,
        Flux.Dense(Dh, 2n, tanh),
        MWFLearning.Scale(α, β),
    ) |> Flux.f64

    models["D"] = Flux.Chain(
        Flux.Dense(n, Dh, Dact),
        hidden(Nh, Dact)...,
        Flux.Dense(Dh, 1, Flux.sigmoid),
    ) |> Flux.f64

    # Initialize `logσ_nbw` linearly spaced kernel bandwidths `logσ` for each `n` channels strictly within the range (logσ_bw[1], logσ_bw[2])
    models["logsigma"] = convert(Matrix{Float64},
        repeat(range(logσ_bw...; length = logσ_nbw+2)[2:end-1], 1, n),
    )
end

#= Test code for loading pre-trained models
let
    # Load pre-trained GAN + MMD
    gan_folder = "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/ganopt-v3/sweep/72"
    mmd_folder = "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/63"
    gan_settings = TOML.parsefile(joinpath(gan_folder, "settings.toml"))
    mmd_settings = TOML.parsefile(joinpath(mmd_folder, "settings.toml"))
    gan_models   = BSON.load(joinpath(gan_folder, "current-model.bson"))
    mmd_models   = BSON.load(joinpath(mmd_folder, "current-model.bson"))
    mmd_prog     = BSON.load(joinpath(mmd_folder, "current-progress.bson"))["progress"]

    # models["G"] = deepcopy(mmd_models["mmd"])
    models["G"] = deepcopy(gan_models["G"])
    models["D"] = deepcopy(gan_models["D"])
    models["logsigma"] = copy(mmd_prog[end, :logsigma])

    # Update settings to match loaded models
    for gan_field in ["hdim", "nhidden", "maxcorr", "noisebounds"]
        settings["gan"][gan_field] = deepcopy(gan_settings["gan"][gan_field])
    end
    for kernel_field in ["bwbounds", "nbandwidth", "losstype"]
        settings["gan"]["kernel"][kernel_field] = deepcopy(mmd_settings["mmd"]["kernel"][kernel_field])
    end

    # Save updated settings
    open(joinpath(settings["data"]["out"], "settings.toml"); write = true) do io
        TOML.print(io, settings)
    end

    # Update sweep settings file, if it exists
    if haskey(ENV, "SETTINGSFILE")
        sweep_settings = TOML.parsefile(ENV["SETTINGSFILE"])
        merge_values_into!(d1::Dict, d2::Dict) = (foreach(k -> d1[k] isa Dict ? merge_values_into!(d1[k], d2[k]) : (d1[k] = deepcopy(d2[k])), keys(d1)); return d1) # recurse keys of d1, copying d2 values into d1. assumes d1 keys are present in d2
        merge_values_into!(sweep_settings, settings)
        cp(ENV["SETTINGSFILE"], ENV["SETTINGSFILE"] * ".bak"; force = true)
        open(ENV["SETTINGSFILE"]; write = true) do io
            TOML.print(io, sweep_settings)
        end
    end
end
=#

# Convenience functions
split_correction_and_noise(μlogϵ) = μlogϵ[1:end÷2, :], exp.(μlogϵ[end÷2+1:end, :])
noise_instance(X, ϵ) = ϵ .* randn(eltype(X), size(X)...)
get_correction_and_noise(X) = split_correction_and_noise(models["G"](X)) # Learning correction + noise
# get_correction_and_noise(X) = models["G"](X), fill(eltype(X)(TOY_NOISE_LEVEL), size(X)...) # Learning correction w/ fixed noise
# get_correction_and_noise(X) = models["G"](X), zeros(eltype(X), size(X)...) # Learning correction only
get_correction(X) = get_correction_and_noise(X)[1]
get_noise(X) = get_correction_and_noise(X)[2]
get_noise_instance(X) = noise_instance(X, get_noise(X))
get_corrected_signal(X) = get_corrected_signal(X, get_correction_and_noise(X)...)
get_corrected_signal(X, dX, ϵ) = get_corrected_signal(abs.(X .+ dX), ϵ)
function get_corrected_signal(X, ϵ)
    ϵR, ϵI = noise_instance(X, ϵ), noise_instance(X, ϵ)
    Xϵ = @. sqrt((X + ϵR)^2 + ϵI^2)
    return Xϵ
end

# Generator and discriminator losses
get_D_Y(Y) = models["D"](Y) # discrim on real data
get_D_G_X(X) = models["D"](get_corrected_signal(X)) # discrim on genatr data (`get_corrected_signal` wraps the generator `models["G"]`)
Dloss(X,Y) = -mean(log.(get_D_Y(Y)) .+ log.(1 .- get_D_G_X(X)))
Gloss(X) = mean(log.(1 .- get_D_G_X(X)))
MMDloss(X,Y) = size(Y,2) * mmd_flux(models["logsigma"], get_corrected_signal(X), Y) # m*MMD^2 on genatr data (`get_corrected_signal` wraps the generator `models["G"]`)

# Global state
timer = TimerOutput()
state = DataFrame(
    :epoch    => Int[], # mandatory field
    :dataset  => Symbol[], # mandatory field
    :time     => Union{Float64, Missing}[],
    :Gloss    => Union{Float64, Missing}[],
    :Dloss    => Union{Float64, Missing}[],
    :D_Y      => Union{Float64, Missing}[],
    :D_G_X    => Union{Float64, Missing}[],
    :MMDsq    => Union{Float64, Missing}[],
    :MMDvar   => Union{Float64, Missing}[],
    :tstat    => Union{Float64, Missing}[],
    :c_alpha  => Union{Float64, Missing}[],
    :P_alpha  => Union{Float64, Missing}[],
    # :logsigma => Union{AbstractVecOrMat{Float64}, Missing}[], #TODO
    :rmse     => Union{Float64, Missing}[],
    :theta_fit_err => Union{Vector{Float64}, Missing}[],
    :signal_fit_logL => Union{Float64, Missing}[],
    :signal_fit_rmse => Union{Float64, Missing}[],
)
optimizers = Dict(
    "G"   => Flux.ADAM(settings["gan"]["stepsize"]),
    "D"   => Flux.ADAM(settings["gan"]["stepsize"]),
    "mmd" => Flux.ADAM(settings["gan"]["stepsize"]),
)

callback = let
    local Xtest, Ytest, θtest, testfits

    if IS_TOY_MODEL
        # X and θ are generated on demand
        Xtest, Ytest, θtest = sampleX(settings["gan"]["batchsize"]), sampleY(settings["gan"]["batchsize"]; dataset = :test), sampleθ(settings["gan"]["batchsize"])
        testfits = nothing
    else
        # Sample X and θ randomly, but choose Ys + corresponding fits consistently in order to compare models, and choose Ys with reasonable agreeance with data in order to not be overconfident in improving terrible fits
        Xtest = sampleX(settings["gan"]["batchsize"]; dataset = :test)
        θtest = sampleθ(settings["gan"]["batchsize"]; dataset = :test)
        iY = if settings["data"]["normalize"]::Bool
            iY = filter(i -> fits_test.loss[i] <= -200.0 && fits_test.rmse[i] <= 0.002, 1:nrow(fits_test)) #TODO
        else
            iY = filter(i -> fits_test.loss[i] <= -125.0 && fits_test.rmse[i] <= 0.15, 1:nrow(fits_test)) #TODO
        end
        iY = sample(MersenneTwister(0), iY, settings["gan"]["batchsize"]; replace = false)
        Ytest = sampleY(nothing; dataset = :test)[..,iY]
        testfits = fits_test[iY,:]
    end

    cb_state = (
        X = Xtest,
        Y = Ytest,
        θ = θtest,
        fits = testfits,
        last_time = Ref(time()),
        last_checkpoint = Ref(-Inf),
        last_best_checkpoint = Ref(-Inf),
        last_θbb = Union{AbstractVecOrMat{Float64}, Nothing}[nothing],
        last_global_i_fits = Union{Vector{Int}, Nothing}[nothing],
    )

    function callback(epoch;
            n          :: Int     = settings["data"]["nsignal"],
            ntheta     :: Int     = settings["data"]["ntheta"],
            m          :: Int     = settings["gan"]["batchsize"],
            outfolder  :: String  = settings["data"]["out"],
            saveperiod :: Float64 = settings["gan"]["saveperiod"],
            ninfer     :: Int     = settings["gan"]["ninfer"],
            nperms     :: Int     = settings["gan"]["nperms"],
            nsamples   :: Int     = settings["gan"]["nsamples"],
            lrthresh   :: Float64 = settings["gan"]["stepthresh"],
            lrdrop     :: Float64 = settings["gan"]["stepdrop"],
            lrdroprate :: Int     = settings["gan"]["steprate"],
        )
        dt, cb_state.last_time[] = time() - cb_state.last_time[], time()
        @unpack X, Y, θ, fits = cb_state

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
        function get_corrected_ν_and_ϵ(ν)
            local _dν, _ϵ = get_correction_and_noise(ν)
            return abs.(ν .+ _dν), _ϵ
        end

        if !isnothing(cb_state.last_θbb[])
            # θbb results from inference on Yθϵ from a previous iteration; use this θbb as a proxy for the current "best guess" θ
            θbb = cb_state.last_θbb[]
            Xθbb = mock_forward_model(θbb, nothing)
            dXθbb, ϵθbb = get_correction_and_noise(Xθbb)
            Xθϵbb = get_corrected_signal(Xθbb, dXθbb, ϵθbb)

            global_i_fits = cb_state.last_global_i_fits[] # use same data as previous mle fits
            mle_err = [sum(.-logpdf.(Rician.(get_corrected_ν_and_ϵ(Xθbb[:,j])...; check_args = false), Yθϵ[:,jY])) for (j,jY) in enumerate(global_i_fits)]
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
            permtest = mmd_perm_test_power(models["logsigma"], Xϵ, Y; batchsize = m, nperms = nperms, nsamples = 1)
            c_α = permtest.c_alpha
            P_α = permtest.P_alpha_approx
            tstat = permtest.MMDsq / permtest.MMDσ
            MMDsq = m * permtest.MMDsq
            MMDvar = m^2 * permtest.MMDvar
        end

        # Compute GAN losses
        d_y = get_D_Y(Y)
        d_g_x = get_D_G_X(X)
        dloss = -mean(log.(d_y) .+ log.(1 .- d_g_x))
        gloss = mean(log.(1 .- d_g_x))

        # Update dataframe
        push!(state, [epoch, :test, dt, gloss, dloss, mean(d_y), mean(d_g_x), MMDsq, MMDvar, tstat, c_α, P_α, rmse, θ_fit_err, sig_fit_logL, sig_fit_rmse])

        function makeplots()
            s = x -> x == round(x) ? round(Int, x) : round(x; sigdigits = 4) # for plotting
            window = 100 # window for plotting error metrics etc.
            nθplot = 4 # number of sets θ to draw for plotting simulated signals
            try
                pgan = nothing
                @timeit timer "gan loss plot" begin
                    dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, state)
                    dfp = dropmissing(dfp[:, [:epoch, :Gloss, :Dloss, :D_Y, :D_G_X]])
                    if !isempty(dfp)
                        pgan = @df dfp plot(:epoch, [:Gloss :Dloss :D_Y :D_G_X]; label = ["G loss" "D loss" "D(Y)" "D(G(X))"], lw = 3)
                        (epoch >= lrdroprate) && vline!(pgan, lrdroprate:lrdroprate:epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                        plot!(pgan; xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                    end
                    # display(pgan) #TODO
                end

                pmodel = @timeit timer "model plot" plot(
                    plot(
                        plot(mean(dX; dims = 2); yerr = std(dX; dims = 2), label = "signal correction", c = :red, title = "model outputs vs. data channel"),
                        plot(mean(ϵ; dims = 2); yerr = std(ϵ; dims = 2), label = "noise amplitude", c = :blue);
                        layout = (2,1),
                    ),
                    plot(permutedims(models["logsigma"]); leg = :none, title = "logσ vs. data channel"),
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

                pmmd = nothing
                @timeit timer "mmd loss plot" begin
                    dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, state)
                    if !isempty(dfp)
                        tstat_nan_outliers = map((_tstat, _mmdvar) -> _mmdvar > eps() ? _tstat : NaN, dfp.tstat, dfp.MMDvar)
                        tstat_drop_outliers = filter(!isnan, tstat_nan_outliers)
                        tstat_median = isempty(tstat_drop_outliers) ? NaN : median(tstat_drop_outliers)
                        tstat_ylim = isempty(tstat_drop_outliers) ? nothing : quantile(tstat_drop_outliers, [0.01, 0.99])
                        p1 = plot(dfp.epoch, dfp.MMDsq; label = "m*MMD²", title = "median loss = $(s(median(state.MMDsq)))") # ylim = quantile(state.MMDsq, [0.01, 0.99])
                        p2 = plot(dfp.epoch, dfp.MMDvar; label = "m²MMDvar", title = "median m²MMDvar = $(s(median(state.MMDvar)))") # ylim = quantile(state.MMDvar, [0.01, 0.99])
                        p3 = plot(dfp.epoch, tstat_nan_outliers; title = "median t = $(s(tstat_median))", label = "t = MMD²/MMDσ", ylim = tstat_ylim)
                        p4 = plot(dfp.epoch, dfp.P_alpha; label = "P_α", title = "median P_α = $(s(median(state.P_alpha)))", ylim = (0,1))
                        foreach([p1,p2,p3,p4]) do p
                            (epoch >= lrdroprate) && vline!(p, lrdroprate:lrdroprate:epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                            plot!(p; xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                        end
                        pmmd = plot(p1, p2, p3, p4)
                        # display(pmmd) #TODO
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
                            Yθϵ[:,global_i_fits], nothing, get_corrected_ν_and_ϵ, θ -> mock_forward_model(θ, nothing);
                            objective = :mle, bounds = theta_bounds_fun(),
                        )
                    end
                    mle_results = map(res) do (bbopt_res, optim_res)
                        x1, ℓ1 = BlackBoxOptim.best_candidate(bbopt_res), BlackBoxOptim.best_fitness(bbopt_res)
                        x2, ℓ2 = Optim.minimizer(optim_res), Optim.minimum(optim_res)
                        (ℓ1 < ℓ2) && @warn "BlackBoxOptim results less than Optim result" #TODO
                        ℓ1 < ℓ2 ? (x = x1, loss = ℓ1) : (x = x2, loss = ℓ2)
                    end
                    θbb = reduce(hcat, (r -> r.x).(mle_results))
                    mle_err = (r -> r.loss).(mle_results)
                    cb_state.last_θbb[] = copy(θbb)
                    cb_state.last_global_i_fits[] = copy(global_i_fits)

                    Xθbb = mock_forward_model(θbb, nothing)
                    dXθbb, ϵθbb = get_correction_and_noise(Xθbb)
                    Xθϵbb = get_corrected_signal(Xθbb, dXθbb, ϵθbb)
                    rmse_err = sqrt.(mean(abs2, Yθϵ[:,global_i_fits] .- Xθϵbb; dims = 1)) |> vec

                    # i_sorted = sortperm(mle_err)[1:7*(ninfer÷8)] # best 7/8 of mle fits (sorted)
                    i_sorted = sortperm(mle_err) # all mle fits (sorted)
                    global_i_sorted = global_i_fits[i_sorted]

                    if IS_TOY_MODEL
                        state[end, :theta_fit_err] = mean(theta_error_fun(θ[:,global_i_sorted], θbb[:,i_sorted]); dims = 2) |> vec |> copy
                    end
                    state[end, :signal_fit_logL] = mean(mle_err[i_sorted])
                    state[end, :signal_fit_rmse] = mean(rmse_err[i_sorted])

                    dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, state)
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
                                    plot(df_inf.epoch, df_inf.signal_fit_logL; title = "min -logL = $(s(minimum(df_inf.signal_fit_logL)))", lab = lab, xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10), ylims = (-Inf, min(-100, maximum(df_inf.signal_fit_logL))))
                                end,
                            )...),
                            layout = @layout([a{0.25w} b{0.75w}]),
                        )
                        # display(pinfer) #TODO
                    end
                end

                return @ntuple(pgan, pmodel, psignals, pinfer, pmmd, pwitness, pheat, pperm)
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
                BSON.bson(joinpath(savefolder, "$(prefix)progress$(suffix).bson"), Dict("progress" => deepcopy(state)))
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
        is_best_model = any(!ismissing, state.signal_fit_logL) && state.signal_fit_logL[findlast(!ismissing, state.signal_fit_logL)] <= minimum(skipmissing(state.signal_fit_logL))

        if is_best_model
            @timeit timer "best model" saveprogress(outfolder, "best-", "")
            if epoch == 0 || time() - cb_state.last_best_checkpoint[] >= saveperiod
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
end

function train_mmd_kernel!(
        logsigma        :: AbstractVecOrMat{Float64};
        m               :: Int             = settings["gan"]["batchsize"],
        lr              :: Float64         = settings["gan"]["kernel"]["stepsize"],
        nbatches        :: Int             = settings["gan"]["kernel"]["nbatches"],
        epochs          :: Int             = settings["gan"]["kernel"]["epochs"],
        kernelloss      :: String          = settings["gan"]["kernel"]["losstype"],
        bwbounds        :: Vector{Float64} = settings["gan"]["kernel"]["bwbounds"],
        recordprogress  :: Bool            = false,
        showprogress    :: Bool            = false,
        plotprogress    :: Bool            = false,
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

    function loss(_logσ, _X, _Y)
        if kernelloss == "tstatistic"
            -mmd_flux_bandwidth_optfun(_logσ, _X, _Y) # Minimize -t = -MMDsq/MMDσ
        else
            -m * mmd_flux(_logσ, _X, _Y) # Minimize -m*MMDsq
        end
    end

    function gradloss(_logσ, _X, _Y)
        ℓ, back = Zygote.pullback(__logσ -> loss(__logσ, _X, _Y), _logσ)
        return ℓ, back
    end

    function checkloss(ℓ)
        if kernelloss == "tstatistic" && abs(ℓ) > 100
            # Denominator has likely shrunk to sqrt(eps), or else we are overtraining
            @info "Loss is too large (ℓ = $ℓ)"
            return false
        else
            return true
        end
    end

    callback = function(epoch, _X, _Y)
        ℓ = loss(logsigma, _X, _Y)
        MMDsq, MMDvar = mmd_and_mmdvar_flux(logsigma, _X, _Y)
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

    function _sampleX̂Y()
        function _tstat_check(_X, _Y)
            (kernelloss != "tstatistic") && return true
            MMDsq, MMDvar = mmd_and_mmdvar_flux(logsigma, _X, _Y)
            MMDsq, MMDvar = m*MMDsq, m^2*MMDvar
            (MMDsq < 0) && return false
            (MMDvar < 100*eps(typeof(MMDvar))) && return false
            return true
        end

        while true
            Y = sampleY(m; dataset = :train)
            X = sampleX(m; dataset = :train)
            X̂ = get_corrected_signal(X)
            _tstat_check(X̂, Y) && return X̂, Y
        end
    end

    for epoch in 1:epochs
        try
            X̂, Y = _sampleX̂Y()
            opt = Flux.ADAM(lr) # new optimizer for each X̂, Y; loss jumps too wildly
            recordprogress && callback(epoch, X̂, Y)
            for _ in 1:nbatches
                ℓ, back = gradloss(logsigma, X̂, Y)
                !checkloss(ℓ) && break
                ∇ℓ = back(1)[1]
                Flux.Optimise.update!(opt, logsigma, ∇ℓ)
                clamp!(logsigma, bwbounds...)
            end
            recordprogress && callback(epoch, X̂, Y)
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

function train_hybrid_gan_model(;
        n          :: Int     = settings["data"]["nsignal"],
        m          :: Int     = settings["gan"]["batchsize"],
        GANrate    :: Int     = settings["gan"]["GANrate"],
        Dsteps     :: Int     = settings["gan"]["Dsteps"],
        kernelrate :: Int     = settings["gan"]["kernel"]["rate"],
        epochs     :: Int     = settings["gan"]["epochs"],
        nbatches   :: Int     = settings["gan"]["nbatches"],
        timeout    :: Float64 = settings["gan"]["traintime"],
        showrate   :: Int     = settings["gan"]["showrate"],
    )
    tstart = Dates.now()

    epoch0 = isempty(state) ? 0 : state.epoch[end]+1
    for epoch in epoch0 .+ (0:epochs)
        try
            if epoch == epoch0
                @timeit timer "initial MMD kernel" train_mmd_kernel!(models["logsigma"])
                @timeit timer "initial callback" callback(epoch0)
                continue
            end

            @timeit timer "epoch" begin
                if mod(epoch, kernelrate) == 0
                    @timeit timer "MMD kernel" train_mmd_kernel!(models["logsigma"])
                end
                @timeit timer "batch loop" for _ in 1:nbatches
                    @timeit timer "MMD generator" begin
                        @timeit timer "sampleX" Xtrain = sampleX(m; dataset = :train)
                        @timeit timer "sampleY" Ytrain = sampleY(m; dataset = :train)
                        @timeit timer "forward" _, back = Zygote.pullback(() -> MMDloss(Xtrain, Ytrain), Flux.params(models["G"]))
                        @timeit timer "reverse" gs = back(1)
                        @timeit timer "update!" Flux.Optimise.update!(optimizers["mmd"], Flux.params(models["G"]), gs)
                    end
                    if mod(epoch, GANrate) == 0
                        @timeit timer "GAN discriminator" for _ in 1:Dsteps
                            @timeit timer "sampleX" Xtrain = sampleX(m; dataset = :train)
                            @timeit timer "sampleY" Ytrain = sampleY(m; dataset = :train)
                            @timeit timer "forward" _, back = Zygote.pullback(() -> Dloss(Xtrain, Ytrain), Flux.params(models["D"]))
                            @timeit timer "reverse" gs = back(1)
                            @timeit timer "update!" Flux.Optimise.update!(optimizers["D"], Flux.params(models["D"]), gs)
                        end
                        @timeit timer "GAN generator" begin
                            @timeit timer "sampleX" Xtrain = sampleX(m; dataset = :train)
                            @timeit timer "forward" _, back = Zygote.pullback(() -> Gloss(Xtrain), Flux.params(models["G"]))
                            @timeit timer "reverse" gs = back(1)
                            @timeit timer "update!" Flux.Optimise.update!(optimizers["G"], Flux.params(models["G"]), gs)
                        end
                    end
                end
                @timeit timer "callback" callback(epoch)
            end

            if mod(epoch, showrate) == 0
                show(stdout, timer); println("\n")
                show(stdout, last(state[:, Not(:theta_fit_err)], 10)); println("\n")
            end
            (epoch == epoch0 + 1) && TimerOutputs.reset_timer!(timer) # throw out initial loop (precompilation, first plot, etc.)

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
    @info "Finished: trained for $(state.epoch[end])/$epochs epochs"

    return nothing
end

train_hybrid_gan_model()

nothing
