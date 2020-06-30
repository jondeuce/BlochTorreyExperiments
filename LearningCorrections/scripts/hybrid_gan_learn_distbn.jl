# Load files
using MMDLearning
pyplot(size=(800,600))
Random.seed!(0);

settings = load_settings(joinpath(@__DIR__, "..", "settings", "hybrid_settings.toml"))
models = Dict{String, Any}()
phys = initialize!(
    ToyModel{Float64}();
    ntrain = 102_400, #TODO settings["gan"]["batchsize"]::Int
    ntest  = 10_240,
    nval   = 10_240,
)

#=
    # Load data samplers
    const IS_TOY_MODEL = false
    const TOY_NOISE_LEVEL = 1e-2
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
=#

# Initialize generator + discriminator + kernel
let
    n  = nsignal(phys)::Int #TODO
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
        MMDLearning.Scale(α, β),
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

# Generator and discriminator losses
ricegen = VectorRicianCorrector(models["G"]) # Generator produces 𝐑^2n outputs parameterizing n Rician distributions
get_D_Y(Y) = models["D"](Y) # discrim on real data
get_D_G_X(X) = models["D"](corrected_signal_instance(ricegen, X)) # discrim on genatr data
Dloss(X,Y) = -mean(log.(get_D_Y(Y)) .+ log.(1 .- get_D_G_X(X)))
Gloss(X) = mean(log.(1 .- get_D_G_X(X)))
MMDloss(X,Y) = size(Y,2) * mmd_flux(models["logsigma"], corrected_signal_instance(ricegen, X), Y) # m*MMD^2 on genatr data

# Global state
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
optimizers = Dict{String,Any}(
    "G"   => Flux.ADAM(settings["gan"]["stepsize"]),
    "D"   => Flux.ADAM(settings["gan"]["stepsize"]),
    "mmd" => Flux.ADAM(settings["gan"]["stepsize"]),
)
cb_state = initialize_callback(phys; nsamples = 2048) #TODO

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

    # Update callback state
    @timeit "update cb state" begin
        update_callback!(cb_state, phys, ricegen; ninfer = ninfer, inferperiod = saveperiod)
    end

    # Perform permutation test
    @timeit "perm test" begin
        permtest = mmd_perm_test_power(models["logsigma"], cb_state["Xθhat"], cb_state["Y"]; batchsize = m, nperms = nperms, nsamples = 1)
        c_α = permtest.c_alpha
        P_α = permtest.P_alpha_approx
        tstat = permtest.MMDsq / permtest.MMDσ
        MMDsq = m * permtest.MMDsq
        MMDvar = m^2 * permtest.MMDvar
    end

    # Compute GAN losses
    d_y = get_D_Y(cb_state["Y"])
    d_g_x = get_D_G_X(cb_state["Xθhat"])
    dloss = -mean(log.(d_y) .+ log.(1 .- d_g_x))
    gloss = mean(log.(1 .- d_g_x))

    # Metrics computed in update_callback!
    @unpack rmse, θ_fit_err, signal_fit_logL, signal_fit_rmse = cb_state["metrics"]
    dt = cb_state["curr_time"] - cb_state["last_time"]

    # Update dataframe
    push!(state, [epoch, :test, dt, gloss, dloss, mean(d_y), mean(d_g_x), MMDsq, MMDvar, tstat, c_α, P_α, rmse, θ_fit_err, signal_fit_logL, signal_fit_rmse])

    function makeplots()
        s = x -> x == round(x) ? round(Int, x) : round(x; sigdigits = 4) # for plotting
        window = 100 # window for plotting error metrics etc.
        try
            pgan = @timeit "gan loss plot" let
                dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, state)
                dfp = dropmissing(dfp[:, [:epoch, :Gloss, :Dloss, :D_Y, :D_G_X]])
                if !isempty(dfp)
                    pgan = @df dfp plot(:epoch, [:Gloss :Dloss :D_Y :D_G_X]; label = ["G loss" "D loss" "D(Y)" "D(G(X))"], lw = 2)
                    (epoch >= lrdroprate) && vline!(pgan, lrdroprate : lrdroprate : epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                    plot!(pgan; xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                else
                    nothing
                end
            end
            !isnothing(pgan) && display(pgan) #TODO

            pmodel = @timeit "model plot" let
                @unpack δθ, ϵθ = cb_state
                plot(
                    plot(
                        plot(mean(δθ; dims = 2); yerr = std(δθ; dims = 2), label = L"signal correction $g_\delta(X)$", c = :red, title = "model outputs vs. data channel"),
                        plot(mean(ϵθ; dims = 2); yerr = std(ϵθ; dims = 2), label = L"noise amplitude $\exp(g_\epsilon(X))$", c = :blue);
                        layout = (2,1),
                    ),
                    plot(permutedims(models["logsigma"]); leg = :none, title = "logσ vs. data channel"),
                )
            end
            display(pmodel) #TODO

            psignals = @timeit "signal plot" let
                @unpack Xθ, Xθhat, δθ, Yθ = cb_state
                nθplot = 4 # number of θ sets to draw for plotting simulated signals
                θplotidx = sample(1:size(Xθ,2), nθplot; replace = false)
                if hasclosedform(phys)
                    plot(
                        [plot(hcat(Yθ[:,j], Xθhat[:,j]); c = [:blue :red], lab = [L"Goal $Y$" L"\hat{X} \sim G(X)"]) for j in θplotidx]...,
                        [plot(hcat(Yθ[:,j] - Xθ[:,j], δθ[:,j]); c = [:blue :red], lab = [L"Goal $Y - X$" L"$g_\delta(X)$"]) for j in θplotidx]...,
                        [plot(Yθ[:,j] - Xθ[:,j] - δθ[:,j]; lab = L"$Y - |X + g_\delta(X)|$") for j in θplotidx]...;
                        layout = (3, nθplot),
                    )
                else
                    plot(
                        [plot(hcat(Xθ[:,j], Xθhat[:,j]); c = [:blue :red], lab = [L"$X$" L"\hat{X} \sim G(X)"]) for j in θplotidx]...,
                        [plot(Xθhat[:,j] - Xθ[:,j]; c = :red, lab = L"$\hat{X} - X$") for j in θplotidx]...,
                        [plot(δθ[:,j]; lab = L"$g_\delta(X)$") for j in θplotidx]...;
                        layout = (3, nθplot),
                    )
                end
            end
            display(psignals) #TODO

            pmmd = @timeit "mmd loss plot" let
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
                    plot(p1, p2, p3, p4)
                else
                    nothing
                end
            end
            !isnothing(pmmd) && display(pmmd) #TODO

            pwitness = nothing #mmd_witness(Xϵ, Y, sigma)
            pheat = nothing #mmd_heatmap(Xϵ, Y, sigma)

            pperm = @timeit "permutation plot" mmd_perm_test_power_plot(permtest)
            display(pperm) #TODO

            pinfer = @timeit "theta inference plot" let
                dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, state)
                df_inf = filter(dfp) do r
                    !ismissing(r.signal_fit_rmse) && !ismissing(r.signal_fit_logL) && !(hasclosedform(phys) && ismissing(r.theta_fit_err))
                end

                if !isempty(dfp) && !isempty(df_inf)
                    @unpack Xθfit, Xθhatfit, Xθδfit, Yfit, Yθfit, Yθhatfit, i_fit, δθfit, θfit, ϵθfit = cb_state
                    @unpack all_signal_fit_logL, all_signal_fit_rmse = cb_state["metrics"]
                    plot(
                        plot(
                            hasclosedform(phys) ?
                                plot(hcat(Yθfit[:,end÷2], Xθfit[:,end÷2]); c = [:blue :red], lab = [L"$Y(\hat{\theta})$ fit" L"$X(\hat{\theta})$ fit"]) :
                                plot(hcat( Yfit[:,end÷2], Xθfit[:,end÷2]); c = [:blue :red], lab = [L"Data $Y$" L"$X(\hat{\theta})$ fit"]),
                            sticks(all_signal_fit_rmse; m = (:circle,4), lab = "rmse: fits"),
                            sticks(all_signal_fit_logL; m = (:circle,4), lab = "-logL: fits"),
                            layout = @layout([a{0.25h}; b{0.375h}; c{0.375h}]),
                        ),
                        let
                            _subplots = Any[]
                            if hasclosedform(phys)
                                prmse = plot(dfp.epoch, dfp.rmse; title = "min rmse = $(s(minimum(dfp.rmse)))", label = L"rmse: $Y(\hat{\theta}) - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$", xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                                pθerr = plot(df_inf.epoch, permutedims(reduce(hcat, df_inf.theta_fit_err)); title = "min max error = $(s(minimum(maximum.(df_inf.theta_fit_err))))", label = permutedims(θlabels(phys)), xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                                append!(_subplots, [prmse, pθerr])
                            end

                            rmselab, logLlab = L"rmse: $Y - \hat{X}(\hat{\theta})$", L"-logL: $Y - \hat{X}(\hat{\theta})$"
                            if false #TODO MRI model
                                rmselab *= "\nrmse prior: $(round(mean(phys.testfits.rmse); sigdigits = 4))"
                                logLlab *= "\n-logL prior: $(round(mean(phys.testfits.loss); sigdigits = 4))"
                            end
                            prmse = plot(df_inf.epoch, df_inf.signal_fit_rmse; title = "min rmse = $(s(minimum(df_inf.signal_fit_rmse)))", lab = rmselab, xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                            plogL = plot(df_inf.epoch, df_inf.signal_fit_logL; title = "min -logL = $(s(minimum(df_inf.signal_fit_logL)))", lab = logLlab, xformatter = x -> string(round(Int, x)), xscale = ifelse(epoch < 10*window, :identity, :log10), ylims = (-Inf, min(-100, maximum(df_inf.signal_fit_logL))))
                            append!(_subplots, [prmse, plogL])

                            plot(_subplots...)
                        end;
                        layout = @layout([a{0.25w} b{0.75w}]),
                    )
                else
                    nothing
                end
            end
            !isnothing(pinfer) && display(pinfer) #TODO

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

    # Save current model + state every `saveperiod` seconds
    if epoch == 0 || time() - cb_state["last_curr_checkpoint"] >= saveperiod
        cb_state["last_curr_checkpoint"] = time()
        @timeit "save current model" saveprogress(outfolder, "current-", "")
        @timeit "make current plots" plothandles = makeplots()
        @timeit "save current plots" saveplots(outfolder, "current-", "", plothandles)
    end

    # Check for and save best model + make best model plots every `saveperiod` seconds
    _is_best_model = collect(skipmissing(state.signal_fit_logL)) |> x -> !isempty(x) && (x[end] <= minimum(x))
    if _is_best_model
        @timeit "save best model" saveprogress(outfolder, "best-", "")
        if time() - cb_state["last_best_checkpoint"] >= saveperiod
            cb_state["last_best_checkpoint"] = time()
            @timeit "make best plots" plothandles = makeplots()
            @timeit "save best plots" saveplots(outfolder, "best-", "", plothandles)
        end
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
            Y = sampleY(phys, m; dataset = :train)
            X = sampleX(phys, m; dataset = :train)
            X̂ = corrected_signal_instance(ricegen, X)
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
                @timeit "initial MMD kernel" train_mmd_kernel!(models["logsigma"])
                @timeit "initial callback" callback(epoch0)
                continue
            end

            @timeit "epoch" begin
                if mod(epoch, kernelrate) == 0
                    @timeit "MMD kernel" train_mmd_kernel!(models["logsigma"])
                end
                @timeit "batch loop" for _ in 1:nbatches
                    @timeit "MMD generator" begin
                        @timeit "sampleX" Xtrain = sampleX(phys, m; dataset = :train)
                        @timeit "sampleY" Ytrain = sampleY(phys, m; dataset = :train)
                        @timeit "forward" _, back = Zygote.pullback(() -> MMDloss(Xtrain, Ytrain), Flux.params(models["G"]))
                        @timeit "reverse" gs = back(1)
                        @timeit "update!" Flux.Optimise.update!(optimizers["mmd"], Flux.params(models["G"]), gs)
                    end
                    if mod(epoch, GANrate) == 0
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

            if mod(epoch, showrate) == 0
                show(stdout, TimerOutputs.get_defaulttimer()); println("\n")
                show(stdout, last(state[:, Not(:theta_fit_err)], 10)); println("\n")
            end
            (epoch == epoch0 + 1) && TimerOutputs.reset_timer!() # throw out initial loop (precompilation, first plot, etc.)

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
