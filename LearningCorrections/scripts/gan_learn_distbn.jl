# Load files
using MMDLearning
pyplot(size=(800,600))
Random.seed!(0);

const IS_TOY_MODEL = true
const TOY_NOISE_LEVEL = 1e-2
const models = Dict{String, Any}()
const settings = load_settings(joinpath(@__DIR__, "..", "settings", "gan_settings.toml"))

# Load data samplers
global sampleX, sampleY, sampleθ, fits_train, fits_test, fits_val
if IS_TOY_MODEL
    _sampleX, _sampleY, _sampleθ = make_toy_samplers(ntrain = settings["gan"]["batchsize"]::Int, epsilon = TOY_NOISE_LEVEL, power = 4.0)
    global sampleX = (m; dataset = :train) -> _sampleX(m) # samples are generated on demand; train/test/val not relevant
    global sampleθ = (m; dataset = :train) -> _sampleθ(m) # samples are generated on demand; train/test/val not relevant
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

# Initialize generator + discriminator
let
    n    = settings["data"]["nsignal"]::Int
    Dh   = settings["gan"]["hdim"]::Int
    Nh   = settings["gan"]["nhidden"]::Int
    dx   = settings["gan"]["maxcorr"]::Float64
    bw   = settings["gan"]["noisebounds"]::Vector{Float64}
    Gact = Flux.relu
    Dact = Flux.relu
    hidden(nlayers, act) = [Flux.Dense(Dh, Dh, act) for _ in 1:nlayers]

    # Slope/intercept for scaling dX to (-dx, dx), logσ to (bw[1], bw[2])
    αbw, βbw = (bw[2]-bw[1])/2, (bw[1]+bw[2])/2
    α = [fill( dx, n); fill(αbw, n)]
    β = [fill(0.0, n); fill(βbw, n)]

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
end

# Generator and discriminator losses
get_D_Y(Y) = models["D"](Y) # discrim on real data
get_D_G_X(X) = models["D"](corrected_signal_instance(X)) # discrim on genatr data (`corrected_signal_instance` wraps the generator `models["G"]`)
Dloss(X,Y) = -mean(log.(get_D_Y(Y)) .+ log.(1 .- get_D_G_X(X)))
Gloss(X) = mean(log.(1 .- get_D_G_X(X)))

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
    :rmse     => Union{Float64, Missing}[],
    :theta_fit_err => Union{Vector{Float64}, Missing}[],
    :signal_fit_logL => Union{Float64, Missing}[],
    :signal_fit_rmse => Union{Float64, Missing}[],
)
optimizers = Dict(
    "G" => Flux.ADAM(settings["gan"]["stepsize"]),
    "D" => Flux.ADAM(settings["gan"]["stepsize"]),
)

callback = let
    local Xval, Yval, θval, valfits

    if IS_TOY_MODEL
        # X and θ are generated on demand
        Xval, Yval, θval = sampleX(settings["gan"]["batchsize"]), sampleY(settings["gan"]["batchsize"]; dataset = :val), sampleθ(settings["gan"]["batchsize"])
        valfits = nothing
    else
        # Sample X and θ randomly, but choose Ys + corresponding fits consistently in order to compare models, and choose Ys with reasonable agreeance with data in order to not be overconfident in improving terrible fits
        Xval = sampleX(settings["gan"]["batchsize"]; dataset = :val)
        θval = sampleθ(settings["gan"]["batchsize"]; dataset = :val)
        iY = if settings["data"]["normalize"]::Bool
            iY = filter(i -> fits_val.loss[i] <= -200.0 && fits_val.rmse[i] <= 0.002, 1:nrow(fits_val)) #TODO
        else
            iY = filter(i -> fits_val.loss[i] <= -125.0 && fits_val.rmse[i] <= 0.15, 1:nrow(fits_val)) #TODO
        end
        iY = sample(MersenneTwister(0), iY, settings["gan"]["batchsize"]; replace = false)
        Yval = sampleY(nothing; dataset = :val)[..,iY]
        valfits = fits_val[iY,:]
    end

    cb_state = (
        X = Xval,
        Y = Yval,
        θ = θval,
        fits = valfits,
        last_time = Ref(time()),
        last_curr_checkpoint = Ref(-Inf),
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
        dX, ϵ = correction_and_noiselevel(X)
        Xϵ = corrected_signal_instance(X, dX, ϵ)

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
        dXθ, ϵθ = correction_and_noiselevel(Xθ)
        Xθϵ = corrected_signal_instance(Xθ, dXθ, ϵθ)

        rmse, θ_fit_err, sig_fit_logL, sig_fit_rmse = missing, missing, missing, missing
        if IS_TOY_MODEL
            rmse = sqrt(mean(abs2, Yθ - (Xθ + dXθ)))
        end

        # Get corrected rician model params; input ν is a model signal
        function get_corrected_ν_and_σ(ν)
            dν, σ = correction_and_noiselevel(ν)
            return abs.(ν .+ dν), σ
        end

        if !isnothing(cb_state.last_θbb[])
            # θbb results from inference on Yθϵ from a previous iteration; use this θbb as a proxy for the current "best guess" θ
            θbb = cb_state.last_θbb[]
            Xθbb = mock_forward_model(θbb, nothing)
            dXθbb, ϵθbb = correction_and_noiselevel(Xθbb)
            Xθϵbb = corrected_signal_instance(Xθbb, dXθbb, ϵθbb)

            global_i_fits = cb_state.last_global_i_fits[] # use same data as previous mle fits
            mle_err = [sum(.-logpdf.(Rician.(get_corrected_ν_and_σ(Xθbb[:,j])...), Yθϵ[:,jY])) for (j,jY) in enumerate(global_i_fits)]
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

        #=
            # Perform permutation test
            @timeit timer "perm test" begin
                permtest = mmd_perm_test_power(logsigma, Xϵ, Y; batchsize = m, nperms = nperms, nsamples = 1)
                c_α = permtest.c_alpha
                P_α = permtest.P_alpha_approx
                tstat = permtest.MMDsq / permtest.MMDσ
                MMDsq = m * permtest.MMDsq
                MMDvar = m^2 * permtest.MMDvar
            end

            # Use permutation test results to compute loss
            #@timeit timer "val loss" ℓ = loss(X, Y)
            reg = lambda * regularizer(dX)
            ℓ = MMDsq + reg

            # Update dataframe
            push!(df, [epoch, dt, ℓ, reg, MMDsq, MMDvar, tstat, c_α, P_α, rmse, copy(logsigma), θ_fit_err, sig_fit_logL, sig_fit_rmse])
        =#

        # Compute losses
        d_y = get_D_Y(Y)
        d_g_x = get_D_G_X(X)
        dloss = -mean(log.(d_y) .+ log.(1 .- d_g_x))
        gloss = mean(log.(1 .- d_g_x))

        # Update dataframe
        push!(state, [epoch, :val, dt, gloss, dloss, mean(d_y), mean(d_g_x), rmse, θ_fit_err, sig_fit_logL, sig_fit_rmse])

        function makeplots()
            s = x -> x == round(x) ? round(Int, x) : round(x; sigdigits = 4) # for plotting
            window = 100 # window for plotting error metrics etc.
            nθplot = 4 # number of sets θ to draw for plotting simulated signals
            try
                pgan = nothing
                @timeit timer "plot gan loss" begin
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
                    plot(mean(dX; dims = 2); yerr = std(dX; dims = 2), label = "signal correction", c = :red, title = "model outputs vs. data channel"),
                    plot(mean(ϵ; dims = 2); yerr = std(ϵ; dims = 2), label = "noise amplitude", c = :blue);
                    layout = (2,1),
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

                #=
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
                =#

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

                    Xθbb = mock_forward_model(θbb, nothing)
                    dXθbb, ϵθbb = correction_and_noiselevel(Xθbb)
                    Xθϵbb = corrected_signal_instance(Xθbb, dXθbb, ϵθbb)
                    mle_err = Optim.minimum.(optres)
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

                return @ntuple(pgan, pmodel, psignals, pinfer) #TODO
                # return @ntuple(pmodel, psignals, ploss, pperm, pwitness, pheat, pinfer) #TODO
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

        # NOTE: `makeplots()` updates `state.signal_fit_logL`, therefore `_check_and_save_best_model()` should be called after `makeplots()`
        _is_best_model() = collect(skipmissing(state.signal_fit_logL)) |> x -> !isempty(x) && (x[end] <= minimum(x))
        _save_best_model() = @timeit timer "save best model" saveprogress(outfolder, "best-", "")
        _check_and_save_best_model() = _is_best_model() ? (_save_best_model(); true) : false

        if epoch == 0 || time() - cb_state.last_curr_checkpoint[] >= saveperiod
            # Save current model + state every `saveperiod` seconds
            cb_state.last_curr_checkpoint[] = time()
            @timeit timer "save current model" saveprogress(outfolder, "current-", "")
            @timeit timer "make current plots" plothandles = makeplots()
            @timeit timer "save current plots" saveplots(outfolder, "current-", "", plothandles)

            # Making plots updates loss; check for best model
            if _check_and_save_best_model()
                cb_state.last_best_checkpoint[] = cb_state.last_curr_checkpoint[]
                @timeit timer "save best plots" saveplots(outfolder, "best-", "", plothandles)
            end
        elseif _check_and_save_best_model()
            # Check for and save best model + make best model plots every `saveperiod` seconds
            if time() - cb_state.last_best_checkpoint[] >= saveperiod
                cb_state.last_best_checkpoint[] = time()
                @timeit timer "make best plots" plothandles = makeplots()
                @timeit timer "save best plots" saveplots(outfolder, "best-", "", plothandles)

                # Making plots updates loss; check for best model
                _check_and_save_best_model()
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
end

function train_gan_model(;
        n          :: Int     = settings["data"]["nsignal"],
        m          :: Int     = settings["gan"]["batchsize"],
        Dsteps     :: Int     = settings["gan"]["Dsteps"],
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
                @timeit timer "initial callback" callback(epoch0)
                continue
            end

            @timeit timer "epoch" begin
                @timeit timer "batch loop" for _ in 1:nbatches
                    @timeit timer "discriminator" for _ in 1:Dsteps
                        @timeit timer "sampleX" Xtrain = sampleX(m; dataset = :train)
                        @timeit timer "sampleY" Ytrain = sampleY(m; dataset = :train)
                        @timeit timer "forward" _, back = Zygote.pullback(() -> Dloss(Xtrain, Ytrain), Flux.params(models["D"]))
                        @timeit timer "reverse" gs = back(1)
                        @timeit timer "update!" Flux.Optimise.update!(optimizers["D"], Flux.params(models["D"]), gs)
                    end
                    @timeit timer "generator" begin
                        @timeit timer "sampleX" Xtrain = sampleX(m; dataset = :train)
                        @timeit timer "forward" _, back = Zygote.pullback(() -> Gloss(Xtrain), Flux.params(models["G"]))
                        @timeit timer "reverse" gs = back(1)
                        @timeit timer "update!" Flux.Optimise.update!(optimizers["G"], Flux.params(models["G"]), gs)
                    end
                end
                @timeit timer "callback" callback(epoch)
            end

            if mod(epoch, showrate) == 0
                show(stdout, timer); println("\n")
                show(stdout, last(state, 10)); println("\n")
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

train_gan_model()

nothing
