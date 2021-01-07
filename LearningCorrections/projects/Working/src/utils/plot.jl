####
#### Defaults
####

Plots.default(fontfamily = "cmu serif")
Plots.default(titlefontsize = 12)
Plots.default(guidefontsize = 12)
Plots.default(legendfontsize = 9)
Plots.default(tickfontsize = 9)

####
#### PyPlot
####

function pyheatmap(
        imdata::AbstractMatrix;
        formatter = nothing,
        filename = nothing,
        clim = nothing,
        cticks = nothing,
        title = nothing,
        cropnan = true,
        aspect = nothing,
        interpolation = "nearest",
        extent = nothing, # [left, right, bottom, top]
        axis = nothing,
        xlabel = nothing,
        ylabel = nothing,
        figsize = (5.0, 8.0),
        dpi = 150.0,
        savetypes = [".png", ".pdf"]
    )

    if cropnan
        xi = .!all(isnan, imdata; dims = 2)
        xj = .!all(isnan, imdata; dims = 1)
        imdata = !any(xi) || !any(xj) ? [NaN] : imdata[findfirst(vec(xi)) : findlast(vec(xi)), findfirst(vec(xj)) : findlast(vec(xj))]
    end

    plt.figure(; figsize, dpi)
    plt.set_cmap("plasma")
    fig, ax = plt.subplots()
    img = ax.imshow(imdata; aspect, interpolation, extent)
    plt.title(title)
    if axis === :off
        ax.set_axis_off()
    end
    if (aspect !== nothing)
        ext = ax.get_images()[1].get_extent()
        ax.set_aspect(abs((ext[2]-ext[1]) / (ext[4]-ext[3])) * aspect)
    end
    (xlabel !== nothing) && (plt.xlabel(xlabel))
    (ylabel !== nothing) && (plt.ylabel(ylabel))

    (formatter isa Function) && (formatter = plt.matplotlib.ticker.FuncFormatter(formatter))
    cbar = fig.colorbar(img, ticks = cticks, format = formatter, aspect = 40)
    cbar.ax.tick_params(labelsize = 10)

    (clim !== nothing) && img.set_clim(clim...)
    (filename !== nothing) && foreach(ext -> plt.savefig(filename * ext; bbox_inches = "tight", dpi), savetypes)
    plt.close("all")

    return nothing
end

####
#### Plots
####

function plot_gan_loss(logger, cb_state, phys; window = 100, lrdroprate = typemax(Int), lrdrop = 1.0, showplot = false)
    @timeit "gan loss plot" try
        !all(c -> c ∈ propertynames(logger), (:epoch, :dataset, :Gloss, :Dloss, :D_Y, :D_G_X)) && return
        dfp = logger[logger.dataset .=== :val, :]
        epoch = dfp.epoch[end]
        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, dfp)
        dfp = dropmissing(dfp[:, [:epoch, :Gloss, :Dloss, :D_Y, :D_G_X]])
        if !isempty(dfp)
            s = x -> (x == round(Int, x) ? round(Int, x) : round(x; sigdigits = 4)) |> string
            ps = [
                plot(dfp.epoch, dfp.Dloss; label = "D loss", lw = 2, c = :red),
                plot(dfp.epoch, dfp.Gloss; label = "G loss", lw = 2, c = :blue),
                plot(dfp.epoch, [dfp.D_Y dfp.D_G_X]; label = ["D(Y)" "D(G(X))"], c = [:red :blue], lw = 2),
                plot(dfp.epoch, dfp.D_Y - dfp.D_G_X; label = "D(Y) - D(G(X))", c = :green, lw = 2),
            ]
            (epoch >= lrdroprate) && map(ps) do p
                plot!(p, lrdroprate : lrdroprate : epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)", seriestype = :vline)
                plot!(p; xscale = ifelse(epoch < 10*window, :identity, :log10))
            end
            p = plot(ps...)
        else
            p = nothing
        end
        showplot && (p !== nothing) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making gan loss plot")
    end
end

function plot_rician_model(logger, cb_state, phys; bandwidths = nothing, showplot = false)
    @timeit "model plot" try
        @unpack δθ, ϵθ = cb_state
        _subplots = []
        let
            δmid, δlo, δhi = eachrow(δθ) |> δ -> (mean.(δ), quantile.(δ, 0.25), quantile.(δ, 0.75))
            ϵmid, ϵlo, ϵhi = eachrow(ϵθ) |> ϵ -> (mean.(ϵ), quantile.(ϵ, 0.25), quantile.(ϵ, 0.75))
            push!(_subplots, plot(
                plot(δmid; yerr = (δmid - δlo, δhi - δmid), label = L"signal correction $g_\delta(X)$", c = :red, title = "model outputs vs. data channel"),
                plot(ϵmid; yerr = (ϵmid - ϵlo, ϵhi - ϵmid), label = L"noise amplitude $\exp(g_\epsilon(X))$", c = :blue);
                layout = (2,1),
            ))
        end
        bandwidth_plot(logσ::AbstractVector) = bandwidth_plot(permutedims(logσ))
        bandwidth_plot(logσ::AbstractMatrix) = size(logσ,1) == 1 ?
            scatter(1:length(logσ), vec(logσ); label = "logσ", title = "logσ") :
            plot(logσ; leg = :none, marker = (1,:circle), title = "logσ vs. data channel")
        if (bandwidths !== nothing)
            push!(_subplots,
                eltype(bandwidths) <: AbstractArray ?
                    plot(bandwidth_plot.(bandwidths)...; layout = (length(bandwidths), 1)) :
                    plot(bandwidths)
            )
        end
        p = plot(_subplots...)
        showplot && (p !== nothing) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making Rician model plot")
    end
end

function plot_rician_signals(logger, cb_state, phys; showplot = false, nsignals = 4)
    @timeit "signal plot" try
        @unpack Y, Xθ, Xθhat, δθ, Yθ = cb_state
        Yplot = hasclosedform(phys) ? Yθ : Y
        θplotidx = sample(1:size(Xθ,2), nsignals; replace = false)
        p = plot(
            [plot(hcat(Yplot[:,j], Xθhat[:,j]); c = [:blue :red], lab = [L"$Y$" L"\hat{X} \sim G(X(\hat{\theta}))"]) for j in θplotidx]...,
            [plot(hcat(Yplot[:,j] - Xθ[:,j], δθ[:,j]); c = [:blue :red], lab = [L"$Y - X(\hat{\theta})$" L"$g_\delta(X(\hat{\theta}))$"]) for j in θplotidx]...,
            [plot(Yplot[:,j] - Xθ[:,j] - δθ[:,j]; lab = L"$Y - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$") for j in θplotidx]...;
            layout = (3, nsignals),
        )
        showplot && (p !== nothing) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making Rician signal plot")
    end
end

function plot_rician_model_fits(logger, cb_state, phys; showplot = false)
    @timeit "signal plot" try
        @unpack Yfit, Xθfit, Xθhatfit, δθfit = cb_state
        nsignals = 4 # number of θ sets to draw for plotting simulated signals
        θplotidx = sample(1:size(Xθfit,2), nsignals; replace = false)
        p = plot(
            [plot(hcat(Yfit[:,j], Xθhatfit[:,j]); c = [:blue :red], lab = [L"$\hat{X}$" L"\hat{X} \sim G(X(\hat{\theta}))"]) for j in θplotidx]...,
            [plot(hcat(Yfit[:,j] - Xθfit[:,j], δθfit[:,j]); c = [:blue :red], lab = [L"$\hat{X} - X(\hat{\theta})$" L"$g_\delta(X(\hat{\theta}))$"]) for j in θplotidx]...,
            [plot(Yfit[:,j] - Xθfit[:,j] - δθfit[:,j]; lab = L"$\hat{X} - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$") for j in θplotidx]...;
            layout = (3, nsignals),
        )
        showplot && (p !== nothing) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making Rician signal plot")
    end
end

function plot_mmd_losses(logger, cb_state, phys; window = 100, lrdroprate = typemax(Int), lrdrop = 1.0, showplot = false)
    @timeit "mmd loss plot" try
        s = x -> x == round(x) ? round(Int, x) : round(x; sigdigits = 4)
        dfp = logger[logger.dataset .=== :val, :]
        epoch = dfp.epoch[end]
        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, dfp)
        if !isempty(dfp)
            tstat_nan_outliers = map((_tstat, _mmdvar) -> _mmdvar > eps() ? _tstat : NaN, dfp.tstat, dfp.MMDvar)
            tstat_drop_outliers = filter(!isnan, tstat_nan_outliers)
            tstat_median = isempty(tstat_drop_outliers) ? NaN : median(tstat_drop_outliers)
            tstat_ylim = isempty(tstat_drop_outliers) ? nothing : quantile(tstat_drop_outliers, [0.01, 0.99])
            p1 = plot(dfp.epoch, dfp.MMDsq; label = "m*MMD²", title = "median loss = $(s(median(dfp.MMDsq)))") # ylim = quantile(dfp.MMDsq, [0.01, 0.99])
            p2 = plot(dfp.epoch, dfp.MMDvar; label = "m²MMDvar", title = "median m²MMDvar = $(s(median(dfp.MMDvar)))") # ylim = quantile(dfp.MMDvar, [0.01, 0.99])
            p3 = plot(dfp.epoch, tstat_nan_outliers; title = "median t = $(s(tstat_median))", label = "t = MMD²/MMDσ", ylim = tstat_ylim)
            p4 = plot(dfp.epoch, dfp.P_alpha; label = "P_α", title = "median P_α = $(s(median(dfp.P_alpha)))", ylim = (0,1))
            foreach([p1,p2,p3,p4]) do p
                (epoch >= lrdroprate) && vline!(p, lrdroprate : lrdroprate : epoch; line = (1, :dot), label = "lr drop ($(lrdrop)X)")
                plot!(p; xscale = ifelse(epoch < 10*window, :identity, :log10))
            end
            p = plot(p1, p2, p3, p4)
        else
            p = nothing
        end
        showplot && (p !== nothing) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making MMD losses plot")
    end
end

function plot_rician_inference(logger, cb_state, phys; window = 100, showplot = false)
    @timeit "theta inference plot" try
        s = x -> (x == round(Int, x) ? round(Int, x) : round(x; sigdigits = 4)) |> string
        dfp = logger[logger.dataset .=== :val, :]
        epoch = dfp.epoch[end]
        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, dfp)
        df_inf = filter(dfp) do r
            !ismissing(r.Xhat_rmse) && !ismissing(r.Xhat_logL) && (
                hasclosedform(phys) ?
                    (!ismissing(r.Yhat_rmse_true) && !ismissing(r.Yhat_theta_err)) :
                    (!ismissing(r.Yhat_rmse) && !ismissing(r.Xhat_theta_err)))
        end

        if !isempty(dfp) && !isempty(df_inf)
            @unpack all_Xhat_logL, all_Xhat_rmse = cb_state["metrics"]
            @unpack Xθ, Y = cb_state
            p = plot(
                plot(
                    plot(hcat(Y[:,end÷2], Xθ[:,end÷2]); c = [:blue :red], lab = [L"Y data" L"$X(\hat{\theta})$ fit"]),
                    sticks(sort(all_Xhat_rmse[sample(1:end, min(128,end))]); m = (:circle,4), lab = "rmse"),
                    sticks(sort(all_Xhat_logL[sample(1:end, min(128,end))]); m = (:circle,4), lab = "-logL"),
                    layout = @layout([a{0.25h}; b{0.375h}; c{0.375h}]),
                ),
                let
                    _subplots = Any[]
                    if hasclosedform(phys)
                        plogL = plot(df_inf.epoch, df_inf.Yhat_logL_true; title = L"True $-logL(Y)$: min = %$(s(minimum(df_inf.Yhat_logL_true)))", label = L"-logL(Y)", xscale = ifelse(epoch < 10*window, :identity, :log10)) # $Y(\hat{\theta}) - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$
                        pθerr = plot(df_inf.epoch, permutedims(reduce(hcat, df_inf.Yhat_theta_err)); title = L"$\hat{\theta}(Y)$: min max = %$(s(minimum(maximum.(df_inf.Yhat_theta_err))))", label = permutedims(θlabels(phys)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                        append!(_subplots, [plogL, pθerr])
                    else
                        plogL = plot(df_inf.epoch, df_inf.Yhat_logL; title = L"$-logL(Y)$: min = %$(s(minimum(df_inf.Yhat_logL)))", label = L"-logL(Y)", xscale = ifelse(epoch < 10*window, :identity, :log10)) # $Y(\hat{\theta}) - |X(\hat{\theta}) + g_\delta(X(\hat{\theta}))|$
                        pθerr = plot(df_inf.epoch, permutedims(reduce(hcat, df_inf.Xhat_theta_err)); title = L"$\hat{\theta}(\hat{X})$: min max = %$(s(minimum(maximum.(df_inf.Xhat_theta_err))))", label = permutedims(θlabels(phys)), xscale = ifelse(epoch < 10*window, :identity, :log10))
                        append!(_subplots, [plogL, pθerr])
                    end

                    if false #TODO MRI model
                        rmselab *= "\nrmse prior: $(round(mean(phys.valfits.rmse); sigdigits = 4))"
                        logLlab *= "\n-logL prior: $(round(mean(phys.valfits.loss); sigdigits = 4))"
                    end
                    logLlab = [L"-logL(\hat{X})" L"-logL(Y)"]
                    rmselab = [L"rmse(\hat{X})" L"rmse(Y)"] # \hat{X}(\hat{\theta})
                    plogL = plot(df_inf.epoch, [df_inf.Xhat_logL df_inf.Yhat_logL]; title = "min = $(s(minimum(df_inf.Xhat_logL))), $(s(minimum(df_inf.Yhat_logL)))", lab = logLlab, xscale = ifelse(epoch < 10*window, :identity, :log10))
                    prmse = plot(df_inf.epoch, [df_inf.Xhat_rmse df_inf.Yhat_rmse]; title = "min = $(s(minimum(df_inf.Xhat_rmse))), $(s(minimum(df_inf.Yhat_rmse)))", lab = rmselab, xscale = ifelse(epoch < 10*window, :identity, :log10))
                    append!(_subplots, [plogL, prmse])

                    plot(_subplots...)
                end;
                layout = @layout([a{0.25w} b{0.75w}]),
            )
        else
            p = nothing
        end

        showplot && (p !== nothing) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making θ inference plot")
    end
end

function plot_all_logger_losses(logger, cb_state, phys;
        colnames = setdiff(propertynames(logger), [:epoch, :iter, :dataset, :time]),
        dataset = :val,
        window = 100,
        showplot = false,
    )
    @timeit "signal plot" try
        s = x -> (x == round(Int, x) ? round(Int, x) : round(x; sigdigits = 4)) |> string
        dfp = logger[logger.dataset .=== dataset, :]
        epoch = dfp.epoch[end]
        dfp = filter(r -> max(1, min(epoch-window, window)) <= r.epoch, dfp)
        ps = map(sort(colnames)) do colname
            i = (!ismissing).(dfp[!, colname])
            if any(i)
                xdata = dfp.epoch[i]
                ydata = dfp[i, colname]
                if ydata[1] isa AbstractArray
                    ydata = permutedims(reduce(hcat, vec.(ydata)))
                end
                plot(xdata, ydata; lab = :none, title = string(colname), titlefontsize = 6, ytickfontsize = 6, xtickfontsize = 6, xscale = ifelse(epoch < 10*window, :identity, :log10))
            else
                nothing
            end
        end
        p = plot(filter(p -> p !== nothing, ps)...)
        showplot && (p !== nothing) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error making Rician signal plot")
    end
end

# TODO: calls sample_batch
function plot_epsilon(phys, derived; knots = (-1.0, 1.0), seriestype = :line, showplot = false)
    function plot_epsilon_inner(; start, stop, zlen = 256, levels = 50)
        #TODO fixed knots, start, stop
        n, nθ, nz = nsignal(phys)::Int, ntheta(phys)::Int, nlatent(derived["ricegen"])::Int
        _, _, Y, Ymeta = sample_batch(:val; batchsize = zlen * nz)
        X, _, Z = sampleXθZ(derived["cvae"], derived["prior"], Ymeta; posterior_θ = true, posterior_Z = false)
        Z = Z[:,1] |> Flux.cpu |> z -> repeat(z, 1, zlen, nz) |> z -> (foreach(i -> z[i,:,i] .= range(start, stop; length = zlen), 1:nz); z)
        _, ϵ = rician_params(derived["ricegen"], X, reshape(Z, nz, :) |> todevice)
        (size(ϵ,1) == 1) && (ϵ = repeat(ϵ, n, 1))
        log10ϵ = log10.(reshape(ϵ, :, zlen, nz)) |> Flux.cpu
        ps = map(1:nz) do i
            zlabs = nz == 1 ? "" : latexstring(" (" * join(map(j -> L"$Z_%$(j)$ = %$(round(Z[1,1,j]; digits = 2))", setdiff(1:nz, i)), ", ") * ")")
            kwcommon = (; leg = :none, colorbar = :right, color = cgrad(:cividis), xlabel = L"$t$", title = L"$\log_{10}\epsilon$ vs. $t$ and $Z_{%$(i)}$%$(zlabs)")
            if seriestype === :surface
                surface(reshape(1:n,n,1), Z[i,:,i]', log10ϵ[:,:,i]; ylabel = L"$Z_{%$(i)}$", fill_z = log10ϵ[:,:,i], camera = (60.0, 30.0), kwcommon...)
            elseif seriestype === :contour
                contourf(repeat(1:n,1,zlen), repeat(Z[i,:,i]',n,1), log10ϵ[:,:,i]; ylabel = L"$Z_{%$(i)}$", levels, kwcommon...)
            else
                plot(log10ϵ[:,:,i]; line_z = Z[i,:,i]', ylabel = L"$\log_{10}\epsilon$", lw = 2, alpha = 0.3, kwcommon...)
            end
        end
        return ps
    end
    ps = mapreduce(vcat, 1:length(knots)-1; init = Any[]) do i
        plot_epsilon_inner(; start = knots[i], stop = knots[i+1])
    end
    p = plot(ps...)
    if showplot; display(p); end
    return p
end

function plot_θZ_histograms(phys, θ, Z; showplot = false)
    pθs = [histogram(θ[i,:]; nbins = 100, label = θlabels(phys)[i], xlim = θbounds(phys)[i]) for i in 1:size(θ,1)]
    pZs = [histogram(Z[i,:]; nbins = 100, label = L"Z_%$i") for i in 1:size(Z,1)] #TODO
    p = plot(pθs..., pZs...)
    if showplot; display(p); end
    return p
end

function plot_priors(phys, derived; showplot = false)
    θ = sampleθprior(derived["prior"], 10000) |> Flux.cpu
    Z = sampleZprior(derived["prior"], 10000) |> Flux.cpu
    plot_θZ_histograms(phys, θ, Z; showplot)
end

function plot_cvaepriors(phys, derived; showplot = false)
    θ = sampleθprior(derived["cvae_prior"], 10000) |> Flux.cpu
    Z = sampleZprior(derived["cvae_prior"], 10000) |> Flux.cpu
    plot_θZ_histograms(phys, θ, Z; showplot)
end

# TODO: calls sample_batch
function plot_posteriors(phys, derived; showplot = false)
    _, _, Y, Ymeta = sample_batch(:val; batchsize = 10000)
    θ, Z = sampleθZ(derived["cvae"], derived["prior"], Ymeta; posterior_θ = true, posterior_Z = true) .|> Flux.cpu
    plot_θZ_histograms(phys, θ, Z; showplot)
end
