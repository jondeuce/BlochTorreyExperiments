function default_savefigs(fig, fname, exts = [".png", ".pdf"])
    for ext in exts
        savefig(fig, fname * ext)
    end
    return fig
end

function plotcircles(circles, bdry = nothing; fname = nothing, disp = (fname == nothing))
    fig = plot(circles; aspectratio = :equal)
    !(bdry == nothing) && plot!(fig, bdry; aspectratio = :equal)
    disp && display(fig)
    !(fname == nothing) && default_savefigs(fig, fname)
    return fig
end

function plotgrids(exteriorgrids, torigrids, interiorgrids; fname = nothing, disp = (fname == nothing))
    numtri = sum(JuAFEM.getncells, exteriorgrids) + sum(JuAFEM.getncells, torigrids) + sum(JuAFEM.getncells, interiorgrids)
    numpts = sum(JuAFEM.getnnodes, exteriorgrids) + sum(JuAFEM.getnnodes, torigrids) + sum(JuAFEM.getnnodes, interiorgrids)
    fig = simpplot(exteriorgrids; colour = :cyan)
    simpplot!(fig, torigrids; colour = :yellow)
    simpplot!(fig, interiorgrids; colour = :red)
    title!("Disjoint Grids: $numtri total triangles, $numpts total points")
    disp && display(fig)
    !(fname == nothing) && default_savefigs(fig, fname)
    return fig
end

function mxplotomega(myelinprob, myelindomains, myelinsubdomains, bdry; titlestr = "Omega", fname = nothing)
    omega = calcomega(myelinprob, myelinsubdomains)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, facecol = omega,
        axis = Float64[mxaxis(bdry)...])
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end

function mxplotmagnitude(sols, btparams, myelindomains, bdry; titlestr = "Magnitude", fname = nothing)
    Umagn = reduce(vcat, norm.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    @unpack R2_sp, R2_lp, R2_Tissue = btparams
    caxis = (0.0, exp(-min(R2_sp, R2_lp, R2_Tissue) * sols[1].t[end]))
    mxsimpplot(getgrid.(myelindomains); newfigure = true, facecol = Umagn,
        axis = Float64[mxaxis(bdry)...], caxis = Float64[caxis...])
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end

function mxplotphase(sols, btparams, myelindomains, bdry; titlestr = "Phase", fname = nothing)
    Uphase = reduce(vcat, angle.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, facecol = Uphase,
        axis = Float64[mxaxis(bdry)...])
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end

function plotbiexp(sols, btparams, myelindomains, outercircles, innercircles, bdry;
        titlestr = "Signal Magnitude vs. Time",
        opts = NNLSRegression(PlotDist = !AVOID_MAT_PLOTS),
        fname = nothing,
        disp = (fname == nothing)
    )
    tspan = get_tspan(opts)
    ts = get_tpoints(opts)
    signals = calcsignal(sols, ts, myelindomains)

    myelin_area = intersect_area(outercircles, bdry) - intersect_area(innercircles, bdry)
    total_area = area(bdry)
    ext_area = total_area - myelin_area

    # In the high diffusion & highly permeable membrane limit, spins are equally
    # likely to be anywhere on the grid, hence experience a decay rate R2_mono
    # on the average, where R2_mono is the area averaged R2 of each compartment
    R2_mono = (btparams.R2_sp * myelin_area + btparams.R2_lp * ext_area) / total_area
    y_monoexp = @. total_area * exp(-ts * R2_mono)

    # In the low diffusion OR impermeable membrane limit, spins are confined to
    # their separate regions and experience their compartment R2 only
    y_biexp = @. ext_area * exp(-ts * btparams.R2_lp) + myelin_area * exp(-ts * btparams.R2_sp)

    if AVOID_MAT_PLOTS
        fig = plot(1000 .* ts, [norm.(signals), y_biexp];
            linewidth = 5, marker = :circle, markersize =10,
            grid = true, minorgrid = true, legend = :topright,
            xticks = 1000 .* ts, xrotation = -60, xlims = 1000 .* tspan,
            labels = ["Simulated", "Bi-Exponential"],
            ylabel = "S(t) Magnitude", xlabel = "Time [ms]",
            title = titlestr)
        
        !(fname == nothing) && default_savefigs(fig, fname)
        disp && display(fig)
    else
        mxcall(:figure, 0)
        mxcall(:plot, 0, collect(1000.0.*ts), [norm.(signals) y_biexp])
        mxcall(:legend, 0, "Simulated", "Bi-Exponential")
        mxcall(:title, 0, titlestr)
        mxcall(:xlabel, 0, "Time [ms]")
        mxcall(:xlim, 0, 1000.0 .* [tspan...])
        mxcall(:ylabel, 0, "S(t) Magnitude")
        !(fname == nothing) && mxsavefig(fname)
    end

    return nothing
end

function plotSEcorr(
        sols, btparams, myelindomains;
        opts::NNLSRegression = NNLSRegression(PlotDist = !AVOID_MAT_PLOTS),
        fname = nothing,
        disp = (fname == nothing)
    )
    tspan = get_tspan(opts)
    ts = get_tpoints(opts)
    signals = calcsignal(sols, ts, myelindomains)

    MWImaps, MWIdist, MWIpart = fitmwfmodel(signals, opts)

    if AVOID_MAT_PLOTS
        mwf = _getmwf(opts, MWImaps, MWIdist, MWIpart)
        T2Vals = 1000 .* get_T2vals(opts)
        xtickvals = length(T2Vals) <= 60 ? T2Vals : T2Vals[1:2:end] # length cannot be more than 120

        fig = plot(T2Vals, MWIdist[:];
            seriestype = :sticks,
            xscale = :log10,
            linewidth = 5, markersize = 5, marker = :circle,
            grid = true, minorgrid = true, legend = :none,    
            xticks = xtickvals, formatter = x -> string(round(x; sigdigits = 3)), xrotation = -60,
            xlim = 1000 .* opts.T2Range,
            xlabel = "T2 [ms]",
            title = "T2 Distribution: nT2 = $(opts.nT2), mwf = $(round(mwf; digits=4))"
        )
        vline!(fig, 1000 .* [opts.SPWin; opts.MPWin];
            xscale = :log10, linewidth = 5, linestyle = :dot, color = :red)
        
        !(fname == nothing) && default_savefigs(fig, fname)
        disp && display(fig)
    elseif opts.PlotDist
        !(fname == nothing) && mxsavefig(fname)
    end

    return MWImaps, MWIdist, MWIpart
end

function plotMWF(params, mwf, mwftrue = nothing;
        plottypes = [:mwf, :mwferror],
        mwfmethod = nothing,
        fname = nothing,
        disp = (fname == nothing)
    )
    mwfplottypes = [:mwf, :mwfplot]
    errorplottypes = [:mwferror, :mwferrorplot, :error, :errorplot]

    # Split params into groups where all fields of params are the same except for :theta
    groups, groupindices = partitionby(params, :theta)
    theta = [[p.theta for p in g] for g in groups]
    MWF = [[mwf[i] for i in gi] for gi in groupindices]
    theta = broadcast!(θ -> θ .= rad2deg.(θ), theta, theta) # change units to degrees
    MWF = broadcast!(mwf -> mwf .= 100 .* mwf, MWF, MWF) # change units to percentages
    
    figs = []
    default_kwargs = Dict(
        :linewidth => 5, :marker => :circle, :markersize => 10,
        :grid => true, :minorgrid => true, :legend => :none,
        :ylabel => "MWF [%]", :xlabel => "Angle [degrees]", :title => "MWF vs. Angle"
    )
    
    for plottype in plottypes
        xdata, ydata = theta, MWF
        props = deepcopy(default_kwargs)
        if plottype ∈ errorplottypes
            if (mwftrue == nothing)
                @warn "MWF error plot requested but true mwf value not given; skipping plottype = $plottype"
                continue
            end
            ydata = map(mwf -> mwf .- (100*mwftrue), MWF)
            props[:title] = "MWF Error vs. Angle"
            props[:ylabel] = "MWF Error [%]"
        else
            if !(plottype ∈ mwfplottypes)
                @warn "Skipping unknown plottype $plottype"
                continue
            end
        end
        !(mwfmethod == nothing) && (props[:title] = string(mwfmethod) * " " * props[:title])
        !(mwftrue == nothing) && (props[:title] *= " (True MWF = $(round(100*mwftrue; sigdigits=4))%)")

        fig = plot(xdata, ydata; props...)
        push!(figs, fig)
        
        if !(fname == nothing)
            _fname = fname
            (mwfmethod != nothing) && (_fname *= "__" * string(mwfmethod))
            (plottype ∈ errorplottypes) ? (_fname *= "__" * "MWFErrorPlot") : (_fname *= "__" * "MWFPlot")
            default_savefigs(fig, _fname)
        end
        disp && display(fig)
    end

    return figs
end

function plotMWF(results::Dict;
        plottypes = [:mwf, :mwferror],
        fname = nothing,
        disp = (fname == nothing)
    )
    # mwfvalues is an array of Dict{Symbol,T}'s, and params is an array of BlochTorreyParameters{T}'s
    @unpack params, mwfvalues = results
    (isempty(params) || isempty(mwfvalues)) && return nothing

    params = convert(Vector{typeof(params[1])}, params) # force proper typing of params
    mwftrue = get(mwfvalues[1], :exact, nothing) # get mwftrue, or nothing if the key doesn't exist
    for key in keys(mwfvalues[1])
        key == :exact && continue # skip plotting horizontal lines of exact value
        mwf = [d[key] for d in mwfvalues]
        plotMWF(params, mwf, mwftrue;
            plottypes = plottypes, mwfmethod = key, disp = disp, fname = fname)
    end
    return nothing
end

function partitionby(s::AbstractVector{S}, field) where {S}
    seenindices = Set{Int}()
    groups, groupindices = [], []
    while length(seenindices) < length(s)
        for i in 1:length(s)
            i ∈ seenindices && continue
            el1 = s[i]
            idx = Int[i]
            group = S[el1]
            for j in 1:length(s)
                ((i == j) || (j ∈ seenindices)) && continue
                el = s[j]
                if all(f -> (f == field) || (getfield(el1, f) == getfield(el, f)), fieldnames(S))
                    push!(idx, j)
                    push!(group, el)
                end
            end
            for k in idx
                push!(seenindices, k)
            end
            push!(groupindices, sort!(idx))
            push!(groups, sort!(group; by = el -> getfield(el, field)))
        end
    end
    return groups, groupindices
end

# Save plot
function mxsavefig(fname; close = true, fig = false, png = true, pdf = false, eps = false)
    flags = String[]
    png && push!(flags, "-png")
    pdf && push!(flags, "-pdf")
    eps && push!(flags, "-eps")
    !isempty(flags) && mxcall(:export_fig, 0, fname, flags...)
    fig && mxcall(:savefig, 0, fname * ".fig")
    close && mxcall(:close, 0)
end