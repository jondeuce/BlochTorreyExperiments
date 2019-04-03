function plotomega(myelinprob, myelindomains, myelinsubdomains, bdry; titlestr = "Omega", fname = nothing)
    omega = calcomega(myelinprob, myelinsubdomains)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = omega)
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end

function plotmagnitude(sols, btparams, myelindomains, bdry; titlestr = "Magnitude", fname = nothing)
    Umagn = reduce(vcat, norm.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = Umagn)
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end

function plotphase(sols, btparams, myelindomains, bdry; titlestr = "Phase", fname = nothing)
    Uphase = reduce(vcat, angle.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = Uphase)
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end

function plotbiexp(sols, btparams, myelindomains, outercircles, innercircles, bdry;
        titlestr = "Signal Magnitude vs. Time",
        disp = false,
        fname = nothing
    )
    opts = NNLSRegression()
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
        
        if !(fname == nothing)
            savefig(fig, fname * ".pdf")
            savefig(fig, fname * ".png")
        end
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
        opts::NNLSRegression = NNLSRegression(),
        disp = false,
        fname = nothing
    )
    tspan = get_tspan(opts)
    ts = get_tpoints(opts)
    signals = calcsignal(sols, ts, myelindomains)

    PLOTDIST = !AVOID_MAT_PLOTS
    MWImaps, MWIdist, MWIpart = fitmwfmodel(signals, opts; PLOTDIST = PLOTDIST)

    if AVOID_MAT_PLOTS
        mwf = _getmwf(opts, MWImaps, MWIdist, MWIpart)
        T2Vals = 1000 .* get_T2vals(opts)
        xtickvals = length(T2Vals) <= 40 ? T2Vals :
                    length(T2Vals) <= 80 ? T2Vals[1:2:end] :
                    T2Vals[1:3:end] # length cannot be more than 120

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
        
        if !(fname == nothing)
            savefig(fig, fname * ".pdf")
            savefig(fig, fname * ".png")
        end
        disp && display(fig)
    else
        (PLOTDIST && !(fname == nothing)) && mxsavefig(fname)
    end

    return MWImaps, MWIdist, MWIpart
end

function plotMWF(params, mwf, mwftrue = nothing; disp = false, fname = nothing)
    groups, groupindices = partitionby(params, :theta)

    theta = [[p.theta for p in g] for g in groups]
    MWF = [[mwf[i] for i in gi] for gi in groupindices]
    theta = broadcast!(θ -> θ .= rad2deg.(θ), theta, theta) # change units to degrees
    MWF = broadcast!(mwf -> mwf .= 100 .* mwf, MWF, MWF) # change units to percentages

    title_str = "Measured MWF vs. Angle"
    !(mwftrue == nothing) && (title_str *= " (True MWF = $(round(mwftrue; sigdigits=4))%)")
    
    fig = plot(theta, MWF;
        linewidth = 5, marker = :circle, markersize =10,
        grid = true, minorgrid = true, legend = :none,
        ylabel = "MWF [%]", xlabel = "Angle [degrees]", title = title_str)
    
    if !(fname == nothing)
        savefig(fig, fname * ".pdf")
        savefig(fig, fname * ".png")
    end
    disp && display(fig)

    return fig
end
function plotMWF(results::Dict; disp = false, fname = nothing)
    # mwfvalues is an array of Dict{Symbol,T}'s, and params is an array of BlochTorreyParameters{T}'s
    @unpack params, mwfvalues = results
    (isempty(params) || isempty(mwfvalues)) && return nothing

    params = convert(Vector{typeof(params[1])}, params) # force proper typing of params
    mwftrue = get(mwfvalues[1], :exact, nothing) # get mwftrue, or nothing if the key doesn't exist
    for key in keys(mwfvalues[1])
        key == :exact && continue # skip plotting horizontal lines of exact value
        mwf = [d[key] for d in mwfvalues]
        fname_appended = fname == nothing ? nothing : fname * "__" * string(key)
        plotMWF(params, mwf, mwftrue; disp = disp, fname = fname_appended)
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
function mxsavefig(fname; close = true, fig = true, png = true, pdf = false, eps = false)
    flags = String[]
    png && push!(flags, "-png")
    pdf && push!(flags, "-pdf")
    eps && push!(flags, "-eps")
    !isempty(flags) && mxcall(:export_fig, 0, fname, flags...)
    fig && mxcall(:savefig, 0, fname * ".fig")
    close && mxcall(:close, 0)
end