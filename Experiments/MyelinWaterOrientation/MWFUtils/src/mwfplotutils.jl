function plotmagnitude(sols, btparams, myelindomains, bdry; titlestr = "Magnitude", fname = nothing)
    Umagn = reduce(vcat, norm.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = Umagn)
    mxcall(:title, 0, titlestr)

    # allgrids = vcat(exteriorgrids[:], torigrids[:], interiorgrids[:])
    # mxsimpplot(allgrids; newfigure = true, axis = mxaxis(bdry))

    !(fname == nothing) && mxsavefig(fname)

    nothing
end

function plotphase(sols, btparams, myelindomains, bdry; titlestr = "Phase", fname = nothing)
    Uphase = reduce(vcat, angle.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = Uphase)
    mxcall(:title, 0, titlestr)

    !(fname == nothing) && mxsavefig(fname)

    nothing
end

function plotbiexp(sols, btparams, myelindomains, outercircles, innercircles, bdry;
        titlestr = "Signal Magnitude vs. Time",
        disp = false,
        fname = nothing
    )

    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    ts = tspan[1]:TE:tspan[2] # signal after each echo
    Stotal = calcsignal(sols, ts, myelindomains)

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
        fig = plot(1000 .* ts, [norm.(Stotal), y_biexp];
            linewidth = 5, marker = :circle, markersize =10,
            grid = true, legend = :topright,
            xlims = 1000 .* tspan,
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
        mxcall(:plot, 0, collect(1000.0.*ts), [norm.(Stotal) y_biexp])
        mxcall(:legend, 0, "Simulated", "Bi-Exponential")
        mxcall(:title, 0, titlestr)
        mxcall(:xlabel, 0, "Time [ms]")
        mxcall(:xlim, 0, 1000.0 .* [tspan...])
        mxcall(:ylabel, 0, "S(t) Magnitude")
        !(fname == nothing) && mxsavefig(fname)
    end

    nothing
end

function plotSEcorr(sols, btparams, myelindomains; disp = false, fname = nothing)
    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    nTE = 32
    ts = tspan[1]:TE:tspan[2] # signal after each echo
    Stotal = calcsignal(sols, ts, myelindomains)

    PLOTDIST = !AVOID_MAT_PLOTS
    T2Range = [8e-3, 2.0]
    spwin = [8e-3, 25e-3]
    mpwin = [25e-3, 200e-3]
    nT2 = 40
    MWImaps, MWIdist, MWIpart = fitmwfmodel(Stotal, NNLSRegression();
        TE = TE, nTE = nTE,
        T2Range = T2Range, nT2 = nT2, spwin = spwin, mpwin = mpwin,
        PLOTDIST = PLOTDIST)
    
    if AVOID_MAT_PLOTS
        mwf = _getmwf(NNLSRegression(), MWImaps, MWIdist, MWIpart)
        T2vals = 1000 .* exp10.(range(log10(T2Range[1]), stop=log10(T2Range[2]), length=nT2))

        fig = plot(T2vals, MWIdist[:];
            xscale = :log10,
            linewidth = 5, markersize = 5, marker = :circle,
            legend = :none,    
            xlim = 1000 .* T2Range,
            seriestype = :sticks,
            xticks = T2vals, formatter = x -> string(round(x; sigdigits = 3)), xrotation = -60,
            xlabel = "T2 [ms]",
            title = "T2 Distribution: nT2 = $nT2, mwf = $(round(mwf; digits=4))"
        )
        vline!(fig, 1000 .* [spwin..., mpwin...]; xscale = :log10, linewidth = 5, linestyle = :dot, color = :red)
        
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

function plotMWF(params, mwfvalues; disp = false, fname = nothing)
    params = convert(Vector{typeof(params[1])}, params) # Ensure proper typing for partitionby
    groups, groupindices = partitionby(params, :theta)

    theta = [[p.theta for p in g] for g in groups]
    MWF = [[mwfvalues[i][:NNLSRegression] for i in gi] for gi in groupindices]
    theta = broadcast!(θ -> θ .= rad2deg.(θ), theta, theta) # change units to degrees
    MWF = broadcast!(mwf -> mwf .= 100 .* mwf, MWF, MWF) # change units to percentages

    true_mwf = 100 * mwfvalues[1][:exact]
    fig = plot(theta, MWF;
        linewidth = 5, marker = :circle, markersize =10, legend = :none,
        ylabel = "MWF [%]", xlabel = "Angle [degrees]",
        title = "Measured MWF vs. Angle: True MWF = $(round(true_mwf; sigdigits=4))%")
    
    if !(fname == nothing)
        savefig(fig, fname * ".pdf")
        savefig(fig, fname * ".png")
    end
    disp && display(fig)

    return theta, MWF
end
plotMWF(results::Dict; kwargs...) = plotMWF(results[:params], results[:mwfvalues]; kwargs...)

function partitionby(s::AbstractVector{S}, field) where {S}
    # vals = [getfield(s,f) for s in s]
    # uniqueset = Set(unique(vals)...)
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
                if all(f -> (f == field) || (getfield(el1,f) == getfield(el,f)), fieldnames(S))
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