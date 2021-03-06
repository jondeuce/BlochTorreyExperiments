function wrap_string(str, len, dlm, newdlm = dlm)
    isempty(str) && return str
    parts = split(str, dlm)
    out = ""
    currlen = 0
    for i in 1:length(parts)-1
        out *= parts[i]
        currlen += length(parts[i])
        out = currlen >= len ? (currlen = 0; out * "\n") : out * newdlm
    end
    out *= parts[end]
    return out
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

function mxplotomega(
        myelinprob, myelindomains, myelinsubdomains, bdry;
        titlestr = "Omega", fname = nothing, kwargs...
    )
    omega = calcomega(myelinprob, myelinsubdomains)
    mxsimpplot(getgrid.(myelindomains);
        facecol = omega, axis = Float64[mxaxis(bdry)...],
        kwargs...)
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end

###
### Magnitude, phase, and longitudinal plotting
###

trans(::Type{uType}, sol::ODESolution, t = sol.t[end]) where {uType} = transverse_signal(reinterpret(uType, sol(t)))
long(::Type{uType}, sol::ODESolution, t = sol.t[end]) where {uType} = longitudinal_signal(reinterpret(uType, sol(t)))
calctimes(sol::ODESolution, length::Int = 100) = range(sol.prob.tspan...; length = length)
calctimes(sol::ODESolution, dt::Real) = sol.prob.tspan[1] : dt : sol.prob.tspan[2]
calcmag(::Type{uType}, sols, ts = sols[1].prob.tspan[2]) where {uType} = reduce(vcat, reduce(hcat, norm.(trans(uType, s, t)) for t in ts) for s in sols)
calcphase(::Type{uType}, sols, ts = sols[1].prob.tspan[2]) where {uType} = reduce(vcat, reduce(hcat, angle.(trans(uType, s, t)) for t in ts) for s in sols)
calclong(::Type{uType}, sols, ts = sols[1].prob.tspan[2]) where {uType} = reduce(vcat, reduce(hcat, long(uType, s, t) for t in ts) for s in sols)

function mxplotmagnitude(
        ::Type{uType}, sols, btparams, myelindomains, bdry;
        titlestr = "Magnitude", fname = nothing, kwargs...
    ) where {uType <: FieldType}
    Umagn = calcmag(uType, sols)
    mxsimpplot(getgrid.(myelindomains);
        facecol = Umagn, axis = Float64[mxaxis(bdry)...], caxis = Float64[0.0, maximum(Umagn)], kwargs...)
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end
mxplotmagnitude(sols, btparams, myelindomains, bdry; kwargs...) =
    mxplotmagnitude(Vec{2,Float64}, sols, btparams, myelindomains, bdry; kwargs...)

function mxplotphase(
        ::Type{uType}, sols, btparams, myelindomains, bdry;
        titlestr = "Phase", fname = nothing, kwargs...
    ) where {uType <: FieldType}
    Uphase = calcphase(uType, sols)
    mxsimpplot(getgrid.(myelindomains);
        facecol = Uphase, axis = Float64[mxaxis(bdry)...], kwargs...)
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end
mxplotphase(sols, btparams, myelindomains, bdry; kwargs...) =
    mxplotphase(Vec{2,Float64}, sols, btparams, myelindomains, bdry; kwargs...)

function mxplotlongitudinal(
        ::Type{uType}, sols, btparams, myelindomains, bdry;
        titlestr = "Longitudinal", fname = nothing, steadystate = 1, kwargs...
    ) where {uType <: Vec{3}}
    Mz = steadystate .- calclong(uType, sols)
    mxsimpplot(getgrid.(myelindomains);
        facecol = Mz, axis = Float64[mxaxis(bdry)...], kwargs...)
    mxcall(:title, 0, titlestr)
    !(fname == nothing) && mxsavefig(fname)
    return nothing
end
mxplotlongitudinal(sols, btparams, myelindomains, bdry; kwargs...) =
    mxplotlongitudinal(Vec{2,Float64}, sols, btparams, myelindomains, bdry; kwargs...)

###
### Magnitude, phase, and longitudinal gifs
###

function mxgifmagnitude(
        ::Type{uType}, sols, btparams, myelindomains, bdry;
        titlestr = "Magnitude", fname = nothing, totaltime = 10.0, kwargs...
    ) where {uType <: FieldType}
    @assert !(fname == nothing)
    ts = calctimes(sols[1], 2.5e-3) # unique!(sort!(round.(sols[1].t; digits=3))) # calctimes(sols[1], 100)
    Umagn = calcmag(uType, sols, ts)
    title = [titlestr * " (t = " * @sprintf("%7.2f", 1000*t) * " ms)" for t in ts]
    mxsimpgif(getgrid.(myelindomains);
        filename = fname, facecol = Umagn, caxistype = "all", title = title, imsize = 0.5 .* [1 1], imscale = 1.0, totaltime = totaltime, kwargs...)
    return nothing
end
mxgifmagnitude(sols, btparams, myelindomains, bdry; kwargs...) =
    mxgifmagnitude(Vec{2,Float64}, sols, btparams, myelindomains, bdry; kwargs...)

function mxgifphase(
        ::Type{uType}, sols, btparams, myelindomains, bdry;
        titlestr = "Phase", fname = nothing, totaltime = 10.0, kwargs...
    ) where {uType <: FieldType}
    @assert !(fname == nothing)
    ts = calctimes(sols[1], 2.5e-3) # unique!(sort!(round.(sols[1].t; digits=3))) # calctimes(sols[1], 100)
    Uphase = calcphase(uType, sols, ts)
    title = [titlestr * " (t = " * @sprintf("%7.2f", 1000*t) * " ms)" for t in ts]
    mxsimpgif(getgrid.(myelindomains);
        filename = fname, facecol = Uphase, caxistype = "all", title = title, imsize = 0.5 .* [1 1], imscale = 1.0, totaltime = totaltime, kwargs...)
    return nothing
end
mxgifphase(sols, btparams, myelindomains, bdry; kwargs...) =
    mxgifphase(Vec{2,Float64}, sols, btparams, myelindomains, bdry; kwargs...)

function mxgiflongitudinal(
        ::Type{uType}, sols, btparams, myelindomains, bdry;
        titlestr = "Longitudinal", fname = nothing, steadystate = 1, totaltime = 10.0, kwargs...
    ) where {uType <: Vec{3}}
    @assert !(fname == nothing)
    ts = calctimes(sols[1], 2.5e-3) # unique!(sort!(round.(sols[1].t; digits=3))) # calctimes(sols[1], 100)
    Mz = steadystate .- calclong(uType, sols, ts)
    title = [titlestr * " (t = " * @sprintf("%7.2f", 1000*t) * " ms)" for t in ts]
    mxsimpgif(getgrid.(myelindomains);
        filename = fname, facecol = Mz, caxistype = "all", title = title, imsize = 0.5 .* [1 1], imscale = 1.0, totaltime = totaltime, kwargs...)
    return nothing
end
mxgiflongitudinal(sols, btparams, myelindomains, bdry; kwargs...) =
    mxgiflongitudinal(Vec{2,Float64}, sols, btparams, myelindomains, bdry; kwargs...)

###
### Total signal plotting
###

function plotmultiexp(sols, btparams, myelindomains, outercircles, innercircles, bdry;
        titlestr = "Signal Magnitude vs. Time",
        opts = NNLSRegression(PlotDist = !AVOID_MAT_PLOTS),
        fname = nothing,
        disp = (fname == nothing)
    )
    # Extract signals from last (0, nTE*TE) of simulation
    signals = transverse_signal(calcsignal(sols, get_tpoints(opts, sols[1].prob.tspan), btparams, myelindomains))
    S0 = norm(signals[1])
    
    # Default timespan/timepoints in range (0, nTE*TE)
    tspan = get_tspan(opts)
    ts = get_tpoints(opts)

    # Extract area fractions
    outercircles_area = intersect_area(outercircles, bdry)
    innercircles_area = intersect_area(innercircles, bdry)
    total_area = area(bdry)
    mvf = (outercircles_area - innercircles_area) / total_area  # myelin (small pool) volume fraction
    ivf = innercircles_area / total_area # intra-cellular (large pool/axonal) volume fraction
    evf = 1 - outercircles_area / total_area  # extra-cellular (tissue) volume fraction
    mwf, iwf, ewf = mvf/(2-mvf), 2*ivf/(2-mvf), 2*evf/(2-mvf) # Assumes PD_sp/PD_lp == 1/2

    # Extract relaxation rates
    R2_mw = btparams.R2_sp # myelin-water (small pool) T2
    R2_iw = btparams.R2_lp # intra-cellular (large pool/axonal) T2
    R2_ew = btparams.R2_Tissue # extra-cellular (tissue) T2

    # In the high diffusion & high permeability limit, spins are equally likely
    # to be anywhere on the grid, and therefore experience a decay rate R2_mono
    # on the average, where R2_mono is the area averaged R2 of each compartment
    R2_mono = mvf * R2_mw + ivf * R2_iw + evf * R2_ew
    y_monoexp = @. S0 * exp(-ts * R2_mono)

    # In the high diffusion & low permeability limit, spins are confined to
    # their separate regions and experience their compartment R2 only
    y_multiexp = @. S0 * (mwf * exp(-ts * R2_mw) + iwf * exp(-ts * R2_iw) + ewf * exp(-ts * R2_ew))

    props = Dict{Symbol,Any}(
        :linewidth => 4, :marker => :circle, :markersize => 10,
        :grid => true, :minorgrid => true, :legend => :topright,
        :xticks => 1000 .* ts, :xrotation => -60, :xlims => 1000 .* tspan,
        :labels => ["Multi-Exponential" "Mono-Exponential" "Simulated"],
        :ylabel => "S(t) Magnitude", :xlabel => "Time [ms]",
        :title => titlestr)
    fig = plot(1000 .* ts, [y_multiexp y_monoexp norm.(signals)]; props...)
    !(fname == nothing) && default_savefigs(fig, fname)
    disp && display(fig)

    return nothing
end

function plotsignal(tpoints, signals;
        timeticks = range(tpoints[1], tpoints[end]; length = 65),
        titlestr = "Complex Signal vs. Time",
        apply_pi_correction = true,
        fname = nothing,
        disp = (fname == nothing)
    )
    trans = transverse_signal(signals)
    allfigs = []

    mag_props = Dict{Symbol,Any}(
        :seriestype => :line, :linewidth => 2, :marker => :none, #:marker => :circle, :markersize => 10,
        :grid => true, :minorgrid => true, :legend => :topright,
        :xticks => 1000 .* timeticks,
        :xrotation => -60, :xlims => 1000 .* extrema(tpoints),
        :formatter => x -> string(round(x; sigdigits = 3)),
        :labels => "Magnitude", :ylabel => L"$S(t)$ Magnitude", :xlabel => "Time [ms]",
        :title => titlestr)
    xdata, ydata = 1000 .* tpoints, norm.(trans)
    push!(allfigs, plot(xdata, ydata; mag_props...))

    pha_props = Dict{Symbol,Any}(mag_props...,
        :seriestype => :steppost, :linewidth => 1, :marker => :none, #:m => :square, :ms => 5,
        :linecolour => :red, :ytick => -180:30:180, :title => "",
        :labels => "Phase (deg)", :ylabel => L"$S(t)$ Phase (deg)")
    xdata, ydata = 1000 .* tpoints, rad2deg.(angle.(trans))
    if apply_pi_correction
        phase_corrections = ifelse.(isodd.(1:length(ydata)), -90, 90)
        ydata = ydata .+ phase_corrections
        pha_props = Dict{Symbol,Any}(pha_props...,
            :ytick => :auto, #:seriestype => :line,
            :labels => L"$\pi$-corrected Phase (deg)", :ylabel => L"$S(t)$ $\pi$-corrected Phase (deg)")
    end
    push!(allfigs, plot(xdata, ydata; pha_props...))

    if eltype(signals) <: Vec{3}
        long_props = Dict{Symbol,Any}(mag_props...,
            :linecolour => :green, :title => "",
            :labels => "Longitudinal", :ylabel => L"$S(t)$ Longitudinal")
        xdata, ydata = 1000 .* tpoints, longitudinal.(signals)
        push!(allfigs, plot(xdata, ydata; long_props...))
    end

    fig = plot(allfigs...; layout = (length(allfigs), 1))
    !(fname == nothing) && default_savefigs(fig, fname)
    disp && display(fig)

    return nothing
end
function plotsignal(results::Dict; kwargs...)
    @unpack tpoints, signals = results
    return plotsignal(tpoints, signals; kwargs...)
end

function plotSEcorr(
        sols, btparams, myelindomains;
        opts::NNLSRegression = NNLSRegression(PlotDist = !AVOID_MAT_PLOTS),
        mwftrue = nothing,
        fname = nothing,
        disp = (fname == nothing)
    )
    signals = transverse_signal(calcsignal(sols, get_tpoints(opts, sols[1].prob.tspan), btparams, myelindomains))
    MWImaps, MWIdist, MWIpart = fitmwfmodel(signals, opts)

    if AVOID_MAT_PLOTS
        mwf = _getmwf(opts, MWImaps, MWIdist, MWIpart)
        T2Vals = 1000 .* get_T2vals(opts)
        
        props = Dict{Symbol,Any}(
            :seriestype => :sticks, :xscale => :log10,
            :linewidth => 5, :markersize => 5, :marker => :circle,
            :grid => true, :minorgrid => true, :legend => :none,
            :xrotation => -60, :xticks => length(T2Vals) <= 60 ? T2Vals : T2Vals[1:2:end], # length never more than 120
            :formatter => x -> string(round(x; sigdigits = 3)),
            :xlim => 1000 .* opts.T2Range,
            :xlabel => "T2 [ms]",
            :title => "T2 Distribution: nT2 = $(opts.nT2), MWF = $(round(100*mwf; sigdigits=4))%"
        )
        !(mwftrue == nothing) && (props[:title] *= " (True MWF = $(round(100*mwftrue; sigdigits=4))%)")

        fig = plot(T2Vals, MWIdist[:]; props...)
        vline!(fig, 1000 .* [opts.SPWin; opts.MPWin];
            xscale = :log10, linewidth = 5, linestyle = :dot, color = :red)
        
        !(fname == nothing) && default_savefigs(fig, fname)
        disp && display(fig)
    elseif opts.PlotDist
        !(fname == nothing) && mxsavefig(fname)
    end

    return MWImaps, MWIdist, MWIpart
end

function plotMWFvsAngle(params, mwf, mwftrue = nothing;
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
    default_kwargs = Dict{Symbol,Any}(
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

function plotMWFvsAngle(results::Dict;
        plottypes = [:mwf, :mwferror],
        fname = nothing,
        disp = (fname == nothing)
    )
    # results[:mwfvalues] is an array of Dict{Symbol,T}'s, and results[:params]
    # is an array of BlochTorreyParameters{T}'s
    #   NOTE: assumes all results correspond to same geometry, i.e. same MWF
    @unpack params, mwfvalues = results
    (isempty(params) || isempty(mwfvalues)) && return nothing

    params = convert(Vector{typeof(params[1])}, params) # force proper typing of params
    mwftrue = get(mwfvalues[1], :exact, nothing) # get mwftrue, or nothing if the key doesn't exist
    for key in keys(mwfvalues[1])
        key == :exact && continue # skip plotting horizontal lines of exact value
        mwf = [d[key] for d in mwfvalues]
        plotMWFvsAngle(params, mwf, mwftrue;
            plottypes = plottypes, mwfmethod = key, disp = disp, fname = fname)
    end
    return nothing
end

function plotMWFvsMethod(mwfvalues::AbstractArray{D};
        fname = nothing,
        disp = (fname == nothing)
    ) where {D <: Dict{Symbol}}
    
    # Extract data
    isempty(mwfvalues) && return nothing
    allmethods = reduce(union, keys(m) for m in mwfvalues)
    data = Dict(k => [get(m, k, nothing) for m in mwfvalues] for k in allmethods)
    methods = filter(m -> m !== :exact, allmethods)

    nmethods = length(keys(data)) - 1 # one is :exact
    markers = [:circle]
    # :diamond, :circle, :star5, :dtriangle, :pentagon, :utriangle, :star4,
    # :octagon, :heptagon, :hexagon, :rtriangle, :ltriangle,
    # :star6, :star7, :star8,
    # :none, :auto, :rect, :cross, :xcross, :vline, :hline, :+, :x,
    props = Dict{Symbol,Any}(
        :seriestype => :scatter, # :ratio => :equal,
        :markersize => 7, :marker => markers[mod1.((1:nmethods)', length(markers))],
        :grid => true, :minorgrid => true, :xrotation => -60,
        :label => reduce(hcat, string(m) for m in methods), :legend => :topleft,
        :xlabel => "Exact MWF [%]", :ylabel => "Computed MWF [%]", :title => "Computed vs. Exact MWF"
    )
    xdata = 100 .* data[:exact]
    ydata = 100 .* reduce(hcat, data[m] for m in methods)
    fig = vline(xdata; lw = 1, ls = :dot, lc = :blue, lab = "")
    plot!(fig, [extrema(xdata)...], [extrema(xdata)...]; lw = 4, ls = :dash, lc = :red, lab = "")
    plot!(fig, xdata, ydata; props...)
    xext, yext = extrema(xdata), extrema(ydata)
    xlims!(fig, xext .+ (-0.5, 0.5))
    ylims!(fig, (min(xext[1], yext[1]), max(xext[2], yext[2])) .+ (-0.5, 0.5) )

    !(fname == nothing) && default_savefigs(fig, fname)
    disp && display(fig)

    return nothing
end
function plotMWFvsMethod(results::Dict; kwargs...)
    @unpack mwfvalues = results
    isempty(mwfvalues) && return nothing
    mwfvalues = convert(Vector{typeof(mwfvalues[1])}, mwfvalues)
    return plotMWFvsMethod(mwfvalues; kwargs...)
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