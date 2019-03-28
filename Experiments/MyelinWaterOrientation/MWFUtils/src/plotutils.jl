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

function plotbiexp(sols, btparams, myelindomains, outercircles, innercircles, bdry; titlestr = "Signal Magnitude vs. Time", fname = nothing)
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

    mxcall(:figure, 0)
    mxcall(:plot, 0, collect(1000.0.*ts), [norm.(Stotal) y_biexp])
    mxcall(:legend, 0, "Simulated", "Bi-Exponential")
    mxcall(:title, 0, titlestr)
    mxcall(:xlabel, 0, "Time [ms]")
    mxcall(:xlim, 0, 1000.0 .* [tspan...])
    mxcall(:ylabel, 0, "S(t) Magnitude")

    !(fname == nothing) && mxsavefig(fname)

    nothing
end

function plotSEcorr(sols, btparams, myelindomains; fname = nothing)
    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    ts = tspan[1]:TE:tspan[2] # signal after each echo
    Stotal = calcsignal(sols, ts, myelindomains)

    MWImaps, MWIdist, MWIpart = fitmwfmodel(Stotal, NNLSRegression();
        T2Range = [8e-3, 2.0],
        spwin = [8e-3, 24.75e-3],
        mpwin = [25.25e-3, 200e-3],
        nT2 = 32,
        RefConAngle = 165.0,
        PLOTDIST = true
    )

    !(fname == nothing) && mxsavefig(fname)

    return MWImaps, MWIdist, MWIpart
end

# Save plot
function mxsavefig(fname; fig = true, png = true, pdf = true, eps = true)
    fig && mxcall(:savefig, 0, fname * ".fig")
    png && mxcall(:export_fig, 0, fname, "-png")
    pdf && mxcall(:export_fig, 0, fname, "-dpdf")
    eps && mxcall(:export_fig, 0, fname, "-eps")
    mxcall(:close, 0)
end

function compareMWFmethods(sols, myelindomains, outercircles, innercircles, bdry)
    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    ts = tspan[1]:TE:tspan[2] # signal after each echo
    Stotal = calcsignal(sols, ts, myelindomains)

    mwfvalues = Dict(
        :exact => getmwf(outercircles, innercircles, bdry),
        :TwoPoolMagnToMagn => getmwf(Stotal, TwoPoolMagnToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolMagnToMagn => getmwf(Stotal, ThreePoolMagnToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolCplxToMagn => getmwf(Stotal, ThreePoolCplxToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolCplxToCplx => getmwf(Stotal, ThreePoolCplxToCplx(); TE = TE, fitmethod = :local)
    )
    return mwfvalues
end