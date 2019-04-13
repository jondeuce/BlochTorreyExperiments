function testbtfinelem2D(opts::BlochTorreyParameters{T};
        Resolution::T       = 1.0,                  # Resolution increase factor
        Domain::NTuple{2,T} = opts.R_mu .* (-2,2),  # Bounds for side of square domain
        Router::T           = opts.R_mu,            # Outer radius; defaults to R_mu
        Time::T             = 60e-3,                # Simulation time
        Plot::Bool          = true                  # Plot resulting magnitude and phase
    ) where {T}

    a, b = Domain
    bdry = Rectangle(Vec{2,T}((a,a)), Vec{2,T}((b,b)))
    outercircles = [Circle{2,T}(zero(Vec{2,T}), Router)]
    exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry =
        creategeometry(opts; outercircles = outercircles, bdry = bdry, RESOLUTION = Resolution)

    myelinprob, myelinsubdomains, myelindomains = createdomains(opts, exteriorgrids, torigrids, interiorgrids, outercircles, innercircles)
    omega = calcomega(myelinprob, myelinsubdomains)

    solve(ms) = solveblochtorrey(myelinprob, ms;
        tspan = (zero(T), Time),
        TE = Time,
        callback = nothing
    )
    sols = solve(myelindomains)
    # sols = solve(myelinsubdomains)

    if Plot
        paramstr = "theta = $(rad2deg(opts.theta)) deg, D = $(opts.D_Tissue) um2/s, K = $(opts.K_perm) um/s"
        mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = omega)
        plotmagnitude(sols, opts, myelindomains, bdry; titlestr = "Magnitude: " * paramstr)
        plotphase(sols, opts, myelindomains, bdry; titlestr = "Phase: " * paramstr)
        # plotSEcorr(sols, opts, myelindomains, fname = "SEcorr")
        # plotbiexp(sols, opts, myelindomains, outercircles, innercircles, bdry; titlestr = "Signal: " * paramstr, fname = "signal")
    end
    # mwfvalues, signals = compareMWFmethods(sols, myelindomains, outercircles, innercircles, bdry)

    # Return a named tupled of geometry structures
    geom = (
        bdry = bdry,
        outercircles = outercircles,
        innercircles = innercircles,
        exteriorgrids = exteriorgrids,
        torigrids = torigrids,
        interiorgrids = interiorgrids,
        myelinprob = myelinprob,
        myelinsubdomains = myelinsubdomains,
        myelindomains = myelindomains,
        omega = omega
    )

    return sols, geom
end
