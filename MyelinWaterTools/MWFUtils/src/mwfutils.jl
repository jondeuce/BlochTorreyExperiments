# Standard date format
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")

function load_results_dict(;
        geomfilename = "geom.bson",
        basedir = ".", # directory to load from (default is current)
        save = false # save reconstructed results
    )
    # Load geometry
    @info "Loading geometry from file: " * geomfilename
    geom = loadgeometry(geomfilename)
    @unpack exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = geom

    # Find btparam filenames and solution filenames
    soldir = isdir(joinpath(basedir, "sol")) ? joinpath(basedir, "sol") : basedir
    paramfiles = filter(s -> endswith(s, "btparams.bson"), joinpath.(soldir, readdir(soldir)))
    solfiles = filter(s -> endswith(s, "odesolution.bson"), joinpath.(soldir, readdir(soldir)))
    numregions = length(solfiles) ÷ length(paramfiles)

    # Initialize results
    allparams = [BSON.load(pfile)[:btparams] for pfile in paramfiles]
    allparams = convert.(typeof(allparams[1]), allparams)
    results = blank_results_dict()
    results[:geom] = geom

    # unpack geometry and create myelin domains
    for (params, solfilebatch) in zip(allparams, Iterators.partition(solfiles, numregions))
        @info "Recreating myelin domains"
        myelinprob, myelinsubdomains, myelindomains = createdomains(params, exteriorgrids, torigrids, interiorgrids, outercircles, innercircles)
        
        @info "Recreating frenquency fields"
        omega = calcomega(myelinprob, myelinsubdomains)
        
        @info "Loading ODE solutions"
        sols = [BSON.load(solfile)[:sol] for solfile in solfilebatch]

        @info "Computing MWF values"
        mwfvalues, signals = compareMWFmethods(sols, myelindomains, outercircles, innercircles, bdry)
        
        push!(results[:params], params)
        push!(results[:myelinprobs], myelinprob)
        push!(results[:myelinsubdomains], myelinsubdomains)
        push!(results[:myelindomains], myelindomains)
        push!(results[:omegas], omega)
        push!(results[:sols], sols)
        push!(results[:signals], signals)
        push!(results[:mwfvalues], mwfvalues)
    end

    if save
        @info "Saving reconstructed dictionary"
        try
            BSON.bson(getnow() * "__results.bson", Dict(:results => deepcopy(results)))
        catch e
            @warn "Error saving results!"
            @warn sprint(showerror, e, catch_backtrace())
        end
    end

    return results
end

# Pack circles
function packcircles(btparams::BlochTorreyParameters = BlochTorreyParameters{Float64}();
        Ncircles    = 20, # number of circles
        goaldensity = btparams.AxonPDensity, # goal packing density
        distthresh  = 0.05 * btparams.R_mu, # overlap occurs when distance between circle edges is ≤ distthresh
        epsilon     = 0.01 * btparams.R_mu, # pack more than necessary by default
        alpha       = 1e-1, # covariance penalty weight (enforces circular distribution)
        beta        = 1e-6, # mutual distance penalty weight
        lambda      = 1.0, # overlap penalty weight (or lagrange multiplier for constrained version)
        greedyiters = 100, # maximum iterations for greedy packing
        maxattempts = 5 # maximum attempts for sampling radii + greedy packing + energy packing
    )

    # Initialize
    circles, domain, η_best = nothing, nothing, 0
    # @info "epsilon = $epsilon" #DEBUG

    for i in 1:maxattempts
        println("\nPacking... (attempt $i/$maxattempts)\n")
        rs = rand(radiidistribution(btparams), Ncircles) # Initial radii distribution
        
        print("GreedyCirclePacking: ")
        @time greedycircles = GreedyCirclePacking.pack(rs; goaldensity = 1.0, iters = greedyiters)

        # print("EnergyCirclePacking: ")
        # @time energycircles = EnergyCirclePacking.pack(greedycircles;
        #     autodiff = false,
        #     secondorder = false,
        #     setcallback = false,
        #     goaldensity = 1.0, #goaldensity # pack as much as possible, scale to goal density after
        #     distancescale = btparams.R_mu,
        #     weights = [alpha, beta, lambda],
        #     epsilon = distthresh # pack as much as possible, penalizing packing tighter than distance distthresh
        # )
        # scaledcircles, domain, _ = CirclePackingUtils.scale_to_density(energycircles, goaldensity, distthresh; MODE = :corners)
        # η_curr = estimate_density(scaledcircles, domain)
        # 
        # println("GreedyCirclePacking density:  $(estimate_density(greedycircles, domain))")
        # println("EnergyCirclePacking density:  $(estimate_density(energycircles, domain))")
        # println("Final scaled circles density: $(estimate_density(scaledcircles, domain))")

        # Scale greedycircles apart before using as initial guess for PeriodicCirclePacking,
        # otherwise circles tend to get caught in local minima where they are close to tangent,
        # but we want to to encourage them to be as evenly packed as possible. This ensures
        # circles don't start out as tangent
        scaledgreedycircles = translate_shape.(greedycircles, 1.1)

        print("PeriodicCirclePacking: ")
        @time periodiccircles, initialdomain = PeriodicCirclePacking.pack(scaledgreedycircles;
            autodiff = false,
            secondorder = false,
            distancescale = btparams.R_mu,
            epsilon = epsilon
        )
        scaledcircles, scaleddomain, _ = periodic_scale_to_density(periodiccircles, initialdomain, goaldensity, distthresh)
        finaldomain, _ = periodic_subdomain(scaledcircles, scaleddomain)
        finalcircles = periodic_circle_repeat(scaledcircles, finaldomain; Nrepeat = 1)
        η_max = periodic_density(periodiccircles, initialdomain)
        η_curr = periodic_density(finalcircles, finaldomain)

        println("")
        println("Distance threshold: $distthresh")
        println("Minimum myelin thickness: $(minimum(radius.(finalcircles))*(1-btparams.g_ratio))")
        println("Minimum circles distance: $(minimum_signed_edge_distance(finalcircles))")
        println("")
        println("Periodic circles density: $η_max")
        println("Final scaled circles density: $η_curr")
        
        (η_curr ≈ goaldensity) && (circles = finalcircles; domain = finaldomain; break)
        (η_curr > η_best) && (η_best = η_curr; circles = finalcircles; domain = finaldomain)
    end

    # Return named tuple of best results
    out = (circles = circles, domain = domain)

    return out
end

function creategeometry(btparams::BlochTorreyParameters{T} = BlochTorreyParameters{Float64}();
        Ncircles = 20, # number of circles
        goaldensity = btparams.AxonPDensity, # goal packing density
        overlapthresh = 0.05, # overlap occurs when distance between circle edges is ≤ overlapthresh * btparams.R_mu
        maxpackiter = 10,
        alpha = 0.5, #DEBUG
        beta = 0.5, #DEBUG
        gamma = 1.0, #DEBUG
        QMIN = 0.4, #DEBUG
        RESOLUTION = 1.0, #DEBUG
        MAXITERS = 1000, #DEBUG
        FIXPOINTSITERS = 250, #DEBUG
        FIXSUBSITERS = 200, #DEBUG
        FORCEDENSITY = false, # If this flag is true, an error is thrown if the reached packing density is not goaldensity
        FORCEAREA = false, # If this flag is true, an error is thrown if the resulting grid area doesn't match the bdry area
        FORCEQUALITY = false # If this flag is true, an error is thrown if the resulting grid doesn't have high enough quality
    ) where {T}

    # Initial set of circles
    outercircles, initialbdry = packcircles(btparams;
        Ncircles = Ncircles, maxattempts = maxpackiter,
        goaldensity = goaldensity, distthresh = overlapthresh * btparams.R_mu)
    innercircles = scale_shape.(outercircles, btparams.g_ratio)
    allcircles = collect(Iterators.flatten(zip(outercircles, innercircles)))

    # Optimize the rectangular subdomain to account for innercircles
    bdry, _ = periodic_subdomain(allcircles, initialbdry)
    outercircles = periodic_circle_repeat(outercircles, bdry)
    innercircles = scale_shape.(outercircles, btparams.g_ratio)
    allcircles = collect(Iterators.flatten(zip(outercircles, innercircles)))

    # # Optimize the rectangular subdomain 
    # bdry, _ = opt_subdomain(allcircles; MODE = :corners)
    # outercircles, bdry, α_best = scale_to_density(outercircles, bdry, btparams.AxonPDensity)
    # innercircles = scale_shape.(outercircles, btparams.g_ratio)
    # allcircles = collect(Iterators.flatten(zip(outercircles, innercircles)))

    if FORCEDENSITY
        density = estimate_density(outercircles, bdry)
        !(density ≈ goaldensity) && error("Packing density not reached: goal density was $goaldensity, reached $density.")
    end

    mincircdist = minimum_signed_edge_distance(outercircles)
    mintoriwidth = (1-btparams.g_ratio) * minimum(radius, outercircles)
    h0 = gamma * min(mincircdist, mintoriwidth)

    dmax = beta * btparams.R_mu
    bbox = [xmin(bdry) ymin(bdry); xmax(bdry) ymax(bdry)]
    pfix = [Vec{2,T}[corners(bdry)...]; reduce(vcat, intersection_points(c,bdry) for c in allcircles)]

    # Increase resolution by a factor RESOLUTION
    h0 /= RESOLUTION
    beta /= RESOLUTION

    # Signed distance function
    fd(x) = drectangle0(x, bdry)

    # Relative edge length function
    function fh(x::Vec{2,T}) where {T}
        douter = dcircles(x, outercircles)
        dinner = dcircles(x, innercircles)
        hallcircles = min(abs(douter), abs(dinner))/T(dmax)
        return alpha + min(hallcircles, one(T))
    end

    # Region and sub-region definitions. Order of `allcircles` is important, as we want to project
    # outer circle points first, followed by inner circle points. Note also that zipping the circles
    # together allows for the anonymous function in the comprehension to be well typed.
    fsubs = [x->dcircle(x,c) for c in allcircles]

    p, t = kmg2d(fd, fsubs, fh, h0, bbox, 1, 0, pfix;
        QMIN = QMIN,
        MAXITERS = MAXITERS,
        FIXPOINTSITERS = FIXPOINTSITERS,
        FIXSUBSITERS = FIXSUBSITERS,
        VERBOSE = true,
        DETERMINISTIC = true,
        PLOT = false,
        PLOTLAST = false
    );

    if FORCEQUALITY
        Qmesh = DistMesh.mesh_quality(p,t)
        !(Qmesh >= QMIN) && error("Grid quality not high enough; Q = $Qmesh < $QMIN.")
    end

    text  = [NTuple{3,Int}[] for _ in 1:1]
    tint  = [NTuple{3,Int}[] for _ in 1:length(outercircles)]
    ttori = [NTuple{3,Int}[] for _ in 1:length(outercircles)]
    for t in t
        @inbounds pmid = (p[t[1]] + p[t[2]] + p[t[3]])/3
        isfound = false
        for j in 1:length(outercircles)
            (fsubs[2j  ](pmid) < 0) && (push!(tint[j],  t); isfound = true; break) # check interior first
            (fsubs[2j-1](pmid) < 0) && (push!(ttori[j], t); isfound = true; break) # then tori
        end
        isfound && continue
        push!(text[1], t) # otherwise, exterior
    end

    function reorder(p, t)
        isempty(t) && (return eltype(p)[], eltype(t)[])
        idx = reinterpret(Int, t) |> copy |> sort! |> unique!
        d = Dict{Int,Int}(idx .=> 1:length(idx))
        return p[idx], [(d[t[1]], d[t[2]], d[t[3]]) for t in t]
    end

    G = Grid{2,3,T,3}
    exteriorgrids = G[Grid(reorder(p,t)...) for t in text]
    interiorgrids = G[Grid(reorder(p,t)...) for t in tint]
    torigrids     = G[Grid(reorder(p,t)...) for t in ttori]

    grid_area = sum(area.(exteriorgrids)) + sum(area.(torigrids)) + sum(area.(interiorgrids))
    bdry_area = area(bdry)
    cell_area_mismatch = bdry_area - grid_area
    if FORCEAREA
        !(grid_area ≈ bdry_area) && error("Grid area is not matched with boundary area; error is $(cell_area_mismatch).")
        dA_max = maximum(1:length(innercircles)) do i
            gin, gout = interiorgrids[i], torigrids[i]
            cin, cout = innercircles[i], outercircles[i]
            Ain, Aout = area(gin), area(gout)
            NCin, NCout = getncells(gin), getncells(gout)
            ain0 = NCin == 0 ? zero(T) : mean(c->area(gin,c), 1:NCin)
            aout0 = NCout == 0 ? zero(T) : mean(c->area(gout,c), 1:NCout)
            Ain0 = intersect_area(cin, bdry)
            Aout0 = intersect_area(cout, bdry) - Ain0
            dAin = ain0 == zero(T) ? Ain0/(h0^2/2) : (Ain-Ain0)/ain0
            dAout = aout0 == zero(T) ? Aout0/(h0^2/2) : (Aout-Aout0)/aout0
            return max(abs(dAin), abs(dAout))
        end
        !(dA_max < one(T)) && error("Grid subregion areas are not close to analytical circle areas; error relative to average triangle area is $(dA_max).")
    end

    # Return named tuple of results
    geom = (
        exteriorgrids = exteriorgrids, 
        torigrids = torigrids, 
        interiorgrids = interiorgrids, 
        outercircles = outercircles, 
        innercircles = innercircles, 
        bdry = bdry
    )
    
    return geom
end

function loadgeometry(fname)
    d = BSON.load(fname)
    G = Grid{2,3,Float64,3} # 2D triangular grid
    C = Circle{2,Float64}
    R = Rectangle{2,Float64}

    # Ensure proper typing of grids, and return NamedTuple of data
    out = ( exteriorgrids = convert(Vector{G}, d[:exteriorgrids][:]),
            torigrids     = convert(Vector{G}, d[:torigrids][:]),
            interiorgrids = convert(Vector{G}, d[:interiorgrids][:]),
            outercircles  = convert(Vector{C}, d[:outercircles][:]),
            innercircles  = convert(Vector{C}, d[:innercircles][:]),
            bdry          = convert(R, d[:bdry]) )
    return out
end

function createdomains(
        btparams::BlochTorreyParameters{Tu},
        exteriorgrids::AbstractArray{G},
        torigrids::AbstractArray{G},
        interiorgrids::AbstractArray{G},
        outercircles::AbstractArray{C},
        innercircles::AbstractArray{C},
        ferritins::AbstractArray{V} = Vec{3,T}[], #Default to geometry float type
        ::Type{uType} = Vec{2,Tu}; #Default btparams float type
        kwargs...
    ) where {T, G<:TriangularGrid{T}, C<:Circle{2,T}, V<:Vec{3,T}, Tu, uType<:FieldType{Tu}}

    myelinprob = MyelinProblem(btparams)
    myelinsubdomains = createmyelindomains(
        vec(exteriorgrids), vec(torigrids), vec(interiorgrids),
        vec(outercircles), vec(innercircles), vec(ferritins),
        uType; kwargs...)

    @info "Assembling MyelinDomain from subdomains"
    print("    Assemble subdomains   "); @time doassemble!.(myelinsubdomains, Ref(myelinprob))
    print("    Factorize subdomains  "); @time factorize!.(getdomain.(myelinsubdomains))
    print("    Assemble combined     "); @time combinedmyelindomain = MyelinDomain(PermeableInterfaceRegion(), myelinprob, myelinsubdomains)
    print("    Factorize combined    "); @time factorize!(combinedmyelindomain)
    myelindomains = [combinedmyelindomain]

    return (myelinprob = myelinprob,
            myelinsubdomains = myelinsubdomains,
            myelindomains = myelindomains)
end

calcomegas(myelinprob, myelinsubdomains) = omegamap.(Ref(myelinprob), myelinsubdomains)
calcomega(myelinprob, myelinsubdomains) = reduce(vcat, calcomegas(myelinprob, myelinsubdomains))

# Vector of signals on each domain.
# NOTE: This are integrals over the region, so the signals are already weighted
#       for the relative size of the region; the total signal is the sum of the
#       signals returned here
function calcsignals(sols, ts, myelindomains)
    Signals = map(sols, myelindomains) do s, m
        [integrate(s(t), m) for t in ts]
    end
    return Signals
end

# Sum signals over all domains
calcsignal(sols, ts, myelindomains) = sum(calcsignals(sols, ts, myelindomains))

# ---------------------------------------------------------------------------- #
# ODEProblem constructor and solver for ParabolicDomain's and MyelinDomain's
# ---------------------------------------------------------------------------- #

# Create an `ODEProblem` from a `ParabolicDomain` representing either
#   du/dt = (M\K)*u   [invertmass = true], or
#   M*du/dt = K*u     [invertmass = false]
function OrdinaryDiffEq.ODEProblem(d::ParabolicDomain, u0, tspan; invertmass = true)
    if !invertmass
        @warn "invertmass = false not yet supported; setting invertmass = true."
        invertmass = true
    end

    A = ParabolicLinearMap(d)
    f = ODEFunction(A)
    return ODEProblem(f, u0, tspan)

    # f!(du,u,p,t) = mul!(du, p[1], u) # RHS action of ODE for general matrix A stored in p[1]
    # A = ParabolicLinearMap(d) # ParabolicLinearMap returns a subtype of LinearMap which acts onto u as (M\K)*u.
    # p = (A,) # ODEProblem parameter tuple
    # return ODEProblem(f!, u0, tspan, p)

    # if invertmass
    #     # ParabolicLinearMap returns a linear operator which acts by (M\K)*u.
    #     A = ParabolicLinearMap(d) # subtype of LinearMap
    #     p = (LinearOperatorWrapper(A),) # wrap LinearMap in an AbstractArray wrapper
    #     F! = ODEFunction{true,true}(f!; # represents M*du/dt = K*u system
    #         mass_matrix = I, # mass matrix
    #         jac = (J,u,p,t) -> J, # Jacobian is constant (LinearOperatorWrapper)
    #         jac_prototype = p[1]
    #     )
    #     return ODEProblem(F!, u0, tspan, p)
    # else
    #     K, M = getstiffness(d), getmass(d)
    #     p = (K, M)
    #     F! = ODEFunction{true,true}(f!; # represents M*du/dt = K*u system
    #         mass_matrix = M, # mass matrix
    #         jac = (J,u,p,t) -> J, # Jacobian is constant (stiffness matrix)
    #         jac_prototype = K
    #     )
    #     return ODEProblem(F!, u0, tspan, p)
    # end
end
OrdinaryDiffEq.ODEProblem(m::MyelinDomain, u0, tspan; kwargs...) = ODEProblem(getdomain(m), u0, tspan; kwargs...)

function solveblochtorrey(
        myelinprob::MyelinProblem, myelindomain::MyelinDomain, alg = default_algorithm(), args...;
        u0 = Vec{2}((0.0, 1.0)), # initial π/2 pulse
        TE = 10e-3, # 10ms echotime
        nTE = 32, # 32 echoes by default
        tspan = TE .* (0, nTE), # time span for ode solution
        saveat = tspan[1]:TE/2:tspan[2], # save every TE/2 by default
        tstops = tspan[1]:TE/2:tspan[2], # default extra points which the integrator must step to; match saveat by default
        callback = MultiSpinEchoCallback(tspan; TE = TE),
        reltol = 1e-8,
        abstol = 0.0,
        kwargs...
    )

    prob = ODEProblem(myelindomain, interpolate(u0, myelindomain), tspan)
    sol = solve(prob, alg, args...;
        dense = false, # don't save all intermediate time steps
        saveat = tstops, # timepoints to save solution at
        tstops = tstops, # ensure stopping at all tstops points
        dt = TE, reltol = reltol, abstol = abstol, callback = callback, kwargs...)
    return sol
end

function solveblochtorrey(myelinprob::MyelinProblem, myelindomains::Vector{<:MyelinDomain}, args...; kwargs...)
    sols = Vector{ODESolution}()
    if length(myelindomains) == 1
        @info "Solving MyelinProblem"
        @time sol = solveblochtorrey(myelinprob, myelindomains[1], args...; kwargs...)
        push!(sols, sol)
    else
        @time for (i,myedom) in enumerate(myelindomains)
            @info "Solving MyelinProblem $i/$(length(myelindomains))"
            @time sol = solveblochtorrey(myelinprob, myedom, args...; kwargs...)
            push!(sols, sol)
        end
    end
    return sols
end

default_cvode_bdf() = CVODE_BDF(method = :Functional)
default_expokit() = ExpokitExpmv(m = 30)
default_higham() = HighamExpmv(precision = :single)
default_algorithm() = default_expokit()