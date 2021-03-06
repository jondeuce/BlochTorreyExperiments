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
        @unpack myelinprob, myelinsubdomains, myelindomains = createdomains(params, exteriorgrids, torigrids, interiorgrids, outercircles, innercircles)
        
        @info "Recreating frenquency fields"
        omega = calcomega(myelinprob, myelinsubdomains)
        
        @info "Loading ODE solutions"
        sols = [BSON.load(solfile)[:sol] for solfile in solfilebatch]

        @info "Computing MWF values"
        mwfvalues, signals = compareMWFmethods(sols, myelindomains, params, outercircles, innercircles, bdry)
        
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

####
#### Rectangular geometry with packed circles
####

function reorder(p, t)
    isempty(t) && (return eltype(p)[], eltype(t)[])
    idx = reinterpret(Int, t) |> copy |> sort! |> unique!
    d = Dict{Int,Int}(idx .=> 1:length(idx))
    return p[idx], [(d[t[1]], d[t[2]], d[t[3]]) for t in t]
end

# Create exteriorgrids, interiorgrids, and torigrids from a grid (p,t) and boundary functions fsubs,
# representing circle distances functions [fouter1, finner1, fouter2, finner2, ...]
function createsubgrids(
        ::AbstractMyelinatedFibresGeometry,
        p::AbstractVector{Vec{2,T}},
        t::AbstractVector{NTuple{3,Int}},
        fsubs
    ) where {T}
    Ncircles = length(fsubs) ÷ 2
    text  = [NTuple{3,Int}[] for _ in 1:1]
    tint  = [NTuple{3,Int}[] for _ in 1:Ncircles]
    ttori = [NTuple{3,Int}[] for _ in 1:Ncircles]

    for t in t
        @inbounds pmid = (p[t[1]] + p[t[2]] + p[t[3]])/3
        isfound = false
        for j in 1:Ncircles
            (fsubs[2j  ](pmid) < 0) && (push!(tint[ j], t); isfound = true; break) # check interior first
            (fsubs[2j-1](pmid) < 0) && (push!(ttori[j], t); isfound = true; break) # then tori
        end
        isfound && continue
        push!(text[1], t) # otherwise, exterior
    end

    G = Grid{2,JuAFEM.Triangle,T}
    exteriorgrids = G[Grid(reorder(p,t)...) for t in text]
    interiorgrids = G[Grid(reorder(p,t)...) for t in tint]
    torigrids     = G[Grid(reorder(p,t)...) for t in ttori]

    return @ntuple(exteriorgrids, torigrids, interiorgrids)
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
        maxattempts = 5, # maximum attempts for sampling radii + greedy packing + energy packing
        verbose     = false,
    )

    # Initialize
    circles, domain, η_best = nothing, nothing, 0
    # @info "epsilon = $epsilon" #DEBUG

    for i in 1:maxattempts
        verbose && println("\nPacking... (attempt $i/$maxattempts)\n")
        rs = rand(radiidistribution(btparams), Ncircles) # Initial radii distribution
        
        verbose && print("GreedyCirclePacking: ")
        greedycircles = GreedyCirclePacking.pack(rs; goaldensity = 1.0, iters = greedyiters)

        # verbose && print("EnergyCirclePacking: ")
        # energycircles = EnergyCirclePacking.pack(greedycircles;
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
        # verbose && println("GreedyCirclePacking density:  $(estimate_density(greedycircles, domain))")
        # verbose && println("EnergyCirclePacking density:  $(estimate_density(energycircles, domain))")
        # verbose && println("Final scaled circles density: $(estimate_density(scaledcircles, domain))")

        # Scale greedycircles apart before using as initial guess for PeriodicCirclePacking,
        # otherwise circles tend to get caught in local minima where they are close to tangent,
        # but we want to to encourage them to be as evenly packed as possible. This ensures
        # circles don't start out as tangent
        scaledgreedycircles = translate_shape.(greedycircles, 1.1)

        verbose && print("PeriodicCirclePacking: ")
        periodiccircles, initialdomain = PeriodicCirclePacking.pack(scaledgreedycircles;
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

        verbose && println("")
        verbose && println("Distance threshold: $distthresh")
        verbose && println("Minimum myelin thickness: $(minimum(radius.(finalcircles))*(1-btparams.g_ratio))")
        verbose && println("Minimum circles distance: $(minimum_signed_edge_distance(finalcircles))")
        verbose && println("")
        verbose && println("Periodic circles density: $η_max")
        verbose && println("Final scaled circles density: $η_curr")
        
        (η_curr ≈ goaldensity) && (circles = finalcircles; domain = finaldomain; break)
        (η_curr > η_best) && (η_best = η_curr; circles = finalcircles; domain = finaldomain)
    end

    # Return named tuple of best results
    out = (circles = circles, domain = domain)

    return out
end

# Create geometry for packed circles on a rectangular domain
function creategeometry(::PeriodicPackedFibres, btparams::BlochTorreyParameters{T} = BlochTorreyParameters{Float64}();
        Ncircles = 20, # number of circles
        goaldensity = btparams.AxonPDensity, # goal packing density
        overlapthresh = 0.05, # overlap occurs when distance between circle edges is ≤ overlapthresh * btparams.R_mu
        maxpackiter = 10,
        alpha = 0.5, #DEBUG
        beta = 0.5, #DEBUG
        gamma = 1.0, #DEBUG
        QMIN = 0.4, #DEBUG
        RESOLUTION = 1.0, #DEBUG
        MAXITERS = 2000, #DEBUG
        FIXSUBSITERS = 450, #DEBUG
        FIXPOINTSITERS = 500, #DEBUG
        DENSITYCTRLFREQ = 250, #DEBUG
        DELTAT = 0.1, #DEBUG
        VERBOSE = false, #DEBUG
        FORCEDENSITY = false, # If this flag is true, an error is thrown if the reached packing density is not goaldensity
        FORCEAREA = false, # If this flag is true, an error is thrown if the resulting grid area doesn't match the bdry area
        FORCEQUALITY = false, # If this flag is true, an error is thrown if the resulting grid doesn't have high enough quality
    ) where {T}

    # Initial set of circles
    outercircles, initialbdry = packcircles(btparams;
        Ncircles = Ncircles,
        maxattempts = maxpackiter,
        goaldensity = goaldensity,
        distthresh = overlapthresh * btparams.R_mu,
        verbose = VERBOSE)
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

    # Increase resolution by a factor RESOLUTION
    h0 /= RESOLUTION
    beta /= RESOLUTION

    dmax = beta * btparams.R_mu
    bbox = [xmin(bdry) ymin(bdry); xmax(bdry) ymax(bdry)]
    pfix = [Vec{2,T}[corners(bdry)...]; reduce(vcat, intersection_points(c,bdry) for c in allcircles)]

    # Signed distance function
    fd(x) = drectangle0(x, bdry)

    # Relative edge length function
    function fh(x::Vec{2,T}) where {T}
        douter = dcircles(x, outercircles)
        dinner = dcircles(x, innercircles)
        hallcircles = min(abs(douter), abs(dinner)) / dmax
        return T(alpha + min(1, hallcircles))
        # return T(alpha + hallcircles)
    end

    # Region and sub-region definitions. Order of `allcircles` is important, as we want to project
    # outer circle points first, followed by inner circle points. Note also that zipping the circles
    # together allows for the anonymous function in the comprehension to be well typed.
    fsubs = [x -> dcircle(x,c) for c in allcircles]
    # fsubs = ntuple(i -> x -> dcircle(x, allcircles[i]), length(allcircles))

    p, t = kmg2d(fd, fsubs, fh, h0, bbox, 1, 0, pfix;
        QMIN = QMIN,
        MAXITERS = MAXITERS,
        FIXSUBSITERS = FIXSUBSITERS,
        FIXPOINTSITERS = FIXPOINTSITERS,
        DENSITYCTRLFREQ = DENSITYCTRLFREQ,
        DELTAT = DELTAT,
        VERBOSE = VERBOSE,
        DETERMINISTIC = true,
        PLOT = false,
        PLOTLAST = false)

    if FORCEQUALITY
        Qmesh = DistMesh.mesh_quality(p,t)
        !(Qmesh >= QMIN) && error("Grid quality not high enough; Q = $Qmesh < $QMIN.")
    end

    # Create subgrids for parent grid (p,t) and fsubs
    @unpack exteriorgrids, torigrids, interiorgrids = createsubgrids(PeriodicPackedFibres(), p, t, fsubs)

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

    return @ntuple(exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry)
end

####
#### Example geometry with only two concentric circles
####

# Create geometry for packed circles on a rectangular domain
function creategeometry(::SingleFibre, btparams::BlochTorreyParameters{T} = BlochTorreyParameters{Float64}();
        radius::T = btparams.R_mu, # radius of fibre
        g_ratio::T = btparams.g_ratio, # g_ratio of fibre
        h0::T = T(1/3 * radius * (1 - g_ratio)),
        QMIN::T = T(0.5),
        MAXITERS = 1000,
        FIXPOINTSITERS = 500,
        FIXSUBSITERS = 450,
        FORCEQUALITY = true, # If this flag is true, an error is thrown if the resulting grid doesn't have high enough quality
        VERBOSE = false,
        PLOT = false,
        PLOTLAST = false,
    ) where {T}

    # Initial set of circles
    outercircles = Circle{2,T}[Circle(zero(Vec{2,T}), radius)]
    innercircles = scale_shape.(outercircles, g_ratio)
    allcircles = [outercircles[1], innercircles[1]]
    bdry = Rectangle{2,T}(radius * Vec{2,T}((-1.5, -1.5)), radius * Vec{2,T}((1.5, 1.5)))
    
    bbox = T[xmin(bdry) ymin(bdry); xmax(bdry) ymax(bdry)]
    pfix = Vec{2,T}[corners(bdry)...]

    # Signed distance function, edge length function, and sub-region definitions
    fd(x) = drectangle0(x, bdry)
    fh(x) = huniform(x)
    # fsubs = ntuple(i -> x -> dcircle(x, allcircles[i]), length(allcircles))
    fsubs = [x -> dcircle(x,c) for c in allcircles]

    p, t = kmg2d(fd, fsubs, fh, h0, bbox, 1, 0, pfix;
        QMIN = QMIN, MAXITERS = MAXITERS, FIXPOINTSITERS = FIXPOINTSITERS, FIXSUBSITERS = FIXSUBSITERS,
        VERBOSE = VERBOSE, PLOT = PLOT, PLOTLAST = PLOTLAST, DETERMINISTIC = true)

    if FORCEQUALITY
        Qmesh = DistMesh.mesh_quality(p,t)
        !(Qmesh >= QMIN) && error("Grid quality not high enough; Q = $Qmesh < $QMIN.")
    end

    # Create subgrids for parent grid (p,t) and fsubs
    @unpack exteriorgrids, torigrids, interiorgrids = createsubgrids(SingleFibre(), p, t, fsubs)

    return @ntuple(exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry)
end

# Load geometry of packed circles on rectangular domain
function geometrytuple(geom::Dict)
    if haskey(geom, :geom) || haskey(geom, "geom")
        @unpack geom = geom # Extract `geom` dict if stored alongside other metadata
    end
    
    # Ensure proper typing of grids, and return NamedTuple of data
    G = Grid{2,JuAFEM.Triangle,Float64} # 2D triangular grid
    C = Circle{2,Float64}
    R = Rectangle{2,Float64}
    exteriorgrids = convert(Vector{G}, geom[:exteriorgrids][:])
    torigrids     = convert(Vector{G}, geom[:torigrids][:])
    interiorgrids = convert(Vector{G}, geom[:interiorgrids][:])
    outercircles  = convert(Vector{C}, geom[:outercircles][:])
    innercircles  = convert(Vector{C}, geom[:innercircles][:])
    bdry          = convert(R, geom[:bdry])

    return @ntuple(exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry)
end
loadgeometry(fname::String) = geometrytuple(BSON.load(fname))

####
#### Create myelin domains from exteriorgrids, torigrids, and interiorgrids
####

function createdomains(
        btparams::BlochTorreyParameters{Tu},
        exteriorgrids::AbstractArray{<:TriangularGrid},
        torigrids::AbstractArray{<:TriangularGrid},
        interiorgrids::AbstractArray{<:TriangularGrid},
        outercircles::AbstractArray{<:Circle{2}},
        innercircles::AbstractArray{<:Circle{2}},
        ferritins::AbstractArray{<:Vec{3}} = Vec{3,Tu}[], #Default to geometry float type
        ::Type{uType} = Vec{2,Tu}; #Default btparams float type
        kwargs...
    ) where {Tu, uType <: FieldType{Tu}}

    myelinprob = MyelinProblem(btparams)
    myelinsubdomains = createmyelindomains(
        vec(exteriorgrids), vec(torigrids), vec(interiorgrids),
        vec(outercircles), vec(innercircles), vec(ferritins),
        uType; kwargs...)

    @info "Assembling MyelinDomain from subdomains"
    @timeit BlochTorreyUtils.TIMER "Assembling MyelinDomain" begin
        @timeit BlochTorreyUtils.TIMER "Assemble subdomains"  foreach(m -> doassemble!(m, myelinprob), myelinsubdomains)
        @timeit BlochTorreyUtils.TIMER "Factorize subdomains" foreach(m -> factorize!(m), myelinsubdomains)
        @timeit BlochTorreyUtils.TIMER "Assemble combined"    myelindomains = [MyelinDomain(PermeableInterfaceRegion(), myelinprob, myelinsubdomains)]
        @timeit BlochTorreyUtils.TIMER "Factorize combined"   foreach(m -> factorize!(m), myelindomains)
    end
    # show(stdout, BlochTorreyUtils.TIMER); println("\n")

    return @ntuple(myelinprob, myelinsubdomains, myelindomains)
end

####
#### Frenquency map calculations
####

calcomegas(myelinprob, myelinsubdomains) = omegamap.(Ref(myelinprob), myelinsubdomains)
calcomega(myelinprob, myelinsubdomains) = reduce(vcat, calcomegas(myelinprob, myelinsubdomains))

####
#### Signal calculation
####

# NOTE: These are integrals over the region, so the signals are already weighted
#       for the relative size of the region; the total signal is the sum of the
#       signals returned here

function calcsignals(sols, ts, btparams, myelindomains::AbstractVector{<:MyelinDomain{R,Tu,uType}}; steadystate = 1) where {R,Tu,uType}
    signals = Vector{uType}[]
    for (s,m) in zip(sols, myelindomains)
        ρ = protondensity(btparams, m) :: Vector{Tu} # vector of nodal proton density values
        signal = if fieldvectype(m) <: Vec{3}
            uType[integrate(ρ .* shift_longitudinal(s(t), steadystate), m) :: uType for t in ts]
        else
            uType[integrate(ρ .* s(t), m) :: uType for t in ts]
        end
        push!(signals, signal)
    end
    return signals
end
calcsignal(sols, ts, btparams, myelindomains; kwargs...) =
    sum(calcsignals(sols, ts, btparams, myelindomains; kwargs...))

# ---------------------------------------------------------------------------- #
# ODEProblem constructor and solver for ParabolicDomain's and MyelinDomain's
# ---------------------------------------------------------------------------- #

const DEBUG_ODEPROBLEM = false

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

    if DEBUG_ODEPROBLEM
        BlochTorreyUtils._reset_all_counters!()
        @time display(@benchmark $f(du,u) setup = (u = copy($u0); du = similar(u)))
        BlochTorreyUtils._display_counters()
        BlochTorreyUtils._reset_all_counters!()
    end

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
        u0 = -1.0im, # initial magnetization
        uType::Type = typeof(u0), # Field type
        TE = 10e-3, # 10ms echotime
        TR = 1000e-3, # 1000ms repetition time
        nTE = 32, # 32 echoes by default
        nTR = 1, # 1 repetition by default
        sliceselectangle = eltype(TE)(uType <: Vec{3} ? π/2 : 0), # initial pulse
        flipangle = π, # flipangle for CPMGCallback
        refocustype = :xyx, # Type of refocusing pulse
        steadystate = (uType <: Vec{3} ? 1 : nothing), # steady state value for z-component of magnetization
        tspan = (zero(TE), (nTR - 1) * TR + nTE * TE), # time span for ode solution
        callback = CPMGCallback(uType, tspan;
            TE = TE, TR = TR, nTE = nTE, nTR = nTR,
            steadystate = steadystate, sliceselectangle = sliceselectangle,
            refocustype = refocustype, flipangle = eltype(TE)(flipangle)),
        reltol = 1e-8,
        abstol = 0.0,
        dt = TE/10, # Maximum timestep
        kwargs...
    )

    # Standardize initial condition and steadystate values; theoretically, these
    # should be free parameters, in practice there is no point changing them
    @assert (uType <: Vec{3}  && u0 ≈ uType((0, 0, 1))) || # Longitudinal magnetization; flipped below
            (uType <: Vec{2}  && u0 ≈ uType((0, -1)))   || # Transverse magnetization following π/2-pulse about x-axis
            (uType <: Complex && u0 ≈ uType(-im))          # Complex transverse magnetization following π/2-pulse about x-axis
    @assert (uType <: Vec{3}  && steadystate ≈ 1) || (steadystate == nothing)

    if !(uType <: Vec{3})
        # Restrictions for when only transverse magnetization is simulated:
        #   -initial pulse (i.e. rotation) cannot be applied
        #   -flip angle must be exactly π, unless refocustype == :xyx
        #   -simulation only runs until nTE * TE, i.e. nTR == 1
        @assert nTR == 1
        @assert flipangle ≈ π || refocustype == :xyx
        @assert sliceselectangle ≈ 0
    end
    
    # Initialize initial magnetization state (in M-space)
    U0 = interpolate(u0, myelindomain)
    (uType <: Vec{3}) && apply_pulse!(U0, sliceselectangle, :x, uType)

    # Our convention is that u₃ = M∞ - M₃. This convenience function shifts U0 from
    # M-space (i.e. [M₁, M₂, M₃]) to u-space (i.e. [u₁, u₂, u₃] = [M₁, M₂, M∞ - M₃])
    (uType <: Vec{3}) && shift_longitudinal!(U0, steadystate)

    # Save solution every dt (an even divisor of TE) as well as every TR by default
    tstops = cpmg_savetimes(tspan, dt, TE, TR, nTE, nTR)

    # Setup problem and solve
    @timeit BlochTorreyUtils.TIMER "solveblochtorrey" begin
        prob = ODEProblem(myelindomain, U0, tspan)
        sol = solve(prob, alg, args...;
            dense = false, # don't save all intermediate time steps
            saveat = tstops, # timepoints to save solution at
            tstops = tstops, # ensure stopping at all tstops points
            dt = dt, reltol = reltol, abstol = abstol, callback = callback, kwargs...)
    end
    # show(stdout, BlochTorreyUtils.TIMER); println("\n")

    if DEBUG_ODEPROBLEM
        BlochTorreyUtils._display_counters()
    end

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

function saveblochtorrey(::Type{uType}, grids::Vector{<:Grid}, sols::Vector{<:ODESolution};
        timepoints = sols[1].t,
        steadystate = 1,
        filename = nothing,
    ) where {uType}

    @assert length(grids) == length(sols)
    @assert !(filename == nothing)
    tostr(x::Int) = @sprintf("%4.4d", x)

    # Create a paraview collection for each (grid, solution) pair, saving for each pair
    # a pvd collection with vtu files for each timepoint
    for i in 1:length(grids)
        vtk_filename_noext = DrWatson.savename(filename, Dict(:grid => i))
        paraview_collection(vtk_filename_noext) do pvd
            for (it,t) in enumerate(timepoints)
                vtk_grid_filename = DrWatson.savename(vtk_filename_noext, Dict(:time => it-1))
                vtk_grid(vtk_grid_filename, grids[i]) do vtk
                    u = copy(reinterpret(uType, sols[i](t)))
                    Mt = transverse.(u)
                    vtk_point_data(vtk, norm.(Mt), "Magnitude")
                    vtk_point_data(vtk, angle.(Mt), "Phase")
                    if uType <: Vec{3}
                        shift_longitudinal!(u, steadystate)
                        Mz = longitudinal.(u)
                        vtk_point_data(vtk, Mz, "Longitudinal")
                    end
                    collection_add_timestep(pvd, vtk, Float64(t))
                end
            end
        end
    end

    return nothing
end
saveblochtorrey(myelindomains::Vector{<:MyelinDomain}, sols, args...; kwargs...) =
    saveblochtorrey(fieldvectype(myelindomains[1]), getgrid.(myelindomains), sols, args...; kwargs...)