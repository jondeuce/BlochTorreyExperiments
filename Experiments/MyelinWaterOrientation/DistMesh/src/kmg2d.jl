#KMG2D 2D-mesh generator using signed distance & size functions.
#--------------------------------------------------------------------
# [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr)
# [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix)
# [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,[],fparam)
# [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix,fparam)
# Input:
#   fd       : Signed distance  d(x,y)  (function handle)
#   fh       : Mesh size function h(x,y) (function handle)
#   h0       : Reference edge length
#   bbox     : Boundary box [xmin,ymin; xmax,ymax]
#   dg       : Type of triangular mesh: dg=1 linear; dg=2 quadratic
#   nr       : Number of refinements (nr=0, without refinement)
#   pfix     : Fixed nodes (nodes that must appear in the mesh)
#   varargin : Additional parameters passed to fd and fh (optional)
# Output:
#   p        : Node coordinates np*2
#   t        : Triangle vertices nt*3 (linear) or nt*6 (quadratic)
#   be       : Boundary edges    ne*2 (linear) or nt*3 (quadratic)
#   bn       : Boundary nodes    nb*1
#--------------------------------------------------------------------
# (c) 2009, Koko J., ISIMA, koko@isima.fr
#--------------------------------------------------------------------

function kmg2d(
        fd, # distance function
        fh, # edge length function
        h0::T, # nominal edge length
        bbox::Matrix{T}, # bounding box (2x2 matrix [xmin ymin; xmax ymax])
        dg::Int = 1, # Type of triangular mesh: dg=1 linear; dg=2 quadratic
        nr::Int = 0, # Number of refinements (nr=0, without refinement)
        pfix::AbstractVector{V} = V[], # fixed points
        pinit::AbstractVector{V} = init_points(bbox, h0), # inital distribution of points (triangular grid by default)
        fsubs::AbstractVector = nothing, # subregion distance functions, the boundaries of which are forced onto the grid
        ∇fd = x -> Tensors.gradient(fd, x), # Gradient of distance function `fd`
        ∇fsubs = [x -> Tensors.gradient(fs, x) for fs in fsubs]; # gradients of subregion distance functions
        PLOT::Bool = false, # plot all triangulations during evolution
        PLOTLAST::Bool = false, # plot resulting triangulation
        DETERMINISTIC::Bool = false, # use deterministic pseudo-random
        MAXSTALLITERS::Int = 500, # max iterations of stalled progress
        RESTARTEDGETHRESH::T = T(1e-4)*maximum(abs, diff(bbox;dims=1)), # min. h0 s.t. restart is allowed
        DENSITYCTRLFREQ::Int = 30, # density check frequency
        DENSITYRELTHRESH::T = T(2.0), # density relative threshold
        DPTOL::T = T(1e-3), # equilibrium step size threshold
        TTOL::T = T(0.1), # large movement tolerance for retriangulation
        FSCALE::T = T(1.2), # scale bar lengths
        DELTAT::T = T(0.2), # relative step size
        GEPS::T = T(1e-3)*h0, # boundary distance threshold
        DEPS::T = sqrt(eps(T))*h0 # finite difference step-length
    ) where {T, V<:Vec{2,T}}

    # 1. Create initial distribution in bounding box (equilateral triangles by default)
    p = copy(pinit)

    # 2. Remove points outside the region, apply the rejection method
    p = filter!(x -> fd(x) < GEPS, p)                      # Keep only d<0 points
    isempty(p) && (return restart())
    r0 = inv.(fh.(p)).^2                                   # Probability to keep point

    randlist = DETERMINISTIC ? rand(MersenneTwister(0), T, length(p)) : rand(T, length(p))
    p = p[maximum(r0) .* randlist .< r0]                   # Rejection method
    isempty(p) && (return restart())

    !isempty(pfix) && (p = setdiff!(p, pfix))              # Remove duplicated nodes
    pfix = sort!(copy(pfix); by = first)
    pfix = threshunique(pfix; rtol = √eps(T), atol = eps(T))
    nfix = length(pfix)

    p = vcat(pfix, p)                                      # Prepend fix points
    t = delaunay2(p)

    # Check that initial distribution is not empty
    (isempty(t) || length(p) == length(pfix)) && (return restart())

    # Plot initial points
    PLOT && display(simpplot(p,t))

    # Initialize buffers
    bars = Vector{NTuple{2,Int}}() # bars indices buffer
    barvec, Fvec, Ftot = Vector{V}(), Vector{V}(), Vector{V}() # vector buffers
    L, L0, hbars, F = Vector{T}(), Vector{T}(), Vector{T}(), Vector{T}() # scalar buffers

    resize_buffers!(buf,len) = map!(x->resize!(x, len), buf, buf) # resize function
    p_buffers = [Ftot] # buffers of length(p)
    bars_buffers = [barvec, L, L0, hbars, F, Fvec] # buffers of length(bars)

    # Initialize variables for first iteration
    count, stallcount = 0, 0
    dtermbest = T(Inf)
    pold = V[V((Inf,Inf))]
    force_triangulation = true
    resize_buffers!(p_buffers, length(p))

    while true
        count += 1

        # Restart if no points are present
        (isempty(p) || isempty(pold)) && (return restart())

        # 3. (Re-)triangulation by the Delaunay algorithm
        if force_triangulation || (√maximum(norm2.(p.-pold)) > h0 * TTOL)      # Any large movement?    
            p = threshunique(p; rtol = √eps(T), atol = h0*eps(T))              # Remove duplicate nodes
            resize_buffers!(p_buffers, length(p)) # Resize buffers of length(p)
            isempty(p) && (return restart())

            pold = copy(p)                                                     # Save current positions
            t = delaunay2!(t, p)                                               # List of triangles
            pmid = V[(p[tt[1]] + p[tt[2]] + p[tt[3]])/3 for tt in t]           # Compute centroids
            t = t[fd.(pmid) .< -GEPS]                                          # Keep interior triangles

            # 4. Describe each bar by a unique pair of nodes
            bars = resize!(bars, 3*length(t))
            @inbounds for (i,tt) in enumerate(t)
                a, b, c = sorttuple(tt)
                bars[3i-2] = (a, b)
                bars[3i-1] = (b, c)
                bars[3i  ] = (c, a)                                            # Interior bars duplicated
            end
            bars = unique!(sort!(bars; by = first))                            # Bars as node pairs
            resize_buffers!(bars_buffers, length(bars))                        # Resize buffers of length(bars)

            # 5. Graphical output of the current mesh
            PLOT && display(simpplot(p,t))
        end

        # Check that there are any points remaining
        (isempty(bars) || isempty(t) || length(p) == length(pfix)) && (return restart())

        # 6. Move mesh points based on bar lengths L and forces F
        @inbounds for (i, b) in enumerate(bars)
            p1, p2 = p[b[1]], p[b[2]]
            barvec[i] = p1 - p2                               # List of bar vectors
            L[i] = norm(barvec[i])                            # L  =  Bar lengths
            hbars[i] = fh((p1+p2)/2)
        end
        L0 .= hbars .* (FSCALE * norm(L)/norm(hbars))         # L0  =  Desired lengths

        # Density control - remove points that are too close
        if mod(count, DENSITYCTRLFREQ) == 0
            b = L0 .> DENSITYRELTHRESH .* L
            if any(b)
                ix = setdiff(reinterpret(Int, bars[b]), 1:nfix)
                deleteat!(p, unique!(sort!(ix)))
                resize_buffers!(p_buffers, length(p))         # Resize buffers of length(p)
                force_triangulation = true                    # Force retriangulation
                continue
            end
        end

        F .= max.(L0 .- L, zero(T))                           # Bar forces (scalars)
        Fvec .= (F./L) .* barvec                              # Bar forces (x,y components)
        Ftot = fill!(Ftot, zero(V))
        @inbounds for (i, b) in enumerate(bars)
            Ftot[b[1]] += Fvec[i]
            Ftot[b[2]] -= Fvec[i]
        end
        @inbounds for i in 1:length(pfix)
            Ftot[i] = zero(V)                                 # Force = 0 at fixed points
        end
        p .+= DELTAT .* Ftot                                  # Update node positions

        # 7. Bring outside points back to the boundary, and
        # 8. Termination criterion: All interior nodes move less than DPTOL (scaled)
        dterm = -T(Inf)
        @inbounds for i in eachindex(p)
            d = fd(p[i])
            if d > zero(T)                                   # Find points outside (d>0)
                ∇d = ∇fd(p[i])                               # ForwardDiff gradient
                p[i] -= ∇d * (d/(∇d⋅∇d))                     # Project
            elseif d < -GEPS
                d_int = DELTAT * norm(Ftot[i])/h0 #TODO sqrt(DELTAT)?
                dterm = max(dterm, d_int)
            end
        end
        dterm >= dtermbest && (stallcount += 1)
        dtermbest = min(dtermbest, dterm)

        if dterm < DPTOL || stallcount >= MAXSTALLITERS
            break
        end
    end

    # Clean up and plot final mesh
    p, t, _ = fixmesh(p,t)
    (PLOT || PLOTLAST) && display(simpplot(p,t))

    return p, t
end
