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
        fd,                             # signed distance function
        fsubs::AbstractVector,          # vector subregion distance functions, the boundaries of which are forced onto the grid
        fh,                             # relative edge length function
        h0::T,                          # nominal edge length
        bbox::Matrix{T},                # bounding box (2x2 matrix [xmin ymin; xmax ymax])
        dg::Int                         = 1, # Type of triangular mesh: dg=1 linear; dg=2 quadratic
        nr::Int                         = 0, # Number of refinements (nr=0, without refinement)
        pfix::AbstractVector{Vec{2,T}}  = Vec{2,T}[], # fixed points
        pinit::AbstractVector{Vec{2,T}} = init_points(bbox, h0), # inital distribution of points (triangular grid by default)
        ∇fd                             = x -> Tensors.gradient(fd, x), # Gradient of distance function `fd`
        ∇fsubs                          = [x -> Tensors.gradient(fs, x) for fs in fsubs]; # gradients of subregion distance functions
        PLOT::Bool                      = false, # plot all triangulations during evolution
        PLOTLAST::Bool                  = false, # plot resulting triangulation
        MAXITERS::Int                   = 500, # max iterations of stalled progress
        MP::Int                         = 5, # equilibrium step size threshold
        DETERMINISTIC::Bool             = false, # use deterministic pseudo-random
        FSCALE::T                       = T(1.2), # scale bar lengths
        DELTAT::T                       = T(0.1), # relative step size
        PEPS::T                         = T(5e-3),
        REPS::T                         = T(0.5),
        QMIN::T                         = T(0.5), # minimum triangle quality (0 < QMIN < 1)
        GEPS::T                         = T(1e-3)*h0, # boundary distance threshold
        DEPS::T                         = sqrt(eps(T))*h0 # finite difference step-length
    ) where {T}

    # Create initial distribution in bounding box (equilateral triangles by default)
    p = copy(pinit)

    # Remove points outside the region, apply the rejection method
    p = filter!(x -> fd(x) < GEPS, p)                      # Keep only d<0 points
    r0 = inv.(fh.(p)).^2                                   # Probability to keep point

    reject_prob = DETERMINISTIC ? rand(MersenneTwister(0), T, length(p)) : rand(T, length(p))
    p = p[maximum(r0) .* reject_prob .< r0]                # Rejection method

    threshunique_(x) = threshunique(sort!(copy(x); by = first); rtol = √eps(T), atol = eps(T))
    p, pfix = threshunique_(p), threshunique_(pfix)        # Remove (approximately) duplicate input nodes

    !isempty(pfix) && (p = setdiff!(p, pfix))              # Remove duplicated nodes between p and pfix
    p = [pfix; p]                                          # Prepend fix points
    t = NTuple{3,Int}[]

    # Initialize buffers
    resize_buffers!(buf,len) = map!(x->resize!(x, len), buf, buf) # resize function
    V = Vec{2,T}
    barvec, L, Lbars, hbars = V[], T[], T[], T[]
    velocity, Δp = V[], T[]
    bars_bufs = [barvec, L, Lbars, hbars] # buffers with length == length(bars)
    node_bufs = [velocity, Δp] # buffers with length == length(p)
    
    # Initialize variables for first iteration
    iter = 0
    ptol, rtol, fixed_nodes = T(1.0), T(1e2), Int[]
    local pold, bars, bdry_nodes, int_nodes
    
    while (iter < MAXITERS && ptol > PEPS)
        iter += 1

        # Re-triangulation by the Delaunay algorithm
        if (rtol > REPS)
            pold = copy(p)                                                     # Save current positions
            
            t = delaunay2!(t, p)                                               # List of triangles
            t = filter!(t) do t
                @inbounds pmid = (p[t[1]] + p[t[2]] + p[t[3]])/3               # Compute centroids
                return fd(pmid) < -GEPS                                        # Keep interior triangles
            end
            t = map!(t, t) do t
                @inbounds A = triangle_area(t, p)
                return A < 0 ? (t[1],t[3],t[2]) : t                            # Flip triangles with incorrect orientation
            end
            
            # Describe each bar by a unique pair of nodes
            bars = unique!(sort!(sortededges(p,t); by=first))                    # sortededges are sorted individually for unique!
            edges = boundedges(p,t)
            bdry_nodes = unique!(sort!(copy(reinterpret(Int, copy(edges)))))
            int_nodes = setdiff!(collect(1:length(p)), bdry_nodes)
            
            # Resize buffers
            resize_buffers!(node_bufs, length(p))
            resize_buffers!(bars_bufs, length(bars))

            # Graphical output of the current mesh
            PLOT && display(simpplot(p,t))
        end

        # 6. Move mesh points based on bar lengths L and forces f
        @inbounds for (i, b) in enumerate(bars)
            p1, p2 = p[b[1]], p[b[2]]
            barvec[i] = p2 - p1
            Lbars[i] = norm(barvec[i])
            hbars[i] = fh((p1+p2)/2)
        end
        ωs = FSCALE * norm(Lbars) / norm(hbars) # FSCALE times scaling factor
        L .= Lbars ./ (ωs .* hbars) # normalized bar lengths

        # Split edges which are too long
        if iter > length(p)
            longbars = bars[findall(ℓ -> ℓ>T(1.5), L)]
            if !isempty(longbars)
                append!(p, [(p[b[1]] + p[b[2]])/2 for b in longbars])
                ptol, rtol = T(1.0), T(1e2)
                continue
            end
        end

        # Compute velocity
        fill!(velocity, zero(V))
        @inbounds for (i, b) in enumerate(bars)
            ℓ, e = L[i], barvec[i]
            # f = max(1 - ℓ, zero(T)) # Persson–Strang force
            f = (1-ℓ^4) * exp(-ℓ^4) # Bossen–Heckbert smoothing function
            v = (f/ℓ) * e
            velocity[b[1]] -= v
            velocity[b[2]] += v
        end

        # Force = 0 at fixed points (both input, and those fixed for convergence acceleration)
        @inbounds for i in 1:length(pfix); velocity[i] = zero(V); end
        (iter > MP * length(p)) && @inbounds for i in fixed_nodes; velocity[i] = zero(V); end
        
        # Update node positions
        p .+= DELTAT .* velocity

        # Check if boundary nodes are far from boundary, and if so, project back
        @inbounds for i in bdry_nodes
            d = fd(p[i])
            while abs(d) > GEPS
                p[i] -= d * ∇fd(p[i])
                d = fd(p[i])
            end
        end

        # 8. Termination criterion: All interior nodes move less than DPTOL (scaled)
        Δp .= DELTAT .* norm.(velocity)
        ptol = maximum(i -> Δp[i], int_nodes)/h0
        rtol = maximum(ps -> norm(ps[1] - ps[2]), zip(p, pold))/h0

        # 9. Check the nodes speed if iter > MP * length(p)
        if iter > MP * length(p)
            fixed_nodes = findall(dx -> norm(dx) < DELTAT * PEPS, Δp)
        end
        
        # 10. check the triangle orientation & quality if ptol < PEPS
        if ptol < PEPS
            Amin = minimum(t->triangle_area(t, p), t)
            Qs = [triangle_quality(t, p) for t in t]
            Qmin, Qidx = findmin(Qs)
            if Amin < 0 || Qmin < QMIN
                if Qmin < QMIN
                    it = setdiff(t[Qidx], union(1:length(pfix), fixed_nodes)) |> sort! |> unique!
                    if length(it) > 1
                        pmid = mean(p[i] for i in it) # 2 or 3 points are unfixed; this is either edge midpoint or triangle midpoint
                        deleteat!(p, it)
                        push!(p, pmid)
                    end
                end
                ptol, rtol, fixed_nodes = T(1.0), T(1e2), Int[]
            end
        end
    end

    # Plot final mesh
    (PLOT || PLOTLAST) && display(simpplot(p,t))

    return p, t
end
