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
        MAXITERS::Int                   = 1000, # max iterations of stalled progress
        MPITERS::Int                    = 100, # equilibrium step size threshold (absolute)
        VERBOSE::Bool                   = false, # print verbose information
        TRIANGLECTRLFREQ::Int           = 100, # period in which triangles are checked for quality
        DENSITYCTRLFREQ::Int            = 50, # period in which points are checked for being too close
        DENSITYRELTHRESH::T             = T(2.5), # bars are too short if DENSITYRELTHRESH times bar length is less than the desired length
        DETERMINISTIC::Bool             = false, # use deterministic pseudo-random
        MP::Int                         = 5, # equilibrium step size threshold (relative to length(p))
        PEPS::T                         = T(5e-3), # relative step size threshold for interior points
        REPS::T                         = T(0.5), # relative point movement threshold between iterations
        QMIN::T                         = T(0.5), # minimum triangle quality (0 < QMIN < 1)
        GEPS::T                         = T(1e-3)*h0, # boundary distance threshold
        DEPS::T                         = sqrt(eps(T))*h0, # finite difference step-length
        FSCALE::T                       = T(1.2), # scale bar lengths
        DELTAT::T                       = T(0.1) # relative step size
    ) where {T}

    # Create initial distribution in bounding box (equilateral triangles by default)
    p = copy(pinit)

    # Remove points outside the region, apply the rejection method
    p = filter!(x -> fd(x) < GEPS, p)                      # Keep only d<0 points
    r0 = inv.(fh.(p)).^2                                   # Probability to keep point

    reject_prob = DETERMINISTIC ? rand(MersenneTwister(0), T, length(p)) : rand(T, length(p))
    p = p[maximum(r0) .* reject_prob .< r0]                # Rejection method
    p = threshunique_all(p,h0)                             # Remove (approximately) duplicate input nodes
    pfix = threshunique_all(pfix,h0)                       # Remove (approximately) duplicate fixed nodes

    !isempty(pfix) && (p = setdiff!(p, pfix))              # Remove duplicated nodes between p and pfix
    p = [pfix; p]                                          # Prepend fix points
    t = NTuple{3,Int}[]

    # Initialize buffers
    resize_buffers!(buf, len) = map!(x->resize!(x, len), buf, buf) # resize function
    V = Vec{2,T}
    barvec, L, Lbars, hbars = V[], T[], T[], T[]
    velocity, Δp = V[], T[]
    bars_bufs = [barvec, L, Lbars, hbars] # buffers with length == length(bars)
    node_bufs = [velocity, Δp] # buffers with length == length(p)
    
    # Initialize variables for first iteration
    iter, tricount = 0, 0
    ptol, rtol, fixed_nodes = T(1.0), T(1e2), Int[]
    local pold, bars, bdry_nodes, int_nodes
    
    MESHTIME = @elapsed while iter < MAXITERS && ptol > PEPS
        iter += 1

        # Re-triangulation by the Delaunay algorithm
        if rtol > REPS
            tricount += 1
            VERBOSE && println("iter = $iter: triangulation #$tricount")
            
            old_length = length(p)
            p = threshunique_fixed(p, union(1:length(pfix), fixed_nodes), h0) # triangulation will fail if two points are within eps() of eachother
            !(length(p) == old_length) && (fixed_nodes = Int[]) # some points were dropped
            pold = copy(p) # Save current positions

            # Update mesh variables
            t = update_mesh!(t, p, fd, GEPS)
            bars = unique!(sort!(sortededges(p,t); by=first)) # `sortededges` only sorts edges individually
            edges = boundedges(p,t)
            bdry_nodes = unique!(sort!(copy(reinterpret(Int, copy(edges)))))
            int_nodes = setdiff!(collect(1:length(p)), bdry_nodes)
            
            # Resize buffers
            resize_buffers!(node_bufs, length(p))
            resize_buffers!(bars_bufs, length(bars))

            # Graphical output of the current mesh
            PLOT && display(simpplot(p,t))
        end

        # Move mesh points based on bar lengths L and forces f
        @inbounds for (i, b) in enumerate(bars)
            p1, p2 = p[b[1]], p[b[2]]
            barvec[i] = p2 - p1
            Lbars[i] = norm(barvec[i])
            hbars[i] = fh((p1+p2)/2)
        end
        ωs = FSCALE * norm(Lbars) / norm(hbars) # FSCALE times scaling factor
        L .= Lbars ./ (ωs .* hbars) # normalized bar lengths

        # Split edges which are too long
        if iter != MAXITERS && mod(iter, DENSITYCTRLFREQ) == 0 #iter > length(p)
            longbars = bars[findall(ℓ -> ℓ > T(1.5), L)]
            if !isempty(longbars)
                append!(p, [(p[b[1]] + p[b[2]])/2 for b in longbars])
                ptol, rtol = T(1.0), T(1e2)
                VERBOSE && println("iter = $iter: splitting $(length(longbars)) bars, now $(length(p)) points")
                continue
            end
        end

        # Density control - remove points that are too close
        if iter != MAXITERS && mod(iter + div(DENSITYCTRLFREQ,2), DENSITYCTRLFREQ) == 0
            shortbars = bars[findall(ℓ -> ℓ < inv(DENSITYRELTHRESH), L)]
            ix = setdiff(reinterpret(Int, shortbars), union(1:length(pfix), fixed_nodes))
            ix = unique!(sort!(ix))
            if !isempty(ix)
                deleteat!(p, ix)
                ptol, rtol, fixed_nodes = T(1.0), T(1e2), Int[]
                VERBOSE && println("iter = $iter: density control, $(length(ix)) points removed, $(length(p)) remaining")
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
        @inbounds for i in 1:length(pfix)
            velocity[i] = zero(V)
        end
        if iter > min(MPITERS, MP * length(p))
            @inbounds for i in fixed_nodes
                velocity[i] = zero(V)
            end
        end
        
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

        # Termination criterion: All interior nodes move less than DPTOL (scaled)
        Δp .= DELTAT .* norm.(velocity)
        
        # Use mean for approximate convergence check
        ptol = mean(i -> Δp[i], int_nodes)/h0 # ptol = maximum(i -> Δp[i], int_nodes)/h0
        rtol = mean(ps -> norm(ps[1] - ps[2]), zip(p, pold))/h0 # rtol = maximum(ps -> norm(ps[1] - ps[2]), zip(p, pold))/h0

        # Check the nodes speed at convergence
        if iter > min(MPITERS, MP * length(p))
            fixed_nodes = findall(dx -> norm(dx) < DELTAT * PEPS, Δp)
        end
        
        # Check the triangle orientation & quality
        if iter != MAXITERS && (ptol < PEPS || mod(iter, TRIANGLECTRLFREQ) == 0)
            Amin = minimum(t->triangle_area(t, p), t)
            Qs = [triangle_quality(t, p) for t in t]
            # Qidx = findall(Q -> Q < QMIN, Qs)
            # Qmin = minimum(Qs)
            # if Amin < 0 || !isempty(Qidx)
            #     if !isempty(Qidx)
            #         idx_to_delete = Int[]
            #         all_fixed_nodes = union(1:length(pfix), fixed_nodes)
            #         for qidx in Qidx
            #             it = setdiff(t[qidx], all_fixed_nodes)
            #             if length(it) > 1
            #                 pmid = mean(p[i] for i in it) # 2 or 3 points are unfixed; this is either edge midpoint or triangle midpoint
            #                 push!(p, pmid) # doesn't affect indexing
            #                 append!(idx_to_delete, it)
            #             end
            #         end
            #         deleteat!(p, unique!(sort!(idx_to_delete)))
            #     end
            #     ptol, rtol, fixed_nodes = T(1.0), T(1e2), Int[]
            #     VERBOSE && println("iter = $iter: removed $(length(Qidx)) low quality trangles, Qmin = $Qmin")
            #     continue
            # end
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
                VERBOSE && println("iter = $iter: removing triangle with Qmin = $Qmin")
                continue
            end
        end

        VERBOSE && println("iter = $iter: ptol = $ptol (goal: $PEPS)")
    end

    # Plot final mesh
    VERBOSE && println(
        "FINISHED:\n" *
        "    Iterations:       $iter\n" *
        "    Number of points: $(length(p))\n" *
        "    Number of cells:  $(length(t))\n" *
        "    Minimum Q:        $(mesh_quality(p,t))\n" *
        "    Maximum rho:      $(mesh_rho(p,t))\n" *
        "    Time:             $MESHTIME")
    (PLOT || PLOTLAST) && display(simpplot(p,t))

    return p, t
end

function update_mesh!(t, p, fd, GEPS)
    # Delaunay triangulation
    t = delaunay2!(t, p)

    # Filter out triangles with midpoints outside of mesh
    t = filter!(t) do t
        @inbounds pmid = (p[t[1]] + p[t[2]] + p[t[3]])/3
        return fd(pmid) < -GEPS
    end

    # Ensure triangles have proper orientation
    t = map!(t, t) do t
        @inbounds A = triangle_area(t, p)
        return A < 0 ? (t[1],t[3],t[2]) : t
    end

    return t
end

threshunique_all(x::AbstractVector{Vec{2,T}}, h0::T) where {T} = threshunique(sort!(copy(x)); rtol = zero(T), atol = h0*√eps(T))
function threshunique_fixed(
        p::AbstractVector{Vec{2,T}},
        ifixed::AbstractVector{Int},
        h0::T
    ) where {T}
    pfix = p[ifixed]
    unfixed = p[setdiff(1:length(p), ifixed)]
    unfixed = threshunique_all(unfixed, h0) # ensure approximately unique unfixed points
    unfixed = filter!(unfixed) do p
        !any(p0 -> norm(p-p0) < h0*√eps(T), pfix) # keep those which aren't too close to fixed points
    end
    return [pfix; unfixed]
end
