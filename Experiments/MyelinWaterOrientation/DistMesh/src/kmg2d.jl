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
#   pfix     : Fixed points (points that must appear in the mesh)
#   varargin : Additional parameters passed to fd and fh (optional)
# Output:
#   p        : Point coordinates np*2
#   t        : Triangle vertices nt*3 (linear) or nt*6 (quadratic)
#   be       : Boundary edges    ne*2 (linear) or nt*3 (quadratic)
#   bn       : Boundary points    nb*1
#--------------------------------------------------------------------
# (c) 2009, Koko J., ISIMA, koko@isima.fr
#--------------------------------------------------------------------

function kmg2d(
        fd,                             # signed distance function
        fsubs::AbstractVector,          # vector of sub-region distance functions, the boundaries of which are forced onto the grid
        fh,                             # relative edge length function
        h0::T,                          # nominal edge length
        bbox::Matrix{T},                # bounding box (2x2 matrix [xmin ymin; xmax ymax])
        dg::Int                         = 1, # Type of triangular mesh: dg=1 linear; dg=2 quadratic
        nr::Int                         = 0, # Number of refinements (nr=0, without refinement)
        pfix::AbstractVector{Vec{2,T}}  = Vec{2,T}[], # fixed points
        pinit::AbstractVector{Vec{2,T}} = init_points(bbox, h0), # inital distribution of points (triangular grid by default)
        ∇fd                             = x -> Tensors.gradient(fd, x), # Gradient of distance function `fd`
        ∇fsubs                          = [x -> Tensors.gradient(fs, x) for fs in fsubs]; # gradients of sub-region distance functions
        PLOT::Bool                      = false, # plot all triangulations during evolution
        PLOTLAST::Bool                  = false, # plot resulting triangulation
        MAXITERS::Int                   = 1000, # max iterations of stalled progress
        MPITERS::Int                    = 250, # after this number of iterations, points which haven't moved far are fixed
        FIXEDSUBSITERS::Int             = 250, # after this number of iterations, interior region boundary points are fixed
        VERBOSE::Bool                   = false, # print verbose information
        TRIANGLECTRLFREQ::Int           = 200, # period in which triangles are checked for quality
        DENSITYCTRLFREQ::Int            = 100, # period in which points are checked for being too close
        DENSITYRELTHRESH::T             = T(3.0), # bars are too short if DENSITYRELTHRESH times bar length is less than the desired length
        DETERMINISTIC::Bool             = false, # use deterministic pseudo-random
        MP::Int                         = 5, # equilibrium step size threshold (relative to length(p))
        PEPS::T                         = T(5e-3), # relative step size threshold for interior points
        REPS::T                         = T(0.5), # relative point movement threshold between iterations
        QMIN::T                         = T(0.5), # minimum triangle quality (0 < QMIN < 1)
        GEPS::T                         = sqrt(eps(T))*h0, # boundary distance threshold (note: can be very low since exact ∇fd should project exactly to boundary)
        LEPS::T                         = T(1e-3), # minimum relative bar length for force calculation (to avoid NaN forces)
        DEPS::T                         = sqrt(eps(T))*h0, # finite difference step-length
        FSCALE::T                       = T(1.2), # scale bar lengths
        DELTAT::T                       = T(0.1) # relative step size
    ) where {T}

    # Create initial distribution in bounding box (equilateral triangles by default)
    p = copy(pinit)

    # Remove points outside the region, apply the rejection method
    p = filter!(x -> fd(x) <= GEPS, p)                     # Keep only d<0 points
    r0 = inv.(fh.(p)).^2                                   # Probability to keep point

    reject_prob = DETERMINISTIC ? rand(MersenneTwister(0), T, length(p)) : rand(T, length(p))
    p = p[maximum(r0) .* reject_prob .< r0]                # Rejection method
    p = gridunique_all(p,h0)                               # Remove (approximately) duplicate input points
    pfix = gridunique_all(pfix,h0)                         # Remove (approximately) duplicate fixed points

    !isempty(pfix) && (p = setdiff!(p, pfix))              # Remove duplicated points between p and pfix
    p = [pfix; p]                                          # Prepend fix points
    t = NTuple{3,Int}[]

    # Initialize buffers
    resize_buffers!(buf, len) = map!(x->resize!(x, len), buf, buf) # resize function
    V = Vec{2,T}
    barvec, L, Lbars, hbars = V[], T[], T[], T[]
    velocity, Δp = V[], T[]
    bars_bufs = [barvec, L, Lbars, hbars] # buffers with length == length(bars)
    point_bufs = [velocity, Δp] # buffers with length == length(p)
    
    # Initialize variables for first iteration
    iter, tricount = 0, 0
    ptol, rtol, fixed_nodes = T(1.0), T(1e2), Int[]
    sub_bdry_nodes = [Int[] for _ in 1:length(fsubs)]
    # subs_int_bdry_nodes = Int[]
    local pold, bars, bdry_nodes, int_nodes
    
    MESHTIME = @elapsed while iter < MAXITERS && ptol > PEPS
        iter += 1

        # Re-triangulation by the Delaunay algorithm
        if rtol > REPS
            tricount += 1
            VERBOSE && println("iter = $iter: triangulation #$tricount")
            
            old_length = length(p)
            p = gridunique_fixed(p, union(1:length(pfix), fixed_nodes), h0) # triangulation will fail if two points are within eps() of eachother
            !(length(p) == old_length) && (fixed_nodes = Int[]) # some points were dropped
            pold = copy(p) # Save current positions

            # Update mesh variables
            t = update_mesh!(t, p, fd, GEPS)

            bars = getedges(t; sorted = true) |> sort! |> unique! # edges sorted individual as tuples, then as vector
            edges = boundedges(p, t)
            bdry_nodes = reinterpret(Int, edges) |> copy |> sort! |> unique! # unique boundary points of domain
            int_nodes = setdiff(1:length(p), bdry_nodes)
            !isempty(fsubs) && update_sub_bdry_nodes!(sub_bdry_nodes, p, t, fsubs, GEPS)

            # Resize buffers
            resize_buffers!(point_bufs, length(p))
            resize_buffers!(bars_bufs, length(bars))
        end

        # Graphical output of the current mesh
        PLOT && display(simpplot(p,t))
        
        # Move mesh points based on bar lengths L and forces f
        @inbounds for (i, b) in enumerate(bars)
            p1, p2 = p[b[1]], p[b[2]]
            barvec[i] = p2-p1
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
        if iter != MAXITERS && iter > DENSITYCTRLFREQ && mod(iter - div(DENSITYCTRLFREQ,2), DENSITYCTRLFREQ) == 0
            shortbars = bars[findall(ℓ -> ℓ < inv(DENSITYRELTHRESH), L)]
            ix = setdiff(reinterpret(Int, shortbars), union(1:length(pfix), fixed_nodes))
            ix = unique!(sort!(ix))
            if !isempty(ix)
                deleteat!(p, ix)
                ptol, rtol, fixed_nodes = T(1.0), T(1e2), Int[]
                # ptol, rtol = T(1.0), T(1e2)
                # fixed_nodes = update_node_order!(fixed_nodes, ix)
                VERBOSE && println("iter = $iter: density control, $(length(ix)) points removed, $(length(p)) remaining")
                continue
            end
        end

        # Compute velocity
        fill!(velocity, zero(V))
        @inbounds for (i, b) in enumerate(bars)
            ℓ, e = max(L[i], LEPS), barvec[i]
            # f = max(1-ℓ, 0) # Persson–Strang force
            ℓ⁴ = (ℓ^2)^2
            f = (1-ℓ⁴) * exp(-ℓ⁴) # Bossen–Heckbert smoothing function
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

        # Project out normal component of velocity on internal sub-boundary nodes.
        # This prevents large movements away from sub-boundaries without having to fix the nodes.
        #   NOTE: `sub_bdry_nodes[j]` may contain fixed nodes, but their velocity was
        #         already set to zero above, so the projection has no effect
        if iter > min(FIXEDSUBSITERS, MP * length(p)) && !all(isempty, sub_bdry_nodes)
            for j in eachindex(fsubs)
                ∇fsub = ∇fsubs[j]
                @inbounds for i in sub_bdry_nodes[j]
                    # NOTE: ∇d should already be a unit vector, but normalizing by norm2(∇d) covers edge cases (e.g. finite differences)
                    ∇d = ∇fsub(p[i])
                    velocity[i] -= ((velocity[i] ⋅ ∇d) / norm2(∇d)) * ∇d # Project out normal component of velocity
                end
            end
        end
        
        # Update point positions
        p .+= DELTAT .* velocity

        # Project sub-region points, restarting if duplicate points are needed
        old_length = length(p)
        for j in eachindex(fsubs)
            p = project_sub_bdry_points!(p, sub_bdry_nodes[j], fd, fsubs[j], ∇fsubs[j], GEPS)
        end
        if length(p) != old_length
            ptol, rtol = T(1.0), T(1e2)
            VERBOSE && println("iter = $iter: sub-region projection, $(length(p) - old_length) new points added, now $(length(p))")
            continue
        end
        
        # Check if boundary points are far from boundary, and if so, project back
        p = project_bdry_points!(p, bdry_nodes, fd, ∇fd, GEPS)
                
        # Termination criterion: All interior points move less than DPTOL (scaled)
        Δp .= DELTAT .* norm.(velocity)
        
        # Use mean for approximate convergence check
        ptol = maximum(i -> Δp[i], int_nodes)/h0
        rtol = maximum(ps -> norm(ps[1] - ps[2]), zip(p, pold))/h0
        _, int_node_ix = findmax(Δp[int_nodes]) #DEBUG
        
        if (iter > 10001)
            #DEBUG show points after projection
            (iter > 10001) && display(simpplot(p,t))
            
            #DEBUG show fixed points
            _pfix = reinterpret(T, p[1:length(pfix)])
            display(plot!(_pfix[1:2:end], _pfix[2:2:end]; seriestype = :scatter, markersize = 5, colour = :green))
            
            #DEBUG show point which is moving the most
            x, y = p[int_nodes[int_node_ix]]
            display(plot!([x], [y]; seriestype = :scatter, markersize = 10, colour = :red))
        end

        # If points have moved sufficiently little since last iteration, fix them in place
        if iter > min(MPITERS, MP * length(p))
            fixed_nodes = findall(dx -> norm(dx) < DELTAT * PEPS, Δp)
        end
        if iter > min(FIXEDSUBSITERS, MP * length(p)) && !all(isempty, sub_bdry_nodes)
            # Fix sub-region boundary nodes which are also on the exterior boundary
            multi_bdry_nodes = reduce(vcat,
                begin
                    fsub, nodes = fsubs[j], sub_bdry_nodes[j]
                    findall(nodes) do i
                        @inbounds x = p[i]
                        abs(fsub(x)) <= GEPS && abs(fd(x)) <= GEPS
                    end
                end
                for j in eachindex(fsubs))
            if !isempty(multi_bdry_nodes)
                multi_bdry_nodes = unique!(sort!(multi_bdry_nodes))
                fixed_nodes = union(fixed_nodes, multi_bdry_nodes)
            end
        end
        
        # Check the triangle orientation & quality
        if iter != MAXITERS && (ptol < PEPS || mod(iter, TRIANGLECTRLFREQ) == 0)
            Amin = minimum(t->triangle_area(t, p), t)
            Qs = [triangle_quality(t, p) for t in t]
            Qmin, Qidx = findmin(Qs)
            if Amin < 0 || Qmin < QMIN
                if Qmin < QMIN
                    ix = setdiff(t[Qidx], union(1:length(pfix), fixed_nodes)) |> sort! |> unique!
                    if length(ix) > 1
                        pmid = mean(p[i] for i in ix) # 2 or 3 points are unfixed; this is either edge midpoint or triangle midpoint
                        deleteat!(p, ix)
                        push!(p, pmid)
                        
                        if (iter > 10001)
                            #DEBUG show point which is moving the most
                            x, y = pmid
                            display(plot!([x], [y]; seriestype = :scatter, markersize = 10, colour = :yellow))
                            sleep(3.0)
                        end
                        VERBOSE && println("iter = $iter: removing triangle with Qmin = $Qmin; removed $(length(ix)) points, now $(length(p))")
                    else
                        if (iter > 10001)
                            #DEBUG show point which is moving the most
                            x, y = mean(p[i] for i in t[Qidx])
                            display(plot!([x], [y]; seriestype = :scatter, markersize = 10, colour = :yellow))
                            sleep(3.0)
                        end
                        VERBOSE && println("iter = $iter: removing triangle with Qmin = $Qmin")
                    end
                else
                    VERBOSE && println("iter = $iter: retriangulating (Amin = $Amin < 0)")
                end
                ptol, rtol, fixed_nodes = T(1.0), T(1e2), Int[]
                continue
            end
        end

        VERBOSE && println("iter = $iter: rtol = $rtol, ptol = $ptol (goal: $PEPS)")
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
    t = delaunay2!(t, p) # Delaunay triangulation
    t = interior_triangles!(t, p, fd, GEPS)  # keep only interior triangles
    t = oriented_triangles!(t, p) # ensure proper orientation
    return t
end

function update_node_order!(nodes, deleted_nodes)
    nodes = setdiff!(nodes, deleted_nodes) # remove nodes which were deleted
    nodes = sort!(nodes) # sorting is allowed, since pfix are always at the front and other fixed point orders don't matter
    shift, shift_idx = 0, 1
    for i in eachindex(nodes)
        while shift_idx <= length(deleted_nodes) && nodes[i] > deleted_nodes[shift_idx]
            shift_idx += 1
            shift += 1
        end
        nodes[i] -= shift
    end
    return nodes
end

function interior_triangles!(t, p, fd, GEPS)
    # Filter out triangles with midpoints outside of mesh
    t = filter!(t) do t
        @inbounds pmid = (p[t[1]] + p[t[2]] + p[t[3]])/3
        return fd(pmid) <= -GEPS
    end
    return t
end
interior_triangles(t, p, fd, GEPS) = interior_triangles!(copy(t), p, fd, GEPS)

function oriented_triangles!(t, p)
    # Ensure triangles have proper orientation
    t = map!(t, t) do t
        @inbounds A = triangle_area(t, p)
        return A < 0 ? (t[1],t[3],t[2]) : t
    end
    return t
end
oriented_triangles(t, p) = oriented_triangles!(copy(t), p)

function update_sub_bdry_nodes!(sub_bdry_nodes, p, t, fsubs, GEPS)
    for (j,fsub) in enumerate(fsubs)
        tsub = interior_triangles(t, p, fsub, GEPS)
        subedges = boundedges(p, tsub)
        sub_bdry_nodes[j] = reinterpret(Int, subedges) |> copy |> sort! |> unique!
    end
    return sub_bdry_nodes
end

@inline function project_point(x, fd, ∇fd, GEPS)
    # Check if boundary points are far from boundary, and if so, project back.
    # Note that if ∇fd computes the exact gradient, e.g. via automatic differentiation,
    # then the inner while loop will always only take one iteration, as ∇d is a unit vector
    # for a signed distance field d, and d is exactly the signed distance to the boundary
    d = fd(x)
    while !(abs(d) <= GEPS)
        ∇d = ∇fd(x)
        ∇d = ∇d/norm(∇d) # NOTE: should already be a unit vector, but this covers edge cases (e.g. finite differences)
        x -= d * ∇d
        d = fd(x)
    end
    return x
end

function project_bdry_points!(p, bdry_nodes, fd, ∇fd, GEPS)
    @inbounds for i in bdry_nodes
        p[i] = project_point(p[i], fd, ∇fd, GEPS)
    end
    return p
end

function project_sub_bdry_points!(p, sub_bdry_nodes, fd, fsub, ∇fsub, GEPS)
    @inbounds for (ix,i) in enumerate(sub_bdry_nodes)
        x = project_point(p[i], fsub, ∇fsub, GEPS)
        if abs(fd(p[i])) <= GEPS && !(abs(fd(x)) <= GEPS) #norm(x - p[i]) > GEPS
            # If the original point p[i] was on the exterior boundary, but the new projected
            # point x no longer is, push x into p and update the corresponding node
            push!(p, x)
            sub_bdry_nodes[ix] = lastindex(p) # x is on the sub-boundary
        else
            # Either p[i] wasn't on the external boundary to begin with, or remained there after projection
            p[i] = x
        end
    end
    return p
end

function gridunique_fixed(
        p::AbstractVector{Vec{2,T}},
        ifixed::AbstractVector{Int},
        h0::T
    ) where {T}
    pfix = p[ifixed]
    unfixed = p[setdiff(1:length(p), ifixed)]
    unfixed = gridunique_all(unfixed, h0) # ensure approximately unique unfixed points
    unfixed = filter!(unfixed) do p
        !any(p0 -> norm(p-p0) < h0*√eps(T), pfix) # keep those which aren't too close to fixed points
    end
    return [pfix; unfixed]
end
gridunique_all(x::AbstractVector{Vec{2,T}}, h0::T) where {T} = gridunique(x, h0*√eps(T))
