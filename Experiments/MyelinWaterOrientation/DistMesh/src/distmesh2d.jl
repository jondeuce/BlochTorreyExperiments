# ---------------------------------------------------------------------------- #
# DISTMESH2D 2-D Mesh Generator using Distance Functions.
#   [P,T] = DISTMESH2D(FD,FH,H0,BBOX,PFIX,FPARAMS)
#
#      P:         Node positions (Nx2)
#      T:         Triangle indices (NTx3)
#      FD:        Distance function d(x,y)
#      FH:        Scaled edge length function h(x,y)
#      H0:        Initial edge length
#      BBOX:      Bounding box [xmin,ymin; xmax,ymax]
#      PFIX:      Fixed node positions (NFIXx2)
#      FPARAMS:   Additional parameters passed to FD and FH
#
#   Example: (Uniform Mesh on Unit Circle)
#      fd = @(p) sqrt(sum(p.^2,2))-1;
#      [p,t] = distmesh2d(fd,@huniform,0.2,[-1,-1;1,1],[]);
#
#   Example: (Rectangle with circular hole, refined at circle boundary)
#      fd = @(p) ddiff(drectangle(p,-1,1,-1,1),dcircle(p,0,0,0.5));
#      fh = @(p) 0.05+0.3*dcircle(p,0,0,0.5);
#      [p,t] = distmesh2d(fd,fh,0.05,[-1,-1;1,1],[-1,-1;-1,1;1,-1;1,1]);
#
#   Example: (Polygon)
#      pv = [-0.4 -0.5;0.4 -0.2;0.4 -0.7;1.5 -0.4;0.9 0.1;
#          1.6 0.8;0.5 0.5;0.2 1;0.1 0.4;-0.7 0.7;-0.4 -0.5];
#      [p,t] = distmesh2d(@dpoly,@huniform,0.1,[-1,-1; 2,1],pv,pv);
#
#   Example: (Ellipse)
#      fd = @(p) p(:,1).^2/2^2+p(:,2).^2/1^2-1;
#      [p,t] = distmesh2d(fd,@huniform,0.2,[-2,-1;2,1],[]);
#
#   Example: (Square, with size function point and line sources)
#      fd = @(p) drectangle(p,0,1,0,1);
#      fh = @(p) min(min(0.01+0.3*abs(dcircle(p,0,0,0)), ...
#                   0.025+0.3*abs(dpoly(p,[0.3,0.7; 0.7,0.5]))),0.15);
#      [p,t] = distmesh2d(fd,fh,0.01,[0,0;1,1],[0,0;1,0;0,1;1,1]);
#
#   Example: (NACA0012 airfoil)
#      hlead = 0.01; htrail = 0.04; hmax = 2; circx = 2; circr = 4;
#      a = .12/.2*[0.2969,-0.1260,-0.3516,0.2843,-0.1036];
#
#      fd = @(p) ddiff(dcircle(p,circx,0,circr),(abs(p(:,2))-polyval([a(5:-1:2),0],p(:,1))).^2-a(1)^2*p(:,1));
#      fh = @(p) min(min(hlead+0.3*dcircle(p,0,0,0),htrail+0.3*dcircle(p,1,0,0)),hmax);
#
#      fixx = 1-htrail*cumsum(1.3.^(0:4)');
#      fixy = a(1)*sqrt(fixx)+polyval([a(5:-1:2),0],fixx);
#      fix = [[circx+[-1,1,0,0]*circr; 0,0,circr*[-1,1]]'; 0,0; 1,0; fixx,fixy; fixx,-fixy];
#      box = [circx-circr,-circr; circx+circr,circr];
#      h0 = min([hlead,htrail,hmax]);
#
#      [p,t] = distmesh2d(fd,fh,h0,box,fix);
#
#
#   See also: MESHDEMO2D, DISTMESHND, DELAUNAYN, TRIMESH.
#
#   distmesh2d.m v1.1
#   Copyright (C) 2004-2012 Per-Olof Persson. See COPYRIGHT.TXT for details.
# ---------------------------------------------------------------------------- #

function distmesh2d(
        fd, # distance function
        fh, # edge length function
        h0::T, # nominal edge length
        bbox::Matrix{T}, # bounding box (2x2 matrix [xmin ymin; xmax ymax])
        pfix = Vec{2,T}[], # fixed points
        ∇fd = x -> Tensors.gradient(fd, x); # Gradient of distance function `fd`
        PLOT::Bool = false, # plot all triangulations during evolution
        PLOTLAST::Bool = false, # plot resulting triangulation
        DETERMINISTIC::Bool = false, # use deterministic pseudo-random
        MAXSTALLITERS::Int = 500, # max iterations of stalled progress
        RESTARTEDGETHRESH::T = T(1e-4)*maximum(diff(bbox;dims=1)), # min. h0 s.t. restart is allowed
        DENSITYCTRLFREQ::Int = 30, # density check frequency
        DPTOL::T = T(1e-3), # equilibrium step size threshold
        TTOL::T = T(0.1), # large movement tolerance for retriangulation
        FSCALE::T = T(1.2), # scale bar lengths
        DELTAT::T = T(0.2), # relative step size
        GEPS::T = T(1e-3)*h0, # boundary distance threshold
        DEPS::T = sqrt(eps(T))*h0 # finite difference step-length
    ) where {T}

    # 0. Useful defines
    V = Vec{2,T}
    InfV = V((Inf,Inf))

    # Function for restarting if h0 is large enough that all points get removed
    function restart()
        if h0/2 < RESTARTEDGETHRESH
            @warn "h0 too small to allow a restart; returning empty grid."
            return pfix, NTuple{3,Int}[]
        else
            # @warn "No points remaining! Shrinking h0 -> h0/2 and retrying..."
            return distmesh2d(fd, fh, h0/2, bbox, pfix, ∇fd;
                PLOT = PLOT,
                PLOTLAST = PLOTLAST,
                DETERMINISTIC = DETERMINISTIC,
                MAXSTALLITERS = MAXSTALLITERS,
                RESTARTEDGETHRESH = RESTARTEDGETHRESH,
                DENSITYCTRLFREQ = DENSITYCTRLFREQ,
                DPTOL = DPTOL,
                TTOL = TTOL,
                FSCALE = FSCALE,
                DELTAT = DELTAT,
                GEPS = GEPS/2, # scales with h0
                DEPS = DEPS/2 # scales with h0
            )
        end
    end

    # 1. Create initial distribution in bounding box (equilateral triangles)
    xrange, yrange = bbox[1,1]:h0:bbox[2,1], bbox[1,2]:h0*T(sqrt(3)/2):bbox[2,2]
    p = zeros(V, length(yrange), length(xrange))
    for (i,y) in enumerate(yrange), (j,x) in enumerate(xrange)
        iseven(i) && (x += h0/2)            # Shift even rows
        p[i,j] = V((x,y))                   # List of node coordinates
    end
    p = vec(p)

    # 2. Remove points outside the region, apply the rejection method
    p = filter!(x -> fd(x) < GEPS, p)       # Keep only d<0 points
    r0 = inv.(fh.(p)).^2                    # Probability to keep point
    isempty(r0) && (return restart())

    randlist = DETERMINISTIC ? mod.(T.(1:length(p)), T(2pi))./T(2pi) : rand(T, length(p))
    p = p[maximum(r0) .* randlist .< r0]                   # Rejection method
    isempty(p) && (return restart())

    !isempty(pfix) && (p = setdiff!(p, pfix))              # Remove duplicated nodes
    pfix = sort!(copy(pfix); by = first)
    pfix = threshunique(pfix; rtol = √eps(T), atol = eps(T))
    nfix = length(pfix)

    p = vcat(pfix, p)                                      # Prepend fix points
    t = delaunay2(p)

    # check that initial distribution is not empty
    (isempty(t) || length(p) == length(pfix)) && (return restart())

    # plot initial points
    PLOT && simpplot(p,t)

    count = 0
    stallcount = 0
    dtermbest = T(Inf)
    pold = V[InfV]                                                             # For first iteration
    bars = Vector{NTuple{2,Int}}()

    while true
        count += 1

        # 3. Retriangulation by the Delaunay algorithm
        # try
        if count == 1 || (√maximum(norm2.(p.-pold)) > h0 * TTOL)               # Any large movement?
            p = threshunique(p; rtol = √eps(T), atol = eps(T))
            isempty(p) && (return restart())

            pold = copy(p)                                                     # Save current positions
            t = delaunay2!(t, p)                                               # List of triangles
            pmid = V[(p[tt[1]] + p[tt[2]] + p[tt[3]])/3 for tt in t]           # Compute centroids
            t = t[fd.(pmid) .< -GEPS]                                          # Keep interior triangles

            # 4. Describe each bar by a unique pair of nodes
            resize!(bars, 3*length(t))
            @inbounds for (i,tt) in enumerate(t)
                a, b, c = sorttuple(tt)
                bars[3i-2] = (a, b)
                bars[3i-1] = (a, c)
                bars[3i  ] = (b, c)                                            # Interior bars duplicated
            end
            unique!(sort!(bars; by = first))                                   # Bars as node pairs

            # 5. Graphical output of the current mesh
            PLOT && simpplot(p,t)
        end

        # Check that there are any points remaining
        (isempty(bars) || isempty(t) || length(p) == length(pfix)) && (return restart())

        # 6. Move mesh points based on bar lengths L and forces F
        barvec = V[p[b[1]] - p[b[2]] for b in bars]           # List of bar vectors
        L = norm.(barvec)                                     # L  =  Bar lengths
        hbars = fh.(V[(p[b[1]] + p[b[2]])/2 for b in bars])
        L0 = hbars * (FSCALE * norm(L)/norm(hbars))           # L0  =  Desired lengths

        # Density control - remove points that are too close
        if mod(count, DENSITYCTRLFREQ) == 0
            b = L0 .> 2 .* L
            if any(b)
                ix = setdiff(reinterpret(Int, bars[b]), 1:nfix)
                deleteat!(p, unique!(sort!(ix)))
                pold = V[InfV]
                continue
            end
        end

        F = max.(L0 .- L, zero(T))                           # Bar forces (scalars)
        Fvec = F./L .* barvec                                # Bar forces (x,y components)
        Ftot = zeros(V, length(p))
        @inbounds for (i, b) in enumerate(bars)
            Ftot[b[1]] += Fvec[i]
            Ftot[b[2]] -= Fvec[i]
        end
        @inbounds for i in 1:length(pfix)
            Ftot[i] = zero(V)                                # Force = 0 at fixed points
        end
        p .+= DELTAT .* Ftot                                 # Update node positions

        # 7. Bring outside points back to the boundary
        d = fd.(p) # distances (used below)
        @inbounds for (i,pᵢ) in enumerate(p)
            if d[i] > zero(T)                                # Find points outside (d>0)
                ∇dᵢ = ∇fd(pᵢ)                                # ForwardDiff gradient
                p[i] -= ∇dᵢ * (d[i]/(∇dᵢ⋅∇dᵢ))               # Project
            end
        end

        # 8. Termination criterion: All interior nodes move less than DPTOL (scaled)
        d_int = DELTAT .* Ftot[d .< -GEPS] #TODO sqrt(DELTAT)?
        dterm = isempty(d_int) ? T(Inf) : sqrt(maximum(norm2, d_int))/h0

        if dterm >= dtermbest
            stallcount = stallcount + 1
        end
        dtermbest = min(dtermbest, dterm)

        if dterm < DPTOL || stallcount >= MAXSTALLITERS
            break
        end
    end

    # Clean up and plot final mesh
    p, t, _ = fixmesh(p,t)
    (PLOT || PLOTLAST) && simpplot(p,t)

    return p, t
end

# ---------------------------------------------------------------------------- #
# Delaunay triangulation
# ---------------------------------------------------------------------------- #

function delaunay2!(
        t::Vector{NTuple{3,Int}},
        p::AbstractVector{Vec{2,T}}
    ) where {T}

    p, pmin, pmax = scaleto(p, min_coord + T(0.1), max_coord - T(0.1))
    P = IndexedPoint2D[IndexedPoint2D(pp[1], pp[2], i) for (i,pp) in enumerate(p)]
    unique!(sort!(P; by = getx))

    tess = DelaunayTessellation2D{IndexedPoint2D}(length(P))
    push!(tess, P)
    t = assign_triangles!(t, tess)

    return t
end
delaunay2(p) = delaunay2!(Vector{NTuple{3,Int}}(), p)

# ---------------------------------------------------------------------------- #
# Assign triangle indicies from Delaunay triangulation
# ---------------------------------------------------------------------------- #

function assign_triangles!(t, tess)
    resize!(t, length(tess))
    @inbounds for (i,tt) in enumerate(tess)
        t[i] = (getidx(geta(tt)), getidx(getb(tt)), getidx(getc(tt)))
    end
    return t
end

# ---------------------------------------------------------------------------- #
# DelaunayTessellation2D iteration protocol
# ---------------------------------------------------------------------------- #

function Base.iterate(tess::DelaunayTessellation2D, ix = 2)
    @inbounds while isexternal(tess._trigs[ix]) && ix <= tess._last_trig_index
        ix += 1
    end
    @inbounds if ix > tess._last_trig_index
        return nothing
    else
        return (tess._trigs[ix], ix + 1)
    end
end

function Base.length(tess::DelaunayTessellation2D)
    len = 0
    for t in tess
        len += 1
    end
    return len
end

Base.eltype(tess::DelaunayTessellation2D{P}) where {P} = VoronoiDelaunay.DelaunayTriangle{P}

# ---------------------------------------------------------------------------- #
# Testing new iteration protocol
# ---------------------------------------------------------------------------- #

# using VoronoiDelaunay
# using BenchmarkTools
#
# function collect_triangles(tess::DelaunayTessellation2D)
#     tris = VoronoiDelaunay.DelaunayTriangle{Point2D}[]
#     for t in tess
#         push!(tris, t)
#     end
#     return tris
# end
#
# function testfun(tess::DelaunayTessellation2D)
#     s = Point2D(0.0, 0.0)
#     for t in tess
#         s = geta(t)
#     end
#     return s
# end
#
# tess = DelaunayTessellation()
# width = max_coord - min_coord
# a = Point2D[Point(min_coord+rand()*width, min_coord+rand()*width) for i in 1:100]
# push!(tess, a)
#
# display(@benchmark testfun($tess))
# tris_old = collect_triangles(tess)
#
# function Base.iterate(tess::DelaunayTessellation2D, ix = 2)
#     @inbounds while isexternal(tess._trigs[ix]) && ix <= tess._last_trig_index
#         ix += 1
#     end
#
#     @inbounds if ix > tess._last_trig_index
#         return nothing
#     else
#         return (tess._trigs[ix], ix + 1)
#     end
# end
#
# display(@benchmark testfun($tess))
# tris_new = collect_triangles(tess)
#
# @show tris_old == tris_new
