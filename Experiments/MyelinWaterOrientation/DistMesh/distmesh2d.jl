#DISTMESH2D 2-D Mesh Generator using Distance Functions.
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
function distmesh2d(
        fd, fh, h0::T, bbox, pfix = Vector{Vec{2,T}}();
        PLOT::Bool = false, # plot all triangulations during evolution
        PLOTLAST::Bool = false, # plot resulting triangulation
        MAXSTALLITERS::Int = 500, # max iterations of stalled progress
        DENSITYCTRLFREQ::Int = 30, # density check frequency
        DPTOL::T = T(0.001), # equilibrium step size threshold
        TTOL::T = T(0.1), # large movement tolerance for retriangulation
        FSCALE::T = T(1.2), # scale bar lengths
        DELTAT::T = T(0.2), # relative step size
        GEPS::T = T(0.001)*h0, # boundary distance threshold
        DEPS::T = sqrt(eps(T))*h0 # finite difference step-length
    ) where {T}

    # Useful constants
    V = Vec{2,T}
    InfV = V((Inf,Inf))

    # 1. Create initial distribution in bounding box (equilateral triangles)
    xrange, yrange = bbox[1,1]:h0:bbox[2,1], bbox[1,2]:h0*sqrt(3)/2:bbox[2,2]
    p = zeros(V, length(yrange), length(xrange))
    for (i,y) in enumerate(yrange), (j,x) in enumerate(xrange)
        iseven(i) && (x += h0/2)            # Shift even rows
        p[i,j] = V((x,y))                   # List of node coordinates
    end
    p = vec(p)

    # 2. Remove points outside the region, apply the rejection method
    p = filter!(x -> fd(x) < GEPS, p)       # Keep only d<0 points
    r0 = inv.(fh.(p)).^2                    # Probability to keep point
    p = p[maximum(r0) .* rand(length(p)) .< r0]            # Rejection method
    # randlist = abs.(sin.(T.(1:length(p))))
    # p = p[maximum(r0) .* randlist .< r0];                # Rejection method
    !isempty(pfix) && (p = setdiff!(p, pfix))              # Remove duplicated nodes
    pfix = unique!(pfix)
    nfix = length(pfix)
    p = vcat(pfix, p)                                      # Prepend fix points
    N = length(p)                                          # Number of points N

    count = 0;
    stallcount = 0;
    dtermbest = T(Inf);

    # Create gradient function for distance function `fd`
    ∇fd = x -> JuAFEM.gradient(fd, x)

    t = delaunay2(p)
    if PLOT
        # clf,view(2),axis equal,axis off
        simpplot(p,t)
    end

    pold = V[InfV]                                                             # For first iteration
    bars = Vector{NTuple{2,Int}}()

    while true
        count += 1

        # 3. Retriangulation by the Delaunay algorithm
        if count == 1 || (√maximum(norm2.(p.-pold)) > h0 * TTOL)               # Any large movement?
            pold = copy(p)                                                     # Save current positions
            t = delaunay2(p)                                                   # List of triangles
            pmid = V[(p[tt[1]] + p[tt[2]] + p[tt[3]])/3 for tt in t]           # Compute centroids
            t = t[fd.(pmid) .< -GEPS]                                          # Keep interior triangles

            # 4. Describe each bar by a unique pair of nodes
            resize!(bars, 3*length(t))
            @inbounds for (i,tt) in enumerate(t)
                bars[3i-2] = (tt[1], tt[2])
                bars[3i-1] = (tt[1], tt[3])
                bars[3i  ] = (tt[2], tt[3])                                    # Interior bars duplicated
            end
            bars .= sorttuple.(bars)
            sort!(bars; by = x -> x[1])
            unique!(bars)                                                      # Bars as node pairs

            # 5. Graphical output of the current mesh
            if PLOT
                simpplot(p,t)
            end
        end

        # 6. Move mesh points based on bar lengths L and forces F
        barvec = V[p[b[1]] - p[b[2]] for b in bars]           # List of bar vectors
        L = norm.(barvec)                                     # L  =  Bar lengths
        hbars = fh.(V[(p[b[1]] + p[b[2]])/2 for b in bars])
        L0 = hbars * (FSCALE * sqrt(sum(L.^2)/sum(hbars.^2))) # L0  =  Desired lengths

        # Density control - remove points that are too close
        if mod(count, DENSITYCTRLFREQ) == 0 && any(L0 .> 2.0 .* L)
            ix = setdiff(reinterpret(Int, bars[L0 .> 2.0 .* L]), 1:nfix)
            deleteat!(p, unique!(sort!(ix)))
            N = size(p,1)
            pold = V[InfV]
            continue
        end

        F = max.(L0.-L, 0.0)                                 # Bar forces (scalars)
        Fvec = F./L .* barvec                                # Bar forces (x,y components)
        Ftot = zeros(V, N)
        for (i, b) in enumerate(bars)
            Ftot[b[1]] += Fvec[i]
            Ftot[b[2]] -= Fvec[i]
        end
        for i in 1:length(pfix)
            Ftot[i] = zero(V)                                # Force  =  0 at fixed points
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
        d_int = sqrt(DELTAT) .* Ftot[d .< -GEPS]
        dterm = isempty(d_int) ? T(Inf) : sqrt(maximum(norm2, d_int))/h0

        if dterm >= dtermbest
            stallcount = stallcount + 1;
        end
        dtermbest = min(dtermbest, dterm);

        if dterm < DPTOL || stallcount >= MAXSTALLITERS
            break
        end
    end

    # Clean up and plot final mesh
    p, t, _ = fixmesh(p,t)
    if PLOT || PLOTLAST
        simpplot(p,t)
    end

    return p, t
end

function delaunay2(p::AbstractVector{Vec{2,T}}) where {T}
    a, b = VoronoiDelaunay.min_coord, VoronoiDelaunay.max_coord
    P, pmin, pmax = scaleto(p, a, b)

    points = IndexedPoint2D[IndexedPoint2D(P[i]..., i) for i in 1:length(P)]
    unique!(points)
    tess = DelaunayTessellation2D{IndexedPoint2D}(length(points))
    push!(tess, points)

    t = Vector{NTuple{3,Int}}()
    sizehint!(t, length(tess._trigs))
    i = 0
    for tt in tess
        i += 1
        push!(t, (getidx(GeometricalPredicates.geta(tt)),
                  getidx(GeometricalPredicates.getb(tt)),
                  getidx(GeometricalPredicates.getc(tt))) )
    end

    return t
end

#FIXMESH  Remove duplicated/unused nodes and fix element orientation.
#   [P,T,PIX]=FIXMESH(P,T)
#
#   Copyright (C) 2004-2012 Per-Olof Persson. See COPYRIGHT.TXT for details.
function fixmesh(
        p::AbstractVector{Vec{2,T}},
        t::AbstractVector{NTuple{3,Int}} = Vector{NTuple{3,Int}}(),
        ptol = 1024*eps(T)
    ) where {T}

    if isempty(p) || isempty(t)
        pix = 1:length(p)
        return p, t, pix
    end

    p_matrix(p) = reshape(reinterpret(T, p), (2, length(p))) |> transpose |> copy
    p_vector(p) = vec(reinterpret(Vec{2,T}, transpose(p))) |> copy
    t_matrix(t) = reshape(reinterpret(Int, t), (3, length(t))) |> transpose |> copy
    t_vector(t) = vec(reinterpret(NTuple{3,Int}, transpose(t))) |> copy

    p = p_matrix(p)
    snap = maximum(maximum(p, dims=1) - minimum(p, dims=1)) * ptol
    _, ix, jx = findunique(p_vector(round.(p./snap).*snap))
    p = p_vector(p)
    p = p[ix]

    if !isempty(t)
        t = t_matrix(t)
        t = reshape(jx[t], size(t))

        pix, ix1, jx1 = findunique(vec(t))
        t = reshape(jx1, size(t))
        p = p[pix]
        pix = ix[pix]

        t = t_vector(t)
        for (i,tt) in enumerate(t)
            d12 = p[tt[2]] - p[tt[1]]
            d13 = p[tt[3]] - p[tt[1]]
            v = (d12[1] * d13[2] - d12[2] * d13[1])/2 # simplex volume
            v < 0 && (t[i] = (tt[2], tt[1], tt[3])) # flip if volume is negative
        end
    end

    return p, t, pix
end

function findunique(A)
    C = unique(A)
    iA = findfirst.(isequal.(C), (A,))
    iC = findfirst.(isequal.(A), (C,))
    return C, iA, iC
end

# Type to keep track of index of initial point. From hack at issue:
#   https://github.com/JuliaGeometry/VoronoiDelaunay.jl/issues/6
struct IndexedPoint2D <: AbstractPoint2D
    _x::Float64
    _y::Float64
    _idx::Int64
    IndexedPoint2D(x, y, idx) = new(x, y, idx)
    IndexedPoint2D(x, y) = new(x, y, 0)
end
GeometricalPredicates.getx(p::IndexedPoint2D) = p._x
GeometricalPredicates.gety(p::IndexedPoint2D) = p._y
getidx(p::IndexedPoint2D) = p._idx

# Simple sorting of 2-tuples
sorttuple(t::Tuple{Int64,Int64}) = t[1] > t[2] ? (t[2], t[1]) : t

# Plotting
function simpplot(
        p::AbstractVector{Vec{2,T}},
        t::AbstractVector{NTuple{3,Int}}
    ) where {T}

    pp = reshape(reinterpret(T, p), (2, length(p))) |> transpose |> Matrix{Float64}
    tt = reshape(reinterpret(Int, t), (3, length(t))) |> transpose |> Matrix{Float64}

    mxcall(:simpplot, 0, pp, tt)
end
