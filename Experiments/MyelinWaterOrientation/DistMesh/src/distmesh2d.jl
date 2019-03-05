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
        pfix::AbstractVector{V} = V[], # fixed points
        pinit::AbstractVector{V} = init_points(bbox, h0), # inital distribution of points (triangular grid by default)
        ∇fd = x -> Tensors.gradient(fd, x); # Gradient of distance function `fd`
        PLOT::Bool = false, # plot all triangulations during evolution
        PLOTLAST::Bool = false, # plot resulting triangulation
        DETERMINISTIC::Bool = false, # deterministically seed the pseudo-random rejection method
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

    # Function for restarting if h0 is large enough that all points get removed
    function restart()
        if h0/2 < RESTARTEDGETHRESH
            @warn "h0 too small to allow a restart; returning empty grid."
            return pfix, NTuple{3,Int}[]
        else
            @warn "No points remaining! Shrinking h0 -> h0/2 and retrying..."
            return distmesh2d(fd, fh, h0/2, bbox, pfix;
                PLOT = PLOT,
                PLOTLAST = PLOTLAST,
                DETERMINISTIC = DETERMINISTIC,
                MAXSTALLITERS = MAXSTALLITERS,
                RESTARTEDGETHRESH = RESTARTEDGETHRESH,
                DENSITYCTRLFREQ = DENSITYCTRLFREQ,
                DENSITYRELTHRESH = DENSITYRELTHRESH,
                DPTOL = DPTOL,
                TTOL = TTOL,
                FSCALE = FSCALE,
                DELTAT = DELTAT,
                GEPS = GEPS/2, # scales with h0
                DEPS = DEPS/2 # scales with h0
            )
        end
    end

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
    pfix = threshunique(pfix; rtol = √eps(T), atol = √eps(T))
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
            bars = unique!(sort!(bars))                            # Bars as node pairs
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
            L[i] = norm(barvec[i])                            # L = Bar lengths
            hbars[i] = fh((p1+p2)/2)
        end
        L0 .= hbars .* (FSCALE * norm(L)/norm(hbars))         # L0 = Desired lengths

        # Density control - remove points that are too close
        if mod(count, DENSITYCTRLFREQ) == 0
            b = DENSITYRELTHRESH .* L .< L0
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
