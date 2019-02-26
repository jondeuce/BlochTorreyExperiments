# ---------------------------------------------------------------------------- #
# DistMesh helper functions
# ---------------------------------------------------------------------------- #

huniform(x::Vec{dim,T}) where {dim,T} = one(T)
norm2(x::Vec) = x⋅x

function to_vec(p::AbstractMatrix{T}) where {T}
    N = size(p, 2)
    P = reinterpret(Vec{N, T}, transpose(p)) |> vec |> copy
    return P
end

function to_tuple(p::AbstractMatrix{T}) where {T}
    N = size(p, 2)
    P = reinterpret(NTuple{N, T}, transpose(p)) |> vec |> copy
    return P
end

function to_mat(P::AbstractVector{NTuple{N,T}}) where {N,T}
    M = length(P)
    p = reshape(reinterpret(T, P), (N, M)) |> transpose |> copy
    return p
end
to_mat(P::AbstractVector{Vec{N,T}}) where {N,T} = to_mat(reinterpret(NTuple{N,T}, P))

function init_points(bbox::Matrix{T}, h0::T) where {T}
    # xrange, yrange = bbox[1,1]:h0:bbox[2,1], bbox[1,2]:h0*sqrt(T(3))/2:bbox[2,2]
    Nx = ceil(Int, (bbox[2,1]-bbox[1,1])/h0)
    Ny = ceil(Int, (bbox[2,2]-bbox[1,2])/(h0*sqrt(T(3))/2))
    xrange, yrange = range(bbox[1,1], bbox[2,1], length=Nx), range(bbox[1,2], bbox[2,2], length=Ny)
    p = zeros(Vec{2,T}, Ny, Nx)
    @inbounds for (i,y) in enumerate(yrange), (j,x) in enumerate(xrange)
        iseven(i) && (x += h0/2)            # Shift even rows
        p[i,j] = Vec{2,T}((x,y))            # Add to list of node coordinates
    end
    return vec(p)
end

#HGEOM - Mesh size function based on medial axis distance
#--------------------------------------------------------------------
# Inputs:
#    p     : points coordinates
#    fd    : signed distance function
#    pmax  : approximate medial axis points
#
# output:
#    fh    : mesh size function of points p
#--------------------------------------------------------------------
#  (c) 2011, Koko J., ISIMA, koko@isima.fr
#--------------------------------------------------------------------
function make_hgeom(
        fd, # distance function
        h0::T, # nominal edge length
        bbox::Matrix{T}, # bounding box (2x2 matrix [xmin ymin; xmax ymax])
        pfix::AbstractVector{V} = V[], # fixed points
        alpha::T = T(1.0) # Relative edge lengths constant; alpha = 0.25 -> rel. length = 5X; 0.5 -> 3X; 1.0 -> 2X
    ) where {T, V<:Vec{2,T}}

    # Make and return hgeom
    pmax, p = medial_axis(fd, h0, bbox, pfix)
    max_fh1 = maximum(x->abs(fd(x)), p)
    max_fh2 = sqrt(maximum(x->minimum(y->norm2(x-y), pmax), p))

    function hgeom(x)
        fh1 = fd(x)/max_fh1 # Normalized signed distance fuction
        fh2 = sqrt(minimum(y->norm2(x-y), pmax))/max_fh2 # Normalized medial axis distance
        fh = alpha + min(abs(fh1), one(T)) + min(fh2, one(T)) # Size function (capped at 1 since max_fh1/max_fh2 are approximate)
        return fh
    end

    return hgeom
end

function medial_axis(
        fd, # distance function
        h0::T, # nominal edge length
        bbox::Matrix{T}, # bounding box (2x2 matrix [xmin ymin; xmax ymax])
        pfix::AbstractVector{V} = V[] # fixed points
    ) where {T, V<:Vec{2,T}}

    # Numerical gradient (NOTE: this is for finding discontinuities in the gradient; can't use auto-diff!)
    e1, e2 = basevec(V,1), basevec(V,2)
    @inline dfd_dx(x::V,h) = (fd(x + h*e1) - fd(x - h*e1))/(2h)
    @inline dfd_dy(x::V,h) = (fd(x + h*e2) - fd(x - h*e2))/(2h)
    @inline    ∇fd(x::V,h) = V((dfd_dx(x,h), dfd_dy(x,h)))

    # Compute the approximate medial axis
    p, pmax = Vector{V}(), Vector{V}()
    h1 = h0
    while isempty(pmax) # shouldn't ever take more than one iteration
        h1 /= 2
        Nx, Ny = ceil(Int, (bbox[2,1]-bbox[1,1])/h1), ceil(Int, (bbox[2,2]-bbox[1,2])/h1)
        p = V[V((xi,yi)) for xi in range(bbox[1,1], bbox[2,1], length=Nx) for yi in range(bbox[1,2], bbox[2,2], length=Ny)]
        pmax = filter(x -> fd(x) <= 0 && norm(∇fd(x,h1/2)) < 0.99, p)
    end
    pmax = unique!(sort!([pfix; pmax]; by = first))

    return pmax, p
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

function assign_triangles!(t, tess)
    resize!(t, length(tess))
    @inbounds for (i,tt) in enumerate(tess)
        t[i] = (getidx(geta(tt)), getidx(getb(tt)), getidx(getc(tt)))
    end
    return t
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
# Simple function for scaling vector of Vec's to range [a,b]
# ---------------------------------------------------------------------------- #

function scaleto!(p::AbstractVector{Vec{2,Float64}}, a, b)
    N = length(p)
    P = reinterpret(Float64, p) # must be Float64
    Pmin, Pmax = minimum(P), maximum(P)
    P .= ((b - a)/(Pmax - Pmin)) .* (P .- Pmin) .+ a
    clamp!(P, a, b) # to be safe
    return p, Pmin, Pmax
end
scaleto(p::AbstractVector{Vec{2,Float64}}, a, b) = scaleto!(copy(p), a, b)

function scaleto(p::AbstractVector{Vec{2,T}}, a, b) where {T}
    x, Xmin, Xmax = scaleto(Vector{Vec{2,Float64}}(p), Float64(a), Float64(b))
    return Vector{Vec{2,T}}(x), T(Xmin), T(Xmax)
end

# ---------------------------------------------------------------------------- #
# Unique, sorting, etc.
# ---------------------------------------------------------------------------- #

function threshunique(
        x::AbstractVector{T};
        rtol = √eps(T),
        atol = eps(T),
        norm = LinearAlgebra.norm
    ) where T

    uniqueset = Vector{T}()
    sizehint!(uniqueset, length(x))
    ex = eachindex(x)
    # idxs = Vector{eltype(ex)}()
    for i in ex
        xi = x[i]
        norm_xi = norm(xi)
        isunique = true
        for xj in uniqueset
            norm_xj = norm(xj)
            if norm(xi - xj) < max(rtol*max(norm_xi, norm_xj), atol)
                isunique = false
                break
            end
        end
        isunique && push!(uniqueset, xi) # push!(idxs, i)
    end

    return uniqueset
end

function findunique(A)
    C = unique(A)
    iA = findfirst.(isequal.(C), (A,))
    iC = findfirst.(isequal.(A), (C,))
    return C, iA, iC
end

# Simple sorting of 2-tuples
sorttuple(t::NTuple{2}) = t[1] > t[2] ? (t[2], t[1]) : t

# Simple sorting of 3-tuples
function sorttuple(t::NTuple{3})
    a, b, c = t
    if a > b
        a, b = b, a
    end
    if b > c
        b, c = c, b
        if a > b
            a, b = b, a
        end
    end
    return (a, b, c)
end

# edges of triangle vector
function getedges(t::AbstractVector{NTuple{3,Int}})
    bars = Vector{NTuple{2,Int}}(undef, 3*length(t))
    @inbounds for (i,tt) in enumerate(t)
        a, b, c = tt
        bars[3i-2] = (a, b)
        bars[3i-1] = (b, c)
        bars[3i  ] = (c, a)
    end
    return bars
end

# ---------------------------------------------------------------------------- #
# AbstractPoint2D subtype which keeps track of index of initial point. From:
#   https://github.com/JuliaGeometry/VoronoiDelaunay.jl/issues/6
# ---------------------------------------------------------------------------- #

struct IndexedPoint2D <: AbstractPoint2D
    _x::Float64
    _y::Float64
    _idx::Int64
    IndexedPoint2D(x, y, idx) = new(x, y, idx)
    IndexedPoint2D(x, y) = new(x, y, 0)
end
VoronoiDelaunay.getx(p::IndexedPoint2D) = p._x
VoronoiDelaunay.gety(p::IndexedPoint2D) = p._y
getidx(p::IndexedPoint2D) = p._idx

# ---------------------------------------------------------------------------- #
# FIXMESH  Remove duplicated/unused nodes and fix element orientation.
#   P, T, PIX = FIXMESH(P,T)
# ---------------------------------------------------------------------------- #

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
        @inbounds for (i,tt) in enumerate(t)
            d12 = p[tt[2]] - p[tt[1]]
            d13 = p[tt[3]] - p[tt[1]]
            v = (d12[1] * d13[2] - d12[2] * d13[1])/2 # simplex volume
            v < 0 && (t[i] = (tt[2], tt[1], tt[3])) # flip if volume is negative
        end
    end

    return p, t, pix
end

# ---------------------------------------------------------------------------- #
# BOUNDEDGES Find boundary edges from triangular mesh
#   E = BOUNDEDGES(P,T)
# ---------------------------------------------------------------------------- #

function boundedges(
        p::AbstractVector{Vec{2,T}},
        t::AbstractVector{NTuple{3,Int}}
    ) where {T}

    # Check for empty inputs
    (isempty(p) || isempty(t)) && return NTuple{2,Int}[]

    # Form all edges, non-duplicates are boundary edges
    p, t = to_mat(p), to_mat(t)
    edges = [t[:,[1,2]]; t[:,[2,3]]; t[:,[3,1]]] #NOTE changed from above, but shouldn't matter
    node3 = [t[:,3]; t[:,1]; t[:,2]]
    edges = sort(edges; dims = 2) # for finding unique edges, make sure they're ordered the same
    _, ix, jx = findunique(to_tuple(edges))
    
    # Histogram edges are 1:max(jx)+1, closed on the left i.e. [a,b). First bin is [1,2), n'th bin is [n,n+1)
    h = fit(Histogram, jx, weights(ones(Int, size(jx))), 1:length(ix)+1; closed = :left)
    ix_unique = ix[h.weights .== 1]
    edges = edges[ix_unique, :]
    node3 = node3[ix_unique]
    
    # Orientation
    v12 = p[edges[:,2],:] - p[edges[:,1],:]
    v13 = p[node3,:] - p[edges[:,1],:]
    b_flip = v12[:,1] .* v13[:,2] .- v12[:,2] .* v13[:,1] .< zero(T) #NOTE this is different in DistMesh; they check for > 0 due to clockwise ordering
    edges[b_flip, [1,2]] = edges[b_flip, [2,1]]
    edges = sort!(to_tuple(edges); by = first)

    return edges
end

# ---------------------------------------------------------------------------- #
# simpplot
# ---------------------------------------------------------------------------- #

# Light grid wrapper structure for simpplot. Non-AbstractPlot arguments to simpplot
# are forwarded to the SimpPlotGrid constructor.
struct SimpPlotGrid{T}
    p::Vector{Vec{2,T}}
    t::Vector{NTuple{3,Int}}
    SimpPlotGrid(g::SimpPlotGrid) = deepcopy(g)
    SimpPlotGrid(p::AbstractVector{Vec{2,T}}, t::AbstractVector{NTuple{3,Int}}) where {T} = new{T}(p,t)
    function SimpPlotGrid(p::AbstractMatrix{T}, t::AbstractMatrix{Int}) where {T}
        size(p,2) == 2 || throw(DimensionMismatch("p must be a matrix of two columns, i.e. representing 2-vectors"))
        size(t,2) == 3 || throw(DimensionMismatch("t must be a matrix of three columns, i.e. representing triangles"))
        new{T}(to_vec(p), to_tuple(t))
    end
end

@userplot SimpPlot

@recipe function f(h::SimpPlot)
    grid = parse_simpplot_args(h.args...)
    if !(typeof(grid) <: SimpPlotGrid)
        error("Arguments not properly parsed; expected two AbstractArray's, got: $unwrapped")
    end
    p, t = grid.p, grid.t
    for tt in t
        p1, p2, p3 = p[tt[1]], p[tt[2]], p[tt[3]]
        x = [p1[1], p2[1], p3[1]]
        y = [p1[2], p2[2], p3[2]]
        @series begin
            seriestype   := :shape
            aspect_ratio := :equal
            legend       := false
            x, y
        end
    end
    # Attributes
    aspect_ratio := :equal
    legend       := false
end

parse_simpplot_args(g::SimpPlotGrid) = g
parse_simpplot_args(args...) = SimpPlotGrid(args...) # construct SimpPlotGrid
parse_simpplot_args(plts::P, args...) where {P <: Union{<:AbstractPlot, <:AbstractArray{<:AbstractPlot}}} = parse_simpplot_args(args...) # skip plot objects
