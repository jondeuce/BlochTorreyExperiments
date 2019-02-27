# ---------------------------------------------------------------------------- #
# DistMesh helper functions
# ---------------------------------------------------------------------------- #

huniform(x::Vec{dim,T}) where {dim,T} = one(T)
norm2(x::Vec) = x⋅x

@inline function triangle_area(p1::Vec{2}, p2::Vec{2}, p3::Vec{2})
    d12, d13 = p2 - p1, p3 - p1
    return (d12[1] * d13[2] - d12[2] * d13[1])/2
end
@inline triangle_area(t::NTuple{3,Int}, p::AbstractArray{V}) where {V<:Vec{2}} = triangle_area(p[t[1]], p[t[2]], p[t[3]])

@inline triangle_quality(a, b, c) = (b+c-a)*(c+a-b)*(a+b-c)/(a*b*c)
@inline triangle_quality(p1::Vec{2}, p2::Vec{2}, p3::Vec{2}) = triangle_quality(norm(p2-p1), norm(p3-p2), norm(p1-p3))
@inline triangle_quality(t::NTuple{3,Int}, p::AbstractArray{V}) where {V<:Vec{2}} = triangle_quality(p[t[1]], p[t[2]], p[t[3]])

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
function hgeom(
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

function sortededges!(
        e::AbstractVector{NTuple{2,Int}},
        p::AbstractVector{Vec{2,T}},
        t::AbstractVector{NTuple{3,Int}}
    ) where {T}
    @assert length(e) == 3*length(t)
    @inbounds for (i,tt) in enumerate(t)
        a, b, c = sorttuple(tt)
        e[3i-2] = (a, b)
        e[3i-1] = (b, c)
        e[3i  ] = (c, a)
    end
    return e
end
function sortededges(p::AbstractVector{Vec{2,T}}, t::AbstractVector{NTuple{3,Int}}) where {T}
    return sortededges!(Vector{NTuple{2,Int}}(undef, 3*length(t)), p, t)
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
