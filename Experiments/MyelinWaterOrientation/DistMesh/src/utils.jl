# ---------------------------------------------------------------------------- #
# DistMesh helper functions
# ---------------------------------------------------------------------------- #

huniform(x::Vec) = one(eltype(x))
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
        bars[3i-1] = (a, c)
        bars[3i  ] = (b, c)
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
    # edges = [t[:,[1,2]]; t[:,[1,3]]; t[:,[2,3]]]
    # node3 = [t[:,3]; t[:,2]; t[:,1]]
    edges = [t[:,[1,2]]; t[:,[2,3]]; t[:,[3,1]]] #NOTE changed from above, but shouldn't matter
    node3 = [t[:,3]; t[:,1]; t[:,2]]
    edges = sort(edges; dims = 2) # for finding unique edges, make sure they're ordered the same
    _, ix, jx = findunique(to_tuple(edges))
    
    # Histogram edges are 1:max(jx)+1, closed on the left i.e. [a,b). First bin is [1,2), n'th bin is [n,n+1)
    h = fit(Histogram, jx, weights(ones(Int, size(jx))), 1:length(ix)+1; closed = :left)
    ix_unique = ix[h.weights .== 1]
    edges = edges[ix_unique, :]
    node3 = node3[ix_unique]
    # qx = findall(w->w==1, h.weights)
    # edges = edges[ix[qx], :]
    # node3 = node3[ix[qx]]
    
    # Orientation
    v12 = p[edges[:,2],:] - p[edges[:,1],:]
    v13 = p[node3,:] - p[edges[:,1],:]
    b_flip = v12[:,1] .* v13[:,2] .- v12[:,2] .* v13[:,1] .< zero(T) #NOTE this is different in DistMesh (they check for > 0) - I believe it's an error in their code
    edges[b_flip, [1,2]] = edges[b_flip, [2,1]]
    edges = sort!(to_tuple(edges); by = first)

    return edges
end

# ---------------------------------------------------------------------------- #
# simpplot
# ---------------------------------------------------------------------------- #

@userplot SimpPlot

@recipe function f(h::SimpPlot)
    unwrapped = unwrap_simpplot_args(h.args...)
    if !(length(unwrapped) == 2 && typeof(unwrapped[1]) <: AbstractArray && typeof(unwrapped[2]) <: AbstractArray)
        error("Arguments not properly parsed; expected two AbstractArray's, got: $unwrapped")
    end
    p, t = unwrapped
    for i in 1:size(t,1)
        x = [p[t[i,j],1] for j in 1:size(t,2)]
        y = [p[t[i,j],2] for j in 1:size(t,2)]
        @series begin
            seriestype := :shape
            x, y
        end
    end
end

# unwrap_simpplot_args(args...) = args # we don't want this fallback; need to make sure it's caught below
unwrap_simpplot_args(p::AbstractArray, t::AbstractArray) = p, t # this is the base case we are looking for
unwrap_simpplot_args(p::AbstractVector{V}, t::AbstractVector{NTuple{3,Int}}) where {V<:Vec{2}} = to_mat(p), to_mat(t)
unwrap_simpplot_args(plts::P, args...) where {P <: Union{<:AbstractPlot, <:AbstractArray{<:AbstractPlot}}} = unwrap_simpplot_args(args...) # skip plot objects
