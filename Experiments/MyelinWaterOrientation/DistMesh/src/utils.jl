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

    # Form all edges, non-duplicates are boundary edges
    p, t = to_mat(p), to_mat(t)
    edges = [t[:,[1,2]]; t[:,[1,3]]; t[:,[2,3]]]
    node3 = [t[:,3]; t[:,2]; t[:,1]]
    edges = sort(edges; dims = 2)
    _, ix, jx = findunique(to_tuple(edges))

    h = fit(Histogram, jx, 1:maximum(jx); closed = :left)
    qx = findall(w -> w==1, h.weights)
    e = edges[ix[qx], :]
    node3 = node3[ix[qx]]

    # Orientation
    v1 = p[e[:,2],:] - p[e[:,1],:]
    v2 = p[node3,:] - p[e[:,1],:]
    ix = findall(v1[:,1] .* v2[:,2] .- v1[:,2] .* v2[:,1] .> zero(T));
    e[ix, [1,2]] = e[ix, [2,1]]
    e = sort!(to_tuple(e); by = first)

    return e
end

# ---------------------------------------------------------------------------- #
# simpplot
# ---------------------------------------------------------------------------- #

function simpplot(p::AbstractMatrix, t::AbstractMatrix;
        newfigure = false,
        hold = false,
        xlim = nothing,
        ylim = nothing,
        axis = nothing
    )
    @assert size(p,2) == 2 && size(t,2) == 3

    newfigure && mxcall(:figure, 0)
    hold && mxcall(:hold, 0, "on")
    mxcall(:simpplot, 0, Matrix{Float64}(p), Matrix{Float64}(t))

    !(xlim == nothing) && mxcall(:xlim, 0, xlim)
    !(ylim == nothing) && mxcall(:ylim, 0, ylim)
    !(axis == nothing) && mxcall(:axis, 0, axis)

    return nothing
end

function simpplot(
        p::AbstractVector{V},
        t::AbstractVector{NTuple{3,Int}};
        kwargs...
    ) where {V<:Vec{2}}
    simpplot(to_mat(p), to_mat(t); kwargs...)
end
