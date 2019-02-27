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
