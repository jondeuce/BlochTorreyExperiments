# ---------------------------------------------------------------------------- #
# Delaunay triangulation
# ---------------------------------------------------------------------------- #

function delaunay2!(
        t::Vector{NTuple{3,Int}},
        p::AbstractVector{Vec{2,T}}
    ) where {T}

    p = delaunay_scale(p) # scale to [min_coord, max_coord]
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

function delaunay_scale!(P::AbstractVector{Float64})
    Pmin, Pmax = minimum(P), maximum(P)
    a, b = VoronoiDelaunay.min_coord + sqrt(eps(Float64)), VoronoiDelaunay.max_coord - sqrt(eps(Float64))
    P .= a .+ ((b - a)/(Pmax - Pmin)) .* (P .- Pmin)
    P = clamp!(P, a, b) # to be safe
    return P
end
function delaunay_scale!(P::AbstractVector{Vec{2,Float64}})
    Pv = reinterpret(Float64, P)
    @views delaunay_scale!(Pv[1:2:end])
    @views delaunay_scale!(Pv[2:2:end])
    return P
end
delaunay_scale(P::AbstractVector{Float64}) = delaunay_scale!(copy(P))
delaunay_scale(P::AbstractVector{Vec{2,Float64}}) = delaunay_scale!(copy(P))
delaunay_scale(P::AbstractVector{T}) where {T} = delaunay_scale(convert(Vector{Float64}, P))
delaunay_scale(P::AbstractVector{V}) where {V<:Vec{2}} = delaunay_scale(convert(Vector{Vec{2,Float64}}, P))

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
