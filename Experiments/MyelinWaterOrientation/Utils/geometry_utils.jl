# ============================================================================ #
# Generic geometry utilities for use within the JuAFEM.jl/Tensors.jl framework
# ============================================================================ #
import Base: maximum, minimum, rand

# Circle based on Vec type from Tensors.jl (code based on GeometryTypes.jl)
struct Circle{dim,T}
    center::Vec{dim,T}
    r::T
end
Circle(center::NTuple{dim,T}, r::T) where {dim,T} = Circle(Vec{dim,T}(center), r)
radius(c::Circle) = c.r
origin(c::Circle) = c.center
radii(c::Circle{dim,T}) where {dim,T} = fill(radius(c), Vec{dim,T})
widths(c::Circle{dim,T}) where {dim,T} = fill(2radius(c), Vec{dim,T})
minimum(c::Circle{dim,T}) where {dim,T} = origin(c) - radii(c)
maximum(c::Circle{dim,T}) where {dim,T} = origin(c) + radii(c)

# Rectangle based on Vec type from Tensors.jl (code based on GeometryTypes.jl)
struct Rectangle{dim,T}
    mins::Vec{dim,T}
    maxs::Vec{dim,T}
end
Rectangle(mins::NTuple{dim,T}, maxs::Vec{dim,T}) where {dim,T} = Rectangle(Vec{dim,T}(mins), maxs)
Rectangle(mins::Vec{dim,T}, maxs::NTuple{dim,T}) where {dim,T} = Rectangle(mins, Vec{dim,T}(maxs))
Rectangle(mins::NTuple{dim,T}, maxs::NTuple{dim,T}) where {dim,T} = Rectangle(Vec{dim,T}(mins), Vec{dim,T}(maxs))

maximum(r::Rectangle) = r.maxs
minimum(r::Rectangle) = r.mins
origin(r::Rectangle) = 0.5(minimum(r) + maximum(r))
widths(r::Rectangle) = maximum(r) - minimum(r)

@inline xmin(r::Rectangle{2}) = minimum(r)[1]
@inline ymin(r::Rectangle{2}) = minimum(r)[2]
@inline xmax(r::Rectangle{2}) = maximum(r)[1]
@inline ymax(r::Rectangle{2}) = maximum(r)[2]

# Random circles and rectangles
rand(::Type{Circle{dim,T}}) where {dim,T} = Circle(2rand(Vec{dim,T})-ones(Vec{dim,T}), rand(T))
rand(::Type{Rectangle{dim,T}}) where {dim,T} = Circle(-rand(Vec{dim,T}), rand(Vec{dim,T}))

# Compute the `skew product` between two 2-dimensional vectors. This is the same
# as computing the third component of the cross product if the vectors `a` and
# `b` were extended to three dimensions
@inline function skewprod(a::Vec{2,T}, b::Vec{2,T}) where T
    @inbounds v = (a × b)[3] # believe it or not, this emits the same llvm code...
    #@inbounds v = a[1]*b[2] - b[1]*a[2]
    return v
end

# Denote the skewproduct using `\boxtimes`
const ⊠ = skewprod

nothing
