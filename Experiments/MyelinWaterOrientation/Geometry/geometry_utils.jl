# ============================================================================ #
# Generic geometry utilities for use within the JuAFEM.jl/Tensors.jl framework
# ============================================================================ #
import Base: maximum, minimum, rand

# ---------------------------------------------------------------------------- #
# Circle based on Vec type from Tensors.jl (code based on GeometryTypes.jl)
# ---------------------------------------------------------------------------- #
struct Circle{dim,T}
    center::Vec{dim,T}
    r::T
end
function Circle(center::Vec{dim,T1}, r::T2) where {dim,T1,T2}
    T = promote_type(T1, T2)
    return Circle(Vec{dim,T}(center), T(r))
end
Circle(::Type{T}, center::Vec{dim}, r::Number) where {dim,T} = Circle(Vec{dim,T}(center), T(r))
Circle(::Type{T}, center::NTuple{dim}, r::Number) where {dim,T} = Circle(Vec{dim,T}(center), T(r))

Circle(center::NTuple{dim,T}, r::T) where {dim,T} = Circle(Vec{dim,T}(center), r)
dimension(c::Circle) = dimension(typeof(c))
floattype(c::Circle) = floattype(typeof(c))
dimension(::Type{Circle{dim,T}}) where {dim,T} = dim
floattype(::Type{Circle{dim,T}}) where {dim,T} = T

radius(c::Circle) = c.r
origin(c::Circle) = c.center
radii(c::Circle{dim,T}) where {dim,T} = fill(radius(c), Vec{dim,T})
widths(c::Circle{dim,T}) where {dim,T} = fill(2radius(c), Vec{dim,T})
minimum(c::Circle{dim,T}) where {dim,T} = origin(c) - radii(c)
maximum(c::Circle{dim,T}) where {dim,T} = origin(c) + radii(c)

@inline xmin(c::Circle{2}) = minimum(c)[1]
@inline ymin(c::Circle{2}) = minimum(c)[2]
@inline xmax(c::Circle{2}) = maximum(c)[1]
@inline ymax(c::Circle{2}) = maximum(c)[2]

# Random circles
rand(::Type{Circle{dim,T}}) where {dim,T} = Circle(2rand(Vec{dim,T})-ones(Vec{dim,T}), rand(T))
rand(::Type{Circle{dim,T}}, N::Int) where {dim,T} = [rand(Circle{dim,T}) for i in 1:N]

# Signed edge distance
@inline function signed_edge_distance(c1::Circle, c2::Circle)
    dx = origin(c1) - origin(c2)
    return norm(dx) - radius(c1) - radius(c2) # zero is when circles are tangent, not overlapping
end
@inline function signed_edge_distance(o1::Vec, r1, o2::Vec, r2)
    dx = o1 - o2
    return norm(dx) - r1 - r2 # zero is when circles are tangent, not overlapping
end

# check if c1 is in c2: distance between origins is less than radius(c2) and
# radius(c1) <= radius(c2)
@inline function is_inside(c1::Circle, c2::Circle, lt = ≤)
    o1, o2, r1, r2 = origin(c1), origin(c2), radius(c1), radius(c2)
    return lt(r1, r2) && lt(norm(o1 - o2) + r1, r2)
end

# check if c1 and c2 are overlapping
@inline function is_overlapping(c1::Circle, c2::Circle, lt = ≤)
    dx = origin(c1) - origin(c2)
    d_min = radius(c1) + radius(c2)
    return lt(dx⋅dx, d_min^2)
end

# bounding circle of a collection of circles. not optimal, but very simple
function bounding_circle(circles::Vector{C}) where {C<:Circle}

    @assert !isempty(circles)
    N = length(circles)
    T = floattype(C)

    N == 1 && return Circle(origin(circles[1]), zero(T))

    circle = circles[1]
    for i in 2:N
        # next circle to consider
        c = circles[i]

        # don't need to do anything if c ∈ circle already
        if !is_inside(c, circle)
            if is_inside(circle, c)
                # if circle ∈ c, set c to be the new circle
                circle = c
            else
                # compute bounding circle
                circle = bounding_circle(circle, c)
            end
        end
    end

    return circle
end
function bounding_circle(c1::Circle, c2::Circle)
    T = promote_type(floattype(c1), floattype(c2))

    o1, o2, r1, r2 = origin(c1), origin(c2), radius(c1), radius(c2)

    r12 = norm(o1-o2)
    if r12 ≈ zero(T)
        return Circle(o1, max(r1,r2))
    end

    α1 = (r12 + r1 - r2)/(2r12)
    α2 = (r12 - r1 + r2)/(2r12)

    rad = T(0.5)*(r1 + r2 + r12)
    center = α1*o1 + α2*o2

    return Circle(center, rad)
end

function crude_bounding_circle(circles::Vector{<:Circle})
    center = mean(c->origin(c), circles)
    rad = maximum(c->norm(origin(c)-center)+radius(c), circles)
    return Circle(center, rad)
end

# ---------------------------------------------------------------------------- #
# Rectangle based on Vec type from Tensors.jl (code based on GeometryTypes.jl)
# ---------------------------------------------------------------------------- #
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

# Random rectangles
rand(::Type{Rectangle{dim,T}}) where {dim,T} = Circle(-rand(Vec{dim,T}), rand(Vec{dim,T}))
rand(::Type{Rectangle{dim,T}}, N::Int) where {dim,T} = [rand(Rectangle{dim,T}) for i in 1:N]

# Bounding box of vector of circles
function bounding_box(circles::Vector{Circle{dim,T}}) where {dim,T}
    min_x = minimum(c->xmin(c), circles)
    min_y = minimum(c->ymin(c), circles)
    max_x = maximum(c->xmax(c), circles)
    max_y = maximum(c->ymax(c), circles)
    return Rectangle((min_x, min_y), (max_x, max_y))
end

# ---------------------------------------------------------------------------- #
# Compute the `skew product` between two 2-dimensional vectors. This is the same
# as computing the third component of the cross product if the vectors `a` and
# `b` were extended to three dimensions
# ---------------------------------------------------------------------------- #
@inline function skewprod(a::Vec{2,T}, b::Vec{2,T}) where T
    @inbounds v = (a × b)[3] # believe it or not, this emits the same llvm code...
    #@inbounds v = a[1]*b[2] - b[1]*a[2]
    return v
end
const ⊠ = skewprod # Denote the skewproduct using `\boxtimes`

nothing
