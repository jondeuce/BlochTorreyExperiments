# ============================================================================ #
# Generic geometry utilities for use within the JuAFEM.jl/Tensors.jl framework
# ============================================================================ #
import Base: maximum, minimum, rand, convert

# ---------------------------------------------------------------------------- #
# Misc. utilities for Vec type from Tensors.jl
# ---------------------------------------------------------------------------- #
norm2(x::Vec) = dot(x,x)
convert(::Type{Vec{dim,T1}}, x::SVector{dim,T2}) where {dim,T1,T2} = Vec{dim,promote_type(T1,T2)}(Tuple(x))
convert(::Type{SVector{dim,T1}}, x::Vec{dim,T2}) where {dim,T1,T2} = SVector{dim,promote_type(T1,T2)}(Tuple(x))

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
function convert(::Type{Circle{dim,T1}}, c::Circle{dim,T2}) where {dim,T1,T2}
    T = promote_type(T1,T2)
    return Circle{dim,T}(Vec{dim,T}(origin(c)), T(radius(c)))
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

# Scale circle by factor `α` relative to it's origin
scale_shape(c::Circle, α::Number) = Circle(origin(c), α * radius(c))

# compute maximal square inscribed inside circle
function inscribed_square(c::Circle{dim,T}) where {dim,T}
    half_widths = radii(c)/T(√dim)
    return Rectangle{dim,T}(origin(c) - half_widths, origin(c) + half_widths)
end

# Random circles
rand(::Type{Circle{dim,T}}) where {dim,T} = Circle(2rand(Vec{dim,T})-ones(Vec{dim,T}), rand(T))
rand(::Type{Circle{dim,T}}, N::Int) where {dim,T} = [rand(Circle{dim,T}) for i in 1:N]

# Signed edge distance
@inline function signed_edge_distance(c1::Circle, c2::Circle)
    dx = origin(c1) - origin(c2)
    return norm(dx) - radius(c1) - radius(c2) # zero is when circles are tangent, not overlapping
end
@inline function signed_edge_distance(o1::Vec, r1, o2::Vec, r2)
    return norm(o1 - o2) - r1 - r2 # zero is when circles are tangent, not overlapping
end

# check if c1 is in c2: r1 < r2 and ||o1 - o2|| < r2 - r1
@inline function is_inside(c1::Circle, c2::Circle, lt = ≤)
    o1, o2, r1, r2 = origin(c1), origin(c2), radius(c1), radius(c2)
    return lt(r1, r2) && lt(norm2(o1 - o2), (r2-r1)^2)
end

@inline function is_inside(x::Vec{dim}, c::Circle{dim}, lt = ≤) where {dim}
    return lt(norm2(x - origin(c)), radius(c)^2)
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
        circle = bounding_circle(circle, circles[i])
    end

    return circle
end

function bounding_circle(c1::Circle, c2::Circle)

    # Check if one circle already contains the other
    is_inside(c1, c2) && return c2
    is_inside(c2, c1) && return c1

    # If not, compute bounding circle for the pair
    T = promote_type(floattype(c1), floattype(c2))
    o1, o2, r1, r2 = origin(c1), origin(c2), radius(c1), radius(c2)

    # Check if circle origins are overlapping
    r12 = norm(o1-o2)
    r12 ≈ zero(T) && return Circle(o1, max(r1,r2)) # arbitarily take o1 for speed. could take average?

    # Otherwise, compute the general case
    rad = T(0.5)*(r1 + r2 + r12)

    # This version seems more numerically unstable...
    α1 = (r12 + r1 - r2)/(2r12)
    α2 = (r12 - r1 + r2)/(2r12)
    center = α1*o1 + α2*o2

    # Compute center
    # center = T(0.5)*(o1+o2) + ((r1-r2)/r12)*(o1-o2)

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

# Scale rectangle by factor `α` relative to it's origin
function scale_shape(r::Rectangle, α::Number)
    o = origin(r)
    new_mins = α * (minimum(r) - o) + o
    new_maxs = α * (maximum(r) - o) + o
    return Rectangle(new_mins, new_maxs)
end

# Bounding box of vector of circles
function bounding_box(circles::Vector{Circle{dim,T}}) where {dim,T}
    min_x = minimum(c->xmin(c), circles)
    min_y = minimum(c->ymin(c), circles)
    max_x = maximum(c->xmax(c), circles)
    max_y = maximum(c->ymax(c), circles)
    return Rectangle((min_x, min_y), (max_x, max_y))
end

# Check if circle is inside rectangle
@inline function is_inside(c::Circle{dim}, r::Rectangle{dim}, lt = ≤) where {dim}
    cmin, cmax, rmin, rmax = minimum(c), maximum(c), minimum(r), maximum(r)
    @inbounds for i in 1:dim
        !(lt(rmin[i], cmin[i]) && lt(cmax[i], rmax[i])) && return false
    end
    return true
end

# Check if circle is outside rectangle
@inline function is_outside(c::Circle{dim}, r::Rectangle{dim}, lt = <) where {dim}
    cmin, cmax, rmin, rmax = minimum(c), maximum(c), minimum(r), maximum(r)
    @inbounds for i in 1:dim
        (lt(cmax[i], rmin[i]) || lt(rmax[i], cmin[i])) && return true
    end
    return false
end

@inline function intersect_area(c::Circle{2}, X0, Y0, X1, Y1)
    # Integration of the intersection of area between the circle `c` and the
    # rectangular region defined by X0 ≤ x ≤ X1, Y0 ≤ y ≤ Y1. It is assumed that
    # X0 ≤ X1 and Y0 ≤ Y1.
    #
    # This code is heavily influenced by the stackoverflow response by user
    # `the swine` at the following link:
    #   https://stackoverflow.com/questions/622287/area-of-intersection-between-circle-and-rectangle

    ZERO, ONE, EPS = zero(X0), one(X0), 5eps(typeof(X0))

    @inline safe_sqrt(x) = x <= EPS ? EPS : √x
    @inline area_section(x, h) = (x*safe_sqrt(1-x^2) + asin(x))/2 - x*h
    @inline area_above(x0, x1, h) = (x = safe_sqrt(1-h^2); area_section(clamp(x1,-x,x), h) - area_section(clamp(x0,-x,x), h))
    @inline area_region(x0, x1, y0, y1) = area_above(x0, x1, y0) - area_above(x0, x1, y1)

    # Get normalized positions
    r, o = radius(c), origin(c)
    x0, x1, y0, y1 = (X0-o[1])/r, (X1-o[1])/r, (Y0-o[2])/r, (Y1-o[2])/r

    x0, x1 = clamp(x0,-ONE,ONE), clamp(x1,-ONE,ONE)
    y0, y1 = clamp(y0,-ONE,ONE), clamp(y1,-ONE,ONE)

    # Formulas above only work assuming 0 ≤ y0 ≤ y1. If this is not the case,
    # the integral needs to be split up
    if y0 < ZERO
        if y1 < ZERO
            # both negative; simply flip about x-axis
            area = area_region(x0, x1, -y1, -y0)
        else
            # integrate from [y0,0] and [0,y1] separately
            area = area_region(x0, x1, ZERO, -y0) + area_region(x0, x1, ZERO, y1)
        end
    else
        area = area_region(x0, x1, y0, y1)
    end

    # Integration is over the unit circle, so scale result by r^2
    area *= r^2

    return area
end

@inline intersect_area(c::Circle{2}, Xmin::Vec{2}, Xmax::Vec{2}) = intersect_area(c, Xmin[1], Xmin[2], Xmax[1], Xmax[2])
@inline intersect_area(c::Circle{2}, r::Rectangle{2}) = intersect_area(c, xmin(r), ymin(r), xmax(r), ymax(r))

function intersect_area_test(c::Circle{2,T}) where {T}
    # allow for points to be inside or slightly outside of circle
    rect = scale_shape(bounding_box([c]), T(1.5))

    x0 = origin(rect)[1] - (xmax(rect)-xmin(rect))/2 * rand(T)
    x1 = origin(rect)[1] + (xmax(rect)-xmin(rect))/2 * rand(T)
    y0 = origin(rect)[2] - (ymax(rect)-ymin(rect))/2 * rand(T)
    y1 = origin(rect)[2] + (ymax(rect)-ymin(rect))/2 * rand(T)
    @assert (x0 <= x1 && y0 <= y1)

    R = radius(c)
    h = R - (min(y1,ymax(c)) - origin(c)[2])

    A_top_test = (intersect_area(c, -Inf,   y1,  x0, Inf) + # 1: top left
                  intersect_area(c,   x0,   y1,  x1, Inf) + # 2: top middle
                  intersect_area(c,   x1,   y1, Inf, Inf))  # 3: top right

    A_top = R^2 * acos((R-h)/R) - (R-h)*√(2*R*h-h^2)
    @assert isapprox(A_top, A_top_test; atol=1e-12)

    A_test = (intersect_area(c, -Inf,   y1,  x0, Inf) + # 1: top left
              intersect_area(c,   x0,   y1,  x1, Inf) + # 2: top middle
              intersect_area(c,   x1,   y1, Inf, Inf) + # 3: top right
              intersect_area(c, -Inf,   y0,  x0,  y1) + # 4: middle left
              intersect_area(c,   x0,   y0,  x1,  y1) + # 5: middle middle
              intersect_area(c,   x1,   y0, Inf,  y1) + # 6: middle right
              intersect_area(c, -Inf, -Inf,  x0,  y0) + # 7: bottom left
              intersect_area(c,   x0, -Inf,  x1,  y0) + # 8: bottom middle
              intersect_area(c,   x1, -Inf, Inf,  y0))  # 9: bottom right

    A_exact = pi*radius(c)^2
    @assert A_test ≈ A_exact

    return (A_test, A_exact)
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
