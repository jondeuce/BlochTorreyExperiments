# ============================================================================ #
# Generic geometry utilities for use within the JuAFEM.jl/Tensors.jl framework
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# Misc. utilities for Vec type from Tensors.jl
# ---------------------------------------------------------------------------- #
Base.convert(::Type{Vec{dim,T1}}, x::SVector{dim,T2}) where {dim,T1,T2} = Vec{dim,promote_type(T1,T2)}(Tuple(x))
Base.convert(::Type{SVector{dim,T1}}, x::Vec{dim,T2}) where {dim,T1,T2} = SVector{dim,promote_type(T1,T2)}(Tuple(x))
@inline norm2(x::Vec) = dot(x,x)
@inline Base.Tuple(v::Vec) = Tensors.get_data(v)

# ---------------------------------------------------------------------------- #
# Circle based on Vec type from Tensors.jl (code based on GeometryTypes.jl)
# ---------------------------------------------------------------------------- #
struct Circle{dim,T}
    center::Vec{dim,T}
    r::T
end
Circle(::Type{T}, center::Vec{dim}, r::Number) where {dim,T} = Circle(Vec{dim,T}(center), T(r))
Circle(::Type{T}, center::NTuple{dim}, r::Number) where {dim,T} = Circle(Vec{dim,T}(center), T(r))

function Circle(center::Vec{dim,T1}, r::T2) where {dim,T1,T2}
    T = promote_type(T1, T2)
    return Circle(Vec{dim,T}(center), T(r))
end
function Base.convert(::Type{Circle{dim,T1}}, c::Circle{dim,T2}) where {dim,T1,T2}
    T = promote_type(T1, T2)
    return Circle{dim,T}(Vec{dim,T}(origin(c)), T(radius(c)))
end
function Base.isapprox(c1::Circle, c2::Circle; kwargs...)
    return isapprox(radius(c1), radius(c2); kwargs...) && isapprox(origin(c1), origin(c2); kwargs...)
end

Circle(center::NTuple{dim,T}, r::T) where {dim,T} = Circle(Vec{dim,T}(center), r)
dimension(c::Circle) = dimension(typeof(c))
floattype(c::Circle) = floattype(typeof(c))
dimension(::Type{Circle{dim,T}}) where {dim,T} = dim
floattype(::Type{Circle{dim,T}}) where {dim,T} = T

radius(c::Circle) = c.r
origin(c::Circle) = c.center
radii(c::Circle{dim,T}) where {dim,T} = fill(radius(c), Vec{dim,T})
widths(c::Circle{dim,T}) where {dim,T} = fill(2radius(c), Vec{dim,T})
Base.minimum(c::Circle{dim,T}) where {dim,T} = origin(c) - radii(c)
Base.maximum(c::Circle{dim,T}) where {dim,T} = origin(c) + radii(c)

@inline xmin(c::Circle{2}) = minimum(c)[1]
@inline ymin(c::Circle{2}) = minimum(c)[2]
@inline xmax(c::Circle{2}) = maximum(c)[1]
@inline ymax(c::Circle{2}) = maximum(c)[2]

@inline area(c::Circle{2}) = pi*radius(c)^2
@inline volume(c::Circle{3}) = 4/3*pi*radius(c)^3

# Scale circle by factor `α` relative to the point `P` (default to it's own origin)
scale_shape(c::Circle{dim}, P::Vec{dim}, α::Number) where {dim} = Circle(α * (origin(c) - P) + P, α * radius(c))
scale_shape(c::Circle, α::Number) = Circle(origin(c), α * radius(c))

# Translate circle by factor `α` relative to the point `P` (defaults to zero vector), or by fixed vector X
translate_shape(c::Circle{dim}, P::Vec{dim}, α::Number) where {dim} = Circle(α * (origin(c) - P) + P, radius(c))
translate_shape(c::Circle{dim}, α::Number) where {dim} = Circle(α * origin(c), radius(c))
translate_shape(c::Circle{dim}, X::Vec{dim}) where {dim} = Circle(origin(c) + X, radius(c))

# compute maximal square inscribed inside circle
function inscribed_square(c::Circle{dim,T}) where {dim,T}
    half_widths = radii(c)/T(√dim)
    return Rectangle{dim,T}(origin(c) - half_widths, origin(c) + half_widths)
end

# Random circles
Base.rand(::Type{Circle{dim,T}}) where {dim,T} = Circle{dim,T}(2*rand(Vec{dim,T})-ones(Vec{dim,T}), rand(T))
Base.rand(::Type{Circle{dim,T}}, N::Int) where {dim,T} = [rand(Circle{dim,T}) for i in 1:N]

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

# check if any circles in `cs` are overlapping
function is_any_overlapping(cs::Vector{C}, lt = ≤) where {C<:Circle}
    @inbounds for i in 1:length(cs)-1
        ci = cs[i]
        @inbounds for j in i+1:length(cs)
            is_overlapping(ci, cs[j], lt) && return true
        end
    end
    return false
end

# bounding circle of a collection of circles. not optimal, but very simple
function bounding_circle(circles::Vector{C}) where {C<:Circle}

    @assert !isempty(circles)
    @inbounds circle = circles[1]

    N = length(circles)
    N == 1 && return circle

    @inbounds for i in 2:N
        circle = bounding_circle(circle, circles[i])
    end

    return circle
end

function bounding_circle(c1::Circle, c2::Circle)

    # Check if one circle already contains the other
    is_inside(c1, c2) && return c2
    is_inside(c2, c1) && return c1

    # If not, check if the circle origins are overlapping
    T = promote_type(floattype(c1), floattype(c2))
    EPS = √eps(T)
    o1, o2, r1, r2 = origin(c1), origin(c2), radius(c1), radius(c2)

    r12 = norm(o1-o2)
    r12 ≤ EPS && return Circle(T(0.5)*(o1+o2), max(r1,r2)+EPS) # arbitarily take o1 for speed. could take average?

    # Otherwise, compute the general case
    rad = T(0.5)*(r1 + r2 + r12)

    # This version is more numerically unstable; lots of cancellation of large +/- numbers
    # α1 = (r12 + r1 - r2)/(2r12)
    # α2 = (r12 - r1 + r2)/(2r12)
    # center = α1*o1 + α2*o2

    # Compute center (note that r12 > EPS)
    center = T(0.5)*(o1+o2) + ((r1-r2)/r12)*(o1-o2)

    return Circle(center, rad)
end

function crude_bounding_circle(circles::Vector{<:Circle})
    center = mean(c->origin(c), circles)
    rad = maximum(c->norm(origin(c)-center)+radius(c), circles)
    return Circle(center, rad)
end

function crude_bounding_circle(origins::AbstractVector{T},
                               radii::AbstractVector{T}) where {T}
    @assert length(origins) >= 2length(radii)
    N = length(radii)
    x̄ = mean(view(origins,1:2:2N))
    ȳ = mean(view(origins,2:2:2N))
    center = Vec{2,T}((x̄, ȳ))
    rad = maximum(i -> sqrt((origins[2i-1]-x̄)^2 + (origins[2i]-ȳ)^2) + radii[i], 1:N)
    return Circle(center, rad)
end

function crude_bounding_circle(c_0::Circle{2,T1},
                               origins::AbstractVector{T2},
                               radii::AbstractVector{T1}) where {T1,T2}
    @assert length(origins) >= 2length(radii)
    N = length(radii)

    x̄ = (sum(view(origins, 1:2:2N)) + origin(c_0)[1])/(N+1)
    ȳ = (sum(view(origins, 2:2:2N)) + origin(c_0)[2])/(N+1)
    center = Vec{2}((x̄, ȳ))
    rad = norm(origin(c_0) - center) + radius(c_0)
    @inbounds for (io, ir) in zip(1:2:2N, 1:N)
        rad = max(rad, sqrt((origins[io] - x̄)^2 + (origins[io+1] - ȳ)^2) + radii[ir])
    end

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

function Base.isapprox(r1::Rectangle, r2::Rectangle; kwargs...)
    return isapprox(minimum(r1), minimum(r2); kwargs...) && isapprox(maximum(r1), maximum(r2); kwargs...)
end

Base.maximum(r::Rectangle) = r.maxs
Base.minimum(r::Rectangle) = r.mins
origin(r::Rectangle) = 0.5*(minimum(r) + maximum(r))
widths(r::Rectangle) = maximum(r) - minimum(r)

@inline xmin(r::Rectangle{2}) = minimum(r)[1]
@inline ymin(r::Rectangle{2}) = minimum(r)[2]
@inline xmax(r::Rectangle{2}) = maximum(r)[1]
@inline ymax(r::Rectangle{2}) = maximum(r)[2]

@inline volume(r::Rectangle{dim}) where {dim} = prod(maximum(r) - minimum(r))
@inline area(r::Rectangle{2}) = volume(r)

# Random rectangles
Base.rand(::Type{Rectangle{dim,T}}) where {dim,T} = Rectangle{dim,T}(-rand(Vec{dim,T}), rand(Vec{dim,T}))
Base.rand(::Type{Rectangle{dim,T}}, N::Int) where {dim,T} = [rand(Rectangle{dim,T}) for i in 1:N]

# Scale rectangle by factor `α` relative to the point `P` (default to it's own origin)
function scale_shape(r::Rectangle{dim}, P::Vec{dim}, α::Number) where {dim}
    new_mins = α * (minimum(r) - P) + P
    new_maxs = α * (maximum(r) - P) + P
    return Rectangle(new_mins, new_maxs)
end
scale_shape(r::Rectangle, α::Number) = scale_shape(r, origin(r), α)

# Translate rectangle by factor `α` relative to the point `P` (defaults to zero vector), or by fixed vector X
function translate_shape(r::Rectangle{dim}, P::Vec{dim}, α::Number) where {dim}
    new_O = α * (origin(r) - P) + P
    new_mins = new_O - 0.5*widths(r)
    new_maxs = new_O + 0.5*widths(r)
    return Rectangle(new_mins, new_maxs)
end
function translate_shape(r::Rectangle, α::Number)
    new_O = α * origin(r)
    new_mins = new_O - 0.5*widths(r)
    new_maxs = new_O + 0.5*widths(r)
    return Rectangle(new_mins, new_maxs)
end
translate_shape(r::Rectangle{dim}, X::Vec{dim}) where {dim} = Rectangle(minimum(r) + X, maximum(r) + X)

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

# Integration of the intersection of area between the circle `c` and the
# rectangular region defined by X0 ≤ x ≤ X1, Y0 ≤ y ≤ Y1. It is assumed that
# X0 ≤ X1 and Y0 ≤ Y1.
#
# This code is heavily influenced by the stackoverflow response by user
# `the swine` at the following link:
#   https://stackoverflow.com/questions/622287/area-of-intersection-between-circle-and-rectangle
function intersect_area(c::Circle{2}, X0, Y0, X1, Y1)

    ZERO, ONE, EPS = zero(X0), one(X0), 5eps(typeof(X0))

    # @inline safe_sqrt(x) = √max(x,EPS)
    # @inline safe_asin(x) = asin(clamp(x,-ONE+EPS,ONE-EPS))
    # @inline area_section(x, h) = (x*safe_sqrt(1-x^2) + safe_asin(x))/2 - x*h
    # @inline area_above(x0, x1, h) = (x = safe_sqrt(1-h^2); area_section(clamp(x1,-x,x), h) - area_section(clamp(x0,-x,x), h))
    # @inline area_region(x0, x1, y0, y1) = area_above(x0, x1, y0) - area_above(x0, x1, y1)

    @inline area_section(x, h) = (x*sqrt(1-x^2) + asin(x))/2 - x*h
    @inline area_above(x0, x1, h) = (x = sqrt(1-h^2); area_section(clamp(x1,-x,x), h) - area_section(clamp(x0,-x,x), h))
    @inline area_region(x0, x1, y0, y1) = area_above(x0, x1, y0) - area_above(x0, x1, y1)

    # Get normalized positions
    r, o = radius(c), origin(c)
    x0, x1, y0, y1 = (X0-o[1])/r, (X1-o[1])/r, (Y0-o[2])/r, (Y1-o[2])/r

    # Clamp just inside ±1 to avoid numerical difficulties with derivatives
    x0, x1 = clamp(x0, -ONE+EPS, ONE-EPS), clamp(x1, -ONE+EPS, ONE-EPS)
    y0, y1 = clamp(y0, -ONE+EPS, ONE-EPS), clamp(y1, -ONE+EPS, ONE-EPS)

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

    # # Equivalent to above, but branch free
    # x0, x1 = min(x0,x1), max(x0,x1)
    # y0, y1 = min(y0,y1), max(y0,y1)
    # FLIP_SIGN = (y0 < ZERO && y1 < ZERO)
    # y0, y1 = FLIP_SIGN * -y1 + !FLIP_SIGN * y0, FLIP_SIGN * -y0 + !FLIP_SIGN * y1
    # area = area_region(x0, x1, ZERO, max(ZERO,-y0)) + area_region(x0, x1, max(y0,ZERO), y1)

    # Integration is over the unit circle, so scale result by r^2
    area *= r^2

    return area
end

@inline intersect_area(c::Circle{2}, Xmin::Vec{2}, Xmax::Vec{2}) = intersect_area(c, Xmin[1], Xmin[2], Xmax[1], Xmax[2])
@inline intersect_area(c::Circle{2}, r::Rectangle{2}) = intersect_area(c, xmin(r), ymin(r), xmax(r), ymax(r))

# Sums `intersect_area` over a vector of circles on a rectangular domain.
#   NOTE: assumes circles are disjoint
function intersect_area(circles::Vector{Circle{2,T1}},
                        domain::Rectangle{2,T2}) where {T1,T2}
    T = promote_type(T1,T2)
    return sum(c -> T(intersect_area_check_inside(c, domain)), circles)
end

function intersect_area(origins::AbstractVector,
                        radii::AbstractVector,
                        domain::Rectangle{2})
    N = length(radii)
    @assert length(origins) >= 2N

    Σ = sum(zip(1:2:2N, 1:N)) do i
        io, ir = i
        o = Vec{2}((origins[io], origins[io+1]))
        r = radii[ir]
        intersect_area_check_inside(Circle(o,r), domain)
    end

    return Σ
end

@inline function intersect_area_check_inside(c::Circle{2,T1},
                                             domain::Rectangle{2,T2}) where {T1,T2}
    T = promote_type(T1,T2)
    if is_inside(c, domain)
        return π*radius(c)^2
    elseif !is_outside(c, domain)
        return intersect_area(c, domain)
    else
        return zero(T)
    end

    # # Equivalent to above, but branch free
    # INSIDE = is_inside(c, domain)
    # OUTSIDE = is_outside(c, domain)
    # return INSIDE * (π*radius(c)^2) + !(INSIDE || OUTSIDE) * intersect_area(c, domain)
end

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
