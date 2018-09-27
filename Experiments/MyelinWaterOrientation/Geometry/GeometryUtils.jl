# ============================================================================ #
# Generic geometry utilities for use within the JuAFEM.jl/Tensors.jl framework
# ============================================================================ #

module GeometryUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using Statistics
using Tensors
using StaticArrays: SVector
using LinearAlgebra
using PolynomialRoots
using Optim
using LineSearches

export Ellipse, Circle, Rectangle
export norm2, hadamardproduct, ⊙, skewprod, ⊠
export origin, radius, radii, widths, corners, geta, getb, getc, getF1, getF2, rotmat
export dimension, floattype, xmin, ymin, xmax, ymax, area, volume
export scale_shape, translate_shape, inscribed_square
export signed_edge_distance, minimum_signed_edge_distance
export bounding_box, bounding_circle, crude_bounding_circle, opt_bounding_ellipse, opt_bounding_circle, intersect_area, intersection_points
export is_inside, is_overlapping, is_any_overlapping, is_on_circle, is_on_any_circle, is_in_circle, is_in_any_circle, is_inside, is_outside

# ---------------------------------------------------------------------------- #
# Types
# ---------------------------------------------------------------------------- #
struct Ellipse{dim,T}
    F1::Vec{dim,T} # focus #1
    F2::Vec{dim,T} # focus #2
    b::T # semi-minor axis
end

struct Circle{dim,T}
    center::Vec{dim,T}
    r::T
end

struct Rectangle{dim,T}
    mins::Vec{dim,T}
    maxs::Vec{dim,T}
end

# ---------------------------------------------------------------------------- #
# Misc. utilities for Vec type from Tensors.jl
# ---------------------------------------------------------------------------- #
Base.convert(::Type{Vec{dim,T1}}, x::SVector{dim,T2}) where {dim,T1,T2} = Vec{dim,promote_type(T1,T2)}(Tuple(x))
Base.convert(::Type{SVector{dim,T1}}, x::Vec{dim,T2}) where {dim,T1,T2} = SVector{dim,promote_type(T1,T2)}(Tuple(x))
@inline Base.Tuple(v::Vec) = Tensors.get_data(v)

@inline norm2(x::Vec) = dot(x,x)
@inline Base.sincos(x::Vec{2}) = (r = norm(x); sinθ = x[2]/r; cosθ = x[1]/r; return (sinθ, cosθ))
@inline function rotmat(x::Vec{2,T}) where {T}
    sinθ, cosθ = sincos(x)
    return Tensor{2,2,T}((cosθ, sinθ, -sinθ, cosθ))
end

@inline function hadamardproduct(S1::Vec{dim}, S2::Vec{dim}) where {dim}
    return Vec{dim}(@inline function(i) v = S1[i] * S2[i]; return v; end)
end
# @inline hadamardproduct2(S1::Vec{2}, S2::Vec{2}) = Vec{2}((S1[1]*S2[1], S1[2]*S2[2]))
# @inline hadamardproduct3(S1::Vec{3}, S2::Vec{3}) = Vec{3}((S1[1]*S2[1], S1[2]*S2[2], S1[3]*S2[3]))
const ⊙ = hadamardproduct

# Compute the `skew product` between two 2-dimensional vectors. This is the same
# as computing the third component of the cross product if the vectors `a` and
# `b` were extended to three dimensions
@inline function skewprod(a::Vec{2,T}, b::Vec{2,T}) where T
    # @inbounds v = (a × b)[3] # believe it or not, this emits the same llvm code...
    @inbounds v = a[1]*b[2] - b[1]*a[2]
    return v
end
const ⊠ = skewprod # Denote the skewproduct using `\boxtimes`

# ---------------------------------------------------------------------------- #
# Ellipse based on Vec type from Tensors.jl (code based on GeometryTypes.jl)
# ---------------------------------------------------------------------------- #
@inline getF1(e::Ellipse) = e.F1
@inline getF2(e::Ellipse) = e.F2
@inline geta(e::Ellipse{2,T}) where {T} = sqrt(e.b^2 + T(0.25)*norm2(e.F1 - e.F2)) # a = √(b² + c²)
@inline getb(e::Ellipse) = e.b
@inline getc(e::Ellipse{dim,T}) where {dim,T} = T(0.5)*norm(e.F1 - e.F2)

@inline area(e::Ellipse{2}) = pi * geta(e) * getb(e)
@inline origin(e::Ellipse{dim,T}) where {dim,T} = T(0.5)*(getF1(e) + getF2(e))
@inline eccentricity(e::Ellipse{2}) = getc(e)/geta(e)

scale_shape(e::Ellipse{dim}, P::Vec{dim}, α::Number) where {dim} = Ellipse(α * (getF1(e) - P) + P, α * (getF2(e) - P) + P, α * getb(e))
scale_shape(e::Ellipse, α::Number) = scale_shape(e, origin(e), α)

@inline Base.sincos(e::Ellipse{2}) = sincos(getF2(e) - getF1(e))
@inline rotmat(e::Ellipse{2}) = rotmat(getF2(e) - getF1(e))

function Base.rand(::Type{Ellipse{dim,T}}) where {dim,T}
    F1 = 2*rand(Vec{dim,T}) - ones(Vec{dim,T})
    F2 = 2*rand(Vec{dim,T}) - ones(Vec{dim,T})
    return Ellipse(F1, F2, rand(T))
end
Base.rand(::Type{E}, N::Int) where {E <: Ellipse} = [rand(E) for i in 1:N]

function signed_edge_distance(X::Vec{2}, e::Ellipse{2})

    # Check for simple cases to skip root finding below
    EPS = 5*eps(getb(e))
    isapprox(X, origin(e); rtol = EPS) && return -getb(e) # origin is a pathological case for rootfinding
    isapprox(X, getF1(e); rtol = EPS) && return getc(e) - geta(e)
    isapprox(X, getF2(e); rtol = EPS) && return getc(e) - geta(e)

    # Vector defining axis direction and therefore orientation
    v = getF2(e) - getF1(e)
    r = norm(v)
    r ≤ EPS && return signed_edge_distance(X, Circle(origin(e), geta(e))) # F1 ≈ F2; return the circle distance

    # Rotate points to standard reference frame where axis vector is parallel to x-axis
    dX = X - origin(e) # distance vector to origin
    sinθ, cosθ = v[2]/r, v[1]/r # (co)sine of angle w.r.t. x-axis
    x0, y0 = dX[1] * cosθ + dX[2] * sinθ, -dX[1] * sinθ + dX[2] * cosθ

    # Solve 4th order polynomial equation
    a, b = geta(e), getb(e)
    t1, t2, t4, t6 = a*a, b*b, y0*y0, x0*x0
    t8, t9, t11 = t1*t1, t2*t1, t2*t2
    t15, t16 = t8*t2, t11*t1
    rts = PolynomialRoots.roots(BigFloat.([
        -t16*t6 + t8*t11 - t15*t4,
        -2*t9*t6 - 2*t9*t4 + 2*t15 + 2*t16,
        -t2*t4 - t1*t6 + t8 + 4*t9 + t11,
        2*t1 + 2*t2,
        1
    ]))
    t = typeof(a)(maximum(r->real(r), rts))

    a², b² = a^2, b^2
    x = a² * x0 / (t + a²)
    y = b² * y0 / (t + b²)
    d = t * sqrt((x/a²)^2+(y/b²)^2)

    return d
end
@inline signed_edge_distance(c::Circle{2}, e::Ellipse{2}) = signed_edge_distance(origin(c), e) - radius(c)

@inline function is_inside(c::Circle{2}, e::Ellipse{2}, lt = ≤)
    !lt(radius(c), getb(e)) && return false # can't be inside if radius is larger than minor axis
    d = signed_edge_distance(origin(c), e)
    return lt(radius(c), -d)
end

# bounding circle of a collection of circles through numerical optimization
function opt_bounding_ellipse(circles::Vector{Circle{dim,T}};
                              epsilon = zero(T),
                              maxiter = 3) where {dim,T}
    # Penalize if inner circle isn't at least a distance ϵ inside the outer ellipse
    function overlap_dist(c_in::Circle, e_bnd::Ellipse, ϵ)
        d12 = signed_edge_distance(origin(c_in), e_bnd) + radius(c_in)
        return d12 > -ϵ ? (d12 + ϵ)^2 : zero(d12)
    end

    function optfun(x::Vector{Tx}, lambda = Tx(1e-3)) where {Tx}
        @inbounds ebound = Ellipse{dim,Tx}(Vec{dim,Tx}((x[1], x[2])), Vec{dim,Tx}((x[3], x[4])), x[5]) # bounding ellipse
        overlap_penalty = sum(c -> overlap_dist(c, ebound, epsilon), circles)
        area_penalty = area(ebound)/length(circles)
        return (overlap_penalty + lambda * area_penalty)/radius(c_initial)^2
    end

    c_initial = crude_bounding_circle(circles)
    O, dX = origin(c_initial), 0.1*radius(c_initial) * ones(Vec{dim,T})
    e_initial = Ellipse(O + dX, O - dX, radius(c_initial)) # need to initialize non-degenerate ellipse
    e_final = e_initial
    e_check = e_opt -> minimum(c -> -signed_edge_distance(origin(c), e_opt) - radius(c), circles) > epsilon/2

    lambda = T(1e-3)
    x0 = [getF1(e_initial)..., getF2(e_initial)..., getb(e_initial)]
    for i in 1:maxiter
        # Can't use autodiff as generic eigenvalue solvers used in signed_edge_distance are too slow
        opt_obj = OnceDifferentiable(x->optfun(x,lambda), x0)
        result = optimize(opt_obj, x0, LBFGS(linesearch = LineSearches.BackTracking(order=3)))

        x = copy(Optim.minimizer(result))
        e_final = Ellipse{dim,T}(Vec{dim,T}((x[1], x[2])), Vec{dim,T}((x[3], x[4])), x[5]) # bounding ellipse
        e_check(e_final) && break

        lambda /= 10
    end

    if !e_check(e_final)
        f = α -> e_check(scale_shape(e_final, α)) - T(0.5)
        α_min = find_zero(f, T.((0.01, 100.0)), Bisection())
        return scale_shape(e_final, α_min + 100*eps(α_min))
    else
        return e_final
    end
end

# ---------------------------------------------------------------------------- #
# Circle based on Vec type from Tensors.jl (code based on GeometryTypes.jl)
# ---------------------------------------------------------------------------- #
Circle(center::NTuple{dim,T}, r::T) where {dim,T} = Circle(Vec{dim,T}(center), r)
Circle(::Type{T}, center::Vec{dim}, r::Number) where {dim,T} = Circle(Vec{dim,T}(Tuple(center)), T(r))
Circle(::Type{T}, center::NTuple{dim}, r::Number) where {dim,T} = Circle(Vec{dim,T}(center), T(r))

function Circle(center::Vec{dim,T1}, r::T2) where {dim,T1,T2}
    T = promote_type(T1, T2)
    return Circle(Vec{dim,T}(Tuple(center)), T(r))
end
function Base.convert(::Type{Circle{dim,T1}}, c::Circle{dim,T2}) where {dim,T1,T2}
    T = promote_type(T1, T2)
    return Circle{dim,T}(Vec{dim,T}(Tuple(origin(c))), T(radius(c)))
end
function Base.isapprox(c1::Circle, c2::Circle; kwargs...)
    return isapprox(radius(c1), radius(c2); kwargs...) && isapprox(origin(c1), origin(c2); kwargs...)
end

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
@inline volume(c::Circle{3}) = 4pi*radius(c)^3/3

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
Base.rand(::Type{C}, N::Int) where {C <: Circle} = [rand(C) for i in 1:N]

# Signed distance from X to circle edge distance
@inline signed_edge_distance(X::Vec, c::Circle) = norm(X - origin(c)) - radius(c)
@inline signed_edge_distance(c::Circle, X::Vec) = signed_edge_distance(X, c)

# Signed distance between circle edges (zero is when circles are tangent, negative when overlapping)
@inline signed_edge_distance(c1::Circle, c2::Circle) = norm(origin(c1) - origin(c2)) - radius(c1) - radius(c2)
@inline signed_edge_distance(o1::Vec, r1, o2::Vec, r2) = norm(o1 - o2) - r1 - r2

function minimum_signed_edge_distance(circles::Vector{C}) where {C<:Circle{dim,T}} where {dim,T}
    min_dist = T(Inf)
    @inbounds for i in 1:length(circles)-1
        c_i = circles[i]
        @inbounds for j in i+1:length(circles)
            c_j = circles[j]
            min_dist = min(min_dist, signed_edge_distance(c_i, c_j))
        end
    end
    min_dist
end

# check if c1 is in c2: r1 < r2 and ||o1 - o2|| < r2 - r1
@inline function is_inside(c1::Circle, c2::Circle, lt = ≤)
    o1, o2, r1, r2 = origin(c1), origin(c2), radius(c1), radius(c2)
    return lt(r1, r2) && lt(norm2(o1 - o2), (r2-r1)^2)
end

@inline function is_inside(x::Vec{dim}, c::Circle{dim}, lt = ≤) where {dim}
    return lt(norm2(x - origin(c)), radius(c)^2)
end

# is_on_circle/is_on_any_circle
@inline function is_on_circle(x::Vec{dim,T},
                              circle::Circle{dim,T},
                              thresh::T=sqrt(eps(T))) where {dim,T}
    return abs(norm(x - origin(circle)) - radius(circle)) <= thresh
end
function is_on_any_circle(x::Vec{dim,T},
                          circles::Vector{Circle{dim,T}},
                          thresh::T=sqrt(eps(T))) where {dim,T}
    return any(circle->is_on_circle(x, circle, thresh), circles)
end

# is_in_circle/is_in_any_circle
@inline function is_in_circle(x::Vec{dim,T},
                              circle::Circle{dim,T},
                              thresh::T=sqrt(eps(T))) where {dim,T}
    dx = x - origin(circle)
    return dx⋅dx <= (radius(circle) + thresh)^2
end
function is_in_any_circle(x::Vec{dim,T},
                          circles::Vector{Circle{dim,T}},
                          thresh::T=sqrt(eps(T))) where {dim,T}
    return any(circle->is_in_circle(x, circle, thresh), circles)
end

# check if c1 and c2 are overlapping
@inline function is_overlapping(c1::Circle{dim,T}, c2::Circle{dim,T}, lt = ≤, thresh = zero(T)) where {dim,T}
    dx = origin(c1) - origin(c2)
    d_min = radius(c1) + radius(c2)
    return lt(dx⋅dx, (d_min + thresh)^2)
end

# check if any circles in `cs` are overlapping
function is_any_overlapping(cs::AbstractVector{C}, lt = ≤, thresh = zero(T)) where {C<:Circle{dim,T}} where {dim,T}
    @inbounds for i in 1:length(cs)-1
        ci = cs[i]
        @inbounds for j in i+1:length(cs)
            is_overlapping(ci, cs[j], lt, thresh) && return true
        end
    end
    return false
end

# bounding circle of a collection of circles through numerical optimization
function opt_bounding_circle(circles::Vector{Circle{dim,T}};
                             epsilon = zero(T),
                             maxiter = 10) where {dim,T}
    # Penalize if inner circle isn't at least a distance ϵ inside the outer circle
    function overlap_dist(c_in::Circle, c_out::Circle, ϵ)
        d12 = signed_edge_distance(origin(c_in), c_out) + radius(c_in)
        return d12 > -ϵ ? (d12 + ϵ)^2 : zero(d12)
    end

    function optfun(x::Vector{Tx}, lambda = Tx(1e-3)) where {Tx}
        @inbounds cbound = Circle{dim,Tx}(Vec{dim,Tx}((x[1], x[2])), x[3]) # bounding circle
        overlap_penalty = sum(c -> overlap_dist(c, cbound, epsilon), circles)
        area_penalty = area(cbound)/length(circles)
        return (overlap_penalty + lambda * area_penalty)/radius(c_initial)^2
    end

    c_initial = crude_bounding_circle(circles)
    c_final = c_initial
    c_check = c_opt -> minimum(c -> -signed_edge_distance(origin(c), c_opt) - radius(c), circles) > epsilon/2

    lambda = T(1e-3)
    x0 = [origin(c_initial)..., radius(c_initial)]
    for i in 1:maxiter
        opt_obj = OnceDifferentiable(x->optfun(x,lambda), x0; autodiff = :forward)
        result = optimize(opt_obj, x0, LBFGS(linesearch = LineSearches.BackTracking(order=3)))

        x = copy(Optim.minimizer(result))
        c_final = Circle{dim,T}(Vec{dim,T}((x[1], x[2])), x[3])
        c_check(c_final) && break

        lambda /= 2
    end

    if !c_check(c_final)
        f = r -> c_check(Circle(origin(c_final), r)) - T(0.5)
        r_min = find_zero(f, T.((0.01, 100.0)).*radius(c_final), Bisection())
        return Circle(origin(c_final), r_min + 100*eps(r_min))
    else
        return c_final
    end
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
    c1 ≈ c2 && return c1

    # If not, check if the circle origins are overlapping
    T = promote_type(floattype(c1), floattype(c2))
    EPS = √eps(T)
    o1, o2, r1, r2 = origin(c1), origin(c2), radius(c1), radius(c2)

    r12 = norm(o1-o2)
    r12 ≤ EPS*max(r1,r2) && return Circle(T(0.5)*(o1+o2), max(r1,r2)*T(1+EPS))

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

function crude_bounding_circle(circles::Vector{C}) where {C<:Circle}
    center = mean(c->origin(c), circles)
    rad = maximum(c->norm(origin(c)-center)+radius(c), circles)
    return Circle(center, rad + 5*eps(rad))
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
Rectangle(mins::NTuple{dim,T}, maxs::Vec{dim,T}) where {dim,T} = Rectangle(Vec{dim,T}(mins), maxs)
Rectangle(mins::Vec{dim,T}, maxs::NTuple{dim,T}) where {dim,T} = Rectangle(mins, Vec{dim,T}(maxs))
Rectangle(mins::NTuple{dim,T}, maxs::NTuple{dim,T}) where {dim,T} = Rectangle(Vec{dim,T}(mins), Vec{dim,T}(maxs))

function Base.isapprox(r1::Rectangle, r2::Rectangle; kwargs...)
    return isapprox(minimum(r1), minimum(r2); kwargs...) && isapprox(maximum(r1), maximum(r2); kwargs...)
end

Base.maximum(r::Rectangle) = r.maxs
Base.minimum(r::Rectangle) = r.mins
origin(r::Rectangle) = (minimum(r) + maximum(r))/2
widths(r::Rectangle) = maximum(r) - minimum(r)

@inline xmin(r::Rectangle{2}) = minimum(r)[1]
@inline ymin(r::Rectangle{2}) = minimum(r)[2]
@inline xmax(r::Rectangle{2}) = maximum(r)[1]
@inline ymax(r::Rectangle{2}) = maximum(r)[2]

@inline volume(r::Rectangle{dim}) where {dim} = prod(maximum(r) - minimum(r))
@inline area(r::Rectangle{2}) = volume(r)

# Corners of rectangle, in counterclockwise order, beginning from bottom left
function corners(r::Rectangle{2,T}) where {T}
    c1, c3, w = minimum(r), maximum(r), widths(r)
    c2 = c1 + Vec{2,T}((w[1], zero(T)))
    c4 = c1 + Vec{2,T}((zero(T), w[2]))
    return (c1, c2, c3, c4)
end

# Random rectangles
Base.rand(::Type{Rectangle{dim,T}}) where {dim,T} = Rectangle{dim,T}(-rand(Vec{dim,T}), rand(Vec{dim,T}))
Base.rand(::Type{Rectangle{dim,T}}, N::Int) where {dim,T} = [rand(Rectangle{dim,T}) for i in 1:N]

# Rectangle intersection
function Base.intersect(r1::Rectangle, r2::Rectangle)
    min1, min2, max1, max2 = minimum(r1), minimum(r2), maximum(r1), maximum(r2)
    V = promote_type(typeof(min1), typeof(min2))
    mins = max.(Tuple(min1), Tuple(min2))
    maxs = min.(Tuple(max1), Tuple(max2))
    maxs = broadcast((mi,mx) -> mx <= mi ? mi : mx, mins, maxs) # in case of null intersection
    return Rectangle(mins, maxs)
end

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
    new_mins = new_O - widths(r)/2
    new_maxs = new_O + widths(r)/2
    return Rectangle(new_mins, new_maxs)
end
function translate_shape(r::Rectangle, α::Number)
    new_O = α * origin(r)
    new_mins = new_O - widths(r)/2
    new_maxs = new_O + widths(r)/2
    return Rectangle(new_mins, new_maxs)
end
translate_shape(r::Rectangle{dim}, X::Vec{dim}) where {dim} = Rectangle(minimum(r) + X, maximum(r) + X)

# Bounding box of vector of circles
bounding_box(c::Circle) = Rectangle((xmin(c), ymin(c)), (xmax(c), ymax(c)))

function bounding_box(circles::Vector{C}) where {C <: Circle}
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

# ---------------------------------------------------------------------------- #
# Area of intersections between shapes (2D)
# ---------------------------------------------------------------------------- #

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
function intersect_area(
        circles::Vector{Circle{2,T1}},
        domain::Rectangle{2,T2}
    ) where {T1,T2}
    T = promote_type(T1,T2)
    return sum(c -> T(intersect_area_check_inside(c, domain)), circles)
end

function intersect_area(
        origins::AbstractVector,
        radii::AbstractVector,
        domain::Rectangle{2}
    )

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

@inline function intersect_area_check_inside(
        c::Circle{2,T1},
        domain::Rectangle{2,T2}
    ) where {T1,T2}

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


# ---------------------------------------------------------------------------- #
# Intersection points between shape boundaries (2D)
# ---------------------------------------------------------------------------- #

function intersection_points(
        circ::Circle{2,T1},
        rect::Rectangle{2,T2},
        ϵ = sqrt(eps(promote_type(T1, T2)))
    ) where {T1, T2}

    @inline fcircle(x, x0, r) = sqrt(r^2 - (x-x0)^2)

    @inline function xpoints!(points::Vector{Vec{2,T}}, x, y0, y1, a, b, r) where {T}
        if a-r <= x && x <= a+r
            d = fcircle(x, a, r)
            (y0 <= b + d <= y1) && push!(points, Vec{2,T}((x, b + d)))
            (y0 <= b - d <= y1) && push!(points, Vec{2,T}((x, b - d)))
        end
        return points
    end

    @inline function ypoints!(points::Vector{Vec{2,T}}, y, x0, x1, a, b, r) where {T}
        if b-r <= y && y <= b+r
            d = fcircle(y, b, r)
            (x0 <= a + d <= x1) && push!(points, Vec{2,T}((a + d, y)))
            (x0 <= a - d <= x1) && push!(points, Vec{2,T}((a - d, y)))
        end
        return points
    end

    T = promote_type(T1, T2)
    x0, x1, y0, y1 = xmin(rect), xmax(rect), ymin(rect), ymax(rect)
    a, b, r = origin(circ)[1], origin(circ)[2], radius(circ)

    # Create initial set of intersection points
    points = Vec{2,T}[]
    xpoints!(points, x0, y0, y1, a, b, r)
    xpoints!(points, x1, y0, y1, a, b, r)
    ypoints!(points, y0, x0, x1, a, b, r)
    ypoints!(points, y1, x0, x1, a, b, r)

    # Remove any duplicates
    if !isempty(points)
        b = fill(true, length(points))
        @inbounds for i in 2:length(points)
            pi = points[i]
            for j in 1:i-1
                pj = points[j]
                (norm(pi - pj) < ϵ * max(norm(pi), norm(pj))) && (b[i] = false; break)
            end
        end
        points = points[b]
    end

    return points
end

@inline intersection_points(r::Rectangle{2}, c::Circle{2}) = intersection_points(c, r)

end # module GeometryUtils

nothing
