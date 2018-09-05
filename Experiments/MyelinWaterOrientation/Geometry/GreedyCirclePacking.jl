# ============================================================================ #
# Greedy circle packing algorithm
# ============================================================================ #

module GreedyCirclePacking

export pack

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using GeometryUtils
using Tensors
# using VoronoiDelaunay
# using Statistics
# using StaticArrays: SVector
# using LinearAlgebra
# using PolynomialRoots
# using Optim
# using LineSearches

const d = signed_edge_distance
function is_tangent(c1::Circle{dim,T}, c2::Circle{dim,T};
                    atol=1e-14, rtol=1e-14) where {dim,T}
    isapprox(d(c1,c2), zero(T); atol=atol, rtol=rtol)
end

# For three tangent circles c1, c2, and c3 with radii v, u, and w, the angle α
# between the origin displacement vectors o12 = o2 - o1 and o13 = o3-o1 is given
# by the below equivalent formulae. cos(α) and sin(α) can also be calculated
# without the use of trigonometric functions.
alpha(v,u,w) = 2*asin(sqrt((u/(v+u))*(w/(v+w))))
# alpha(v,u,w) = acos(((v+u)^2+(v+w)^2-(u+w)^2)/(2*(v+u)*(v+w))) # equivalent, slower form
cos_alpha(v,u,w) = ((v+u)^2+(v+w)^2-(u+w)^2)/(2*(v+u)*(v+w))
sin_alpha(v,u,w) = 2*sqrt(u*v*w*(u+v+w))/((u+v)*(v+w))

# Equivalent formulae for a circle c3 which is tangent to two non-tangent
# circles c1 and c2
function alpha(c1::Circle{dim,T}, c2::Circle{dim,T}, c3::Circle{dim,T}) where {dim,T}
    #TODO @assert (valid conditions...)
    return atan(cos_alpha(c1,c2,c3), sin_alpha(c1,c2,c3))
end
function cos_alpha(c1::Circle{dim,T}, c2::Circle{dim,T}, c3::Circle{dim,T}) where {dim,T}
    #TODO @assert (valid conditions...)
    # Simple law of cosines:
    a, c = radius(c1) + radius(c3), radius(c2) + radius(c3)
    b² = norm2(origin(c2) - origin(c1))
    b = sqrt(b²)
    return (a^2 + b² - c^2)/(2a*b)
end
function sin_alpha(c1::Circle{dim,T}, c2::Circle{dim,T}, c3::Circle{dim,T}) where {dim,T}
    #TODO @assert (valid conditions...)
    a, b, c = radius(c1) + radius(c3), norm(origin(c2) - origin(c1)), radius(c2) + radius(c3)
    return sqrt((c-a+b)*(a+b-c)*(a-b+c)*(a+b+c))/(2a*b)
end


function pack(r::Vector{T}) where {T}
    # Initialize and handle degenerate case
    c = Vector{Circle{2,T}}(undef, length(r))
    length(c) == 0 && return c

    # One circle
    c[1] = Circle{2,T}(Vec{2,T}((zero(T), zero(T))), r[1])
    length(c) == 1 && return c

    # Two circles
    c[2] = Circle{2,T}(origin(c[1]) + Vec{2,T}((r[1]+r[2], zero(T))), r[2])
    length(c) == 2 && return c

    # Three circles
    r3, c3, s3 = r[1] + r[3], cos_alpha(r[1], r[2], r[3]), sin_alpha(r[1], r[2], r[3])
    c[3] = Circle{2,T}(origin(c[1]) + r3 * Vec{2,T}((c3, s3)), r[3])
    length(c) == 3 && return c

    # Four or more circles

end

# Return the pair of circles tangent to c1 and c2
function tangent_circles(c1::Circle{2,T}, c2::Circle{2,T}, r3::T) where {T}
    o1, r1, r2 = origin(c1), radius(c1), radius(c2)
    cosα, sinα = cos_alpha(r1,r2,r3), sin_alpha(r1,r2,r3)
    dx = origin(c2) - origin(c1)
    R = rotmat(dx)
    o3A = o1 + (r1+r3) * (R' ⋅ Vec{2,T}((cosα,  sinα)))
    o3B = o1 + (r1+r3) * (R' ⋅ Vec{2,T}((cosα, -sinα)))
    return (Circle{2,T}(o3A, r3), Circle{2,T}(o3B, r3))
end


end # module GreedyCirclePacking

nothing
