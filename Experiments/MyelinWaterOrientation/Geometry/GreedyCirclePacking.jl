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

function pack(r::Vector{T}) where {T}
    # Initialize and handle degenerate case
    N = length(r)
    c = Vector{Circle{2,T}}(undef, length(r))
    N == 0 && return c

    # One circle
    c[1] = Circle{2,T}(Vec{2,T}((zero(T), zero(T))), r[1])
    N == 1 && return c

    # Two circles
    c[2] = Circle{2,T}(origin(c[1]) + Vec{2,T}((r[1]+r[2], zero(T))), r[2])
    N == 2 && return c

    # Three circles
    r3, c3, s3 = r[1] + r[3], cos_alpha(r[1], r[2], r[3]), sin_alpha(r[1], r[2], r[3])
    c[3] = Circle{2,T}(origin(c[1]) + r3 * Vec{2,T}((c3, s3)), r[3])
    N == 3 && return c

    # Four or more circles
    ATOL = 1e-12 * minimum(r)
    @inbounds for i = 4:N
        Σd²_best = T(Inf)
        # for t in Iterators.flatten(tangent_circles(c[j],c[k],r[i],ATOL) for j in 1:i-1 for k in 1:j-1)
        for jj in 1:i-1 for ii in 1:jj-1 for t in tangent_circles(c[jj], c[ii], r[i], 2*ATOL)
            overlap = false
            Σd² = zero(T)
            for j = 1:i-1
                d²_tj = norm2(origin(t) - origin(c[j]))
                d²_tj < (r[i] + r[j] - ATOL)^2 && (overlap = true; break)
                Σd² += d²_tj
            end
            if !overlap && Σd² < Σd²_best
                Σd²_best = Σd²
                c[i] = t
            end
        end end end
    end

    return c
end

# For three tangent circles c1, c2, and c3 with radii v, u, and w, the angle α
# between the origin displacement vectors o12 = o2-o1 and o13 = o3-o1 is given
# by the below equivalent formulae. cos(α) and sin(α) can also be calculated
# without the use of trigonometric functions.
alpha(v,u,w) = 2*asin(sqrt((u/(v+u))*(w/(v+w))))
# alpha(v,u,w) = acos(((v+u)^2+(v+w)^2-(u+w)^2)/(2*(v+u)*(v+w))) # equivalent, slower form
cos_alpha(v,u,w) = ((v+u)^2+(v+w)^2-(u+w)^2)/(2*(v+u)*(v+w))
sin_alpha(v,u,w) = 2*sqrt(u*v*w*(u+v+w))/((u+v)*(v+w))

# Equivalent formulae for a circle with radius r3 which is tangent to two
# non-tangent circles c1 and c2.
#NOTE: Assumes that c1 and c2 are close enough that this is valid.
function alpha(c1::Circle, c2::Circle, r3)
    #TODO @assert (valid conditions...)
    return atan(cos_alpha(c1,c2,r3), sin_alpha(c1,c2,r3))
end
function cos_alpha(c1::Circle, c2::Circle, r3)
    #TODO @assert (valid conditions...)
    # Simple law of cosines:
    a, c = radius(c1) + r3, radius(c2) + r3
    b² = norm2(origin(c2) - origin(c1))
    b = sqrt(b²)
    return (a^2+b²-c^2)/(2a*b)
end
function sin_alpha(c1::Circle, c2::Circle, r3)
    #TODO @assert (valid conditions...)
    a, b, c = radius(c1) + r3, norm(origin(c2) - origin(c1)), radius(c2) + r3
    return sqrt((c-a+b)*(a+b-c)*(a-b+c)*(a+b+c))/(2a*b)
end

# Return the pair of circles nearest c1 and c2 with radius r3. If c1 and c2 are
# close enough, the two circles tangent to c1 and c2 are returned. Otherwise,
# the circles tangent only one of c1 or c3 along the line between c1 and c3
# are returned.
function tangent_circles(c1::Circle{2,T}, c2::Circle{2,T}, r3::T, ATOL::T = 1e-14*r3) where {T}
    D = d(c1, c2)
    D <= -ATOL && error("Circles must be non-overlapping")

    dx = origin(c2) - origin(c1)
    r1, r2 = radius(c1), radius(c2)

    if D >= 2*r3
        dx /= norm(dx)
        o3A = origin(c1) + (r1+r3) * dx
        o3B = origin(c2) - (r2+r3) * dx
    else
        if D < ATOL # -ATOL < D < ATOL
            cosα, sinα = cos_alpha(r1,r2,r3), sin_alpha(r1,r2,r3)
        else # ATOL <= D < 2*r3
            cosα, sinα = cos_alpha(c1,c2,r3), sin_alpha(c1,c2,r3)
        end
        R = rotmat(dx)
        o3A = origin(c1) + (r1+r3) * (R' ⋅ Vec{2,T}((cosα,  sinα)))
        o3B = origin(c1) + (r1+r3) * (R' ⋅ Vec{2,T}((cosα, -sinα)))
    end

    return (Circle{2,T}(o3A, r3), Circle{2,T}(o3B, r3))
end

end # module GreedyCirclePacking

nothing
