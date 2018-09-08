# ============================================================================ #
# Manifold circle packing algorithm
# ============================================================================ #

module ManifoldCirclePacking

export pack

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using GeometryUtils
using CirclePackingUtils
using Tensors
# using VoronoiDelaunay
# using Statistics
# using StaticArrays: SVector
# using LinearAlgebra
# using PolynomialRoots
using Optim
using LineSearches

@inline d(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) = norm(o1 - o2) - r1 - r2 - ϵ
@inline ∇d(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) = (o1 - o2)/norm(o1 - o2) # ∇ w.r.t `o1`
@inline α(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) = (r1 + r2 + ϵ)/norm(o1 - o2)

function pack(
        r::AbstractVector{T},
        x0 = randn(T, 2*length(r)),
        ϵ = zero(T);
        manifold = MinimallyTangent(ϵ, r),
        Alg = Optim.LBFGS(
            manifold = manifold,
            linesearch = LineSearches.BackTracking(order=3)),
        Opts = Optim.Options(
            iterations = 100_000,
            x_tol = 1e-10 * minimum(r),
            g_tol = 1e-12,
            allow_f_increases = true)
        ) where {T}

    # Check inputs. x0 is a vector of unknown circle origins and r is a vector
    # of known radii. Must have at least 2 circles
    @assert length(x0) == 2*length(r) >= 4

    # Initial functions and derivatives
    D = (o1,o2,r1,r2) -> d(o1,o2,r1,r2,ϵ)
    ∇D = (o1,o2,r1,r2) -> ∇d(o1,o2,r1,r2,ϵ)
    f = (x) -> pairwise_sum(D, x, r)
    g! = (out, x) -> pairwise_grad!(out, ∇D, x, r)
    fg! = (out, x) -> (g!(out, x); return f(x))

    # Form *Differentiable object
    opt_obj = OnceDifferentiable(f, g!, fg!, x0)

    # Optimize and get results
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = result.minimizer
    c = [Circle{2,T}(Vec{2,T}((x[2i-1],x[2i])), r[i]) for i in 1:length(r)]
    return c
end

function pack(
        c::AbstractVector{Circle{2,T}},
        ϵ = zero(T);
        kwargs...) where {T}
    # Unpack input vector of circles to radius and x0 vectors
    N = length(c)
    r, x0 = zeros(N), zeros(Vec{2,T}, N)
    for (i, ci) in enumerate(c)
        r[i], x0[i] = radius(ci), origin(ci)
    end
    x0 = copy(reinterpret(T, x0))
    return pack(r, x0, ϵ; kwargs...)
end

struct MinimallyTangent{T} <: Manifold
    ϵ::T # minimum distance
    r::AbstractVector{T} # circle radii
end

# Compute scale factor `α` such that the minimum distance between circles is m.ϵ
# and scale `x` accordingly
function Optim.retract!(m::MinimallyTangent, x::AbstractVector{T}) where {T}
    # Check inputs and initialize
    @assert length(x) == 2*length(m.r) >= 4
    o = reinterpret(Vec{2,T}, x)

    # Pull first iteration outside of the loop for easy initialization
    @inbounds alpha = α(o[1], o[2], m.r[1], m.r[2], m.ϵ)
    for j = 3:length(o)
        for i in 1:j-1
            @inbounds alpha = max(alpha, α(o[i], o[j], m.r[i], m.r[j], m.ϵ))
        end
    end

    # Scale x and return
    return x .*= alpha
end

# # The gradient with respect to each circle origin will be projected onto the
# # average tangent direction with respect to all nearby circles, weighted by
# # 1/(d^2 + λ), where d is the signed circle distance, and λ > 0 serves to both
# # to avoid singularities, and handle multiple tangent vectors.
# function Optim.project_tangent!(
#         m::MinimallyTangent,
#         g::AbstractVector{T},
#         x::AbstractVector{T}) where {T}
#
#     # Check inputs and initialize
#     @assert length(g) == length(x) == 2*length(m.r) >= 4
#     G = reinterpret(Vec{2,T}, g)
#     o = reinterpret(Vec{2,T}, x)
#     N = length(G)
#     λ = √eps(T)
#
#     @inbounds for i in 1:N
#         t = zero(Vec{2,T})
#         Σw = zero(T)
#         for j in 1:i-1
#             dt = o[j] - o[i] # difference vector
#             dt /= norm(dt) # normalized distance vector
#             w = 1/(d(o[i], o[j], m.r[i], m.r[j], m.ϵ)^2 + λ) # weight
#             t += w * dt
#             Σw += w
#         end
#         for j in i+1:N
#             dt = o[j] - o[i] # difference vector
#             dt /= norm(dt) # normalized distance vector
#             w = 1/(d(o[i], o[j], m.r[i], m.r[j], m.ϵ)^2 + λ) # weight
#             t += w * dt
#             Σw += w
#         end
#         t /= norm(t) # normalized weighted difference
#         # W = tanh(2*Σw*λ) # W approaches 1 when Σw > 1/λ, 0 when Σw << 1/λ
#         W = one(T)
#         G[i] -= (W * dot(t, G[i])) * t # project out component parallel to t with weight W
#     end
#
#     return g
# end

# Gradient of squared distance function is scale invariant, so just return g
Optim.project_tangent!(m::MinimallyTangent, g, x) = g

end # module ManifoldCirclePacking

nothing
