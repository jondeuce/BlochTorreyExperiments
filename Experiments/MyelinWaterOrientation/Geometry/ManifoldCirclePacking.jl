Σw# ============================================================================ #
# Manifold circle packing algorithm
# ============================================================================ #

module ManifoldCirclePacking

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
using Optim
using LineSearches

@inline d(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) = norm(o1 - o2) - r1 - r2 - ϵ
@inline ∇d(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) = (o1 - o2)/norm(o1 - o2) # ∇ w.r.t `o1`
@inline α(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) = (r1 + r2 + ϵ)/norm(o1 - o2)

function pack(
        r::AbstractVector{T},
        x0 = randn(T, 2*length(r)),
        ϵ = zero(T);
        m = MinimallyTangent(ϵ, r),
        Alg = Optim.LBFGS(
            manifold = m,
            linesearch = LineSearches.BackTracking(order=3)),
        Opts = Optim.Options(
            iterations = 100_000,
            x_tol = 1e-8 * minimum(r),
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
    # x_init = copy(x0)
    # x_init = Optim.retract!(m, x_init) # Valid initial starting point
    # x_init .*= 2.0
    opt_obj = OnceDifferentiable(f, g!, fg!, x0)

    # Optimize and get results
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    return result
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

# # Gradient of squared distance function is scale invariant, so just return g
# Optim.project_tangent!(m::MinimallyTangent, g, x) = g

# The gradient with respect to each circle origin will be projected onto the
# average tangent direction with respect to all nearby circles, weighted by
# 1/(d^2 + λ), where d is the signed circle distance, and λ > 0 serves to both
# to avoid singularities, and handle multiple tangent vectors.
function Optim.project_tangent!(
        m::MinimallyTangent,
        g::AbstractVector{T},
        x::AbstractVector{T}) where {T}

    # Check inputs and initialize
    @assert length(g) == length(x) == 2*length(m.r) >= 4
    G = reinterpret(Vec{2,T}, g)
    o = reinterpret(Vec{2,T}, x)
    N = length(G)
    λ = √eps(T)

    @inbounds for i in 1:N
        t = zero(Vec{2,T})
        Σw = zero(T)
        for j in 1:i-1
            dt = o[j] - o[i] # difference vector
            dt /= norm(dt) # normalized distance vector
            w = 1/(d(o[i], o[j], m.r[i], m.r[j], m.ϵ)^2 + λ) # weight
            t += w * dt
            Σw += w
        end
        for j in i+1:N
            dt = o[j] - o[i] # difference vector
            dt /= norm(dt) # normalized distance vector
            w = 1/(d(o[i], o[j], m.r[i], m.r[j], m.ϵ)^2 + λ) # weight
            t += w * dt
            Σw += w
        end
        t /= norm(t) # normalized weighted difference
        # W = tanh(2*Σw*λ) # W approaches 1 when Σw > 1/λ, 0 when Σw << 1/λ
        W = one(T)
        G[i] -= W * dot(t, G[i]) * t # project out component parallel to t with weight W
    end

    return g
end



# struct AverageTangent{T} <: Manifold
#     ϵ::T # minimum distance
#     r::AbstractVector{T} # circle radii
# end
#
# # Compute scale factor `α` such that the minimum distance between circles is m.ϵ
# # and scale `x` accordingly
# function Optim.retract!(m::AverageTangent, x::AbstractVector{T}) where {T}
#     @assert length(x) == 2*length(m.r) >= 4
#     o = reinterpret(Vec{2,T}, x)
#     N = length(o)
#
#     # Pull first iteration outside of the loop for easy initialization
#     @inbounds alpha = α(o[1], o[2], m.r[1], m.r[2], m.ϵ)
#     for j = 3:N
#         for i in 1:j-1
#             @inbounds alpha += α(o[i], o[j], m.r[i], m.r[j], m.ϵ)
#         end
#     end
#     alpha /= div(N*(N-1), 2) # average alpha
#
#     # Scale x and return
#     return x .*= alpha
# end
#
# # Gradient of squared distance function is scale invariant, so just return g
# Optim.project_tangent!(m::AverageTangent, g, x) = g


# Symmetric pairwise sum of the four argument function `f` where the first two
# arguments are Vec{2,T}'s and the second two are scalars. The output is the sum
# over i<j of f(o[i], o[j], r[i], r[j]) where o[i] is the Vec{2,T} formed from
# x[2i-1:2i].
# `f` is symmetric in both the first two arguments and the second two, i.e.
# f(o1,o2,r1,r2) == f(o2,o1,r1,r2) and f(o1,o2,r1,r2) == f(o1,o2,r2,r1).
function pairwise_sum(f,
                      x::AbstractVector{T},
                      r::AbstractVector) where {T}
    @assert length(x) == 2*length(r) >= 4
    o = reinterpret(Vec{2,T}, x)

    # Pull first iteration outside of the loop for easy initialization
    Σ = f(o[1], o[2], r[1], r[2])
    @inbounds for j = 3:length(o)
        for i in 1:j-1
            Σ += f(o[i], o[j], r[i], r[j])
        end
    end

    return Σ
end

# Gradient of the pairwise_sum function w.r.t. `x`. Result is stored in `g`.
function pairwise_grad!(g::AbstractVector{T},
                        ∇f,
                        x::AbstractVector{T},
                        r::AbstractVector) where {T}
    @assert length(g) == length(x) == 2*length(r) >= 4
    fill!(g, zero(T))
    G = reinterpret(Vec{2,T}, g)
    o = reinterpret(Vec{2,T}, x)

    @inbounds for i in 1:length(o)
        for j in 1:i-1
            G[i] += ∇f(o[i], o[j], r[i], r[j])
        end
        for j in i+1:length(o)
            G[i] += ∇f(o[i], o[j], r[i], r[j])
        end
    end

    return g
end

end # module ManifoldCirclePacking

nothing
