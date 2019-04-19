# ---------------------------------------------------------------------------- #
# Symmetric functions on circles
#   Symmetric pairwise sum of the four argument function f(o1,o2,r1,r2) where
#   o1, o2 are Vec{2,T} representing circle origins and r1, r2 and circle radii.
#   The output is the sum over i<j of f(o[i], o[j], r[i], r[j]), where o[i] is
#   the Vec{2,T} formed from x[2i-1:2i].
#   Additionally, it is assumed that `f` is symmetric in both o1, o2 and r1, r2.
# ---------------------------------------------------------------------------- #

const Tensor2D = Tensor{2,2}
const SymTensor2D = SymmetricTensor{2,2}
const MaybeSymTensor2D{T} = Union{<:Tensor2D{T}, <:SymTensor2D{T}} where {T}

# Symmetric pairwise sum of the four argument function f(o1,o2,r1,r2)
function pairwise_sum(f,
                      x::AbstractVector{T},
                      r::AbstractVector) where {T}
    @assert length(x) == 2*length(r) >= 4
    o = reinterpret(Vec{2,T}, x)
    N = length(o)

    # Pull first iteration outside of the loop for easy initialization
    @inbounds Σ = f(o[1], o[2], r[1], r[2])
    @inbounds for i = 3:N
        oᵢ, rᵢ = o[i], r[i]
        for j in 1:i-1
            Σ += f(oᵢ, o[j], rᵢ, r[j])
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
    N = length(o)

    @inbounds for i in 1:N
        oᵢ, rᵢ = o[i], r[i]
        for j in 1:i-1
            G[i] += ∇f(oᵢ, o[j], rᵢ, r[j])
        end
        for j in i+1:N
            G[i] += ∇f(oᵢ, o[j], rᵢ, r[j])
        end
    end

    return g
end

# Hessian of the pairwise_sum function w.r.t. `x`. Result is stored in `H`.
function pairwise_hess!(H::AbstractMatrix{T},
                        ∇²f,
                        x::AbstractVector{T},
                        r::AbstractVector) where {T}
    @assert size(H,1) == size(H,2) == length(x) == 2*length(r) >= 4
    fill!(H, zero(T))
    o = reinterpret(Vec{2,T}, x)
    N = length(o)

    # Hoist l=k=1 outside of main loop to get htype for free
    oₖ, rₖ = o[1], r[1]
    @inbounds h = ∇²f(oₖ, o[2], rₖ, r[2])
    @inbounds for j in 3:N
        h += ∇²f(oₖ, o[j], rₖ, r[j])
    end
    unsafe_assignat!(H, h, 1, 1)
    htype = typeof(h)

    @inbounds for k in 2:N
        oₖ, rₖ, K = o[k], r[k], 2k-1

        # lower off-diagonal terms (l<k)
        for l in 1:k-1
            h = -∇²f(oₖ, o[l], rₖ, r[l]) # off-diags have minus sign
            unsafe_assignat!(H, h, K, 2l-1)
        end

        # diagonal terms (l==k): sum over j ≠ k
        h = zero(htype)
        for j in 1:k-1
            h += ∇²f(oₖ, o[j], rₖ, r[j])
        end
        for j in k+1:N
            h += ∇²f(oₖ, o[j], rₖ, r[j])
        end
        unsafe_assignat!(H, h, K, K)
    end

    # upper off-diagonal (l>k) terms
    unsafe_symmetrize_upper!(H)

    return H
end

function unsafe_symmetrize_upper!(H)
    @inbounds for j in 2:size(H,1), i in 1:j-1
        H[i,j] = H[j,i]
    end
    return H
end

@inline function unsafe_assignat!(
        H::AbstractMatrix{T},
        h::Tensor2D{T},
        i::Int,
        j::Int) where {T}
    @inbounds H[i,   j]   = h[1]
    @inbounds H[i+1, j]   = h[2]
    @inbounds H[i,   j+1] = h[3]
    @inbounds H[i+1, j+1] = h[4]
    return H
end

@inline function unsafe_assignat!(
        H::AbstractMatrix{T},
        h::SymTensor2D{T},
        i::Int,
        j::Int) where {T}
    @inbounds H[i,   j]   = h.data[1]
    @inbounds H[i+1, j]   = h.data[2]
    @inbounds H[i,   j+1] = h.data[2]
    @inbounds H[i+1, j+1] = h.data[3]
    return H
end

function unsafe_assignall!(H, h::MaybeSymTensor2D{T}) where {T}
    @inbounds for j in 1:2:size(H,2), i in 1:2:size(H,1)
        unsafe_assignat!(H,h,i,j)
    end
    return H
end

function Base.:-(J::UniformScaling{Bool}, A::SymmetricTensor{2,2})
    T, d = typeof(A), Tensors.get_data(A)
    @inbounds B = T((J - d[1], -d[2], J - d[3]))
    return B
end

function Base.:-(A::SymmetricTensor{2,2}, J::UniformScaling{Bool})
    T, d = typeof(A), Tensors.get_data(A)
    @inbounds B = T((d[1] - J, d[2], d[3] - J))
    return B
end

####
#### Circle edge distance function
####

# Symmetric circle distance function on the circles (o1,r1) and (o2,r2)
@inline d(dx::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = norm(dx) - r1 - r2 - ϵ
@inline d(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = d(o1 - o2, r1, r2, ϵ)

# Gradient w.r.t `o1`
@inline ∇d(dx::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = dx/norm(dx)
@inline ∇d(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = ∇d(o1 - o2, r1, r2, ϵ)

# Hessian w.r.t `o1`, i.e. ∂²d/∂o1², given by:
#   1/|o1-o2|*I - (o1-o2)*(o1-o2)'/|o1-o2|^3
# Or, writing e12 = o1 - o2 and d12 = |e12|, we have:
#   (I - e12 * e12')/d12
# NOTE: ∂²d/∂o1o2 == ∂²d/∂o2o1 == -∂²/∂o1²
@inline function ∇²d(dx::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1)))
    α = inv(norm(dx))
    h = α * (I - otimes(α * dx))
    return h
end
@inline ∇²d(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = ∇²d(o1 - o2, r1, r2, ϵ)

####
#### Squared circle edge distance function
####

# Symmetric squared circle distance function on the circles (o1,r1) and (o2,r2)
@inline d²(dx::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = d(dx,r1,r2,ϵ)^2
@inline d²(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = d²(o1 - o2, r1, r2, ϵ)

# Gradient w.r.t `o1`
@inline ∇d²(dx::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = 2 * d(dx,r1,r2,ϵ) * ∇d(dx,r1,r2,ϵ)
@inline ∇d²(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = ∇d²(o1 - o2, r1, r2, ϵ)

# Hessian w.r.t `o1`, i.e. ∂²d/∂o1²
@inline ∇²d²(dx::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = 2 * d(dx,r1,r2,ϵ) * ∇²d(dx,r1,r2,ϵ) + 2 * otimes(∇d(dx,r1,r2,ϵ))
@inline ∇²d²(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = ∇²d²(o1 - o2, r1, r2, ϵ)

####
#### Squared circle edge distance function which is zero if they aren't overlapping
####

# Symmetric squared circle distance function on the circles (o1,r1) and (o2,r2)
@inline function d²_overlap(dx::Vec{dim,T}, r1::Real, r2::Real, ϵ = zero(typeof(r1))) where {dim,T}
    _d = d(dx,r1,r2,ϵ)
    return _d >= zero(T) ? zero(T) : _d^2
end
@inline d²_overlap(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = d²_overlap(o1 - o2, r1, r2, ϵ)

# Gradient w.r.t `o1`
@inline function ∇d²_overlap(dx::Vec{dim,T}, r1::Real, r2::Real, ϵ = zero(typeof(r1))) where {dim,T}
    _d = d(dx,r1,r2,ϵ)
    return _d >= zero(T) ? zero(Vec{dim,T}) : 2 * _d * ∇d(dx,r1,r2,ϵ)
end
@inline ∇d²_overlap(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = ∇d²_overlap(o1 - o2, r1, r2, ϵ)

# Hessian w.r.t `o1`, i.e. ∂²d/∂o1²
@inline function ∇²d²_overlap(dx::Vec{dim,T}, r1::Real, r2::Real, ϵ = zero(typeof(r1))) where {dim,T}
    _d = d(dx,r1,r2,ϵ)
    return _d >= zero(T) ? zero(SymmetricTensor{2,dim,T}) : 2 * _d * ∇²d(dx,r1,r2,ϵ) + 2 * otimes(∇d(dx,r1,r2,ϵ))
end
@inline ∇²d²_overlap(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ = zero(typeof(r1))) = ∇²d²_overlap(o1 - o2, r1, r2, ϵ)

####
#### Exponential barrier function
####

# Exponential barrier which satisfies the following (where μ = r1 + r2 and Δ = |dx| - μ):
#   f(Δ = -1ϵ) = μ^2 * (μ/ϵ)^2 >> μ^2
#   f(Δ =  0ϵ) = μ^2
#   f(Δ = +1ϵ) = ϵ^2
@inline function barrier(dx::Vec, r1::Real, r2::Real, ϵ::Real)
    μ = r1 + r2
    α = -2 * log(ϵ/μ) / ϵ # decay rate
    Δ = d(dx,r1,r2,ϵ) + ϵ # signed edge distance: Δ = d + ϵ = |dx| - μ
    b = μ^2 * exp(-α * Δ) # exponential barrier
    # b = max(b, eps(typeof(b))) # avoid subnormals
    # b = min(b, inv(eps(typeof(b)))) # avoid overflow
    return b
end
@inline barrier(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ::Real) = barrier(o1 - o2, r1, r2, ϵ)

# Gradient w.r.t `o1`
@inline function ∇barrier(dx::Vec, r1::Real, r2::Real, ϵ::Real)
    μ = r1 + r2
    α = -2 * log(ϵ/μ) / ϵ
    return barrier(dx,r1,r2,ϵ) * (-α * ∇d(dx,r1,r2,ϵ))
end
@inline ∇barrier(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ::Real) = ∇barrier(o1 - o2, r1, r2, ϵ)

# Hessian w.r.t `o1`, i.e. ∂²d/∂o1²
@inline function ∇²barrier(dx::Vec, r1::Real, r2::Real, ϵ::Real)
    μ = r1 + r2
    α = -2 * log(ϵ/μ) / ϵ
    return barrier(dx,r1,r2,ϵ) * (α^2 * otimes(∇d(dx,r1,r2,ϵ)) - α * ∇²d(dx,r1,r2,ϵ))
end
@inline ∇²barrier(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ::Real) = ∇²barrier(o1 - o2, r1, r2, ϵ)

####
#### Softplus barrier function
####

# Softplus barrier which satisfies the following (where μ = r1 + r2 and Δ = |dx| - μ):
#   f(Δ = 0)   = μ^2
#   f(Δ = ϵ/2) = ϵ^2
#   f(Δ = ϵ)   ≈ 0
@inline function softplusbarrier(dx::Vec, r1::Real, r2::Real, ϵ::Real)
    μ = r1 + r2
    Δ = d(dx,r1,r2,ϵ) + ϵ # signed edge distance: Δ = d + ϵ = |dx| - μ
    A = ϵ^2 / logtwo
    α = (10 * μ^2 / (ϵ/2)) / A
    # z = -α * (Δ - ϵ/2)
    # b = z > invsoftplus(eps(inv(A))) ? A*softplus(z) : zero(z)
    b = A * softplus(-α * (Δ - ϵ/2))
    # b = max(b, eps(typeof(b)))
    return b
end
@inline softplusbarrier(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ::Real) = softplusbarrier(o1 - o2, r1, r2, ϵ)

# Gradient w.r.t `o1`
@inline function ∇softplusbarrier(dx::Vec, r1::Real, r2::Real, ϵ::Real)
    μ = r1 + r2
    Δ = d(dx,r1,r2,ϵ) + ϵ # signed edge distance: Δ = d + ϵ = |dx| - μ
    A = ϵ^2 / logtwo
    α = (10 * μ^2 / (ϵ/2)) / A
    ∂b = A * logistic(-α * (Δ - ϵ/2))
    # ∂b = max(∂b, eps(typeof(∂b)))
    return ∂b * (-α * ∇d(dx,r1,r2,ϵ))
end
@inline ∇softplusbarrier(o1::Vec, o2::Vec, r1::Real, r2::Real, ϵ::Real) = ∇softplusbarrier(o1 - o2, r1, r2, ϵ)
