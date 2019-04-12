# ---------------------------------------------------------------------------- #
# ParabolicLinearMap methods: LinearMap's for solving M*du/dt = K*u ODE systems
# ---------------------------------------------------------------------------- #

# Properties
Base.size(A::ParabolicLinearMap) = size(A.K)
LinearAlgebra.issymmetric(A::ParabolicLinearMap) = false
LinearAlgebra.ishermitian(A::ParabolicLinearMap) = false
LinearAlgebra.isposdef(A::ParabolicLinearMap) = false

# AdjointMap and TransposeMap wrapper definitions (shouldn't need these definitions since ParabolicLinearMap <: LinearMap)
# LinearAlgebra.adjoint(A::ParabolicLinearMap) = LinearMaps.AdjointMap(A)
# LinearAlgebra.transpose(A::ParabolicLinearMap) = LinearMaps.TransposeMap(A)

# LinearMaps doesn't define A_mul_B etc. for abstract LinearMap's for some reason... Need to explicitly define them here
LinearMaps.A_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector)  = mul!(y, A, x)
LinearMaps.A_mul_B!(y::AbstractMatrix, A::ParabolicLinearMap, x::AbstractMatrix)  = mul!(y, A, x)
LinearMaps.At_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector) = mul!(y, transpose(A), x)
LinearMaps.At_mul_B!(y::AbstractMatrix, A::ParabolicLinearMap, x::AbstractMatrix) = mul!(y, transpose(A), x)
LinearMaps.Ac_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector) = mul!(y, adjoint(A), x)
LinearMaps.Ac_mul_B!(y::AbstractMatrix, A::ParabolicLinearMap, x::AbstractMatrix) = mul!(y, adjoint(A), x)

# LinearAlgebra.ldiv! is not defined for SuiteSparse.CHOLMOD.Factor; define in terms of \ for simplicity
LinearAlgebra.ldiv!(y::AbstractVecOrMat, A::SuiteSparse.CHOLMOD.Factor, x::AbstractVecOrMat) = copyto!(y, A\x)
LinearAlgebra.ldiv!(A::SuiteSparse.CHOLMOD.Factor, x::AbstractVecOrMat) = copyto!(x, A\x)

# Multiplication action
Minv_K_mul_u!(Y, X, K, Mfact) = (mul!(Y, K, X); ldiv!(Mfact, Y); return Y)
Kt_Minv_mul_u!(Y, X, K, Mfact) = (mul!(Y, transpose(K), Mfact\X); return Y)
Kc_Minv_mul_u!(Y, X, K, Mfact) = (mul!(Y, adjoint(K), Mfact\X); return Y)

# Multiplication with Vector or Matrix
LinearAlgebra.mul!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
LinearAlgebra.mul!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
LinearAlgebra.mul!(Y::AbstractVector, A::LinearMaps.TransposeMap{T, <:ParabolicLinearMap{T}}, X::AbstractVector) where {T} = Kt_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
LinearAlgebra.mul!(Y::AbstractMatrix, A::LinearMaps.TransposeMap{T, <:ParabolicLinearMap{T}}, X::AbstractMatrix) where {T} = Kt_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
LinearAlgebra.mul!(Y::AbstractVector, A::LinearMaps.AdjointMap{T, <:ParabolicLinearMap{T}}, X::AbstractVector) where {T} = Kc_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
LinearAlgebra.mul!(Y::AbstractMatrix, A::LinearMaps.AdjointMap{T, <:ParabolicLinearMap{T}}, X::AbstractMatrix) where {T} = Kc_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)

function Base.show(io::IO, d::ParabolicLinearMap)
    compact = get(io, :compact, false)
    if compact
        print(io, size(d,1), "×", size(d,2), " ", typeof(d))
    else
        print(io, "$(typeof(d)) with:")
        print(io, "\n     M: "); _compact_show_sparse(io, d.M)
        print(io, "\n Mfact: "); _compact_show_factorization(io, d.Mfact)
        print(io, "\n     K: "); _compact_show_sparse(io, d.K)
    end
end

# ---------------------------------------------------------------------------- #
# LinearMap methods: helper functions for LinearMap's
# ---------------------------------------------------------------------------- #

# For taking literal powers of LinearMaps, e.g. A^2
Base.to_power_type(A::Union{<:LinearMaps.AdjointMap, <:LinearMaps.TransposeMap, <:LinearMap}) = A

# For making FunctionMap's callable (for direct use within ODEProblem's)
(A::LinearMaps.FunctionMap{T,F1,F2})(du,u) where {T,F1,F2} = mul!(du,A,u)
(A::LinearMaps.FunctionMap{T,F1,F2})(du,u,p) where {T,F1,F2} = mul!(du,A,u)
(A::LinearMaps.FunctionMap{T,F1,F2})(du,u,p,t) where {T,F1,F2} = mul!(du,A,u)

function LinearAlgebra.tr(A::LinearMap{T}, t::Int = 10) where {T}
    # Approximate trace using mat-vec's with basis vectors
    N = size(A, 2)
    t = min(t, N)
    x = zeros(T, N)
    y = similar(x)
    est = zero(T)
    for ix in StatsBase.sample(1:N, t; replace = false)
        x[ix] = one(T)
        y = mul!(y, A, x)
        est += y[ix]
        x[ix] = zero(T)
    end
    return N * (est/t)
end

function expmv_opnorm(A, p)
    if p == 1
        return ExpmV.norm1est(A, 1)
    elseif p == Inf
        return ExpmV.norm1est(A', 1)
    else
        error("p=1 or p=Inf required; got p=$p")
    end
end

# Default to p = 2 for consistency with Base, even though it would throw an error
LinearAlgebra.opnorm(A::LinearMap, p::Real = 2) = expmv_opnorm(A, p)
# LinearAlgebra.opnorm(A::LinearMap, p::Real = 2, t::Int = 10) = normest1_norm(A, p, t)
# LinearAlgebra.norm(A::LinearMap, p::Real = 2, t::Int = 10) = normest1_norm(A, p, t)

# ---------------------------------------------------------------------------- #
# DiffEqParabolicLinearMapWrapper methods: Effectively a simplified LinearMap,
# but subtypes AbstractMatrix so that it can be passed to DiffEq* solvers
# ---------------------------------------------------------------------------- #

struct DiffEqParabolicLinearMapWrapper{T,Atype} <: AbstractMatrix{T}
    A::Atype
    DiffEqParabolicLinearMapWrapper(A::Atype) where {Atype} = new{eltype(A), Atype}(A)
end
LinearAlgebra.adjoint(A::DiffEqParabolicLinearMapWrapper) = DiffEqParabolicLinearMapWrapper(A.A')
LinearAlgebra.transpose(A::DiffEqParabolicLinearMapWrapper) = DiffEqParabolicLinearMapWrapper(transpose(A.A))

# NOTE: purposefully not forwarding getindex; wrapper type should behave like a LinearMap
Lazy.@forward DiffEqParabolicLinearMapWrapper.A (Base.size, LinearAlgebra.tr, LinearAlgebra.issymmetric, LinearAlgebra.ishermitian, LinearAlgebra.isposdef)

# TODO: Lazy.@forward gives ambiguity related errors when trying to forward these methods: mul!, norm, opnorm
LinearAlgebra.mul!(Y::AbstractVector, A::DiffEqParabolicLinearMapWrapper, X::AbstractVector) = mul!(Y, A.A, X)
LinearAlgebra.mul!(Y::AbstractMatrix, A::DiffEqParabolicLinearMapWrapper, X::AbstractMatrix) = mul!(Y, A.A, X)
LinearAlgebra.opnorm(A::DiffEqParabolicLinearMapWrapper, p::Real) = expmv_opnorm(A.A, p)
# LinearAlgebra.opnorm(A::DiffEqParabolicLinearMapWrapper, p::Real, t::Int = 10) = normest1_norm(A.A, p, t)
# LinearAlgebra.norm(A::DiffEqParabolicLinearMapWrapper, p::Real, t::Int = 10) = normest1_norm(A.A, p, t)

Base.show(io::IO, A::DiffEqParabolicLinearMapWrapper) = print(io, "$(typeof(A))")
Base.show(io::IO, ::MIME"text/plain", A::DiffEqParabolicLinearMapWrapper) = print(io, "$(size(A,1)) × $(size(A,2)) $(typeof(A))")
