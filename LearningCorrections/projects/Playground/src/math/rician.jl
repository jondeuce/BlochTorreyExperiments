####
#### Rician Distribution
####

#### Rician distribution: https://en.wikipedia.org/wiki/Rice_distribution
struct Rician{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    ν::T
    σ::T
    Rician{T}(ν::T, σ::T) where {T<:Real} = new{T}(ν, σ)
end

@inline function Rician(ν::T, σ::T; check_args = true) where {T <: Real}
    check_args && Distributions.@check_args(Rician, σ >= zero(σ) && ν >= zero(ν))
    return Rician{T}(ν, σ)
end

#### Outer constructors
@inline Rician(ν::Real, σ::Real) = Rician(promote(ν, σ)...)
@inline Rician(ν::Integer, σ::Integer) = Rician(float(ν), float(σ))
@inline Rician(ν::T) where {T <: Real} = Rician(ν, one(T))
@inline Rician() = Rician(0.0, 1.0, check_args = false)

#### Conversions
@inline Base.convert(::Type{Rician{T}}, ν::S, σ::S) where {T <: Real, S <: Real} = Rician(T(ν), T(σ))
@inline Base.convert(::Type{Rician{T}}, d::Rician{S}) where {T <: Real, S <: Real} = Rician(T(d.ν), T(d.σ), check_args = false)

# Distributions.@distr_support Rician 0 Inf
@inline Base.minimum(d::Union{Rician, Type{Rician}}) = 0
@inline Base.maximum(d::Union{Rician, Type{Rician}}) = Inf

#### Parameters
@inline Distributions.params(d::Rician) = (d.ν, d.σ)
@inline Distributions.partype(d::Rician{T}) where {T<:Real} = T

@inline Distributions.location(d::Rician) = d.ν
@inline Distributions.scale(d::Rician) = d.σ

@inline Base.eltype(::Type{Rician{T}}) where {T} = T

#### Laguerre/Bessel functions
_besselix(ν::Real, x::Real) = besselix(oftype(float(x), ν), float(x)) # besselix(ν, x) = Iν(x) * exp(-|x|)
_logbesseli(ν::Real, x::Real) = log(_besselix(ν, x)) + abs(x) # since log(besselix(ν, x)) = log(Iν(x)) - |x|
_logbesseli0(x::Real) = _logbesseli(0, x)
_logbesseli1(x::Real) = _logbesseli(1, x)
_logbesseli2(x::Real) = _logbesseli(2, x)
_besseli1i0m1(x::Real) = _besselix(1, x) / _besselix(0, x) - 1
_besseli2i0(x::Real) = _besselix(2, x) / _besselix(0, x)
_laguerre½(x::Real) = (x < 0 ? one(x) : exp(x)) * ((1 - x) * _besselix(0, -x/2) - x * _besselix(1, -x/2)) # besselix(ν, ±x/2) = Iν(±x/2) * exp(-|±x/2|) = Iν(-x/2) * exp(∓x/2)

#### Statistics
@inline Distributions.mean(d::Rician) = d.σ * sqrt(pi/2) * _laguerre½(-d.ν^2 / 2d.σ^2)
# @inline Distributions.mode(d::Rician) = ?
# @inline Distributions.median(d::Rician) = ?

@inline Distributions.var(d::Rician) = 2 * d.σ^2 + d.ν^2 - pi * d.σ^2 * _laguerre½(-d.ν^2 / 2d.σ^2)^2 / 2
@inline Distributions.std(d::Rician) = sqrt(var(d))
# @inline Distributions.skewness(d::Rician{T}) where {T<:Real} = ?
# @inline Distributions.kurtosis(d::Rician{T}) where {T<:Real} = ?
# @inline Distributions.entropy(d::Rician) = ?

#### Evaluation
@inline _rician_logpdf(x::Real, ν::Real, σ::Real) = log(x/σ^2) + _logbesseli0(x*ν/σ^2) - (x*x + ν*ν)/2σ^2
@inline Distributions.logpdf(d::Rician, x::Real) = _rician_logpdf(x, d.ν, d.σ)
@inline Distributions.pdf(d::Rician, x::Real) = exp(logpdf(d, x)) # below version errors for large x (besseli throws); otherwise is consistent
# @inline Distributions.pdf(d::Rician, x::Real) = x * besseli(0, x * d.ν / d.σ^2) * exp(-(x^2 + d.ν^2) / (2*d.σ^2)) / d.σ^2

#### Gradient
@inline function _∇logpdf(d::Rician, x::Real)
    σ, ν = d.σ, d.ν
    σ2 = σ^2
    B = _besseli1i0m1(x*ν/σ2)
    return (σ, ν, σ2, B)
end
@inline function ∇logpdf(d::Rician, x::Real)
    σ, ν, σ2, B = _∇logpdf(d, x)
    dνx, σ3 = ν-x, σ*σ2
    ∇ν = (B*x - dνx) / σ2
    ∇σ = (dνx^2 - 2*(σ2 + ν*B*x)) / σ3
    return (∇ν, ∇σ)
end
@inline function ∂logpdf_∂ν(d::Rician, x::Real)
    σ, ν, σ2, B = _∇logpdf(d, x)
    return (B*x - (ν-x)) / σ2
end
@inline function ∂logpdf_∂σ(d::Rician, x::Real)
    σ, ν, σ2, B = _∇logpdf(d, x)
    dνx, σ3 = ν-x, σ*σ2
    return ((ν-x)^2 - 2*(σ2 + ν*B*x)) / σ3
end

#### Hessian
@inline function ∂²logpdf_∂ν²(d::Rician, x::Real)
    σ, ν = d.σ, d.ν
    σ2 = σ^2
    z = x*ν/σ2
    B = _besseli1i0m1(z)
    B′ = (1 + _besseli2i0(z))/2 - (B + 1)^2
    return (x^2 * B′ - σ2) / σ2^2
end

#### Sampling
@inline Distributions.rand(rng::Random.AbstractRNG, d::Rician{T}) where {T} = sqrt((d.ν + d.σ * randn(rng, T))^2 + (d.σ * randn(rng, T))^2)

#### ChainRules

# Define chain rule (Zygote > v"0.4.20" reads ChainRules directly)
ChainRules.@scalar_rule(
    _besselix(ν::Real, x::Real),
    (ChainRules.DoesNotExist(), (_besselix(ν - 1, x) + _besselix(ν + 1, x)) / 2 - sign(x) * Ω),
)

# Define corresponding dual rule
function _besselix(ν::Real, x::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    y, ẏ = ChainRules.frule((ChainRules.NO_FIELDS, ChainRules.DoesNotExist(), 1), _besselix, ν, ForwardDiff.value(x))
    return ForwardDiff.Dual{T,V,N}(y, ẏ * ForwardDiff.partials(x))
end

#### CUDA-friendly native julia Besselix functions

"Approximation of besselix(0, x) = exp(-|x|) * besseli(0, x)"
@inline function _besselix0_cuda_unsafe(x)
    ax = abs(x)
    if ax < 3.75f0
        y = (ax / 3.75f0)^2
        out = exp(-ax) * evalpoly(y, (1.0f0, 3.5156229f0, 3.0899424f0, 1.2067492f0, 0.2659732f0, 0.360768f-1, 0.45813f-2))
        # out = exp(-ax) * evalpoly(y, (1.0000000f+0, 3.5156250f+0, 3.0899048f+0, 1.2069941f+0, 2.6520866f-1, 3.7294969f-2, 3.6420866f-3)) # derived from taylor series directly, but is less accurate; why the mismatch?
    else
        y = 3.75f0 / ax
        out = evalpoly(y, (0.39894228f0, 0.1328592f-1, 0.225319f-2, -0.157565f-2, 0.916281f-2, -0.2057706f-1, 0.2635537f-1, -0.1647633f-1, 0.392377f-2))
        out /= sqrt(ax)
    end
    return out
end
ChainRules.@scalar_rule(_besselix0_cuda_unsafe(x), _besselix1_cuda_unsafe(x) - sign(x) * Ω)

"Approximation of besselix(1, x) = exp(-|x|) * besseli(1, x)"
@inline function _besselix1_cuda_unsafe(x)
    ax = abs(x)
    if ax < 3.75f0
        y = (ax / 3.75f0)^2
        out = exp(-ax) * ax * evalpoly(y, (0.5f0, 0.87890594f0, 0.51498869f0, 0.15084934f0, 0.2658733f-1, 0.301532f-2, 0.32411f-3))
        # out = exp(-ax) * ax * evalpoly(y, (5.0000000f-1, 8.7890625f-1, 5.1498413f-1, 1.5087426f-1, 2.6520865f-2, 3.1079140f-3, 2.6014904f-4)) # derived from taylor series directly, but is less accurate; why the mismatch?
    else
        y = 3.75f0 / ax
        out = evalpoly(y, (0.39894228f0, -0.3988024f-1, -0.362018f-2, 0.163801f-2, -0.1031555f-1, 0.2282967f-1, -0.2895312f-1, 0.1787654f-1, -0.420059f-2))
        out /= sqrt(ax)
    end
    return x < 0 ? -out : out
end

@inline _logbesseli0_cuda_unsafe(x) = log(_besselix0_cuda_unsafe(x)) + abs(x) # since log(besselix(ν, x)) = log(Iν(x)) - |x|
@inline _logbesseli1_cuda_unsafe(x) = log(_besselix1_cuda_unsafe(x)) + abs(x) # since log(besselix(ν, x)) = log(Iν(x)) - |x|
@inline _laguerre½_cuda_unsafe(x) = (x < 0 ? one(x) : exp(x)) * ((1 - x) * _besselix0_cuda_unsafe(-x/2) - x * _besselix1_cuda_unsafe(-x/2))
@inline _rician_mean_cuda_unsafe(ν, σ) = sqrthalfπ * σ * _laguerre½_cuda_unsafe(-ν^2 / 2σ^2)
@inline _rician_logpdf_cuda_unsafe(x, ν, σ) = _logbesseli0_cuda_unsafe(x*ν/σ^2) - (x^2 + ν^2)/(2*σ^2) + log(x/σ^2)

# Define chain rule (Zygote > v"0.4.20" reads ChainRules directly)
ChainRules.@scalar_rule(
    _rician_logpdf_cuda_unsafe(x, ν, σ),
    @setup(c = inv(σ^2), z = c*x*ν, r = _besselix1_cuda_unsafe(z) / _besselix0_cuda_unsafe(z)),
    # (c * (ν * r - x) + (1/x), c * (x * r - ν), c * (-2 * ν * x * r + x^2 + ν^2) / σ - 2/σ), # assumes strictly positive values
    (c * (ν * r - x) + (1/x), c * (x * r - ν), c * ((x - ν)^2 + 2 * ν * x * (1 - r)) / σ - 2/σ), # assumes strictly positive values
)

@inline _rician_mean_cuda(ν, σ) = (ϵ = 1f-6; return _rician_mean_cuda_unsafe(abs(ν) + ϵ, abs(σ) + ϵ)) # abs(⋅)+ϵ instead of e.g. max(⋅,ϵ) to avoid completely dropping gradient when <ϵ
@inline _rician_logpdf_cuda(x, ν, σ) = (ϵ = 1f-6; return _rician_logpdf_cuda_unsafe(abs(x) + ϵ, abs(ν) + ϵ, abs(σ) + ϵ)) # abs(⋅)+ϵ instead of e.g. max(⋅,ϵ) to avoid completely dropping gradient when <ϵ
