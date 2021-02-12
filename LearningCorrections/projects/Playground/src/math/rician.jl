####
#### Rician Distribution
####

#### Rician distribution: https://en.wikipedia.org/wiki/Rice_distribution
struct Rician{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    ν::T
    σ::T
end

#### Outer constructors
@inline Rician(ν::Real, σ::Real) = Rician(promote(ν, σ)...)
@inline Rician(ν::Integer, σ::Integer) = Rician(float(ν), float(σ))
@inline Rician(ν::Real) = Rician(ν, one(typeof(ν)))
@inline Rician() = Rician(0.0, 1.0)

#### Conversions
@inline Base.convert(::Type{Rician{T}}, ν::S, σ::S) where {T <: Real, S <: Real} = Rician(T(ν), T(σ))
@inline Base.convert(::Type{Rician{T}}, d::Rician{S}) where {T <: Real, S <: Real} = Rician(T(d.ν), T(d.σ))

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
@inline_cufunc _besselix(ν::Real, x::Real) = besselix(oftypefloat(x, ν), float(x)) # besselix(ν, x) = Iν(x) * exp(-|x|)
@inline_cufunc _besselix0(x::Real) = _besselix(0, x)
@inline_cufunc _besselix1(x::Real) = _besselix(1, x)
@inline_cufunc _besselix2(x::Real) = _besselix(2, x)
@inline_cufunc _logbesseli(ν::Real, x::Real) = log(_besselix(ν, x)) + abs(x) # since log(besselix(ν, x)) = log(Iν(x)) - |x|
@inline_cufunc _logbesseli0(x::Real) = _logbesseli(0, x)
@inline_cufunc _logbesseli1(x::Real) = _logbesseli(1, x)
@inline_cufunc _logbesseli2(x::Real) = _logbesseli(2, x)
@inline_cufunc _besseli1i0m1(x::Real) = _besselix1(x) / _besselix0(x) - 1
@inline_cufunc _besseli2i0(x::Real) = _besselix2(x) / _besselix0(x)
@inline_cufunc _laguerre½(x::Real) = ifelse(x < 0, one(x), exp(x)) * ((1 - x) * _besselix0(-x/2) - x * _besselix1(-x/2)) # besselix(ν, ±x/2) = Iν(±x/2) * exp(-|±x/2|) = Iν(-x/2) * exp(∓x/2)

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

#### Sampling
@inline Distributions.rand(rng::Random.AbstractRNG, d::Rician{T}) where {T} = sqrt((d.ν + d.σ * randn(rng, T))^2 + (d.σ * randn(rng, T))^2)

#### CUDA-friendly native julia Besselix functions

"Approximation of besselix(0, x) = exp(-|x|) * besseli(0, x)"
function _besselix0_cuda_unsafe end

"Approximation of besselix(1, x) = exp(-|x|) * besseli(1, x)"
function _besselix1_cuda_unsafe end

@inline function _besselix0_cuda_unsafe(x)
    ax = abs(x)
    y1 = (ax / 3.75f0)^2
    y1 = exp(-ax) * evalpoly(y1, (1.0f0, 3.5156229f0, 3.0899424f0, 1.2067492f0, 0.2659732f0, 0.360768f-1, 0.45813f-2))
    y2 = 3.75f0 / ax
    y2 = evalpoly(y2, (0.39894228f0, 0.1328592f-1, 0.225319f-2, -0.157565f-2, 0.916281f-2, -0.2057706f-1, 0.2635537f-1, -0.1647633f-1, 0.392377f-2))
    y2 /= sqrt(ax)
    return LoopVectorization.VectorizationBase.vifelse(ax < 3.75f0, y1, y2)
end

@cufunc function _besselix0_cuda_unsafe(x)
    ax = abs(x)
    if ax < 3.75f0
        y = (ax / 3.75f0)^2
        y = exp(-ax) * evalpoly(y, (1.0f0, 3.5156229f0, 3.0899424f0, 1.2067492f0, 0.2659732f0, 0.360768f-1, 0.45813f-2))
        # y = exp(-ax) * evalpoly(y, (1.0000000f+0, 3.5156250f+0, 3.0899048f+0, 1.2069941f+0, 2.6520866f-1, 3.7294969f-2, 3.6420866f-3)) # derived from taylor series directly, but is less accurate; why the mismatch?
    else
        y = 3.75f0 / ax
        y = evalpoly(y, (0.39894228f0, 0.1328592f-1, 0.225319f-2, -0.157565f-2, 0.916281f-2, -0.2057706f-1, 0.2635537f-1, -0.1647633f-1, 0.392377f-2))
        y /= sqrt(ax)
    end
    return y
end

@inline function _besselix1_cuda_unsafe(x)
    ax = abs(x)
    y1 = (ax / 3.75f0)^2
    y1 = exp(-ax) * ax * evalpoly(y1, (0.5f0, 0.87890594f0, 0.51498869f0, 0.15084934f0, 0.2658733f-1, 0.301532f-2, 0.32411f-3))
    y2 = 3.75f0 / ax
    y2 = evalpoly(y2, (0.39894228f0, -0.3988024f-1, -0.362018f-2, 0.163801f-2, -0.1031555f-1, 0.2282967f-1, -0.2895312f-1, 0.1787654f-1, -0.420059f-2))
    y2 /= sqrt(ax)
    y = LoopVectorization.VectorizationBase.vifelse(ax < 3.75f0, y1, y2)
    return LoopVectorization.VectorizationBase.vifelse(x < 0, -y, y)
end

@cufunc function _besselix1_cuda_unsafe(x)
    ax = abs(x)
    if ax < 3.75f0
        y = (ax / 3.75f0)^2
        y = exp(-ax) * ax * evalpoly(y, (0.5f0, 0.87890594f0, 0.51498869f0, 0.15084934f0, 0.2658733f-1, 0.301532f-2, 0.32411f-3))
        # y = exp(-ax) * ax * evalpoly(y, (5.0000000f-1, 8.7890625f-1, 5.1498413f-1, 1.5087426f-1, 2.6520865f-2, 3.1079140f-3, 2.6014904f-4)) # derived from taylor series directly, but is less accurate; why the mismatch?
    else
        y = 3.75f0 / ax
        y = evalpoly(y, (0.39894228f0, -0.3988024f-1, -0.362018f-2, 0.163801f-2, -0.1031555f-1, 0.2282967f-1, -0.2895312f-1, 0.1787654f-1, -0.420059f-2))
        y /= sqrt(ax)
    end
    return x < 0 ? -y : y
end

@inline_cufunc _logbesseli0_cuda_unsafe(x) = log(_besselix0_cuda_unsafe(x)) + abs(x) # since log(besselix(ν, x)) = log(Iν(x)) - |x|
@inline_cufunc _logbesseli1_cuda_unsafe(x) = log(_besselix1_cuda_unsafe(x)) + abs(x) # since log(besselix(ν, x)) = log(Iν(x)) - |x|
@inline_cufunc _laguerre½_cuda_unsafe(x) = ifelse(x < 0, one(x), exp(x)) * ((1 - x) * _besselix0_cuda_unsafe(-x/2) - x * _besselix1_cuda_unsafe(-x/2))

#### ChainRules and ForwardDiff

@inline_cufunc ∂x_besselix(ν::Real, x::Real, Ω::Real) = (_besselix(ν - 1, x) + _besselix(ν + 1, x)) / 2 - sign(x) * Ω
ChainRules.@scalar_rule _besselix(ν::Real, x::Real) (ChainRules.DoesNotExist(), ∂x_besselix(ν, x, Ω))

@inline_cufunc ∂x_besselix0(x::Real, Ω::Real) = _besselix1(x) - sign(x) * Ω
@inline_cufunc f_∂x_besselix0(x::Real) = (Ω = _besselix0(x); return (Ω, ∂x_besselix0(x, Ω)))
ChainRules.@scalar_rule _besselix0(x::Real) ∂x_besselix0(x, Ω)
@primal_dual_gradient _besselix0 f_∂x_besselix0

@inline_cufunc ∂x_besselix1(x::Real, Ω::Real) = (_besselix0(x) + _besselix2(x))/2 - sign(x) * Ω
@inline_cufunc f_∂x_besselix1(x::Real) = (Ω = _besselix1(x); return (Ω, ∂x_besselix1(x, Ω)))
ChainRules.@scalar_rule _besselix1(x::Real) ∂x_besselix1(x, Ω)
@primal_dual_gradient _besselix1 f_∂x_besselix1

@inline_cufunc ∂x_besselix0_cuda_unsafe(x::Real, Ω::Real) = _besselix1_cuda_unsafe(x) - sign(x) * Ω
@inline_cufunc f_∂x_besselix0_cuda_unsafe(x::Real) = (Ω = _besselix0_cuda_unsafe(x); return (Ω, ∂x_besselix0_cuda_unsafe(x, Ω)))
ChainRules.@scalar_rule _besselix0_cuda_unsafe(x::Real) ∂x_besselix0_cuda_unsafe(x, Ω)
@primal_dual_gradient _besselix0_cuda_unsafe f_∂x_besselix0_cuda_unsafe

@inline_cufunc ∂x_laguerre½_cuda_unsafe(x::Real) = ifelse(x < 0, one(x), exp(x)) * (_besselix1_cuda_unsafe(x/2) - _besselix0_cuda_unsafe(x/2)) / 2
ChainRules.@scalar_rule _laguerre½_cuda_unsafe(x::Real) ∂x_laguerre½_cuda_unsafe(x)
@dual_gradient _laguerre½_cuda_unsafe ∂x_laguerre½_cuda_unsafe

#### Misc. derivatives

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

@inline function ∂²logpdf_∂ν²(d::Rician, x::Real)
    σ, ν = d.σ, d.ν
    σ2 = σ^2
    z = x*ν/σ2
    B = _besseli1i0m1(z)
    B′ = (1 + _besseli2i0(z))/2 - (B + 1)^2
    return (x^2 * B′ - σ2) / σ2^2
end
