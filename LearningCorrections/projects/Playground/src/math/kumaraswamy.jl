####
#### Kumaraswamy Distribution
####

#### Kumaraswamy distribution: https://en.wikipedia.org/wiki/Rice_distribution
struct Kumaraswamy{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    a::T
    b::T
end

#### Outer constructors
@inline Kumaraswamy(a::Real, b::Real) = Kumaraswamy(promote(a, b)...)
@inline Kumaraswamy(a::Integer, b::Integer) = Kumaraswamy(float(a), float(b))
@inline Kumaraswamy(a::Real) = Kumaraswamy(a, one(typeof(a)))
@inline Kumaraswamy() = Kumaraswamy(1.0, 1.0)

#### Conversions
@inline Base.convert(::Type{Kumaraswamy{T}}, a::S, b::S) where {T <: Real, S <: Real} = Kumaraswamy(T(a), T(b))
@inline Base.convert(::Type{Kumaraswamy{T}}, d::Kumaraswamy{S}) where {T <: Real, S <: Real} = Kumaraswamy(T(d.a), T(d.b))

# Distributions.@distr_support Kumaraswamy 0 1
@inline Base.minimum(d::Union{Kumaraswamy, Type{Kumaraswamy}}) = 0
@inline Base.maximum(d::Union{Kumaraswamy, Type{Kumaraswamy}}) = 1

#### Parameters
@inline Distributions.params(d::Kumaraswamy) = (d.a, d.b)
@inline Distributions.partype(::Kumaraswamy{T}) where {T<:Real} = T

#TODO @inline Distributions.location(d::Kumaraswamy) = ...
#TODO @inline Distributions.scale(d::Kumaraswamy) = ...

@inline Base.eltype(::Type{Kumaraswamy{T}}) where {T} = T

#### Statistics
@inline Distributions.mean(d::Kumaraswamy) = d.b * SpecialFunctions.beta(1 + inv(d.a), d.b)
@inline Distributions.mode(d::Kumaraswamy) = ((d.a - 1) / (d.a * d.b - 1)) ^ inv(d.a)
@inline Distributions.median(d::Kumaraswamy{T}) where {T} = (1 - T(2)^(-inv(d.b))) ^ inv(d.a)

@inline Distributions.var(d::Kumaraswamy) = d.b * SpecialFunctions.beta(1 + 2/d.a, d.b) - mean(d)^2
@inline Distributions.std(d::Kumaraswamy) = sqrt(var(d))
# @inline Distributions.skewness(d::Kumaraswamy{T}) where {T<:Real} = ...
# @inline Distributions.kurtosis(d::Kumaraswamy{T}) where {T<:Real} = ...
# @inline Distributions.entropy(d::Kumaraswamy) = ...

#### Evaluation
@inline Distributions.logpdf(d::Kumaraswamy, x::Real) = -log(d.a) - log(d.b) - (d.a-1)*log(x) - (d.b-1)*log(1-x^d.a)
@inline Distributions.pdf(d::Kumaraswamy, x::Real) = d.a * d.b * x^(d.a-1) * (1 - x^d.a)^(d.b-1)

#### Sampling
@inline Distributions.rand(rng::Random.AbstractRNG, d::Kumaraswamy{T}) where {T} = (1 - rand(rng, T) ^ inv(d.b)) ^ inv(d.a)
