####
#### Truncated Gaussian distribution
####

#### Truncated Gaussian distribution: https://en.wikipedia.org/wiki/Truncated_normal_distribution
struct TruncatedGaussian{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    σ::T
    a::T
    b::T
end
