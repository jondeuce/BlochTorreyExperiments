####
#### Rician Distribution
####

#### Rician distribution: https://en.wikipedia.org/wiki/Rice_distribution
struct Rician{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    ν::T
    σ::T
    Rician{T}(ν::T, σ::T) where {T<:Real} = new{T}(ν, σ)
end

function Rician(ν::T, σ::T; check_args = true) where {T <: Real}
    check_args && Distributions.@check_args(Rician, σ >= zero(σ) && ν >= zero(ν))
    return Rician{T}(ν, σ)
end

#### Outer constructors
Rician(ν::Real, σ::Real) = Rician(promote(ν, σ)...)
Rician(ν::Integer, σ::Integer) = Rician(float(ν), float(σ))
Rician(ν::T) where {T <: Real} = Rician(ν, one(T))
Rician() = Rician(0.0, 1.0, check_args = false)

#### Conversions
Base.convert(::Type{Rician{T}}, ν::S, σ::S) where {T <: Real, S <: Real} = Rician(T(ν), T(σ))
Base.convert(::Type{Rician{T}}, d::Rician{S}) where {T <: Real, S <: Real} = Rician(T(d.ν), T(d.σ), check_args = false)

# Distributions.@distr_support Rician 0 Inf
Base.minimum(d::Union{Rician, Type{Rician}}) = 0
Base.maximum(d::Union{Rician, Type{Rician}}) = Inf

#### Parameters
Distributions.params(d::Rician) = (d.ν, d.σ)
@inline Distributions.partype(d::Rician{T}) where {T<:Real} = T

Distributions.location(d::Rician) = d.ν
Distributions.scale(d::Rician) = d.σ

Base.eltype(::Type{Rician{T}}) where {T} = T

#### Bessel function
_L½_bessel_kernel(x) = exp(x/2) * ((1-x) * besseli(0, -x/2) - x * besseli(1, -x/2))
_L½_series_kernel(x) = sqrt(-x/pi) * (256 - 64/x + 8/x^2 - 6/x^3 + 75/x^4) / 128
laguerre½(x, t = 20) = -x < t ? _L½_bessel_kernel(x) : _L½_series_kernel(x)

_logI0_bessel_kernel(z) = log(besseli(0, z) + eps(eltype(z)))
_logI0_series_kernel(z) = z - log(2*(pi*z) + eps(eltype(z)))/2 + log1p(1/8z + 9/(2*(8z)^2) - 9*25/(6*(8z)^3))
logbesseli0(z, t = 20)  = z < t ? _logI0_bessel_kernel(z) : _logI0_series_kernel(z)

#### Statistics
Distributions.mean(d::Rician) = d.σ * sqrt(pi/2) * laguerre½(-d.ν^2 / 2d.σ^2)
# Distributions.mode(d::Rician) = ?
# Distributions.median(d::Rician) = ?

Distributions.var(d::Rician) = 2 * d.σ^2 + d.ν^2 - pi * d.σ^2 * laguerre½(-d.ν^2 / 2d.σ^2)^2 / 2
Distributions.std(d::Rician) = sqrt(var(d))
# Distributions.skewness(d::Rician{T}) where {T<:Real} = ?
# Distributions.kurtosis(d::Rician{T}) where {T<:Real} = ?
# Distributions.entropy(d::Rician) = ?

#### Evaluation
Distributions.logpdf(d::Rician, x::Real) = log(x / d.σ^2 + eps(eltype(x))) + logbesseli0(x * d.ν / d.σ^2) - (x^2 + d.ν^2) / (2*d.σ^2)
# Distributions.logpdf(d::Rician, x::AbstractVector{<:Real}) = logpdf.(d, x)
Distributions.pdf(d::Rician, x::Real) = exp(logpdf(d, x)) # below version errors for large x (besseli throws); otherwise is consistent
# Distributions.pdf(d::Rician, x::Real) = x * besseli(0, x * d.ν / d.σ^2) * exp(-(x^2 + d.ν^2) / (2*d.σ^2)) / d.σ^2

#### Sampling
Distributions.rand(rng::Distributions.AbstractRNG, d::Rician{T}) where {T} = sqrt((d.ν + d.σ * randn(rng, T))^2 + (d.σ * randn(rng, T))^2)

#### Testing
#= laguerre½
let
    f₊ = x -> laguerre½(-x, x + sqrt(eps()))
    f₋ = x -> laguerre½(-x, x - sqrt(eps()))
    df = x -> abs((f₊(x) - f₋(x))/f₊(x))
    # xs = range(1.0, 1000.0; length = 100)
    xs = range(1.0f0, 50.0f0; length = 100)
    p = plot()
    plot!(p, xs, (x -> laguerre½(-x)).(xs); lab = "laguerre½(-x)")
    plot!(p, xs, f₊.(xs); lab = "f_+")
    plot!(p, xs, f₋.(xs); lab = "f_-")
    display(p)
    plot(xs, log10.(df.(xs)); lab = "df") |> display
    log10.(df.(xs))
end
=#

#= logbesseli0
let
    f₊ = z -> logbesseli0(z, z + sqrt(eps()))
    f₋ = z -> logbesseli0(z, z - sqrt(eps()))
    df = z -> abs((f₊(z) - f₋(z))/f₊(z))
    # xs = range(1.0, 500.0; length = 100)
    xs = range(1.0f0, 50.0f0; length = 100)
    # plot(xs, (z -> logbesseli0(-z, 1000)).(xs); lab = "logbesseli0(-z)")
    # plot!(xs, f₊.(xs); lab = "f_+")
    # plot!(xs, f₋.(xs); lab = "f_-")
    plot(xs, log10.(df.(xs)); lab = "df") |> display
    log10.(df.(xs))
end
=#

#= (log)pdf
let
    p = plot()
    σ = 0.23
    xs = range(0.0, 8.0; length = 500)
    for ν in [0.0, 0.5, 1.0, 2.0, 4.0]
        d = Rician(ν, σ)
        plot!(p, xs, pdf.(d, xs); lab = "nu = $ν, sigma = $σ")
        x = 8.0 #rand(Uniform(xs[1], xs[end]))
        @show log(pdf(d, x))
        @show logpdf(d, x)
        @assert log(pdf(d, x)) ≈ logpdf(d, x)
    end
    display(p)
end
=#

#= mean/std/rand
using Plots
for ν in [0, 1, 10], σ in [1e-3, 1.0, 10.0]
    d = Rician(ν, σ)
    vline!(histogram([mean(rand(d,1000)) for _ in 1:1000]; nbins = 50), [mean(d)], line = (:black, :solid, 5), title = "nu = $ν, sigma = $σ") |> display
    vline!(histogram([std(rand(d,1000)) for _ in 1:1000]; nbins = 50), [std(d)], line = (:black, :solid, 5), title = "nu = $ν, sigma = $σ") |> display
end
=#
