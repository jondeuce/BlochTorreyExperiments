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
@inline _L½_bessel_kernel(x::T) where {T} = exp(-x/2) * ((1 + x) * T(besseli(0, x/2)) + x * T(besseli(1, x/2)))
@inline _L½_series_kernel(x::T) where {T} = (y = inv(x); @evalpoly(y, T(2), T(1)/2, T(1)/16, T(3)/64, T(75)/1024, T(735)/4096) * sqrt(x/pi))
@inline laguerre½(x, t = 150) = -x < t ? _L½_bessel_kernel(float(-x)) : _L½_series_kernel(float(-x)) # note negative arguments for kernels

@inline _logI0_bessel_kernel(x::T) where {T} = log(T(besseli(0, x)) + eps(T))
@inline _logI0_series_kernel(x::T) where {T} = (y = inv(x); log(@evalpoly(y, T(1), T(1)/8, T(9)/128, T(75)/1024, T(3675)/32768, T(59535)/262144, T(2401245)/4194304, T(57972915)/33554432)) + x - log(x)/2 - log(2*T(π))/2)
@inline logbesseli0(x, t = 75)  = x < t ? _logI0_bessel_kernel(float(x)) : _logI0_series_kernel(float(x))

@inline _logI1_bessel_kernel(x::T) where {T} = log(T(besseli(1, x)) + eps(T))
@inline _logI1_series_kernel(x::T) where {T} = (y = inv(x); log(@evalpoly(y, T(1), -T(3)/8, -T(15)/128, -T(105)/1024, -T(4725)/32768, -T(72765)/262144, -T(2837835)/4194304, -T(66891825)/33554432, )) + x - log(x)/2 - log(2*T(π))/2)
@inline logbesseli1(x, t = 75)  = x < t ? _logI1_bessel_kernel(float(x)) : _logI1_series_kernel(float(x))

@inline _I1I0m1_bessel_kernel(x::T) where {T} = T(besseli(1, x)) / T(besseli(0, x)) - 1
@inline _I1I0m1_series_kernel(x::T) where {T} = (y = inv(x); y * @evalpoly(y, -T(1)/2, -T(1)/8, -T(1)/8, -T(25)/128, -T(13)/32, -T(1073)/1024, -T(103)/32, -T(375733)/32768, -T(23797)/512, -T(55384775)/262144))
@inline besseli1i0m1(x, t = 50) = x < t ? _I1I0m1_bessel_kernel(float(x)) : _I1I0m1_series_kernel(float(x))

@inline _I2I0_bessel_kernel(x::T) where {T} = T(besseli(2, x)) / T(besseli(0, x))
@inline _I2I0_series_kernel(x::T) where {T} = (y = inv(x); @evalpoly(y, T(1), -T(2), T(1), T(1)/4, T(1)/4, T(25)/64, T(13)/16, T(1073)/512, T(103)/16, T(375733)/16384, T(23797)/256))
@inline besseli2i0(x, t = 50)   = x < t ? _I2I0_bessel_kernel(float(x)) : _I2I0_series_kernel(float(x))

#### Statistics
@inline Distributions.mean(d::Rician) = d.σ * sqrt(pi/2) * laguerre½(-d.ν^2 / 2d.σ^2)
# @inline Distributions.mode(d::Rician) = ?
# @inline Distributions.median(d::Rician) = ?

@inline Distributions.var(d::Rician) = 2 * d.σ^2 + d.ν^2 - pi * d.σ^2 * laguerre½(-d.ν^2 / 2d.σ^2)^2 / 2
@inline Distributions.std(d::Rician) = sqrt(var(d))
# @inline Distributions.skewness(d::Rician{T}) where {T<:Real} = ?
# @inline Distributions.kurtosis(d::Rician{T}) where {T<:Real} = ?
# @inline Distributions.entropy(d::Rician) = ?

#### Evaluation
@inline Distributions.logpdf(d::Rician, x::Real) = log(x / d.σ^2 + eps(float(typeof(x)))) + logbesseli0(x * d.ν / d.σ^2) - (x^2 + d.ν^2) / (2*d.σ^2)
@inline Distributions.pdf(d::Rician, x::Real) = exp(logpdf(d, x)) # below version errors for large x (besseli throws); otherwise is consistent
# @inline Distributions.pdf(d::Rician, x::Real) = x * besseli(0, x * d.ν / d.σ^2) * exp(-(x^2 + d.ν^2) / (2*d.σ^2)) / d.σ^2

#### Gradient
@inline function _∇logpdf(d::Rician, x::Real)
    σ, ν = d.σ, d.ν
    σ2 = σ^2
    B = besseli1i0m1(x*ν/σ2)
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
    B = besseli1i0m1(z)
    B′ = (1 + besseli2i0(z))/2 - (B + 1)^2
    return (x^2 * B′ - σ2) / σ2^2
end

#### Sampling
@inline Distributions.rand(rng::Distributions.AbstractRNG, d::Rician{T}) where {T} = sqrt((d.ν + d.σ * randn(rng, T))^2 + (d.σ * randn(rng, T))^2)

#### Testing
#= laguerre½
let
    vals = Dict{Int,BigFloat}(
        1       => big"1.44649134408317183335863878689608088636624386024796164711195852984091354675405604850815082040598215708783541465221",
        150     => big"13.8428182132579207630845698226782480139394357529419478259979332171281698876990601582933968786450293076448578693085",
        250     => big"17.8590913502406046821168814494005149903283070693484246034663777670678881083836244124944028011196109619555364318691",
        1000    => big"35.6914040595513768651551927899531513343638822880127244775680481255076084811577905488106037306888201625798303423912",
        1000000 => big"1128.37944919033960964972053998058678715837204482409899836336271985095168033327696330342629488845179765606231172765",
    )
    for (x,v) in sort(vals; by = first)
        println("\nx = $x:")
        for T in [Float64, Float32]
            F = x == 150 ? 250 : x == 250 ? 10 : 2
            if x <= 150
                display(abs(_L½_bessel_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_L½_bessel_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
            if x >= 150
                display(abs(_L½_series_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_L½_series_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
        end
    end
end;
=#

#= logbesseli0
let
    vals = Dict{Int,BigFloat}(
        1       => big"0.235914358507178648689414846199951780553937315868442236933",
        75      => big"71.92399534542726979766263061737789216870198000778955948380",
        250     => big"246.3208320120570875328350064326308479165696536189728583253",
        1000    => big"995.6273088898694646714677644808475148830463306781655261607",
        1000000 => big"999992.1733063128132527062308001677706659748509246985475905",
    )
    for (x,v) in sort(vals; by = first)
        println("\nx = $x:")
        for T in [Float64, Float32]
            F = x == 1 ? 5 : 2
            if x <= 75
                display(abs(_logI0_bessel_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_logI0_bessel_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
            if x >= 75
                display(abs(_logI0_series_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_logI0_series_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
        end
    end
end;
=#

#= logbesseli1
let
    vals = Dict{Int,BigFloat}(
        1       => big"-0.57064798749083128142317323666480514121275701003814227921",
        75      => big"71.91728368097706026635707716796022782798273695917775824800",
        250     => big"246.3188279973098207462626972558328961077394570824750455984",
        1000    => big"995.6268086396399849229481182362161219299624703016781038929",
        1000000 => big"999992.1733058128130027060016331886034188380540456910701362",
    )
    for (x,v) in sort(vals; by = first)
        println("\nx = $x:")
        for T in [Float64, Float32]
            F = x == 1 ? 5 : 2
            if x <= 75
                display(abs(_logI1_bessel_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_logI1_bessel_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
            if x >= 75
                display(abs(_logI1_series_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_logI1_series_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
        end
    end
end;
=#

#= besseli1i0m1
let
    vals = Dict{Int,BigFloat}(
        1       => big"-0.55361003410346549295231820480735733022374685259961217713",
        50      => big"-0.01005103262150224740734407053531738330479593697870198617",
        75      => big"-0.00668919153535887090022356792413031251065008144225797039",
        250     => big"-0.00200200805042034549987616998460352026581291107756201664",
        1000    => big"-0.00050012512519571980108182565204402140552990313131753794",
        1000000 => big"-5.000001250001250001953129062510478547812614665076603e-7"
    )
    for (x,v) in sort(vals; by = first)
        println("\nx = $x:")
        for T in [Float64, Float32]
            F = x == 50 ? 250 : x == 75 ? 10 : 2
            if x <= 50
                display(abs(_I1I0m1_bessel_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_I1I0m1_bessel_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
            if x >= 50
                display(abs(_I1I0m1_series_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_I1I0m1_series_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
        end
    end
end;
=#

#= besseli2i0
let
    vals = Dict{Int,BigFloat}(
        1       => big"0.107220068206930985904636409614714660447493705199224354277",
        50      => big"0.960402041304860089896293762821412695332191837479148079447",
        75      => big"0.973511711774276236557339295144643475000284002171793545877",
        250     => big"0.992016016064403362763999009359876828162126503288620496133",
        1000    => big"0.998001000250250391439602163651304088042811059806262635075",
        1000000 => big"0.999998000001000000250000250000390625812502095709562522933"
    )
    for (x,v) in sort(vals; by = first)
        for T in [Float64, Float32]
            println("\nx = $x ($T):")
            F = x == 1 ? 5 : 2
            if x <= 50
                display(abs(_I2I0_bessel_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_I2I0_bessel_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
            if x >= 50
                display(abs(_I2I0_series_kernel(T(x)) - v) < F * eps(T(v)))
                # display(abs(_I2I0_series_kernel(T(x)) - v))
                # display(eps(T(v)))
            end
        end
    end
end;
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

#= ∇logpdf
let
    ν, σ, x = 100*rand(), 100*rand(), 100*rand()
    d = Rician(ν, σ)
    δ = cbrt(eps())
    ∇ = ∇logpdf(d, x)
    ∇ν = (logpdf(Rician(ν + δ, σ), x) - logpdf(Rician(ν - δ, σ), x)) / 2δ
    ∇σ = (logpdf(Rician(ν, σ + δ), x) - logpdf(Rician(ν, σ - δ), x)) / 2δ
    ∇δ = (ν = ∇ν, σ = ∇σ)
    display(map((x,y) -> (x-y)/y, values(∇δ), values(∇)))
end;
=#

#= ∇²logpdf
let
    ν, σ, x = 100*rand(), 100*rand(), 100*rand()
    d = Rician(ν, σ)
    δ = eps(ν)^(1/4)
    ∇ν² = ∂²logpdf_∂ν²(d, x)
    ∇νδ² = (logpdf(Rician(ν + δ, σ), x) - 2 * logpdf(Rician(ν, σ), x) + logpdf(Rician(ν - δ, σ), x)) / δ^2
    display(∇ν²)
    display(∇νδ²)
    display(∇ν² - ∇νδ²)
end;
=#

#= mean/std/rand
using Plots
for ν in [0, 1, 10], σ in [1e-3, 1.0, 10.0]
    d = Rician(ν, σ)
    vline!(histogram([mean(rand(d,1000)) for _ in 1:1000]; nbins = 50), [mean(d)], line = (:black, :solid, 5), title = "nu = $ν, sigma = $σ") |> display
    vline!(histogram([std(rand(d,1000)) for _ in 1:1000]; nbins = 50), [std(d)], line = (:black, :solid, 5), title = "nu = $ν, sigma = $σ") |> display
end
=#
