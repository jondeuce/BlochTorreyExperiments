using StatsPlots, Distributions, OnlineStats, LaTeXStrings
using StaticArrays, LinearAlgebra
using QuadGK
using DrWatson: @dict, @ntuple
using Parameters: @unpack

tounit(x) = x / norm(x)
fielddir(α) = SVector{3}(sin(α), 0, cos(α))
function angle(x::SVector{3}, y::SVector{3})
    u, v = tounit(x), tounit(y)
    θ = 2 * atan(norm(u - v), norm(u + v))
end
sinangle(x::SVector{3}, y::SVector{3}) = norm(x × y) / (norm(x) * norm(y))
cosangle(x::SVector{3}, y::SVector{3}) = (x ⋅ y) / (norm(x) * norm(y))

function sin²α(B̂, θ, ϕ)
    sinθ, cosθ = sin(θ), cos(θ)
    sinϕ, cosϕ = sin(ϕ), cos(ϕ)
    v = SVector{3}(sinθ*cosϕ, sinθ*sinϕ, cosθ)
    sinα = sinangle(B̂, v)
    return sinα^2
end

function sample(α₀, ρ, Nb, dθ, dϕ, dr, dℓ)
    B̂ = fielddir(α₀)
    ν = (ρ^2 - 1)
    num = sin(α₀)^2
    den = one(α₀)
    for i in 1:Nb
        θ, ϕ, r, ℓ = rand(dθ), rand(dϕ), rand(dr), rand(dℓ)
        A = r^2 * ν * ℓ
        num += A * sin²α(B̂, θ, ϕ)
        den += A
    end
    return num / den
end

function runsample(Ns, α₀, ρ, Nb, dθ, dϕ, dr, dℓ)
    S = KHist(25)
    for i in 1:Ns
        fit!(S, sample(α₀, ρ, Nb, dθ, dϕ, dr, dℓ))
    end
    return S
end

function main(
        Ns = 10_000, Na = 25, Nb = 10, ρ = 2,
        dθ = Normal(45.0, 10.0), dϕ = Uniform(0.0, 360.0),
        dr = Uniform(1/6, 1/4), dℓ = Uniform(1/10, 1/4)
    )
    αF = range(0, π/2; length = Na)
    runs = [runsample(Ns, α, ρ, Nb, dθ, dϕ, dr, dℓ) for α in αF]
    μ_sin²α = mean.(runs)
    σ_sin²α = std.(runs)

    h(y) = asin(√y)
    dh(y) = inv(2 * √(y * (1 - y)))
    moments = [begin
        fX(x) = exp(-(x - μ)^2 / (2σ^2)) / √(2π * σ^2)
        fY(y) = fX(h(y)) * abs(dh(y))
        μα, _ = quadgk(fY, 0.0, 1.0)
        σ²α, _ = quadgk(y -> y^2 * fY(y), 0.0, 1.0)
        (μ = μα, σ = √σ²α)
    end
    for (μ, σ) in zip(μ_sin²α, σ_sin²α)]
    
    αV = [m.μ for m in moments]
    σV = [m.σ for m in moments]

    return @ntuple(runs, αF, αV, σV)
end

function plotmain(args...; kwargs...)
    str(x) = string(round(x; sigdigits = 3))
    title(x) = "μ = " * str(mean(x)) # * ", " * "σ = " * str(std(x))
    
    @unpack runs, αF, αV, σV = main(args...)
    
    plot(map(S -> begin
            plot(S; title = title(S), legend = :none)
            plot!(Normal(mean(S), std(S)), legend = :none)
        end, runs)...
    ) |> display
    
    plot([αF αF], [αF αV];
        ribbon = [0 1] .* σV,
        ls = [:dash :solid],
        legend = :topleft,
        label = ["Identity" "Simulated"],
        title = L"Simulated $\alpha_{Vessel}$ vs. $\alpha_{Fibre}$",
        kwargs...
    ) |> display
    
    return runs
end