using StatsPlots, Distributions, OnlineStats, LaTeXStrings
using StaticArrays, LinearAlgebra
using QuadGK
using DrWatson: @dict, @ntuple
using Parameters: @unpack
pyplot(size=(800,600))

tounit(x) = x / norm(x)
fielddir(α) = SVector{3}(sin(α), 0, cos(α))
function angle(x::SVector{3}, y::SVector{3})
    u, v = tounit(x), tounit(y)
    θ = 2 * atan(norm(u - v), norm(u + v))
end
sinangle(x::SVector{3}, y::SVector{3}) = norm(x × y) / (norm(x) * norm(y))
cosangle(x::SVector{3}, y::SVector{3}) = (x ⋅ y) / (norm(x) * norm(y))

g(x) = asin(√x) # Transformation y(x) applied to x = sin²α
h(y) = sin(y)^2 # Inverse transformation x(y) applied to y = α (monotonically increasing)
dh(y) = sin(2y) # Derivative dh/dy (needed for transformation)

function sin²α(B̂, θ, ϕ)
    sinθ, cosθ = sin(θ), cos(θ)
    sinϕ, cosϕ = sin(ϕ), cos(ϕ)
    v = SVector{3}(sinθ*cosϕ, sinθ*sinϕ, cosθ)
    sinα = sinangle(B̂, v)
    return sinα^2
end

function sample(α₀, dists)
    @unpack db, dθ, dϕ, dr, dℓ = dists

    B̂ = fielddir(α₀) # Main magnetic field direction
    num = sin(α₀)^2
    den = one(α₀)

    for i in 1:rand(db)
        θ, ϕ, r, ℓ = rand(dθ), rand(dϕ), rand(dr), rand(dℓ)
        A = r^2 * ℓ
        num += A * sin²α(B̂, θ, ϕ)
        den += A
    end

    return num / den
end
runsample(α₀, opts, dists) = [sample(α₀, dists) for _ in 1:opts.Ns]

# Transform sin²α-distribution to α-distribution by fitting a distribution D
# to the observed sin²α values and then transforming variables
function fitsample(S, D = Beta)
    # Fit x = sin²α samples to distribution D and truncate to x ∈ [0, 1] if necessary
    DX = fit(D, S) 
    !(D <: Beta) && (DX = Truncated(DX, 0.0, 1.0))
    
    # Compute transformed pdf and moments
    fX(x) = pdf(DX, x) # Pdf for x = sin²α, x ∈ [0, 1]
    fY(y) = fX(h(y)) * abs(dh(y)) # Transformed pdf for y = α, y ∈ [0, π/2]
    # μY = quadgk(y -> y * fY(y), 0.0, π/2)[1] # Compute mean of y
    # σY = √quadgk(y -> (y - μY)^2 * fY(y), 0.0, π/2)[1] # Compute std of y
    
    # Compute moments directly from data
    μY, σY = mean(g(x) for x in S), std(g(x) for x in S)
    μX, σX = mean(S), std(S)
    
    return @ntuple(μY, σY, fY, μX, σX, DX, fX)
end

function main(
        αF = range(0, π/2; length = 16), # Field angles to simulate
        Ns = 100_000, # Number of samples per field angle
        db = 5:20, # Number of branches per vessel
        dθ = Normal(deg2rad(45.0), deg2rad(10.0)), # Branching angle distribution (polar angle)
        dϕ = Uniform(0.0, 2π), # Azimuthal angle distribution
        dr = Uniform(1/6, 1/4), # Relative radius size distribution
        dℓ = Uniform(1/10, 1/4), # Relative length distribution
    )
    # Run Ns samples of volume-weighted sin²α computations
    opts = @ntuple(Ns) # Hyperparameters
    dists = @ntuple(db, dθ, dϕ, dr, dℓ) # Distributions to sample from
    runs = [runsample(α₀, opts, dists) for α₀ in αF]
    data = [α => fitsample(S) for (α, S) in zip(αF, runs)]

    return @ntuple(runs, data)
end

function plotmain(args...; kwargs...)
    str(x) = string(round(x; sigdigits = 3))
    title(S, μ = mean(S)) = "μ = " * str(μ) # * ", " * "σ = " * str(std(x))
    
    # Run simulations
    @unpack runs, data = main(args...)
    αF = [d[1] for d in data]
    αV = [d[2].μY for d in data]
    σV = [d[2].σY for d in data]
    
    plot(map((S,d) -> begin
            @unpack fX = d[2]
            p = histogram(S; title = title(S), legend = :none, normalized = true)
            plot!(fX, xlims(p)..., legend = :none)
        end, runs, data)...;
        titlefontsize = 10, xtickfontsize = 6, ytickfontsize = 6,
    ) |> display
    
    plot(map((S,d) -> begin
            @unpack μY, σY, fY = d[2]
            p = histogram(rad2deg.(g.(S)); title = title(nothing, rad2deg(μY)), legend = :none, normalized = true)
            plot!(p, y -> deg2rad(fY(deg2rad(y))))
        end, runs, data)...;
        titlefontsize = 10, xtickfontsize = 6, ytickfontsize = 6,
    ) |> display

    plot(rad2deg.([αF αF]), rad2deg.([αF αV]);
        ribbon = [0 1] .* rad2deg.(σV),
        ls = [:dash :solid],
        legend = :topleft,
        label = ["Identity" "Simulated"],
        title = L"Simulated $\alpha_{Vessel}$ vs. $\alpha_{Fibre}$",
        xticks = 0:15:90, yticks = 0:15:90,
        xlims = (0,90), ylims = (0,90),
        kwargs...
    ) |> display
    
    return @ntuple(runs, data)
end