import Pkg
Pkg.activate(realpath(joinpath(@__DIR__, "../../../MyelinWaterTools")))

import Dates, BSON, MAT
using StatsPlots, Distributions, OnlineStats, LaTeXStrings
using StaticArrays, LinearAlgebra
using QuadGK
using DrWatson: @dict, @ntuple
using Parameters: @unpack
pyplot(size=(800,600))

tounit(x) = x / norm(x)
fielddir(α) = SVector{3}(sind(α), 0, cosd(α))
function angle(x::SVector{3}, y::SVector{3})
    u, v = tounit(x), tounit(y)
    θ = 2 * atand(norm(u - v), norm(u + v))
end
sinangle(x::SVector{3}, y::SVector{3}) = norm(x × y) / (norm(x) * norm(y))
cosangle(x::SVector{3}, y::SVector{3}) = (x ⋅ y) / (norm(x) * norm(y))

g(x) = asind(√x) # Transformation y(x) applied to x = sin²α
h(y) = sind(y)^2 # Inverse transformation x(y) applied to y = α (monotonically increasing)
dh(y) = (π/180) * sind(2y) # Derivative dh/dy (needed for transformation)

function sin²α(B̂, θ, ϕ)
    sinθ, cosθ = sind(θ), cosd(θ)
    sinϕ, cosϕ = sind(ϕ), cosd(ϕ)
    v = SVector{3}(sinθ*cosϕ, sinθ*sinϕ, cosθ)
    sinα = sinangle(B̂, v)
    return sinα^2
end

function sample(α₀, dists)
    @unpack db, dθ, dϕ, dr, dℓ = dists

    B̂ = fielddir(α₀) # Main magnetic field direction
    num = sind(α₀)^2
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
    fY(y) = fX(h(y)) * abs(dh(y)) # Transformed pdf for y = α, y ∈ [0, 90]
    # μY = quadgk(y -> y * fY(y), 0.0, 90.0)[1] # Compute mean of y
    # σY = √quadgk(y -> (y - μY)^2 * fY(y), 0.0, 90.0)[1] # Compute std of y
    
    # Compute moments directly from data
    μY, σY = mean(g(x) for x in S), std(g(x) for x in S)
    μX, σX = mean(S), std(S)
    
    return @ntuple(μY, σY, fY, μX, σX, DX, fX)
end

function main(
        αF = 0.0:0.25:90.0, # Field angles to simulate
        Ns = 100_000, # Number of samples per field angle
        db = 1:5, # Number of branches per vessel
        dθ = Normal(90.0, 0.0), # Branching angle distribution (polar angle)
        dϕ = Uniform(0.0, 360.0), # Azimuthal angle distribution
        dr = Uniform(0.1, 0.5), # Relative radius size distribution
        dℓ = Uniform(0.1, 0.5), # Relative length distribution
    )
    # Run Ns samples of volume-weighted sin²α computations
    params = @ntuple(αF, Ns, db, dθ, dϕ, dr, dℓ) # All inputs
    opts = @ntuple(Ns) # Simulation hyperparameters
    dists = @ntuple(db, dθ, dϕ, dr, dℓ) # Distributions to sample from
    runs = [runsample(α₀, opts, dists) for α₀ in αF]
    data = [α => fitsample(S) for (α, S) in zip(αF, runs)]

    return @ntuple(runs, data, params)
end

function plotmain(args...; kwargs...)
    str(x) = string(round(x; sigdigits = 3))
    title(S, μ = mean(S)) = "μ = " * str(μ) # * ", " * "σ = " * str(std(x))
    
    # Run simulations
    main_output = main(args...)
    @unpack runs, data, params = main_output
    αF = [d[1] for d in data]
    αV = [d[2].μY for d in data]
    σV = [d[2].σY for d in data]
    
    # plot(map((S,d) -> begin
    #         @unpack fX = d[2]
    #         p = histogram(S; title = title(S), legend = :none, normalized = true)
    #         plot!(fX, xlims(p)..., legend = :none)
    #     end, runs, data)...;
    #     titlefontsize = 8, xtickfontsize = 6, ytickfontsize = 6,
    # ) |> display
    
    # plot(map((S,d) -> begin
    #         @unpack μY, σY, fY = d[2]
    #         p = histogram(g.(S); title = title(nothing, μY), legend = :none, normalized = true)
    #         plot!(p, y -> fY(y))
    #     end, runs, data)...;
    #     titlefontsize = 8, xtickfontsize = 6, ytickfontsize = 6,
    # ) |> display

    ann_text = """
    Number of Branches: b ∈ $(params.db)
    Branching Polar Angle: θ ∈ $(replace(string(params.dθ), "{Float64}" => ""))
    Branching Azimuthal Angle: ϕ ∈ $(replace(string(params.dϕ), "{Float64}" => ""))
    Relative Branch Radii: r ∈ $(replace(string(params.dr), "{Float64}" => ""))
    Relative Lengths: ℓ ∈ $(replace(string(params.dℓ), "{Float64}" => ""))
    """

    p = plot([αF αF], [αF αV];
        ribbon = [0 1] .* σV,
        ls = [:dash :solid],
        legend = :topleft,
        # grid = true,
        # minorgrid = true,
        # minorgridalpha = 1,
        # minorticks = 5,
        # minorgridalpha = 1,
        xlabel = "Fibre Angle [degrees]", ylabel = "Effective Angle [degrees]",
        label = ["Identity" "Simulated"],
        title = L"Simulated $\alpha_{Effective}$ vs. $\alpha_{Fibre}$",
        titlefontsize = 16, tickfontsize = 10, legendfontsize = 12, guidefontsize = 12,
        xticks = 0:5:90, yticks = 0:5:90,
        xlims = (0,90), ylims = (0,90),
        annotations = (60, 15, text(ann_text, font(12))),
        kwargs...)
    display(p)

    # Save figure
    now = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
    fname = now * "." * "angledistribution"
    for ext in [".png", ".pdf"]
        savefig(p, fname * ext)
    end

    # Save data to bson and mat files
    data = Dict("FibreAngle" => αF, "EffectiveAngle" => αV, "EffectiveAngleSTD" => σV)
    BSON.bson(fname * ".bson", data)
    MAT.matwrite(fname * ".mat", data)

    return main_output
end