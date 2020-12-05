# Initialization project/code loading
import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
# Pkg.instantiate()

using MWFLearning
using StatsBase
using StatsPlots
pyplot(size = (600,400))

function plot_T2_dist(d; α = 0.05, σ = 0, T2Range = (1e-3, 2000e-3), nT2 = 120)
    η = d[:sweepparams][:TE]
    t = η .* (1:d[:sweepparams][:nTE])
    x = init_signal(d[:signals])
    x .+= σ.*randn(Random.MersenneTwister(0), size(x))
    τ = log10range(T2Range...; length = nT2);
    @time y = ilaplace(x, τ, η, α)
    @btime ilaplace($x, $τ, $η, $α)
    props = Dict(:legendfontsize => 14, :tickfontsize => 10, :minorgrid => true)
    @time display(plot(
        plot(1000 .* t, x;
            label = "\$\\sigma = $σ\$",
            props...),
        plot(τ, y;
            xscale = :log10, lt = :sticks, lab = "\$\\alpha = $α\$",
            props...);
        layout = (2,1)
    ))
    return nothing
end

sweepdir = joinpath(BTHOME, "BlochTorreyResults/Experiments/MyelinWaterLearning/generate-mwf-signal-1/")
measdir = "measurables"
fulldir = joinpath(sweepdir, measdir)
# files = sample(readdir(fulldir), 3)
files = readdir(fulldir)[500:500]
d = [BSON.load(joinpath(fulldir, f)) for f in files]
# σrange = [0.0, 1e-3, 5e-3]
σrange = [0.0]
# σrange = [5e-3]
# αrange = [0.03, 0.04, 0.05]
αrange = [0.05]
for d in d, σ in σrange, α in αrange
    @time plot_T2_dist(d; α = α, σ = σ, T2Range = (1e-3, 2000e-3), nT2 = 120)
end
