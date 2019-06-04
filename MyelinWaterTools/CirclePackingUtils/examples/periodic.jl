using CirclePackingUtils
using Distributions
using StatsPlots
gr(legend=:none, size=(1200,800), ratio = :equal)

# Radii distribution for typical myelinated axons
function radiidistribution(k = 5.7, θ = 0.46/5.7, lower = -1.5, upper = 2.5)
    d = Distributions.Gamma(k, θ)
    if !(lower == -Inf && upper == Inf)
        μ, σ = k*θ, √k*θ
        d = Distributions.Truncated(d, μ + lower*σ, μ + upper*σ)
    end
    return d
end

function example(;
        radii = rand(radiidistribution(), 25),
        plotpacked = true
    )
    @time begin
        cs_greedy = GreedyCirclePacking.pack(radii)
        cs_scaled = map(c -> Circle(1.1*origin(c), radius(c)), cs_greedy)
        # cs_scaled = map(t -> Circle(1.1*t[1], t[2]), zip(initialize_origins(radii), radii))
        cs_opt, boundary_rectangle = PeriodicCirclePacking.pack(cs_scaled;
            distancescale = mean(radii),
            epsilon = 0.05 * mean(radii)
        )
        cs_periodic = periodic_circles(cs_opt, boundary_rectangle)
        if plotpacked
            p = plot(periodic_circle_repeat(cs_opt, boundary_rectangle))
            p = plot!(p, boundary_rectangle)
            w = widths(boundary_rectangle)
            xlims!(p, -0.3w[1], 1.3w[1])
            ylims!(p, -0.3w[2], 1.3w[2])
            display(p)
        end
        @info "Density = $(periodic_density(cs_periodic, boundary_rectangle))"
        @info "MinDist = $(minimum_signed_edge_distance(cs_periodic))"
    end
    return (radii = radii, circles = cs_periodic, domain = boundary_rectangle)
end
