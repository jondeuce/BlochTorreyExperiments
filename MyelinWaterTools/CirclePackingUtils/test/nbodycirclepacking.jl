using StatsPlots
gr(size=(800,600), leg = false, labels = nothing)

make_circles(rs, u) = [Circle(Vec{2,Float64}((u[4i-3],u[4i-2])), rs[i]) for i in 1:length(rs)]
function plot_circles_trajectory(rs, sol, domain; N = 100)
    for t in range(sol.t[[1,end]]..., length = N)
        @info "t = $t"
        p = plot(domain)
        cs = make_circles(rs, sol(t))
        cs = periodic_circle_repeat(cs, domain)
        plot!(p, cs)
        display(p)
    end
    return nothing
end

N = 20
rs = 1.0 .+ rand(N)

@time out = NBodyCirclePacking.pack(rs;
    alg = BS3(),#Rodas5(),Tsit5()
    init_speed = 5.0, min_speed = 1e-2, damping = 0.5
)
plot_circles_trajectory(rs, out.sol, out.domain)

geom = periodic_scale_to_threshold(out.circles, out.domain, 0)

@time out = NBodyCirclePacking.pack(rs;
    initial_origins = origin.(geom.circles),
    domain = geom.domain,
    alg = BS3(),#Rodas5(),Tsit5()
    init_speed = 1.0, min_speed = 1e-2, damping = 0.8
)
plot_circles_trajectory(rs, out.sol, out.domain)
