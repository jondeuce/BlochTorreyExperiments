ConstrainedOpts = Optim.Options(iterations = 100_000,
                                g_tol = 1e-12,
                                allow_f_increases = false)

@time circles_opt_con, opt_result_con = pack_circles(
    rs;
    initial_origins = origin.(circles_opt),
    constrained = true,
    epsilon = ϵ,
    Opts = ConstrainedOpts);

for method in [:suave, :vegas, :divonne, :cuhre]
    res = estimate_density_monte_carlo(cs_plot; integrator = Cuba_integrator(method))
    @show (method, res...)
end

@benchmark estimate_density_monte_carlo(cs_plot; integrator = Cuba_integrator(:suave))
@benchmark estimate_density_monte_carlo(cs_plot; integrator = Cuba_integrator(:vegas))
@benchmark estimate_density_monte_carlo(cs_plot; integrator = Cuba_integrator(:divonne))
@benchmark estimate_density_monte_carlo(cs_plot; integrator = Cuba_integrator(:cuhre))
