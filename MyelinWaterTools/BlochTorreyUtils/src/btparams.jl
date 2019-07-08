# ---------------------------------------------------------------------------- #
# BlochTorreyParameters methods
# ---------------------------------------------------------------------------- #

"""
By default, truncate the radii distribution (of Gamma type) so as to prevent the
generated radii being too large/small for the purpose of mesh generation
With the default lower = -1.5 (in units of sigma) and upper = 2.5 and the default
BlochTorreyParameters, this corresponds to rmean/rmin ~ 2.7 and rmax/rmin ~ 2.0,
and captures ~95% of the pdf mass.
"""
function radiidistribution(p::BlochTorreyParameters, lower = -1.5, upper = 2.5)
    k, θ = p.R_shape, p.R_scale # shape and scale parameters
    d = Distributions.Gamma(k, θ)
    if !(lower == -Inf && upper == Inf)
        μ, σ = k*θ, √k*θ
        d = Distributions.Truncated(d, μ + lower*σ, μ + upper*σ)
    end
    return d
end
@inline radiidistribution(p::MyelinProblem) = radiidistribution(p.params)

"""
Functions for computing each of mwf, g_ratio, and packing density in terms of
the other two parameters, assuming periodic circle packing and constant g_ratio
"""
periodic_mwf(density, g_ratio) = density * (1 - g_ratio^2)
periodic_g_ratio(density, mwf) = √(1 - mwf / density)
periodic_packdensity(mwf, g_ratio) = mwf / (1 - g_ratio^2)
periodic_mwf(;density = 0.7, g_ratio = 0.8) = periodic_mwf(density, g_ratio)
periodic_g_ratio(;density = 0.7, mwf = 0.252) = periodic_g_ratio(density, mwf)
periodic_packdensity(;mwf = 0.252, g_ratio = 0.8) = periodic_packdensity(mwf, g_ratio)

"""
If we want to generate a grid with a given MWF, we have two free parameters to
choose, namely the g-ratio and the packing density. Ideally, the MWF would
uniquely determine the other corresponding parameters; this requires an assumed
relationship between two of the parameters.

Here, we choose simply to minimize the difference between the g-ratio and the
packing density in order to prevent extreme cases. In other words, we solve

    min (g_goal - g_ratio)^2 + (density_goal - density)^2

subject to

    density = mwf / (1 - g_ratio^2)
    g_ratio ∈ g_ratio_bounds
    density ∈ density_bounds
"""
function optimal_g_ratio_packdensity_linesearch(mwf;
        g_ratio_goal = 0.75,
        density_goal = 0.65,
        g_ratio_bounds = (0.6, 0.9),
        density_bounds = (0.6, 0.8)
    )
    g_low, g_high = g_ratio_bounds
    η_low, η_high = density_bounds
    if periodic_packdensity(mwf, g_ratio_bounds[1]) < η_low
        @show periodic_packdensity(mwf, g_ratio_bounds[1])
        @show η_low
        @assert η_low > mwf
        g_low = min(g_ratio_bounds[2], max(g_ratio_bounds[1], periodic_g_ratio(max(η_low, mwf + √eps(mwf)), mwf)))
    end
    if periodic_packdensity(mwf, g_ratio_bounds[2]) > η_high
        @show periodic_packdensity(mwf, g_ratio_bounds[2])
        @show η_high
        @assert η_high > mwf
        g_high = max(g_ratio_bounds[1], min(g_ratio_bounds[2], periodic_g_ratio(max(η_high, mwf + √eps(mwf)), mwf)))
    end
    g_ratio_bounds = (g_low, g_high)
    @show g_ratio_bounds

    if g_ratio_bounds[1] >= g_ratio_bounds[2]
        @warn "No solution found in search range; returning highest density solution"
        g_ratio = max(g_ratio_bounds...)
        AxonPDensity = periodic_packdensity(mwf, g_ratio)
    else
        optfun = g -> (g - g_ratio_goal)^2 + (periodic_packdensity(mwf, g) - density_goal)^2
        res = Optim.optimize(optfun, g_ratio_bounds...)
        
        display(res) #TODO
    
        g_ratio = Optim.minimizer(res)
        AxonPDensity = periodic_packdensity(mwf, g_ratio)
    end
    MWF = periodic_mwf(AxonPDensity, g_ratio)

    return @ntuple(g_ratio, AxonPDensity, MWF)
end

function optimal_g_ratio_packdensity_gridsearch(mwf;
        g_ratio_goal = 0.78,
        density_goal = 0.70,
        g_ratio_bounds = (0.5, 0.9),
        density_bounds = (0.6, 0.8)
    )
    
    x_goal = [g_ratio_goal, density_goal]
    fun(x) =  sum(abs2, x .- x_goal)
    fun_grad!(g, x) = (g .= 2 .* (x .- x_goal))
    fun_hess!(h, x) = (h .= Matrix(I,2,2))

    CON_FACT = 1e3 # Hack to make constraint more strongly enforced
    con!(c, x) = (c[1] = CON_FACT * (x[1] - periodic_g_ratio(x[2], mwf)); c)
    con_J!(J, x) = (J[1,1] = CON_FACT; J[1,2] = -CON_FACT * mwf / (2 * x[2]^2 * periodic_g_ratio(x[2], mwf)); J)
    con_H!(h, x, λ) = h[2,2] += λ[1] * CON_FACT * (mwf^2 / (4 * x[2]^4 * periodic_g_ratio(x[2], mwf)^3) + mwf / (x[2]^3 * periodic_g_ratio(x[2], mwf)))
    
    x0 = [g_ratio_goal, density_goal]
    lx = [g_ratio_bounds[1], density_bounds[1]]
    ux = [g_ratio_bounds[2], density_bounds[2]]
    df = Optim.TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
    dfc = Optim.TwiceDifferentiableConstraints(con!, con_J!, con_H!, lx, ux, [0.0], [0.0])
    res = Optim.optimize(df, dfc, x0, Optim.IPNewton())
    
    g_ratio, AxonPDensity = Optim.minimizer(res)
    MWF = periodic_mwf(AxonPDensity, g_ratio)

    if !isapprox(mwf, MWF; atol = 1e-3)
        @warn "Desired MWF ($mwf) couldn't be reached: MWF = $MWF"
    end

    return @ntuple(g_ratio, AxonPDensity, MWF)
end