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
Functions for computing each of mvf, g_ratio, and packing density in terms of
the other two parameters, assuming periodic circle packing and constant g_ratio
"""
periodic_mvf(density, g_ratio) = density * (1 - g_ratio^2)
periodic_g_ratio(density, mvf) = √(1 - mvf / density)
periodic_packdensity(mvf, g_ratio) = mvf / (1 - g_ratio^2)
periodic_mvf(;density = 0.7, g_ratio = 0.8) = periodic_mvf(density, g_ratio)
periodic_g_ratio(;density = 0.7, mvf = 0.252) = periodic_g_ratio(density, mvf)
periodic_packdensity(;mvf = 0.252, g_ratio = 0.8) = periodic_packdensity(mvf, g_ratio)

"""
If we want to generate a grid with a given MVF, we have two free parameters to
choose, namely the g-ratio and the packing density. Ideally, the MVF would
uniquely determine the other corresponding parameters; this requires an assumed
relationship between two of the parameters. The keyword `solution_choice` defines
how the final solution is chosen:
    :max_density    fix density to maximum and compute corresponding g-ratio
    :min_g_ratio    fix g-ratio to minimum and compute corresponding density
    :random_g_ratio fix a random g-ratio and compute the corresponding density
    :random_density fix a random density and compute the corresponding g-ratio
    :median         choose the solution which has g-ratio and density nearest to
                    the median of the solution space
    :random         return a random valid solution
"""
function optimal_g_ratio_packdensity(goal_mvf;
        g_ratio_bounds = (0.60, 0.92),
        density_bounds = (0.15, 0.82),
        solution_choice = :max_density,
    )

    mvf_min = (1 - g_ratio_bounds[2]^2) * density_bounds[1]
    mvf_max = (1 - g_ratio_bounds[1]^2) * density_bounds[2]
    if !(mvf_min <= goal_mvf <= mvf_max)
        error("Goal mvf doesn't permit a solution for given g-ratio and density bounds; mvf = $goal_mvf must be in [$mvf_min, $mvf_max]")
    end

    g(density) = sqrt(1 - goal_mvf/density)
    d(gratio)  = goal_mvf / (1 - gratio^2)

    g_min = max(g(max(goal_mvf, density_bounds[1])), g_ratio_bounds[1])
    g_max = min(g(density_bounds[2]), g_ratio_bounds[2])
    d_min = max(d(g_ratio_bounds[1]), density_bounds[1])
    d_max = min(d(g_ratio_bounds[2]), density_bounds[2])

    # Finds nearest (g, d) pair to (g0, d0) that is a solution,
    # also ensuring that (g, d) is within the boundaries
    function fixed_point_iter(g0, d0)
        g0, d0 = clamp(g0, g_min, g_max), clamp(d0, d_min, d_max)
        while abs((1 - g0^2) * d0 - goal_mvf) > 1e-14
            _g0, _d0 = g0, d0
            g0 = clamp((g(_d0) + _g0) / 2, g_min, g_max)
            d0 = clamp((d(_g0) + _d0) / 2, d_min, d_max)
        end
        return g0, d0
    end

    rand_g() = g_min + rand() * (g_max - g_min)
    rand_d() = d_min + rand() * (d_max - d_min)

    g_opt, d_opt =
        if solution_choice == :max_density
            fixed_point_iter(g(d_max), d_max)
        elseif solution_choice == :min_g_ratio
            fixed_point_iter(g_min, d(g_min))
        elseif solution_choice == :random_g_ratio
            _g = rand_g()
            fixed_point_iter(_g, d(_g))
        elseif solution_choice == :random_density
            _d = rand_d()
            fixed_point_iter(g(_d), _d)
        elseif solution_choice == :median
            fixed_point_iter((g_min+g_max)/2, (d_min+d_max)/2)
        elseif solution_choice == :random
            fixed_point_iter(rand_g(), rand_d())
        else
            error("Unknown solution choice: $solution_choice")
        end

    g_ratio, AxonPDensity, MVF = g_opt, d_opt, goal_mvf
    return @ntuple(g_ratio, AxonPDensity, MVF)
end

"""
Compute g-ratio and packing density from desired myelin volume fraction
using a refining grid search method. In the general case, this will produce
multiple solutions; the keyword `solution_choice` defines how the final
solution is chosen:
    :median     choose the solution which has g-ratio and density nearest to
                the median of all possible solutions
    :mean       same as :median, but using the mean
    :random     return a random valid solution
"""
function optimal_g_ratio_packdensity_gridsearch(goal_mvf;
        g_ratio_bounds = (0.60, 0.92),
        density_bounds = (0.15, 0.82),
        solution_choice = :median,
        iterations = 3, #number of resolution doublings during grid search
    )
    mvf_min = (1 - g_ratio_bounds[2]^2) * density_bounds[1]
    mvf_max = (1 - g_ratio_bounds[1]^2) * density_bounds[2]
    if !(mvf_min <= goal_mvf <= mvf_max)
        error("Goal mvf doesn't permit a solution for the given " *
              "g-ratio and density bounds; need mvf in [$mvf_min, $mvf_max]")
    end
    if solution_choice ∉ (:mean, :median, :random)
        error("Solution choice method must be one of :mean, :median, :random")
    end

    function F(gratio, density, mvf)
        return (1 - gratio^2) * density - mvf
    end
    ax1 = MDBM.Axis(range(g_ratio_bounds..., length = 10), "g") # initial grid in g-direction
    ax2 = MDBM.Axis(range(density_bounds..., length = 10), "d") # initial grid in d-direction
    prob = MDBM.MDBM_Problem((g,d) -> F(g,d,goal_mvf), [ax1,ax2])
    MDBM.solve!(prob, iterations)
    g_sol, d_sol = MDBM.getinterpolatedsolution(prob) # approximate interpolated solution points

    function fixed_point_iter!(gbuf,dbuf,g,d,mvf)
        gbuf .= sqrt.(1 .- mvf ./ d) # nearest g value
        dbuf .= mvf ./ (1 .- g.^2) # nearest d value
        g .= clamp.((gbuf .+ g) ./ 2, g_ratio_bounds...)
        d .= clamp.((dbuf .+ d) ./ 2, density_bounds...)
        return nothing
    end

    # Perform a simple fixed point iteration to increase accuracy
    # and to ensure candidate solutions are within desired bounds
    gbuf, dbuf = copy(g_sol), copy(d_sol)
    while maximum(abs, F.(g_sol, d_sol, goal_mvf)) > 1e-14
        fixed_point_iter!(gbuf, dbuf, g_sol, d_sol, goal_mvf)
    end

    g_ratio, AxonPDensity = if solution_choice == :random
        rand(g_sol), rand(d_sol)
    else
        goal_g, goal_d = if solution_choice == :median
            median(g_sol), median(d_sol)
        elseif solution_choice == :mean
            mean(g_sol), mean(d_sol)
        end
        _, index = findmin((g_sol .- goal_g).^2 .+ (d_sol .- goal_d).^2)
        g_sol[index], d_sol[index]
    end
    MVF = goal_mvf

    return @ntuple(g_ratio, AxonPDensity, MVF)
end

# function optimal_g_ratio_packdensity_gridsearch(goal_mvf;
#         g_ratio_goal = 0.78,
#         density_goal = 0.70,
#         g_ratio_bounds = (0.5, 0.9),
#         density_bounds = (0.6, 0.8)
#     )
#     
#     x_goal = [g_ratio_goal, density_goal]
#     fun(x) =  sum(abs2, x .- x_goal)
#     fun_grad!(g, x) = (g .= 2 .* (x .- x_goal))
#     fun_hess!(h, x) = (h .= Matrix(I,2,2))
# 
#     CON_FACT = 1e3 # Hack to make constraint more strongly enforced
#     con!(c, x) = (c[1] = CON_FACT * (x[1] - periodic_g_ratio(x[2], goal_mvf)); c)
#     con_J!(J, x) = (J[1,1] = CON_FACT; J[1,2] = -CON_FACT * goal_mvf / (2 * x[2]^2 * periodic_g_ratio(x[2], goal_mvf)); J)
#     con_H!(h, x, λ) = h[2,2] += λ[1] * CON_FACT * (goal_mvf^2 / (4 * x[2]^4 * periodic_g_ratio(x[2], goal_mvf)^3) + goal_mvf / (x[2]^3 * periodic_g_ratio(x[2], goal_mvf)))
#     
#     x0 = [g_ratio_goal, density_goal]
#     lx = [g_ratio_bounds[1], density_bounds[1]]
#     ux = [g_ratio_bounds[2], density_bounds[2]]
#     df = Optim.TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
#     dfc = Optim.TwiceDifferentiableConstraints(con!, con_J!, con_H!, lx, ux, [0.0], [0.0])
#     res = Optim.optimize(df, dfc, x0, Optim.IPNewton())
#     
#     g_ratio, AxonPDensity = Optim.minimizer(res)
#     MVF = periodic_mvf(AxonPDensity, g_ratio)
# 
#     if !isapprox(goal_mvf, MVF; rtol = 1e-3)
#         @warn "Desired MVF ($goal_mvf) couldn't be reached: MVF = $MVF"
#     end
# 
#     return @ntuple(g_ratio, AxonPDensity, MVF)
# end

# """
# If we want to generate a grid with a given MVF, we have two free parameters to
# choose, namely the g-ratio and the packing density. Ideally, the MVF would
# uniquely determine the other corresponding parameters; this requires an assumed
# relationship between two of the parameters.
# 
# Here, we choose to minimize the differences between the g-ratio and the
# packing density from a prior goal value. In other words, we solve
# 
#     min (g_goal - g_ratio)^2 + (density_goal - density)^2
# 
# subject to
# 
#     density = mvf / (1 - g_ratio^2)
#     g_ratio ∈ g_ratio_bounds
#     density ∈ density_bounds
# """
# function optimal_g_ratio_packdensity_linesearch(mvf;
#         g_ratio_goal = 0.75,
#         density_goal = 0.65,
#         g_ratio_bounds = (0.6, 0.9),
#         density_bounds = (0.6, 0.8)
#     )
#     g_low, g_high = g_ratio_bounds
#     η_low, η_high = density_bounds
#     if periodic_packdensity(mvf, g_ratio_bounds[1]) < η_low
#         @show periodic_packdensity(mvf, g_ratio_bounds[1])
#         @show η_low
#         @assert η_low > mvf
#         g_low = min(g_ratio_bounds[2], max(g_ratio_bounds[1], periodic_g_ratio(max(η_low, mvf + √eps(mvf)), mvf)))
#     end
#     if periodic_packdensity(mvf, g_ratio_bounds[2]) > η_high
#         @show periodic_packdensity(mvf, g_ratio_bounds[2])
#         @show η_high
#         @assert η_high > mvf
#         g_high = max(g_ratio_bounds[1], min(g_ratio_bounds[2], periodic_g_ratio(max(η_high, mvf + √eps(mvf)), mvf)))
#     end
#     g_ratio_bounds = (g_low, g_high)
#     @show g_ratio_bounds
# 
#     if g_ratio_bounds[1] >= g_ratio_bounds[2]
#         @warn "No solution found in search range; returning highest density solution"
#         g_ratio = max(g_ratio_bounds...)
#         AxonPDensity = periodic_packdensity(mvf, g_ratio)
#     else
#         optfun = g -> (g - g_ratio_goal)^2 + (periodic_packdensity(mvf, g) - density_goal)^2
#         res = Optim.optimize(optfun, g_ratio_bounds...)
#         
#         display(res) #TODO
#     
#         g_ratio = Optim.minimizer(res)
#         AxonPDensity = periodic_packdensity(mvf, g_ratio)
#     end
#     MVF = periodic_mvf(AxonPDensity, g_ratio)
# 
#     return @ntuple(g_ratio, AxonPDensity, MVF)
# end