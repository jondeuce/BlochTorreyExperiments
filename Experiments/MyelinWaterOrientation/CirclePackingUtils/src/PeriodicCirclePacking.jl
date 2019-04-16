# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

module PeriodicCirclePacking

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using ..CirclePackingUtils
using GeometryUtils
using LinearAlgebra, Statistics
using DiffResults, Optim, LineSearches, ForwardDiff
using Tensors

export pack

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #

function pack(
        radii::AbstractVector;
        initial_origins::AbstractVector{V} = initialize_origins(radii; distribution = :uniformsquare),
        distancescale = mean(radii),
        epsilon::Real = 0.1*distancescale,
        autodiff::Bool = false,
        chunksize::Int = 10,
        secondorder::Bool = false,
        Alg = secondorder ? Newton(linesearch = LineSearches.BackTracking(order=3))
                          : LBFGS(linesearch = LineSearches.BackTracking(order=3)),
        Opts = Optim.Options(iterations = 100_000,
                             x_tol = 1e-6*distancescale,
                             g_tol = 1e-12,
                             allow_f_increases = true)
    ) where {V <: Vec{2}}

    # Initial unknowns
    Rmax = maximum(radii)
    xmin, xmax = extrema((x->x[1]).(initial_origins))
    ymin, ymax = extrema((x->x[2]).(initial_origins))
    W0, H0 = (xmax - xmin) + 2*Rmax, (ymax - ymin) + 2*Rmax
    x0 = [reinterpret(eltype(V), initial_origins .- Ref(V((xmin, ymin)))); W0; H0]

    # Create energy function and gradient/hessian
    energy = barrier_energy(radii, epsilon)
    # energy, ∇energy!, ∇²energy! = barrier_energy(radii, epsilon)

    # Form (*)Differentiable object
    if autodiff
        # Forward mode automatic differentiation
        if secondorder
            opt_obj = TwiceDifferentiable(energy, x0; autodiff = :forward)
        else
            opt_obj = OnceDifferentiable(energy, x0; autodiff = :forward)
        end
    else
        if secondorder
            opt_obj = TwiceDifferentiable(energy, ∇energy!, ∇²energy!, x0)
        else
            opt_obj = OnceDifferentiable(energy, ∇energy!, x0)
        end
    end

    # Optimize and get results
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = copy(Optim.minimizer(result))
    packed_circles = tocircles(x[1:end-2], radii)

    centre = mean(origin, packed_circles)
    widths = Vec{2}((x[end-1], x[end]))
    boundary_rectangle = Rectangle(centre - widths/2, centre + widths/2)

    return packed_circles, boundary_rectangle
end

function pack(c::AbstractVector{Circle{2,T}}; kwargs...) where {T}
    o, r = tovectors(c)
    o = reinterpret(Vec{2,T}, o) |> copy
    return pack(r; initial_origins = o, kwargs...)
end

# function barrier_energy(r::AbstractVector, ϵ::Real)
#     # Mutual distance and overlap distance squared functions, gradients, and hessians
#     @inline b(o1,o2,r1,r2)   = CirclePackingUtils.d²(o1,o2,r1,r2,ϵ) + CirclePackingUtils.barrier(o1,o2,r1,r2,ϵ)
#     @inline ∇b(o1,o2,r1,r2)  = CirclePackingUtils.∇d²(o1,o2,r1,r2,ϵ) + CirclePackingUtils.∇barrier(o1,o2,r1,r2,ϵ)
#     @inline ∇²b(o1,o2,r1,r2) = CirclePackingUtils.∇²d²(o1,o2,r1,r2,ϵ) + CirclePackingUtils.∇²barrier(o1,o2,r1,r2,ϵ)
# 
#     # Energy function/gradient/hessian
#     energy(x) = pairwise_sum(b, x, r)
#     ∇energy!(g, x) = pairwise_grad!(g, ∇b, x, r)
#     ∇²energy!(h, x) = pairwise_hess!(h, ∇²b, x, r)
# 
#     return energy, ∇energy!, ∇²energy!
# end

function dx_periodic(x1::Vec{2}, x2::Vec{2}, P::Vec{2})
    dx = x1 - x2
    dx = Vec{2}((
        min(abs(dx[1]), abs(dx[1] + P[1]), abs(dx[1] - P[1])),
        min(abs(dx[2]), abs(dx[2] + P[2]), abs(dx[2] - P[2]))
    ))
    return dx
end

function barrier_energy(r::AbstractVector, ϵ::Real)
    function energy(x)
        P = Vec{2}((x[end-1], x[end]))
        @inline function b(o1,o2,r1,r2)
            dx = dx_periodic(o1,o2,P)
            CirclePackingUtils.d²(dx,r1,r2,ϵ) + CirclePackingUtils.barrier(dx,r1,r2,ϵ)
        end
        pairwise_sum(b, @views(x[1:end-2]), r)
    end
end

end # module PeriodicCirclePacking

nothing
