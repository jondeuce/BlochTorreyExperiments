# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

module PeriodicCirclePacking

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using ..CirclePackingUtils
using ..CirclePackingUtils: d², ∇d², ∇²d², barrier, ∇barrier, ∇²barrier
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
    xmin, xmax = extrema(x[1] for x in initial_origins)
    ymin, ymax = extrema(x[2] for x in initial_origins)
    W0, H0 = (xmax - xmin) + 2*Rmax, (ymax - ymin) + 2*Rmax # initial width/height of box
    os = initial_origins .- Ref(V((xmin - Rmax, ymin - Rmax))) # ensure all circles are in box
    x0 = [reinterpret(eltype(V), os); W0; H0] # initial unknowns

    # Create energy function and gradient/hessian
    local energy, ∇energy!, ∇²energy!
    if autodiff
        energy = autodiff_barrier_energy(radii, epsilon)
    else
        energy, ∇energy!, ∇²energy! = barrier_energy(radii, epsilon)
    end

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

# @inline function dx_periodic(x1::Number, x2::Number, P::Number)
#     T = promote_type(typeof(x1), typeof(x2), typeof(P))
#     dx = x1 - x2
#     return T(dx > P/2 ? dx - P : dx < -P/2 ? dx + P : dx)
# end
@inline dx_periodic(x1::Number, x2::Number, P::Number) = mod(x1 - x2 + P/2, P) - P/2
@inline dx_periodic(x1::Vec{2}, x2::Vec{2}, P::Vec{2}) = Vec{2}((dx_periodic(x1[1],x2[1],P[1]), dx_periodic(x1[2],x2[2],P[2])))

function barrier_energy(r::AbstractVector, ϵ::Real)
    # Mutual distance and overlap distance squared functions, gradients, and hessians
    @inline get_b(P) = (o1,o2,r1,r2) -> (dx = dx_periodic(o1,o2,P); d²(dx,r1,r2,ϵ) + barrier(dx,r1,r2,ϵ))
    @inline get_∇b(P) = (o1,o2,r1,r2) -> (dx = dx_periodic(o1,o2,P); ∇d²(dx,r1,r2,ϵ) + ∇barrier(dx,r1,r2,ϵ))
    @inline get_∇²b(P) = (o1,o2,r1,r2) -> (dx = dx_periodic(o1,o2,P); ∇²d²(dx,r1,r2,ϵ) + ∇²barrier(dx,r1,r2,ϵ))

    # Energy function/gradient/hessian
    function energy(x)
        o, P = @views(x[1:end-2]), Vec{2}((x[end-1], x[end]))
        return pairwise_sum(get_b(P), o, r)
    end

    function ∇energy!(g, x)
        o, P = @views(x[1:end-2]), Vec{2}((x[end-1], x[end]))
        pairwise_grad!(@views(g[1:end-2]), get_∇b(P), o, r)
        @views g[end-1:end] .= Tensors.gradient(P -> pairwise_sum(get_b(P), o, r), P)
        return g
    end

    function ∇²energy!(h, x)
        o, P = @views(x[1:end-2]), Vec{2}((x[end-1], x[end]))
        pairwise_hess!(@views(h[1:end-2, 1:end-2]), get_∇²b(P), o, r)
        @views h[end-1:end, end-1:end] .= Tensors.hessian(P -> pairwise_sum(get_b(P), o, r), P)
        return h
    end

    return energy, ∇energy!, ∇²energy!
end

function autodiff_barrier_energy(r::AbstractVector, ϵ::Real)
    @inline get_b(P) = (o1,o2,r1,r2) -> (dx = dx_periodic(o1,o2,P); d²(dx,r1,r2,ϵ) + barrier(dx,r1,r2,ϵ))
    function energy(x)
        P = Vec{2}((x[end-1], x[end]))
        return pairwise_sum(get_b(P), @views(x[1:end-2]), r)
    end
end

end # module PeriodicCirclePacking

nothing
