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
        energy, ∇energy!, _ = barrier_energy(radii, epsilon)
    end

    # Form (*)Differentiable object
    opt_obj = if autodiff
        secondorder ? TwiceDifferentiable(energy, x0; autodiff = :forward) : OnceDifferentiable(energy, x0; autodiff = :forward)
    else
        secondorder && @warn "Hessian not implemented; set autodiff = true for second order. Defaulting to first order."
        OnceDifferentiable(energy, ∇energy!, x0)
    end

    # Optimize and get results
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = copy(Optim.minimizer(result))
    widths = V((x[end-1], x[end]))
    origins = reinterpret(V, x[1:end-2]) |> copy
    origins .= periodic_mod.(origins, Ref(widths)) # force origins to be contained in [0,W] x [0,H]
    packed_circles = Circle.(origins, radii)
    boundary_rectangle = Rectangle(zero(V), widths)

    return packed_circles, boundary_rectangle
end

function pack(c::AbstractVector{Circle{2,T}}; kwargs...) where {T}
    o, r = tovectors(c)
    o = reinterpret(Vec{2,T}, o) |> copy
    return pack(r; initial_origins = o, kwargs...)
end

function barrier_energy(r::AbstractVector, ϵ::Real)
    # Mutual distance and overlap distance squared functions, gradients, and hessians
    @inline get_b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); d²(dx,r1,r2,ϵ) + barrier(dx,r1,r2,ϵ))
    @inline get_∇b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); ∇d²(dx,r1,r2,ϵ) + ∇barrier(dx,r1,r2,ϵ))
    @inline get_∇²b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); ∇²d²(dx,r1,r2,ϵ) + ∇²barrier(dx,r1,r2,ϵ))

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
        @views h[1:end-2, end-1:end] .= 0 #TODO this is an incorrect assumption
        @views h[end-1:end, 1:end-2] .= 0 #TODO this is an incorrect assumption
        return h
    end

    return energy, ∇energy!, ∇²energy!
end

function autodiff_barrier_energy(r::AbstractVector, ϵ::Real)
    @inline get_b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); d²(dx,r1,r2,ϵ) + barrier(dx,r1,r2,ϵ))
    function energy(x)
        P = Vec{2}((x[end-1], x[end]))
        return pairwise_sum(get_b(P), @views(x[1:end-2]), r)
    end
end

end # module PeriodicCirclePacking

nothing
