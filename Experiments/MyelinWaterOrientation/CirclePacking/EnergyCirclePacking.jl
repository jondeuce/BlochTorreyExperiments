# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

module EnergyCirclePacking

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using GeometryUtils
using CirclePackingUtils
using LinearAlgebra, Statistics
using DiffBase, Optim, LineSearches, ForwardDiff, Roots
using Tensors
# using Parameters: @with_kw
# using JuAFEM

export pack

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #

function pack(
        radii::AbstractVector,
        ::Val{DIM} = Val(2);
        initial_origins::AbstractVector{<:Vec{DIM}} = initialize_origins(radii),
        goaldensity = 0.8,
        distancescale = mean(radii),
        weights::AbstractVector = [1.0, 1e-6, 1.0],
        epsilon::Real = 0.1*distancescale,
        autodiff::Bool = false,
        chunksize::Int = 10,
        secondorder::Bool = false,
        setcallback = false,
        Alg = secondorder ? Newton(linesearch = LineSearches.BackTracking(order=3))
                          : LBFGS(linesearch = LineSearches.BackTracking(order=3)),
        Opts = Optim.Options(iterations = 100_000,
                             x_tol = 1e-6*distancescale,
                             g_tol = 1e-12,
                             allow_f_increases = true)
        ) where {DIM}

    # Initial circles
    T, N = eltype(radii), length(initial_origins)
    initial_circles = Circle.(T, initial_origins, radii)
    x0 = copy(reinterpret(T, origin.(initial_circles)))

    # `pairwise_sum` and derivatives
    D²_mutual = (o1,o2,r1,r2) -> CirclePackingUtils.d²(o1,o2,r1,r2,epsilon)
    ∇D²_mutual = (o1,o2,r1,r2) -> CirclePackingUtils.∇d²(o1,o2,r1,r2,epsilon)
    ∇²D²_mutual = (o1,o2,r1,r2) -> CirclePackingUtils.∇²d²(o1,o2,r1,r2,epsilon)
    D²_overlap = (o1,o2,r1,r2) -> CirclePackingUtils.d²_overlap(o1,o2,r1,r2,epsilon)
    ∇D²_overlap = (o1,o2,r1,r2) -> CirclePackingUtils.∇d²_overlap(o1,o2,r1,r2,epsilon)
    ∇²D²_overlap = (o1,o2,r1,r2) -> CirclePackingUtils.∇²d²_overlap(o1,o2,r1,r2,epsilon)

    f_mutual = (x) -> pairwise_sum(D²_mutual, x, radii)
    g_mutual! = (out, x) -> pairwise_grad!(out, ∇D²_mutual, x, radii)
    h_mutual! = (out, x) -> pairwise_hess!(out, ∇²D²_mutual, x, radii)
    # fg_mutual! = (out, x) -> (g_mutual!(out, x); return f_mutual(x))
    f_overlap = (x) -> pairwise_sum(D²_overlap, x, radii)
    g_overlap! = (out, x) -> pairwise_grad!(out, ∇D²_overlap, x, radii)
    h_overlap! = (out, x) -> pairwise_hess!(out, ∇²D²_overlap, x, radii)

    # Scaling factor: average over all pairs, and normalize by typical squared distance
    Npairs = div(N*(N-1), 2)
    scalefactor = inv(Npairs * distancescale^2)

    function energy(x)
        T = eltype(x)
        E_total = zero(T)
        # !(weights[1] ≈ zero(T)) && (E_total += energy_covariance(circles))
        !(weights[2] ≈ zero(T)) && (E_total += weights[2] * f_mutual(x))
        !(weights[3] ≈ zero(T)) && (E_total += weights[3] * f_overlap(x))
        return scalefactor * E_total
    end

    g_buffer, h_buffer = zeros(T, DIM*N), zeros(T, DIM*N, DIM*N)
    up_g!(g, f!, x, w) = (f!(g_buffer, x); return axpy!(w, g_buffer, g))
    up_h!(h, f!, x, w) = (f!(h_buffer, x); return axpy!(w, h_buffer, h))

    function ∇energy!(g, x)
        T = eltype(x)
        g = fill!(g, zero(T))
        # !(weights[1] ≈ zero(T)) && (E_total += energy_covariance(circles))
        !(weights[2] ≈ zero(T)) && (g = up_g!(g, g_mutual!, x, scalefactor * weights[2]))
        !(weights[3] ≈ zero(T)) && (g = up_g!(g, g_overlap!, x, scalefactor * weights[3]))
        return g
    end

    function ∇²energy!(h, x)
        T = eltype(x)
        h = fill!(h, zero(T))
        # !(weights[1] ≈ zero(T)) && (E_total += energy_covariance(circles))
        !(weights[2] ≈ zero(T)) && (h = up_h!(h, h_mutual!, x, scalefactor * weights[2]))
        !(weights[3] ≈ zero(T)) && (h = up_h!(h, h_overlap!, x, scalefactor * weights[3]))
        return h
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
    if setcallback
        optfields = fieldnames(typeof(Opts))
        optvalues = getfield.(Ref(Opts), optfields)
        optdict = Dict(zip(optfields, optvalues))
        Opts = Optim.Options(; optdict...,
            callback = state -> check_density_callback(state, radii, T(goaldensity), T(epsilon/2)),
            extended_trace = true)
    end
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = copy(Optim.minimizer(result))
    packed_circles = tocircles(x, radii)

    # Scale to desired density, if possible
    packed_circles = scale_to_density(packed_circles, goaldensity)

    return packed_circles
end

function pack(c::AbstractVector{Circle{DIM,T}}; kwargs...) where {DIM,T}
    x0, r = tovectors(c)
    x0 = Vector(reinterpret(Vec{DIM,T}, x0))
    return pack(r, Val(DIM); initial_origins = x0, kwargs...)
end

end # module EnergyCirclePacking

nothing
