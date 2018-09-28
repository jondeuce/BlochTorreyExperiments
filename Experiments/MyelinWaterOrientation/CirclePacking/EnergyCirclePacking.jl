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
        radii::AbstractVector;
        initial_origins::AbstractVector{V} = initialize_origins(radii),
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
    ) where {V <: Vec{2}}

    # Initial circles
    T, N = eltype(radii), length(initial_origins)
    initial_circles = Circle.(T, initial_origins, radii)
    x0 = copy(reinterpret(T, origin.(initial_circles)))

    # Scaling factor: average over all pairs, and normalize by typical squared distance
    Npairs = div(N*(N-1), 2)
    scalefactor = inv(Npairs * distancescale^2)

    # Create energy function and gradient/hessian
    energy, ∇energy!, ∇²energy! = create_energy(radii; w = weights, ϵ = epsilon, β = scalefactor)

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

    # Add callback which stops minimization when goaldensity is reached/exceeded
    setcallback && (Opts = add_callback(Opts, radii; η = goaldensity, ϵ = epsilon))

    # Optimize and get results
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = copy(Optim.minimizer(result))
    packed_circles = tocircles(x, radii)

    # Scale to desired density, if possible
    packed_circles = scale_to_density(packed_circles, goaldensity)

    return packed_circles
end

function pack(c::AbstractVector{Circle{2,T}}; kwargs...) where {T}
    x0, r = tovectors(c)
    x0 = Vector(reinterpret(Vec{2,T}, x0))
    return pack(r; initial_origins = x0, kwargs...)
end

function add_callback(
        Opts, # options to update
        r::AbstractVector{T}; # circle radii
        η::T = one(T), # goal density
        ϵ::T = zero(T) # circle overlap distance threshold
    ) where {T}

    # TODO: this is such a hack... better way?
    optfields = fieldnames(typeof(Opts))
    optvalues = getfield.(Ref(Opts), optfields)
    optdict = Dict(zip(optfields, optvalues))

    return Optim.Options(; optdict...,
        callback = state -> check_density_callback(state, r, η, ϵ),
        extended_trace = true)
end

function create_energy(
        r::AbstractVector{T}; # circle radii
        w::AbstractVector{T} = zeros(T, 3), # weight for each energy term
        ϵ::T = zero(T), # circle overlap distance threshold
        β::T = one(T) # overall scale factor
    ) where {T}

    # Mutual distance and overlap distance squared functions, gradients, and hessians
    @inline D²_mutual(o1,o2,r1,r2)    = CirclePackingUtils.d²(o1,o2,r1,r2,ϵ)
    @inline ∇D²_mutual(o1,o2,r1,r2)   = CirclePackingUtils.∇d²(o1,o2,r1,r2,ϵ)
    @inline ∇²D²_mutual(o1,o2,r1,r2)  = CirclePackingUtils.∇²d²(o1,o2,r1,r2,ϵ)
    @inline D²_overlap(o1,o2,r1,r2)   = CirclePackingUtils.d²_overlap(o1,o2,r1,r2,ϵ)
    @inline ∇D²_overlap(o1,o2,r1,r2)  = CirclePackingUtils.∇d²_overlap(o1,o2,r1,r2,ϵ)
    @inline ∇²D²_overlap(o1,o2,r1,r2) = CirclePackingUtils.∇²d²_overlap(o1,o2,r1,r2,ϵ)

    # `pairwise_sum` and derivatives
    f_mutual(x)        = pairwise_sum(D²_mutual, x, r)
    g_mutual!(out, x)  = pairwise_grad!(out, ∇D²_mutual, x, r)
    h_mutual!(out, x)  = pairwise_hess!(out, ∇²D²_mutual, x, r)
    f_overlap(x)       = pairwise_sum(D²_overlap, x, r)
    g_overlap!(out, x) = pairwise_grad!(out, ∇D²_overlap, x, r)
    h_overlap!(out, x) = pairwise_hess!(out, ∇²D²_overlap, x, r)

    # Buffers for gradient and hessian calculations, as well as update functions
    N = length(r)
    g_buffer = zeros(T, 2N)
    h_buffer = zeros(T, 2N, 2N)
    @inline up_g!(g, f!, x, α) = (f!(g_buffer, x); return axpy!(α, g_buffer, g))
    @inline up_h!(h, f!, x, α) = (f!(h_buffer, x); return axpy!(α, h_buffer, h))

    # Energy function
    function energy(x)
        T = eltype(x)
        E_total = zero(T)
        # !(w[1] ≈ zero(T)) && (E_total += energy_covariance(circles))
        !(w[2] ≈ zero(T)) && (E_total += w[2] * f_mutual(x))
        !(w[3] ≈ zero(T)) && (E_total += w[3] * f_overlap(x))
        return β * E_total
    end

    # Energy function gradient
    function ∇energy!(g, x)
        T = eltype(x)
        g = fill!(g, zero(T))
        # !(w[1] ≈ zero(T)) && (E_total += energy_covariance(circles))
        !(w[2] ≈ zero(T)) && (g = up_g!(g, g_mutual!, x, β * w[2]))
        !(w[3] ≈ zero(T)) && (g = up_g!(g, g_overlap!, x, β * w[3]))
        return g
    end

    # Energy function hessian
    function ∇²energy!(h, x)
        T = eltype(x)
        h = fill!(h, zero(T))
        # !(w[1] ≈ zero(T)) && (E_total += energy_covariance(circles))
        !(w[2] ≈ zero(T)) && (h = up_h!(h, h_mutual!, x, β * w[2]))
        !(w[3] ≈ zero(T)) && (h = up_h!(h, h_overlap!, x, β * w[3]))
        return h
    end

    return energy, ∇energy!, ∇²energy!
end

end # module EnergyCirclePacking

nothing