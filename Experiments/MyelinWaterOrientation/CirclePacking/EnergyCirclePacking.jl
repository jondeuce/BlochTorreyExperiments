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
        autodiff::Bool = true,
        chunksize::Int = 10,
        secondorder::Bool = false,
        setcallback = true,
        Alg = secondorder ? Newton(linesearch = LineSearches.BackTracking(order=3))
                          : LBFGS(linesearch = LineSearches.BackTracking(order=3)),
        Opts = Optim.Options(iterations = 100_000,
                             #x_tol = 1e-6*distancescale,
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
    D²_overlap = (o1,o2,r1,r2) -> CirclePackingUtils.d²_overlap(o1,o2,r1,r2,epsilon)
    ∇D²_overlap = (o1,o2,r1,r2) -> CirclePackingUtils.∇d²_overlap(o1,o2,r1,r2,epsilon)

    f_mutual = (x) -> pairwise_sum(D²_mutual, x, radii)
    g_mutual! = (out, x) -> pairwise_grad!(out, ∇D²_mutual, x, radii)
    h_mutual! = (out, x) -> pairwise_hess!(out, ∇D²_mutual, x, radii)
    # fg_mutual! = (out, x) -> (g_mutual!(out, x); return f_mutual(x))
    f_overlap = (x) -> pairwise_sum(D²_overlap, x, radii)
    g_overlap! = (out, x) -> pairwise_grad!(out, ∇D²_overlap, x, radii)
    h_overlap! = (out, x) -> pairwise_hess!(out, ∇D²_overlap, x, radii)

    # Scaling factor: average over all pairs, and normalize by typical squared distance
    Npairs = div(N*(N-1), 2)
    scalefactor = inv(Npairs * distancescale^2)

    # packing_energy(x, radii, goaldensity, distancescale, weights, epsilon)
    function packing_energy(x)
        T = eltype(x)
        E_total = zero(T)
        # !(weights[1] ≈ zero(T)) && (E_total += energy_covariance(circles))
        !(weights[2] ≈ zero(T)) && (E_total += f_mutual(x))
        !(weights[3] ≈ zero(T)) && (E_total += f_overlap(x))
        return scalefactor * E_total
    end

    gbuffer = similar(x0)
    function ∇packing_energy!(g, x)
        T = eltype(x)
        fill!(g, zero(T))
        # !(weights[1] ≈ zero(T)) && (E_total += energy_covariance(circles))
        !(weights[2] ≈ zero(T)) && (g .+= g_mutual!(g_buffer, x))
        !(weights[3] ≈ zero(T)) && (g .+= g_overlap!(g_buffer, x))
        rmul!(g, scalefactor)
        return g
    end

    # Form *Differentiable object
    if autodiff
        if secondorder
            opt_obj = TwiceDifferentiable(packing_energy, x0; autodiff = :forward)
        else
            # Forward mode automatic differentiation

            # ---- Use buffer of circles to avoid allocations ---- #
            # chunksize = min(chunksize, length(x0))
            # dualcircles = Vector{Circle{2,ForwardDiff.Dual{Nothing,T,chunksize}}}(undef, numcircles(data))
            # realcircles = Vector{Circle{2,T}}(undef, numcircles(data))
            #
            # function f_buffered(x::Vector{T}) where {T<:AbstractFloat}
            #     getcircles!(realcircles, x, data)
            #     return packing_energy(realcircles, goaldensity, distancescale, weights, epsilon)
            # end
            #
            # function f_buffered(x::Vector{D}) where {D<:ForwardDiff.Dual}
            #     dualcircles = reinterpret(Circle{DIM,D}, dualcircles)
            #     getcircles!(dualcircles, x, data)
            #     return packing_energy(dualcircles, goaldensity, distancescale, weights, epsilon)
            # end
            #
            # checktag = true
            # g!, fg!, cfg = wrap_gradient(f_buffered, x0, Val{chunksize}, Val{checktag}; isforward = true)
            # opt_obj = OnceDifferentiable(f_buffered, g!, fg!, x0)

            # ---- Simple precompiled gradient ---- #
            # g!, fg! = wrap_gradient(packing_energy, x0, Val{chunksize}; isforward = true)
            # opt_obj = OnceDifferentiable(packing_energy, g!, fg!, x0)

            # ---- Simple precompiled gradient, but can't configure chunk ---- #
            opt_obj = OnceDifferentiable(packing_energy, x0; autodiff = :forward)
        end
    else
        opt_obj = OnceDifferentiable(packing_energy, ∇packing_energy!, x0)
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

    # Reconstruct resulting circles
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

# ---------------------------------------------------------------------------- #
# Energies on circles
# ---------------------------------------------------------------------------- #

# # Packing energy (unconstrained problem)
# function packing_energy(
#         circles::Vector{Circle{DIM,Tx}},
#         goaldensity::Real = 0.8,
#         distancescale::Real = mean(radii),
#         weights::AbstractVector = Tx[1.0, 1e-6, 1.0],
#         epsilon::Real = 0.1*distancescale
#         ) where {DIM,Tx}
#     # Initialize energies
#     E_density, E_mutual, E_overlap = zero(Tx), zero(Tx), zero(Tx)
#
#     # # Penalize by goaldensity
#     # if !(weights[1] ≈ zero(Tf))
#     #     # E_density = (goaldensity - estimate_density(c_0,origins,radii))^2
#     #     E_density = (goaldensity - estimate_density(circles))^2
#     # end
#
#     # # Penalize by non-circularness of distribution
#     # if !(weights[1] ≈ zero(Tx))
#     #     E_density = energy_covariance(circles)/distancescale^2
#     # end
#
#     # Using the overlap as the only metric clearly will not work, as any
#     # isolated set of circles will have zero energy. Therefore, we penalize by
#     # the total squared distances to encourage the circles to stay close
#     if !(weights[2] ≈ zero(Tx))
#         # E_mutual = energy_sum_squared_distances(c_0,origins,radii,Val(DIM))/distancescale^2
#         E_mutual = energy_sum_squared_distances(circles,Val(DIM))/distancescale^2
#     end
#     if !(weights[3] ≈ zero(Tx))
#         # E_overlap = energy_sum_overlap_squared_distances(c_0,origins,radii,epsilon,Val(DIM))/distancescale^2
#         E_overlap = energy_sum_overlap_squared_distances(circles,epsilon,Val(DIM))/distancescale^2
#     end
#
#     # We could also interpret the "packing energy" instead as the Lagrangian
#     # for the constrained problem where lambda is a Lagrange multiplier and the
#     # overlap energy is constrained to be exactly zero (which occurs whenever
#     # there are no overlapping circles)
#     E_total = weights[1]*E_density + weights[2]*E_mutual + weights[3]*E_overlap
#
#     return E_total
#
# end
#
# # Packing energy (unconstrained problem)
# function packing_energy(
#         x::AbstractVector{Tx},
#         data::OptData{DIM,Tf},
#         goaldensity::Real = 0.8,
#         distancescale::Real = mean(radii),
#         weights::AbstractVector = Tx[1.0, 1e-6, 1.0],
#         epsilon::Real = 0.1*distancescale
#         ) where {DIM,Tx,Tf}
#     # Initialize circles
#     circles = getcircles(x, data)
#     return packing_energy(circles, goaldensity, distancescale, weights, epsilon)
# end
#
#
# # ---------------------------------------------------------------------------- #
# # Energies on circles: ForwardDiff friendly with circles vector
# # ---------------------------------------------------------------------------- #
#
# # Sum squared circle distances
# function energy_covariance(circles::Vector{Circle{DIM,T}}) where {DIM,T}
#     circlepoints = reshape(reinterpret(T, circles), (DIM+1, length(circles))) # reinterp as DIM+1 x Ncircles array
#     @views origins = circlepoints[1:2, :] # DIM x Ncircles view of origin points
#     Σ = cov(origins; dims = 2) # covariance matrix of origin locations
#     σ² = T(tr(Σ)/DIM) # mean variance
#     return sum(abs2, Σ - σ²*I) # penalize non-diagonal covariance matrices
# end
#
# # Sum squared circle distances
# function energy_sum_squared_distances(
#         circles::Vector{Circle{DIM,T}},
#         ::Val{DIM} = Val(2)
#         ) where {DIM,T}
#
#     N = length(circles)
#     E = zero(T)
#
#     @inbounds for i in 1:N-1
#         c_i = circles[i]
#         for j in i+1:N
#             c_j = circles[j]
#             d_ij = signed_edge_distance(c_i, c_j)
#             E += d_ij^2
#         end
#     end
#
#     return E
# end
#
# # Sum squared distances only from overlapping circles. The parameter `epsilon`,
# # which defaults to zero, allows for a overlapping threshold. If epsilon > 0,
# # then the overlapping energy will be counted if the circles are closer than
# # `epsilon` distance apart. Similarly, if epsilon < 0, then the overlapping
# # energy will only be counted only if the circles are overlapping by more than a
# # distance of abs(epsilon)
# function energy_sum_overlap_squared_distances(
#         circles::Vector{Circle{DIM,T}},
#         epsilon::Real = zero(T),
#         ::Val{DIM} = Val(2)
#         ) where {DIM,T}
#
#     N = length(circles)
#     E = zero(T)
#     ϵ = T(epsilon)
#
#     @inbounds for i in 1:N-1
#         c_i = circles[i]
#         d²_overlap = zero(T)
#         for j in i+1:N
#             c_j = circles[j]
#             d_ij = signed_edge_distance(c_i, c_j)
#             d_ij < ϵ && (d²_overlap += (d_ij-ϵ)^2)
#         end
#         E += d²_overlap
#     end
#
#     return E
# end

end # module EnergyCirclePacking

nothing
