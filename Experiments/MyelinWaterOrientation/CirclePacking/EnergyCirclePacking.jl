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
# Types
# ---------------------------------------------------------------------------- #

# Struct for holding parameters which are fixed w.r.t. the minimization
struct OptData{dim,T}
    radii::AbstractVector{T}
    first_origin::Vec{dim,T}
    is_first_origin_fixed::Bool
    is_x_coord_fixed::Bool
end
numcircles(data::OptData) = length(data.radii)

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
        constrained::Bool = false, # autodiff && secondorder,
        reversemode::Bool = false, # autodiff && !secondorder,
        setcallback = true,
        Alg = secondorder ? Newton(linesearch = LineSearches.BackTracking(order=3))
                          : LBFGS(linesearch = LineSearches.BackTracking(order=3)),
        Opts = Optim.Options(iterations = 100_000,
                             #x_tol = 1e-6*distancescale,
                             g_tol = 1e-12,
                             allow_f_increases = true)
        ) where {DIM}

    # Initial circles
    T = eltype(radii)
    initial_circles = Circle.(T, initial_origins, radii)

    # One circle must be fixed, else optimization is ill-defined (metric would
    # be translation invariant)
    c_fixed = initial_circles[1]
    c_variable = initial_circles[2:end]

    # x0 = copy(reinterpret(T, origin.(c_variable)))[2:end]; o_fixed = true; x_fixed = true
    # x0 = copy(reinterpret(T, origin.(c_variable))); o_fixed = true; x_fixed = false
    x0 = copy(reinterpret(T, origin.(initial_circles))); o_fixed = false; x_fixed = false
    data = OptData(radii, origin(c_fixed), o_fixed, x_fixed)

    if constrained
        # Constrained problem using Lagrange multipliers
        push!(x0, one(T)) # push initial Lagrange multiplier
        g = x -> packing_energy(x, data, goaldensity, distancescale, cat(1, weights[1:2], x[end]), epsilon)
        f = x -> sum(abs2, ForwardDiff.gradient(g, x))
    else
        # Unconstrained problem with penalty on overlap
        f = x -> packing_energy(x, data, goaldensity, distancescale, weights, epsilon)
    end

    # Form *Differentiable object
    if autodiff
        if secondorder
            opt_obj = TwiceDifferentiable(f, x0; autodiff = :forward)
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
            # g!, fg! = wrap_gradient(f, x0, Val{chunksize}; isforward = true)
            # opt_obj = OnceDifferentiable(f, g!, fg!, x0)

            # ---- Simple precompiled gradient, but can't configure chunk ---- #
            opt_obj = OnceDifferentiable(f, x0; autodiff = :forward)
        end
    else
        opt_obj = f
    end

    # Optimize and get results
    if setcallback
        optfields = fieldnames(typeof(Opts))
        optvalues = getfield.(Ref(Opts), optfields)
        optdict = Dict(zip(optfields, optvalues))
        Opts = Optim.Options(; optdict...,
            callback = state -> check_density_callback(state, data.radii, T(goaldensity), T(epsilon/2)),
            extended_trace = true)
    end
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = constrained ? copy(Optim.minimizer(result)[1:end-1]) : copy(Optim.minimizer(result))

    # Reconstruct resulting circles
    packed_circles = getcircles(x, data)

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

function getcircles!(
        circles::Vector{Circle{DIM,Tx}},
        x::AbstractVector{Tx},
        data::OptData{DIM,Tf}
        ) where {DIM,Tx,Tf}
    # There are two sets of a fixed data:
    #   -> The fixed circle c_0, the first circle == circles[1]
    #   -> The x-coordinate of the second circle == 0.0 to fix the rotation of the system
    # Therefore, x is a vector where x[1] is the y-coordinate of the second circle, and
    # then alternating x/y coordinates. I.e., x == [y2,x3,y3,x4,y4,...,xN,yN]
    N = numcircles(data)
    @assert length(circles) == N

    if data.is_first_origin_fixed
        @inbounds circles[1] = Circle{DIM,Tx}(Vec{DIM,Tx}(Tuple(data.first_origin)), Tx(data.radii[1]))
        if data.is_x_coord_fixed
            @inbounds circles[2] = Circle{DIM,Tx}(Vec{DIM,Tx}((Tx(data.first_origin[1]), x[1])), Tx(data.radii[2]))
            @inbounds for (j,i) in enumerate(3:N)
                circles[i] = Circle{DIM,Tx}(Vec{DIM,Tx}((x[2j], x[2j+1])), Tx(data.radii[i]))
            end
        else
            @inbounds for (j,i) in enumerate(2:N)
                circles[i] = Circle{DIM,Tx}(Vec{DIM,Tx}((x[2j-1], x[2j])), Tx(data.radii[i]))
            end
        end
    else
        @inbounds for i in 1:N
            circles[i] = Circle{DIM,Tx}(Vec{DIM,Tx}((x[2i-1], x[2i])), Tx(data.radii[i]))
        end
    end

    return circles
 end

function getcircles(
        x::AbstractVector{Tx},
        data::OptData{DIM,Tf}
        ) where {DIM,Tx,Tf}
    circles = Vector{Circle{DIM,Tx}}(undef, numcircles(data))
    getcircles!(circles, x, data)
    return circles
end

# Packing energy (unconstrained problem)
function packing_energy(
        circles::Vector{Circle{DIM,Tx}},
        goaldensity::Real = 0.8,
        distancescale::Real = mean(radii),
        weights::AbstractVector = Tx[1.0, 1e-6, 1.0],
        epsilon::Real = 0.1*distancescale
        ) where {DIM,Tx}
    # Initialize energies
    E_density, E_mutual, E_overlap = zero(Tx), zero(Tx), zero(Tx)

    # # Penalize by goaldensity
    # if !(weights[1] ≈ zero(Tf))
    #     # E_density = (goaldensity - estimate_density(c_0,origins,radii))^2
    #     E_density = (goaldensity - estimate_density(circles))^2
    # end

    # Penalize by non-circularness of distribution
    if !(weights[1] ≈ zero(Tx))
        E_density = energy_covariance(circles)/distancescale^2
    end

    # Using the overlap as the only metric clearly will not work, as any
    # isolated set of circles will have zero energy. Therefore, we penalize by
    # the total squared distances to encourage the circles to stay close
    if !(weights[2] ≈ zero(Tx))
        # E_mutual = energy_sum_squared_distances(c_0,origins,radii,Val(DIM))/distancescale^2
        E_mutual = energy_sum_squared_distances(circles,Val(DIM))/distancescale^2
    end
    if !(weights[3] ≈ zero(Tx))
        # E_overlap = energy_sum_overlap_squared_distances(c_0,origins,radii,epsilon,Val(DIM))/distancescale^2
        E_overlap = energy_sum_overlap_squared_distances(circles,epsilon,Val(DIM))/distancescale^2
    end

    # We could also interpret the "packing energy" instead as the Lagrangian
    # for the constrained problem where lambda is a Lagrange multiplier and the
    # overlap energy is constrained to be exactly zero (which occurs whenever
    # there are no overlapping circles)
    E_total = weights[1]*E_density + weights[2]*E_mutual + weights[3]*E_overlap

    return E_total

end

# Packing energy (unconstrained problem)
function packing_energy(
        x::AbstractVector{Tx},
        data::OptData{DIM,Tf},
        goaldensity::Real = 0.8,
        distancescale::Real = mean(radii),
        weights::AbstractVector = Tx[1.0, 1e-6, 1.0],
        epsilon::Real = 0.1*distancescale
        ) where {DIM,Tx,Tf}
    # Initialize circles
    circles = getcircles(x, data)
    return packing_energy(circles, goaldensity, distancescale, weights, epsilon)
end


# ---------------------------------------------------------------------------- #
# Energies on circles: ForwardDiff friendly with circles vector
# ---------------------------------------------------------------------------- #

# Sum squared circle distances
function energy_covariance(circles::Vector{Circle{DIM,T}}) where {DIM,T}
    circlepoints = reshape(reinterpret(T, circles), (DIM+1, length(circles))) # reinterp as DIM+1 x Ncircles array
    @views origins = circlepoints[1:2, :] # DIM x Ncircles view of origin points
    Σ = cov(origins; dims = 2) # covariance matrix of origin locations
    σ² = T(tr(Σ)/DIM) # mean variance
    return sum(abs2, Σ - σ²*I) # penalize non-diagonal covariance matrices
end

# Sum squared circle distances
function energy_sum_squared_distances(
        circles::Vector{Circle{DIM,T}},
        ::Val{DIM} = Val(2)
        ) where {DIM,T}

    N = length(circles)
    E = zero(T)

    @inbounds for i in 1:N-1
        c_i = circles[i]
        for j in i+1:N
            c_j = circles[j]
            d_ij = signed_edge_distance(c_i, c_j)
            E += d_ij^2
        end
    end

    return E
end

# Sum squared distances only from overlapping circles. The parameter `epsilon`,
# which defaults to zero, allows for a overlapping threshold. If epsilon > 0,
# then the overlapping energy will be counted if the circles are closer than
# `epsilon` distance apart. Similarly, if epsilon < 0, then the overlapping
# energy will only be counted only if the circles are overlapping by more than a
# distance of abs(epsilon)
function energy_sum_overlap_squared_distances(
        circles::Vector{Circle{DIM,T}},
        epsilon::Real = zero(T),
        ::Val{DIM} = Val(2)
        ) where {DIM,T}

    N = length(circles)
    E = zero(T)
    ϵ = T(epsilon)

    @inbounds for i in 1:N-1
        c_i = circles[i]
        d²_overlap = zero(T)
        for j in i+1:N
            c_j = circles[j]
            d_ij = signed_edge_distance(c_i, c_j)
            d_ij < ϵ && (d²_overlap += (d_ij-ϵ)^2)
        end
        E += d²_overlap
    end

    return E
end

end # module EnergyCirclePacking

nothing
