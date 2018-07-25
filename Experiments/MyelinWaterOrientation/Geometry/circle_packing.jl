# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #
function pack_circles(radii::AbstractVector,
                      λ::Real = convert(eltype(radii), 1e-4),
                      ::Type{Val{DIM}} = Val{2};
                      initial_origins::AbstractVector = initialize_origins(radii, Val{DIM}),
                      autodiff::Bool = true,
                      secondorder::Bool = false,
                      Alg = secondorder
                            ? Newton(linesearch = LineSearches.BackTracking(order=3))
                            : LBFGS(linesearch = LineSearches.BackTracking(order=3)),
                      Opts = Optim.Options(iterations = 100_000,
                                           g_tol = 1e-12,
                                           allow_f_increases = true)
                      ) where {DIM}
    # Initial circles
    initial_circles = Circle.(initial_origins, radii)

    # One circle must be fixed, else optimization is ill-defined (metric would
    # be translation invariant)
    c_fixed = initial_circles[1]
    c_variable = initial_circles[2:end]

    # Set optimization function and initial guess
    f = x -> packing_energy(c_fixed, x, radius.(c_variable), λ, Val{DIM})
    x0 = copy(reinterpret(floattype(c_fixed), origin.(c_variable)))

    # Optimize and get results
    if autodiff
        if secondorder
            diff_obj = TwiceDifferentiable(f, x0; autodiff = :forward)
            result = optimize(diff_obj, x0, Alg, Opts)
        else
            diff_obj = OnceDifferentiable(f, x0; autodiff = :forward)
            result = optimize(diff_obj, x0, Alg, Opts)
        end
    else
        result = optimize(f, x0, Alg, Opts)
    end
    x = copy(Optim.minimizer(result))

    # Reconstruct resulting circles
    packed_origins = [origin(c_fixed), copy(reinterpret(Vec{DIM,floattype(c_fixed)}, x))...]
    packed_circles = Circle.(packed_origins, radii)

    return packed_circles, result
end

function initialize_origins(radii::AbstractVector{T},
                            ::Type{Val{DIM}} = Val{2}) where {DIM,T}
    # Initialize with random origins
    Ncircles = length(radii)
    mesh_scale = T(2.0)*maximum(radii)*sqrt(Ncircles)
    initial_origins = mesh_scale .* (T(2.0).*rand(T,DIM*Ncircles).-one(T))
    initial_origins = reinterpret(Vec{DIM,T}, initial_origins)
    initial_origins .-= [initial_origins[1]] # shift such that initial_origins[1] is at the origin
    return initial_origins
end

# ---------------------------------------------------------------------------- #
# Energies on circles
# ---------------------------------------------------------------------------- #

# Packing energy
function packing_energy(c_0::Circle,
                        origins::AbstractVector,
                        radii::AbstractVector,
                        λ::Real = convert(eltype(radii), 1e-4),
                        ::Type{Val{DIM}} = Val{dimension(c_0)}) where {DIM}
    # Using the overlap as the only metric clearly will not work, as any
    # isolated set of circles will have zero energy. Therefore, we penalize by
    # the total squared distances to encourage the circles to stay close
    E_overlap = energy_sum_overlap_squared_distances(c_0,origins,radii,Val{DIM})
    E_mutual = energy_sum_squared_distances(c_0,origins,radii,Val{DIM})
    return E_overlap + λ*E_mutual
end

# Sum squared circle distances
function energy_sum_squared_distances(c_0::Circle,
                                      origins::AbstractVector,
                                      radii::AbstractVector,
                                      ::Type{Val{DIM}}) where {DIM}
    T = promote_type(eltype(origins), eltype(radii))
    N = length(radii)
    E = zero(T)

    @assert length(origins) == N*DIM
    origins = reinterpret(Vec{DIM,T}, origins)

    # One circle must be fixed, otherwise problem is ill-posed (translation invariant)
    @inbounds for j in 1:N
        d_ij = signed_edge_distance(origin(c_0), radius(c_0), origins[j], radii[j])
        E += d_ij^2
    end

    @inbounds for i in 1:N-1
        origin_i = origins[i]
        radius_i = radii[i]
        @inbounds for j in i+1:N
            d_ij = signed_edge_distance(origin_i, radius_i, origins[j], radii[j])
            E += d_ij^2
        end
    end

    return E

end

# Sum inverse squared circle distances
function energy_sum_inv_squared_distances(c_0::Circle,
                                          origins::AbstractVector,
                                          radii::AbstractVector,
                                          ::Type{Val{DIM}}) where {DIM}
    T = promote_type(eltype(origins), eltype(radii))
    N = length(radii)
    E = zero(T)

    @assert length(origins) == N*DIM
    origins = reinterpret(Vec{DIM,T}, origins)

    # One circle must be fixed, otherwise problem is ill-posed (translation invariant)
    @inbounds for j in 1:N
        dx = origin(c_0) - origins[j]
        a² = min(radius(c_0), radii[j])^2
        r² = max(dx⋅dx, a²)
        E += inv(r²)
    end

    @inbounds for i in 1:N-1
        origin_i = origins[i]
        radius_i = radii[i]
        @inbounds for j in i+1:N
            dx = origin_i - origins[j]
            a² = min(radius_i, radii[j])^2
            r² = max(dx⋅dx, a²)
            E += inv(r²)
        end
    end

    return E

end

# Sum squared distances only from overlapping circles
function energy_sum_overlap_squared_distances(c_0::Circle,
                                              origins::AbstractVector,
                                              radii::AbstractVector,
                                              ::Type{Val{DIM}}) where {DIM}
    T = promote_type(eltype(origins), eltype(radii))
    N = length(radii)
    E = zero(T)

    @assert length(origins) == N*DIM
    origins = reinterpret(Vec{DIM,T}, origins)

    # One circle must be fixed, otherwise problem is ill-posed (translation invariant)
    d²_overlap = zero(T)
    @inbounds for j in 1:N
        d_ij = signed_edge_distance(origin(c_0), radius(c_0), origins[j], radii[j])
        if d_ij < zero(T)
            d_ij² = d_ij^2
            d²_overlap += d_ij²
        end
    end
    E += d²_overlap

    @inbounds for i in 1:N-1
        origin_i = origins[i]
        radius_i = radii[i]
        d²_min = T(Inf)
        d²_overlap = zero(T)
        @inbounds for j in i+1:N
            d_ij = signed_edge_distance(origin_i, radius_i, origins[j], radii[j])
            if d_ij < zero(T)
                d_ij² = d_ij^2
                d²_overlap += d_ij²
            end
        end
        E += d²_overlap
    end

    return E

end
