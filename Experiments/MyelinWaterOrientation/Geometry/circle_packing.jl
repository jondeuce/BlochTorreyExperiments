# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #
function pack_circles(radii::AbstractVector, ::Type{Val{DIM}} = Val{2};
                      initial_origins::AbstractVector{<:Vec{DIM}} = initialize_origins(radii, Val{DIM}),
                      alpha::Real = convert(eltype(radii), 1e-4),
                      epsilon::Real = convert(eltype(radii), 0.05minimum(radii)),
                      distancescale = mean(radii),
                      autodiff::Bool = true,
                      secondorder::Bool = false,
                      constrained::Bool = false,
                      Alg = secondorder ? Newton(linesearch = LineSearches.BackTracking(order=3))
                                        : LBFGS(linesearch = LineSearches.BackTracking(order=3)),
                      Opts = Optim.Options(iterations = 100_000,
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

    x0 = copy(reinterpret(T, origin.(c_variable)))
    if constrained
        # Constrained problem using Lagrange multipliers
        push!(x0, one(T)) # push initial Lagrange multiplier
        g = x -> packing_energy(c_fixed, x, radius.(c_variable), distancescale, alpha, x[end], epsilon, Val{DIM})
        f = x -> sum(abs2, ForwardDiff.gradient(g, x))
    else
        # Unconstrained problem with penalty on overlap
        const lambda = one(T) # lagrange multipliers not used
        f = x -> packing_energy(c_fixed, x, radius.(c_variable), distancescale, alpha, lambda, epsilon, Val{DIM})
    end

    # Optimize and get results
    if autodiff
        if secondorder
            opt_obj = TwiceDifferentiable(f, x0; autodiff = :forward)
        else
            opt_obj = OnceDifferentiable(f, x0; autodiff = :forward)
        end
    else
        opt_obj = f
    end
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = constrained ? copy(Optim.minimizer(result)[1:end-1]) : copy(Optim.minimizer(result))

    # Reconstruct resulting circles
    packed_origins = [origin(c_fixed), copy(reinterpret(Vec{DIM,T}, x))...]
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

function wrap_gradient(f, x, ::Type{Val{N}} = Val{10}) where {N}
    cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{N}())
    g! = (out, x) -> ForwardDiff.gradient!(out, f, x, cfg)
    return g!
end

# ---------------------------------------------------------------------------- #
# Energies on circles
# ---------------------------------------------------------------------------- #

# Packing energy (unconstrained problem)
function packing_energy(c_0::Circle,
                        origins::AbstractVector,
                        radii::AbstractVector,
                        distancescale = mean(radii),
                        alpha::Real = convert(eltype(radii), 1e-4),
                        lambda::Real = one(eltype(radii)),
                        epsilon::Real = zero(eltype(radii)),
                        ::Type{Val{DIM}} = Val{dimension(c_0)}) where {DIM}
    # Using the overlap as the only metric clearly will not work, as any
    # isolated set of circles will have zero energy. Therefore, we penalize by
    # the total squared distances to encourage the circles to stay close
    E_overlap = energy_sum_overlap_squared_distances(c_0,origins,radii,epsilon,Val{DIM})
    E_mutual = energy_sum_squared_distances(c_0,origins,radii,Val{DIM})

    # @show (minimum(ForwardDiff.value.(origins)), maximum(ForwardDiff.value.(origins)))
    # @show (ForwardDiff.value(E_overlap), ForwardDiff.value(E_mutual))

    # We could also interpret the "packing energy" instead as the Lagrangian
    # for the constrained problem where lambda is a Lagrange multiplier and the
    # overlap energy is constrained to be exactly zero (which occurs whenever
    # there are no overlapping circles)
    E_total = (alpha*E_mutual + lambda*E_overlap)/distancescale^2

    return E_total
end

# Sum squared circle distances
function energy_sum_squared_distances(c_0::Circle,
                                      origins::AbstractVector,
                                      radii::AbstractVector,
                                      ::Type{Val{DIM}} = Val{2}) where {DIM}

    T = promote_type(eltype(origins), eltype(radii)) # need this for ForwardDiff
    N = length(radii)
    E = zero(T)

    @assert length(origins) >= N*DIM # allow for extra variables to be at the end of origins
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
                                          ::Type{Val{DIM}} = Val{2}) where {DIM}

    T = promote_type(eltype(origins), eltype(radii)) # need this for ForwardDiff
    N = length(radii)
    E = zero(T)

    @assert length(origins) >= N*DIM # allow for extra variables to be at the end of origins
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

# Sum squared distances only from overlapping circles. The parameter `epsilon`,
# which defaults to zero, allows for a overlapping threshold. If epsilon > 0,
# then the overlapping energy will be counted if the circles are closer than
# `epsilon` distance apart. Similarly, if epsilon < 0, then the overlapping
# energy will only be counted only if the circles are overlapping by more than a
# distance of abs(epsilon)
function energy_sum_overlap_squared_distances(c_0::Circle,
                                              origins::AbstractVector,
                                              radii::AbstractVector,
                                              epsilon::Real = zero(eltype(radii)),
                                              ::Type{Val{DIM}} = Val{2}) where {DIM}

    T = promote_type(eltype(origins), eltype(radii)) # need this for ForwardDiff
    N = length(radii)
    E = zero(T)
    ϵ = T(epsilon)

    @assert length(origins) >= N*DIM # allow for extra variables to be at the end of origins
    origins = reinterpret(Vec{DIM,T}, origins)

    # One circle must be fixed, otherwise problem is ill-posed (translation invariant)
    d²_overlap = zero(T)
    @inbounds for j in 1:N
        d_ij = signed_edge_distance(origin(c_0), radius(c_0), origins[j], radii[j])
        if d_ij < ϵ
            d_ij² = (d_ij-ϵ)^2
            d²_overlap += d_ij²
        end
    end
    E += d²_overlap

    @inbounds for i in 1:N-1
        origin_i = origins[i]
        radius_i = radii[i]
        d²_overlap = zero(T)
        @inbounds for j in i+1:N
            d_ij = signed_edge_distance(origin_i, radius_i, origins[j], radii[j])
            if d_ij < ϵ
                d_ij² = (d_ij-ϵ)^2
                d²_overlap += d_ij²
            end
        end
        E += d²_overlap
    end

    return E

end
