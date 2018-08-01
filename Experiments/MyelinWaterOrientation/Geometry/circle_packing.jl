# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #
function pack_circles(radii::AbstractVector, ::Type{Val{DIM}} = Val{2};
                      initial_origins::AbstractVector{<:Vec{DIM}} = initialize_origins(radii, Val{DIM}),
                      goaldensity = 0.8,
                      distancescale = mean(radii),
                      weights::AbstractVector = [1.0, 1e-6, 1.0],
                      epsilon::Real = 0.1*distancescale,
                      autodiff::Bool = true,
                      secondorder::Bool = false,
                      constrained::Bool = false, # autodiff && secondorder,
                      reversemode::Bool = false, # autodiff && !secondorder,
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

    x0 = copy(reinterpret(T, origin.(c_variable)))
    if constrained
        # Constrained problem using Lagrange multipliers
        push!(x0, one(T)) # push initial Lagrange multiplier
        g = x -> packing_energy(c_fixed, x, radius.(c_variable), goaldensity, distancescale, cat(1, weights[1:2], x[end]), epsilon, Val{DIM})
        f = x -> sum(abs2, ForwardDiff.gradient(g, x))
    else
        # Unconstrained problem with penalty on overlap
        r_variable = radius.(c_variable)
        f = x -> packing_energy(c_fixed, x, r_variable, goaldensity, distancescale, weights, epsilon, Val{DIM})
    end

    # Form *Differentiable object
    if autodiff
        if secondorder
            opt_obj = TwiceDifferentiable(f, x0; autodiff = :forward)
        else
            if reversemode
                # Reverse mode automatic differentiation
                g!, fg! = wrap_gradient(f, x0, isforward = false, isdynamic = true)
                opt_obj = OnceDifferentiable(f, g!, fg!, x0)
            else
                # # Forward mode automatic differentiation
                # g!, fg! = wrap_gradient(f, x0, isforward = true)
                # opt_obj = OnceDifferentiable(f, g!, fg!, x0)

                # Forward mode automatic differentiation
                opt_obj = OnceDifferentiable(f, x0; autodiff = :forward)
            end
        end
    else
        opt_obj = f
    end

    # Optimize and get results
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = constrained ? copy(Optim.minimizer(result)[1:end-1]) : copy(Optim.minimizer(result))

    # Reconstruct resulting circles
    packed_origins = [origin(c_fixed), copy(reinterpret(Vec{DIM,T}, x))...]
    packed_circles = Circle.(packed_origins, radii)

    # Set origin to be the mean of the circles
    P = mean(origin.(packed_circles))
    packed_circles = translate_shape.(packed_circles, [-P]) # shift mean to origin

    # Check that desired desired packing density can be attained
    expand_circles = (α) -> translate_shape.(packed_circles, α)
    density = (α) -> estimate_density(expand_circles(α))

    α_min = find_zero(α -> is_any_overlapping(expand_circles(α)) - 0.5, (0.01, 100.0), Bisection())
    ϵ = 100*eps(α_min)
    if density(α_min + ϵ) ≤ goaldensity
        warn("Density cannot be reached without overlapping circles; can " *
             "only reach $(density(α_min + ϵ)) < $goaldensity. Decrease " *
             "density goal, or adjust mutual distance penalty weight.")
        packed_circles = expand_circles(α_min + ϵ)
    else
        # Find α which results in the desired packing density
        α_best = find_zero(α -> density(α) - goaldensity, (α_min + ϵ, 100.0), Bisection())
        packed_circles = expand_circles(α_best)
    end

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

function wrap_gradient(f, x0::AbstractArray{T},
        ::Type{Val{N}} = Val{min(10,length(x0))};
        isforward = false,
        isdynamic = !isforward) where {T,N}

    if isforward
        # ForwardDiff gradient (pre-recorded config; faster, type stable, but static)
        const cfg = ForwardDiff.GradientConfig(f, x0, ForwardDiff.Chunk{N}())
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, cfg)

        # # ForwardDiff gradient (dynamic config; slower, type unstable, but dynamic chunk sizing)
        # g! = (out, x) -> ForwardDiff.gradient!(out, f, x)
    else
        if isdynamic
            # ReverseDiff gradient (pre-recorded config; slower, but dynamic call graph)
            const cfg = ReverseDiff.GradientConfig(x0)
            g! = (out, x) -> ReverseDiff.gradient!(out, f, x, cfg)
        else
            # ReverseDiff gradient (pre-recorded tape; faster, but static call graph)
            const f_tape = ReverseDiff.GradientTape(f, x0)
            const compiled_f_tape = ReverseDiff.compile(f_tape)
            g! = (out, x) -> ReverseDiff.gradient!(out, compiled_f_tape, x)
        end
    end

    fg! = (G, x) -> begin
        # `DiffResult` is both a light wrapper around `G` and storage for the forward pass
        all_results = DiffBase.DiffResult(zero(T), G)
        g!(all_results, x) # update G(x) == ∇f(x)
        return DiffBase.value(all_results) # return f(x)
    end

    return g!, fg!
end

# ---------------------------------------------------------------------------- #
# Estimate packing density
# ---------------------------------------------------------------------------- #

function estimate_density(circles::Vector{Circle{dim,T}}, α = T(0.75)) where {dim,T}
    # For this estimate, we compute the inscribed square of the bounding circle
    # which bounds all of the `circles`. Then, the square is scaled down a small
    # amount with the hope that this square contains a relatively large and
    # representative region of circles for which to integrate over to obtain the
    # packing density, but not so large that there is much empty space remaining
    boundary_circle = crude_bounding_circle(circles)
    inner_square = inscribed_square(boundary_circle)
    domain = scale_shape(inner_square, α)
    A = prod(maximum(domain) - minimum(domain)) # domain area
    Σ = intersect_area(circles, domain) # total circle areas
    return T(Σ/A)
end

function estimate_density(c_0::Circle{2},
                          origins::AbstractVector,
                          radii::AbstractVector,
                          α = eltype(radii)(0.75))
    # For this estimate, we compute the inscribed square of the bounding circle
    # which bounds all of the `circles`. Then, the square is scaled down a small
    # amount with the hope that this square contains a relatively large and
    # representative region of circles for which to integrate over to obtain the
    # packing density, but not so large that there is much empty space remaining
    boundary_circle = crude_bounding_circle(c_0, origins, radii)
    inner_square = inscribed_square(boundary_circle)
    domain = scale_shape(inner_square, α)
    A = prod(maximum(domain) - minimum(domain)) # domain area
    Σ = intersect_area_check_inside(c_0, domain) # add fixed circle area
    Σ += intersect_area(origins, radii, domain) # total (variable) circle areas

    return Σ/A
end

function estimate_density_monte_carlo(
        circles::Vector{Circle{dim,T}},
        α = T(0.75);
        integrator = Cuba_integrator()
        ) where {dim,T}
    # For this estimate, we compute the inscribed square of the bounding circle
    # which bounds all of the `circles`. Then, the square is scaled down a small
    # amount with the hope that this square contains a relatively large and
    # representative region of circles for which to integrate over to obtain the
    # packing density
    boundary_circle = crude_bounding_circle(circles)
    inner_square = inscribed_square(boundary_circle)
    domain = scale_shape(inner_square, α)

    # Integrand is simply boolean true if inside circle, and false otherwise
    lb, ub = minimum(domain), maximum(domain) # domain bounds
    A = prod(ub - lb) # domain area

    f = x -> convert(T, any(c -> is_inside(x,c), circles)::Bool)
    I, E, P = integrator(f, lb, ub)

    return I/A, E/A, P
end

function HCubature_integrator(;norm = Base.norm, rtol = sqrt(eps()), atol = 0.0, maxevals = typemax(Int))
    return (f, lb, ub) -> HCubature.hcubature(x->f(Vec{2}(x)), lb, ub; norm=norm, rtol=rtol, atol=atol, maxevals=maxevals)
end

function Cuba_integrand!(x,F,f,lb,ub)
    y = lb + (ub - lb) .* x # shift x ∈ [0,1]^2 -> y ∈ [lb,ub]
    J = prod(ub - lb) # Jacobian of linear transformation
    X = Vec{2}((y[1], y[2]))
    F[1] = f(X) * J
    return nothing
end

function Cuba_integrator(method = :cuhre; kwargs...)
    const ndim = 2 # number of dimensions of domain
    const ncomp = 1 # number of components of f
    unwrap(I) = (I.integral[1], I.error[1], I.probability[1])
    integrator(method,f,lb,ub) = unwrap(method((x,F)->Cuba_integrand!(x,F,f,lb,ub), ndim, ncomp; kwargs...))

    if method == :cuhre
        integrator = (f,lb,ub) -> integrator(Cuba.cuhre,f,lb,ub) #unwrap(Cuba.cuhre((x,F)->Cuba_integrand!(x,F,f,lb,ub), ndim, ncomp; kwargs...))
    elseif method == :vegas
        integrator = (f,lb,ub) -> integrator(Cuba.vegas,f,lb,ub) #unwrap(Cuba.vegas((x,F)->Cuba_integrand!(x,F,f,lb,ub), ndim, ncomp; kwargs...))
    elseif method == :divonne
        integrator = (f,lb,ub) -> integrator(Cuba.divonne,f,lb,ub) #unwrap(Cuba.divonne((x,F)->Cuba_integrand!(x,F,f,lb,ub), ndim, ncomp; kwargs...))
    elseif method == :suave
        integrator = (f,lb,ub) -> integrator(Cuba.suave,f,lb,ub) #unwrap(Cuba.suave((x,F)->Cuba_integrand!(x,F,f,lb,ub), ndim, ncomp; kwargs...))
    else
        error("Invalid method `$method`")
    end
end

# ---------------------------------------------------------------------------- #
# Energies on circles
# ---------------------------------------------------------------------------- #

# Packing energy (unconstrained problem)
function packing_energy(c_0::Circle,
                        origins::AbstractVector,
                        radii::AbstractVector,
                        goaldensity::Real = 0.8,
                        distancescale::Real = mean(radii),
                        weights::AbstractVector = [1.0, 1e-6, 1.0],
                        epsilon::Real = 0.1*distancescale,
                        ::Type{Val{DIM}} = Val{dimension(c_0)}) where {DIM}
    # Float type of fixed radii
    T = eltype(radii)
    E_density, E_mutual, E_overlap = zero(T), zero(T), zero(T)

    # Penalize by goaldensity
    if !(weights[1] ≈ zero(T))
        E_density = (goaldensity - estimate_density(c_0,origins,radii))^2
    end

    # Using the overlap as the only metric clearly will not work, as any
    # isolated set of circles will have zero energy. Therefore, we penalize by
    # the total squared distances to encourage the circles to stay close
    if !(weights[2] ≈ zero(T))
        E_mutual = energy_sum_squared_distances(c_0,origins,radii,Val{DIM})/distancescale^2
    end
    if !(weights[3] ≈ zero(T))
        E_overlap = energy_sum_overlap_squared_distances(c_0,origins,radii,epsilon,Val{DIM})/distancescale^2
    end

    # We could also interpret the "packing energy" instead as the Lagrangian
    # for the constrained problem where lambda is a Lagrange multiplier and the
    # overlap energy is constrained to be exactly zero (which occurs whenever
    # there are no overlapping circles)
    E_total = weights[1]*E_density + weights[2]*E_mutual + weights[3]*E_overlap

    return E_total

end

# Sum squared circle distances
function energy_sum_squared_distances(c_0::Circle,
                                      origins::AbstractVector,
                                      radii::AbstractVector,
                                      ::Type{Val{DIM}} = Val{2}) where {DIM}

    T = promote_type(eltype(origins), eltype(radii)) # need this for autodiff
    N = length(radii)
    E = zero(T)

    @assert length(origins) >= N*DIM # allow for extra variables to be at the end of origins
    origins = reinterpret(Vec{DIM,T}, origins)
    # origins = [Vec{DIM,T}((origins[i], origins[i+1])) for i in 1:2:2N]

    # One circle must be fixed, otherwise problem is ill-posed (translation invariant)
    @inbounds for j in 1:N
        # origin_j = Vec{2}((origins[2j-1], origins[2j]))
        # radius_j = radii[j]
        # d_ij = signed_edge_distance(origin(c_0), radius(c_0), origin_j, radius_j)
        d_ij = signed_edge_distance(origin(c_0), radius(c_0), origins[j], radii[j])
        E += d_ij^2
    end

    @inbounds for i in 1:N-1
        # origin_i = Vec{2}((origins[2i-1], origins[2i]))
        origin_i = origins[i]
        radius_i = radii[i]
        @inbounds for j in i+1:N
            # origin_j = Vec{2}((origins[2j-1], origins[2j]))
            # radius_j = radii[j]
            # d_ij = signed_edge_distance(origin_i, radius_i, origin_j, radius_j)
            d_ij = signed_edge_distance(origin_i, radius_i, origins[j], radii[j])
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
    # origins = [Vec{DIM,T}((origins[i], origins[i+1])) for i in 1:2:2N]

    # One circle must be fixed, otherwise problem is ill-posed (translation invariant)
    d²_overlap = zero(T)
    @inbounds for j in 1:N
        # origin_j = Vec{2}((origins[2j-1], origins[2j]))
        # radius_j = radii[j]
        # d_ij = signed_edge_distance(origin(c_0), radius(c_0), origin_j, radius_j)
        d_ij = signed_edge_distance(origin(c_0), radius(c_0), origins[j], radii[j])

        if d_ij < ϵ
            d_ij² = (d_ij-ϵ)^2
            d²_overlap += d_ij²
        end

        # # same as above, but branch-free
        # d²_overlap += (d_ij < ϵ) * (d_ij-ϵ)^2
    end
    E += d²_overlap

    @inbounds for i in 1:N-1
        # origin_i = Vec{2}((origins[2i-1], origins[2i]))
        origin_i = origins[i]
        radius_i = radii[i]
        d²_overlap = zero(T)
        @inbounds for j in i+1:N
            # origin_j = Vec{2}((origins[2j-1], origins[2j]))
            # radius_j = radii[j]
            # d_ij = signed_edge_distance(origin_i, radius_i, origin_j, radius_j)
            d_ij = signed_edge_distance(origin_i, radius_i, origins[j], radii[j])

            if d_ij < ϵ
                d_ij² = (d_ij-ϵ)^2
                d²_overlap += d_ij²
            end

            # same as above, but branch-free
            # d²_overlap += (d_ij < ϵ) * (d_ij-ϵ)^2
        end
        E += d²_overlap
    end

    return E

end

nothing
