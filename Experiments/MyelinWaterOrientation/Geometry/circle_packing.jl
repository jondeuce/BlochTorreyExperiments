# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #

struct FixedOptData{dim,T}
    first_origin::Vec{dim,T}
    second_x_coord::T
    radii::AbstractVector{T}
    x_coord_fixed::Bool
end
numcircles(data::FixedOptData) = length(data.radii)

function pack_circles(radii::AbstractVector, ::Type{Val{DIM}} = Val{2};
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

    # x0 = copy(reinterpret(T, origin.(c_variable)))[2:end]; x_coord_fixed = true
    x0 = copy(reinterpret(T, origin.(c_variable))); x_coord_fixed = false
    data = FixedOptData(origin(c_fixed), origin(c_variable[1])[1], radii, x_coord_fixed)

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
            if reversemode
                # Reverse mode automatic differentiation
                g!, fg! = wrap_gradient(f, x0; isforward = false, isdynamic = true)
                opt_obj = OnceDifferentiable(f, g!, fg!, x0)
            else
                # Forward mode automatic differentiation

                # ---- Use buffer of circles to avoid allocations ---- #
                chunksize = min(chunksize, length(x0))
                const dualcircles = Vector{Circle{2,ForwardDiff.Dual{Void,T,chunksize}}}(numcircles(data))
                const realcircles = Vector{Circle{2,T}}(numcircles(data))

                function f_buffered(x::Vector{T}) where {T<:AbstractFloat}
                    getcircles!(realcircles, x, data)
                    return packing_energy(realcircles, goaldensity, distancescale, weights, epsilon)
                end

                function f_buffered(x::Vector{D}) where {D<:ForwardDiff.Dual}
                    dualcircles = reinterpret(Circle{DIM,D}, dualcircles)
                    getcircles!(dualcircles, x, data)
                    return packing_energy(dualcircles, goaldensity, distancescale, weights, epsilon)
                end

                const checktag = true
                g!, fg!, cfg = wrap_gradient(f_buffered, x0, Val{chunksize}, Val{checktag}; isforward = true)
                opt_obj = OnceDifferentiable(f_buffered, g!, fg!, x0)

                # ---- Simple precompiled gradient ---- #
                # g!, fg! = wrap_gradient(f, x0, Val{chunksize}; isforward = true)
                # opt_obj = OnceDifferentiable(f, g!, fg!, x0)

                # ---- Simple precompiled gradient, but can't configure chunk ---- #
                # opt_obj = OnceDifferentiable(f, x0; autodiff = :forward)
            end
        end
    else
        opt_obj = f
    end

    # Optimize and get results
    if setcallback
        optfields = fieldnames(Opts)
        optvalues = getfield.(Opts, optfields)
        optdict = Dict(zip(optfields, optvalues))
        Opts = Optim.Options(; optdict...,
            callback = state -> check_density_callback(state, data, T(goaldensity), T(epsilon/2)),
            extended_trace = true)
    end
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = constrained ? copy(Optim.minimizer(result)[1:end-1]) : copy(Optim.minimizer(result))

    # Reconstruct resulting circles
    packed_circles = getcircles(x, data)

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

function initialize_origins(radii::AbstractVector{T};
                            distribution = :uniformsquare) where {T}
    # Initialize with random origins
    Ncircles = length(radii)
    Rmax = maximum(radii)
    mesh_scale = T(2Rmax*sqrt(Ncircles))

    if distribution == :random
        # Randomly distributed origins
        initial_origins = mesh_scale .* (T(2.0).*rand(T,2*Ncircles).-one(T))
        initial_origins = reinterpret(Vec{2,T}, initial_origins)
        initial_origins .-= [initial_origins[1]] # shift such that initial_origins[1] is at the origin
        R = getrotmat(initial_origins[2]) # rotation matrix for initial_origins[2]
        broadcast!(o -> R' ⋅ o, initial_origins, initial_origins) # rotate such that initial_origins[2] is on the x-axis
    elseif distribution == :uniformsquare
        # Uniformly distributed, non-overlapping circles
        Nx, Ny = ceil(Int, √Ncircles), floor(Int, √Ncircles)
        initial_origins = zeros(Vec{2,T}, Ncircles)
        ix = 0;
        for j in 0:Ny-1, i in 0:Nx-1
            (ix += 1) > Ncircles && break
            initial_origins[ix] = Vec{2,T}((2Rmax*i, 2Rmax*j))
        end
    else
        error("Unknown initial origins distribution: $distribution.")
    end
    return initial_origins
end

function wrap_gradient(f, x0::AbstractArray{T},
        ::Type{Val{N}} = Val{min(10,length(x0))},
        ::Type{Val{TF}} = Val{true};
        isforward = true,
        isdynamic = !isforward) where {T,N,TF}

    if isforward
        # ForwardDiff gradient (pre-recorded config; faster, type stable, but static)
        const cfg = ForwardDiff.GradientConfig(f, x0, ForwardDiff.Chunk{min(N,length(x0))}())
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, cfg)

        # # ForwardDiff gradient (dynamic config; slower, type unstable, but dynamic chunk sizing)
        # g! = (out, x) -> ForwardDiff.gradient!(out, f, x)

        fg! = (out, x) -> begin
            # `DiffResult` is both a light wrapper around gradient `out` and storage for the forward pass
            all_results = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(all_results, f, x, cfg, Val{TF}()) # update out == ∇f(x)
            return DiffBase.value(all_results) # return f(x)
        end
    else
        if isdynamic
            # ReverseDiff gradient (pre-recorded config; slower, but dynamic call graph)
            const cfg = ReverseDiff.GradientConfig(x0)
            g! = (out, x) -> ReverseDiff.gradient!(out, f, x, cfg)
        else
            # ReverseDiff gradient (pre-recorded tape; faster, but static call graph)
            const f_tape = ReverseDiff.GradientTape(f, x0)
            const compiled_f_tape = ReverseDiff.compile(f_tape)
            const cfg = compiled_f_tape # for returning
            g! = (out, x) -> ReverseDiff.gradient!(out, compiled_f_tape, x)
        end

        fg! = (out, x) -> begin
            # `DiffResult` is both a light wrapper around gradient `out` and storage for the forward pass
            all_results = DiffBase.DiffResult(zero(T), out)
            g!(all_results, x) # update out == ∇f(x)
            return DiffBase.value(all_results) # return f(x)
        end
    end


    return g!, fg!, cfg
end

# ---------------------------------------------------------------------------- #
# Estimate packing density
# ---------------------------------------------------------------------------- #

function check_density_callback(state, data::FixedOptData, goaldensity, epsilon)
    isa(state, AbstractArray) && (state = state[end])
    circles = getcircles(state.metadata["x"], data)
    return !is_any_overlapping(circles, <, epsilon) && (estimate_density(circles) > goaldensity)
end

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

# ---------------------------------------------------------------------------- #
# Energies on circles
# ---------------------------------------------------------------------------- #

function getcircles!(circles::Vector{Circle{DIM,Tx}},
                     x::AbstractVector{Tx},
                     data::FixedOptData{DIM,Tf}) where {DIM,Tx,Tf}
    # There are two sets of a fixed data:
    #   -> The fixed circle c_0, the first circle == circles[1]
    #   -> The x-coordinate of the second circle == 0.0 to fix the rotation of the system
    # Therefore, x is a vector where x[1] is the y-coordinate of the second circle, and
    # then alternating x/y coordinates. I.e., x == [y2,x3,y3,x4,y4,...,xN,yN]
    N = numcircles(data)
    @assert length(circles) == N

    @inbounds circles[1] = Circle{DIM,Tx}(Vec{DIM,Tx}(data.first_origin), Tx(data.radii[1]))
    if data.x_coord_fixed
        @inbounds circles[2] = Circle{DIM,Tx}(Vec{DIM,Tx}((Tx(data.second_x_coord), x[1])), Tx(data.radii[2]))
        @inbounds for (j,i) in enumerate(3:N)
            circles[i] = Circle{DIM,Tx}(Vec{DIM,Tx}((x[2j], x[2j+1])), Tx(data.radii[i]))
        end
    else
        @inbounds for (j,i) in enumerate(2:N)
            circles[i] = Circle{DIM,Tx}(Vec{DIM,Tx}((x[2j-1], x[2j])), Tx(data.radii[i]))
        end
    end

    return circles
 end

function getcircles(x::AbstractVector{Tx},
                    data::FixedOptData{DIM,Tf}) where {DIM,Tx,Tf}
    circles = Vector{Circle{DIM,Tx}}(numcircles(data))
    getcircles!(circles, x, data)
    return circles
end

# Packing energy (unconstrained problem)
function packing_energy(circles::Vector{Circle{DIM,Tx}},
                        goaldensity::Real = 0.8,
                        distancescale::Real = mean(radii),
                        weights::AbstractVector = Tx[1.0, 1e-6, 1.0],
                        epsilon::Real = 0.1*distancescale) where {DIM,Tx}
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
        # E_mutual = energy_sum_squared_distances(c_0,origins,radii,Val{DIM})/distancescale^2
        E_mutual = energy_sum_squared_distances(circles,Val{DIM})/distancescale^2
    end
    if !(weights[3] ≈ zero(Tx))
        # E_overlap = energy_sum_overlap_squared_distances(c_0,origins,radii,epsilon,Val{DIM})/distancescale^2
        E_overlap = energy_sum_overlap_squared_distances(circles,epsilon,Val{DIM})/distancescale^2
    end

    # We could also interpret the "packing energy" instead as the Lagrangian
    # for the constrained problem where lambda is a Lagrange multiplier and the
    # overlap energy is constrained to be exactly zero (which occurs whenever
    # there are no overlapping circles)
    E_total = weights[1]*E_density + weights[2]*E_mutual + weights[3]*E_overlap

    return E_total

end

# Packing energy (unconstrained problem)
function packing_energy(x::AbstractVector{Tx},
                        data::FixedOptData{DIM,Tf},
                        goaldensity::Real = 0.8,
                        distancescale::Real = mean(radii),
                        weights::AbstractVector = Tx[1.0, 1e-6, 1.0],
                        epsilon::Real = 0.1*distancescale) where {DIM,Tx,Tf}
    # Initialize circles
    circles = getcircles(x, data)
    return packing_energy(circles, goaldensity, distancescale, weights, epsilon)
end


# ---------------------------------------------------------------------------- #
# Energies on circles: ForwardDiff friendly with circles vector
# ---------------------------------------------------------------------------- #

# Sum squared circle distances
function energy_covariance(circles::Vector{Circle{DIM,T}}) where {DIM,T}
    circlepoints = reinterpret(T, circles, (DIM+1, length(circles))) # reinterp as DIM+1 x Ncircles array
    @views origins = circlepoints[1:2, :] # DIM x Ncircles view of origin points
    Σ = cov(origins, 2) # covariance matrix of origin locations
    σ² = T(trace(Σ)/DIM) # mean variance
    return sum(abs2, Σ - σ²*I) # penalize non-diagonal covariance matrices
end

# Sum squared circle distances
function energy_sum_squared_distances(circles::Vector{Circle{DIM,T}},
                                      ::Type{Val{DIM}} = Val{2}) where {DIM,T}
    N = length(circles)
    E = zero(T)

    @inbounds for i in 1:N-1
        c_i = circles[i]
        @inbounds for j in i+1:N
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
function energy_sum_overlap_squared_distances(circles::Vector{Circle{DIM,T}},
                                              epsilon::Real = zero(T),
                                              ::Type{Val{DIM}} = Val{2}) where {DIM,T}
    N = length(circles)
    E = zero(T)
    ϵ = T(epsilon)

    @inbounds for i in 1:N-1
        c_i = circles[i]
        d²_overlap = zero(T)
        @inbounds for j in i+1:N
            c_j = circles[j]
            d_ij = signed_edge_distance(c_i, c_j)
            d_ij < ϵ && (d²_overlap += (d_ij-ϵ)^2)
        end
        E += d²_overlap
    end

    return E
end

# ---------------------------------------------------------------------------- #
# Energies on circles: ForwardDiff friendly
# ---------------------------------------------------------------------------- #

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
        d_ij < ϵ && (d²_overlap += (d_ij-ϵ)^2)
    end
    E += d²_overlap

    @inbounds for i in 1:N-1
        origin_i = origins[i]
        radius_i = radii[i]
        d²_overlap = zero(T)
        @inbounds for j in i+1:N
            d_ij = signed_edge_distance(origin_i, radius_i, origins[j], radii[j])
            d_ij < ϵ && (d²_overlap += (d_ij-ϵ)^2)
        end
        E += d²_overlap
    end

    return E

end

# ---------------------------------------------------------------------------- #
# Energies on circles: ReverseDiff friendly
# ---------------------------------------------------------------------------- #

# Sum squared circle distances
function energy_sum_squared_distances_reversediff(
        c_0::Circle,
        origins::AbstractVector,
        radii::AbstractVector,
        ::Type{Val{DIM}} = Val{2}) where {DIM}

    T = promote_type(eltype(origins), eltype(radii)) # need this for autodiff
    N = length(radii)
    E = zero(T)

    @assert length(origins) >= N*DIM # allow for extra variables to be at the end of origins

    @inline compute_dij²(c1::Circle, c2::Circle) = ReverseDiff.@forward(signed_edge_distance)(c1, c2)^2
    # cs = Circle{DIM,T}[Circle{DIM,T}(Vec{2}((origins[2i-1], origins[2i])), radii[i]) for i in 1:N]

    # One circle must be fixed, otherwise problem is ill-posed (translation invariant)
    @inbounds for j in 1:N
        # c_j = cs[j]
        c_j = Circle{DIM,T}(Vec{2}((origins[2j-1], origins[2j])), radii[j])
        E += compute_dij²(c_i, c_j)
    end

    @inbounds for i in 1:N-1
        # c_i = cs[i]
        c_i = Circle{DIM,T}(Vec{2}((origins[2i-1], origins[2i])), radii[i])
        @inbounds for j in i+1:N
            # c_j = cs[j]
            c_j = Circle{DIM,T}(Vec{2}((origins[2j-1], origins[2j])), radii[j])
            E += compute_dij²(c_i, c_j)
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
function energy_sum_overlap_squared_distances_reversediff(
        c_0::Circle,
        origins::AbstractVector,
        radii::AbstractVector,
        epsilon::Real = zero(eltype(radii)),
        ::Type{Val{DIM}} = Val{2}) where {DIM}

    T = promote_type(eltype(origins), eltype(radii)) # need this for ForwardDiff
    N = length(radii)
    E = zero(T)
    ϵ = T(epsilon)

    @assert length(origins) >= N*DIM # allow for extra variables to be at the end of origins
    # cs = Circle{DIM,T}[Circle{DIM,T}(Vec{2}((origins[2i-1], origins[2i])), radii[i]) for i in 1:N]

    # @inline _thresh_d_ij²(d_ij, ϵ) = d_ij < ϵ ? (d_ij-ϵ)^2 : zero(d_ij)
    # @inline thresh_d_ij²(d_ij, ϵ) = ReverseDiff.@forward(_thresh_d_ij²)(d_ij, ϵ)
    # @inline signed_edge_dij(c1::Circle, c2::Circle) = ReverseDiff.@forward(signed_edge_distance)(c1, c2)

    @inline thresh_d_ij²(d_ij, ϵ) = d_ij < ϵ ? (d_ij-ϵ)^2 : zero(d_ij)
    @inline function compute_overlap_dij²(c1::Circle, c2::Circle)
        d_ij = ReverseDiff.@forward(signed_edge_distance)(c1, c2)
        return ReverseDiff.@forward(thresh_d_ij²)(d_ij, ϵ)
    end

    # One circle must be fixed, otherwise problem is ill-posed (translation invariant)
    @inbounds for j in 1:N
        c_j = Circle{DIM,T}(Vec{2}((origins[2j-1], origins[2j])), radii[j])
        E += compute_overlap_dij²(c_0, c_j)
        # E += thresh_d_ij²(signed_edge_dij(c_0, c_j), ϵ)
    end

    @inbounds for i in 1:N-1
        c_i = Circle{DIM,T}(Vec{2}((origins[2i-1], origins[2i])), radii[i])
        @inbounds for j in i+1:N
            c_j = Circle{DIM,T}(Vec{2}((origins[2j-1], origins[2j])), radii[j])
            E += compute_overlap_dij²(c_i, c_j)
            # E += thresh_d_ij²(signed_edge_dij(c_i, c_j), ϵ)
        end
    end

    return E

end

# ---------------------------------------------------------------------------- #
# Estimate density using Monte Carlo integration
# ---------------------------------------------------------------------------- #

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

nothing
