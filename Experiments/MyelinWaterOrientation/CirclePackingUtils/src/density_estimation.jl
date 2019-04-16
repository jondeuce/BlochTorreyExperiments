# ---------------------------------------------------------------------------- #
# Circle packing density tools
# ---------------------------------------------------------------------------- #

function opt_subdomain(
        circles::AbstractVector{Circle{2,T}},
        α0::Real = T(0.75),
        α_lb::Real = T(0.65),
        α_ub::Real = T(0.85),
        d_thresh::Real = T(minimum(radius, circles)/2),
        λ_relative = T(0.5);
        # λ2_relative = 50 * length(circles)
        MODE = :scale
    ) where {T}
    
    # # Simple method
    # opt_rectangle = scale_shape(inscribed_square(crude_bounding_circle(circles)), T(α0))
    
    # Brute force search method which tries to force circle/rectangle intersection angles
    # to be as close to 90 degrees as possible, and also to maximimze the distance between
    # interior circles (within d_thresh of the boundary) and the square boundary
    function get_inscribed_rect(α)
        # Simply scale the inscribed square by a constant
        return scale_shape(inscribed_square(crude_bounding_circle(circles)), T(α))
    end

    rect0 = scale_shape(inscribed_square(crude_bounding_circle(circles)), T(α0))
    dx0, dy0 = widths(rect0)
    e1, e2 = basevec(Vec{2,T})
    function get_inscribed_rect(α1, α2, α3, α4)
        # Allow all four corners to move my relative amounts αs
        pmin = rect0.mins + α1 * dx0 * e1 + α2 * dy0 * e2
        pmax = rect0.maxs + α3 * dx0 * e1 + α4 * dy0 * e2
        return Rectangle(pmin, pmax)
    end

    function energy(args...)
        r = get_inscribed_rect(args...)
        drect(x) = min(x[1] - xmin(r), xmax(r) - x[1], x[2] - ymin(r), ymax(r) - x[2])
        
        # # This seems to be too strong of a condition
        # centred_energy = sum(circles) do c
        #     out = zero(T)
        #     if is_on_boundary(c, r)
        #         d = drect(origin(c)) #distance btwn circle origin and rect bdry
        #         out += (d/d_thresh)^2 # penalize away from zero
        #     end
        #     return out
        # end

        # # Strongly penalize corners not being contained in circles
        # corner_energy = sum(corners(r)) do p
        #     is_in_any_circle(p, circles) ? zero(T) : T(1e6)*length(circles)
        # end

        # # Strongly penalize corners not being contained in circles
        # corner_energy = sum(corners(r)) do p
        #     d² = minimum(c->norm2(p-origin(c)), circles)
        #     return d²/d_thresh^2
        # end

        # Corners don't need to be inside circles
        corner_energy = zero(T)

        angle_count = 0
        angle_energy = sum(circles) do c
            out = zero(T)
            for p in intersection_points(r, c)
                t1 = (p[1] ≈ xmin(r) || p[1] ≈ xmax(r)) ? # check if intersect is on left/right side 
                    Vec{2,T}((0,1)) : # left/right: tangent vector to rect is vertical
                    Vec{2,T}((1,0)) # top/bottom: tangent vector to rect is horizontal.
                n2 = p - origin(c) # normal vector to circle at p
                t2 = Vec{2,T}((-n2[2], n2[1])) # tangent vector to circle at p
                θ = acos(abs(t1⋅t2)/(norm(t1)*norm(t2))) # (smallest) angle between t1 and t2
                
                out += (1 - θ/(T(π)/2))^2
                angle_count += 1 # count number of terms
            end
            return out
        end
        angle_energy /= max(angle_count, 1)
        
        # Do we need this?
        angle_energy = zero(T)

        distance_count = 0
        distance_energy = sum(circles) do c
            out = zero(T)
            # if is_inside(c, r)
            #     d = drect(origin(c)) - radius(c) #distance btwn circle and rect bdry
            #     if d < d_thresh
            #         out += (1 - d/d_thresh)^2 # penalize away from d_thresh
            #         distance_count += 1 # count number of terms
            #     end
            # end
            if is_on_boundary(c, r)
                d = drect(origin(c)) #distance btwn circle origin and rect bdry
                out += (d/d_thresh)^2 # penalize away from zero
                distance_count += 1 # count number of terms
            end
            return out
        end
        distance_energy /= max(distance_count, 1)

        total_energy = angle_energy + λ_relative * distance_energy + corner_energy
        # println("$angle_energy, $distance_energy, $total_energy") #DEBUG
        
        # total_energy = centred_energy + λ_relative * distance_energy + corner_energy
        # println("$centred_energy, $distance_energy, $corner_energy, $total_energy") #DEBUG
        
        return total_energy
    end

    local opt_rectangle, α
    if MODE == :scale
        α_range = range(T(α_lb), T(α_ub), length=151)
        _, i_opt = findmin([energy(α) for α in α_range])
        α = α_range[i_opt]
        opt_rectangle = get_inscribed_rect(α)
    else
        α_range = range(T(α_lb - α0), T(α_ub - α0), length=5)
        _, i_opt = findmin([energy(α...) for α in Iterators.product([α_range for _ in 1:4]...)])
        α = [α_range[i] for i in Tuple(i_opt)]
        opt_rectangle = get_inscribed_rect(α...)
    end

    # # Energy method (not very effective)
    # starting_rectangle = bounding_box(circles)
    # mean_radius = mean(radius, circles)
    # N_total = length(circles)
    #
    # function energy(α)
    #     inner = scale_shape(starting_rectangle, α)
    #     c1, c2, c3, c4 = corners(inner)
    #     x0, x1, y0, y1 = xmin(inner), xmax(inner), ymin(inner), ymax(inner)
    #
    #     d = fill(T(Inf), 8)
    #     N_inside = 0
    #     for c in circles
    #         o = origin(c)
    #         d[1] = min(d[1], norm(c1 - o)) # corner distances
    #         d[2] = min(d[2], norm(c2 - o))
    #         d[3] = min(d[3], norm(c3 - o))
    #         d[4] = min(d[4], norm(c4 - o))
    #         d[5] = min(d[5], abs(x0 - o[1])) # edge distances
    #         d[6] = min(d[6], abs(x1 - o[1]))
    #         d[7] = min(d[7], abs(y0 - o[2]))
    #         d[8] = min(d[8], abs(y1 - o[2]))
    #         is_inside(o, inner) && (N_inside += 1)
    #     end
    #
    #     return sum(d)/mean_radius - N_inside/N_total
    # end
    #
    # # Find the optimal α using a Bisection method
    # α = Optim.minimizer(optimize(energy, α_lb, α_ub, Brent()))
    # opt_rectangle = scale_shape(starting_rectangle, α)

    return opt_rectangle, α
end

# For this estimate, we compute the inscribed square of the bounding circle
# which bounds all of the `circles`. Then, the square is scaled down a small
# amount with the hope that this square contains a relatively large and
# representative region of circles for which to integrate over to obtain the
# packing density, but not so large that there is much empty space remaining.
# If α is given, simply use this square. Otherwise, compute the optimal
# subdomain using the above helper function
function estimate_density(
        circles::AbstractVector{C},
        α::Union{<:Number,Nothing} = nothing;
        MODE = :scale
    ) where {C<:Circle{2}}
    domain = α == nothing ?
        opt_subdomain(circles; MODE = MODE)[1] :
        scale_shape(inscribed_square(crude_bounding_circle(circles)), α)
    return estimate_density(circles, domain)
end

function estimate_density(
        circles::AbstractVector{C},
        domain::Rectangle{2}
    ) where {C<:Circle{2}}
    A = prod(maximum(domain) - minimum(domain)) # domain area
    Σ = intersect_area(circles, domain) # total circle areas
    return Σ/A
end

function scale_to_density(circles::Vector{C}, goaldensity, distthresh = 0; MODE = :scale) where {C<:Circle{2}}
    domain = opt_subdomain(circles; MODE = MODE)[1]
    packed_circles, packed_domain, α_best = scale_to_density(circles, domain, goaldensity, distthresh)
    return packed_circles, packed_domain, α_best
end

function scale_to_density(circles::Vector{C}, domain::Rectangle{2}, goaldensity, distthresh = 0) where {C<:Circle{2}}
    # Check that desired desired packing density can be attained
    expand_circles = (α) -> translate_shape.(circles, α) # circle origins are expanded uniformly by a factor α
    expand_domain = (α) -> scale_shape(domain, α) # domain is scaled uniformly by a factor α
    density = (α) -> estimate_density(expand_circles(α), expand_domain(α))#, α_inner_domain)

    α_min = find_zero(α -> is_any_overlapping(expand_circles(α), ≤, distthresh) - 0.5, (1.0e-3, 1.0e3), Bisection())
    α_eps = 100 * eps(α_min)
    α_best = α_min + α_eps
    
    if density(α_best) ≤ goaldensity
        # Goal density can't be reached; shrinking as much as possible
        msg = "Density cannot be reached without overlapping circles more than distthresh = $distthresh; " *
              "shrinking as much as possible to $(density(α_min + α_eps)) < $goaldensity"
        @warn msg
    else
        # Find α which results in the desired packing density
        α_best = find_zero(α -> density(α) - goaldensity, (α_min + α_eps, 1.0e3), Bisection())
    end
    
    packed_circles = expand_circles(α_best)
    packed_domain = expand_domain(α_best)
    return packed_circles, packed_domain, α_best
end

function covariance_energy(circles::Vector{Circle{DIM,T}}) where {DIM,T}
    circlepoints = reshape(reinterpret(T, circles), (DIM+1, length(circles))) # reinterp as DIM+1 x Ncircles array
    @views origins = circlepoints[1:DIM, :] # DIM x Ncircles view of origin points
    Σ = cov(origins; dims = 2) # covariance matrix of origin locations
    σ² = T(tr(Σ)/DIM) # mean variance
    return sum(abs2, Σ - σ²*I) # penalize non-diagonal covariance matrices
end

# ---------------------------------------------------------------------------- #
# Optim tools
# ---------------------------------------------------------------------------- #

function check_density_callback(state, r, goaldensity, epsilon)
    if isa(state, AbstractArray) && !isempty(state)
        currstate = state[end]
        resize!(state, 1) # only store last state
        state[end] = currstate
    else
        currstate = state
    end
    circles = tocircles(currstate.metadata["x"], r)
    return (estimate_density(circles; MODE = :corners) > goaldensity) && !is_any_overlapping(circles, ≤, epsilon)
end