# ---------------------------------------------------------------------------- #
# Tensors tools
# ---------------------------------------------------------------------------- #

@inline periodic_diff(x1::Number, x2::Number, P::Number) = mod(x1 - x2 + P/2, P) - P/2
@inline function periodic_diff(x1::Vec{dim}, x2::Vec{dim}, P::Vec{dim}) where {dim}
    return Vec{dim}(@inline function (i) @inbounds periodic_diff(x1[i], x2[i], P[i]) end)
end
@inline function periodic_mod(x::Vec{dim}, P::Vec{dim}) where {dim}
    return Vec{dim}(@inline function (i) @inbounds mod(x[i], P[i]) end)
end

# ---------------------------------------------------------------------------- #
# Periodic circle tools
# ---------------------------------------------------------------------------- #

function periodic_circles(cs::AbstractVector{C}, bdry::Rectangle{2}) where {C <: Circle{2}}
    X0, X1 = minimum(bdry), maximum(bdry)
    e1, e2 = basevec(typeof(X0))
    P = X1 - X0
    N = length(cs)
    cs = copy(cs)
    @inbounds for i in 1:N
        c = cs[i]
        o, r = periodic_mod(origin(c) - X0, P) + X0, radius(c)
        c = Circle(o, r)
        cs[i] = c # center
        push!(cs, Circle(o + P[1]*e1, r)) # right
        push!(cs, Circle(o - P[1]*e1, r)) # left
        push!(cs, Circle(o + P[2]*e2, r)) # up
        push!(cs, Circle(o - P[2]*e2, r)) # down
        push!(cs, Circle(o + P[1]*e1 + P[2]*e2, r)) # right-up
        push!(cs, Circle(o - P[1]*e1 + P[2]*e2, r)) # left-up
        push!(cs, Circle(o + P[1]*e1 - P[2]*e2, r)) # right-down
        push!(cs, Circle(o - P[1]*e1 - P[2]*e2, r)) # left-down
    end
    cs = filter!(c -> !is_outside(c, bdry), cs)
    idx = filter(i -> !any(cs[i] ≈ cs[j] for j in 1:i-1), 1:length(cs))
    cs = cs[idx]
    return cs
end

function periodic_unique_circles(cs::AbstractVector{C}, bdry::Rectangle{2}) where {C <: Circle{2}}
    X0, P = minimum(bdry), widths(bdry)
    cs = periodic_circles(cs, bdry)
    cs = [Circle(periodic_mod(origin(c) - X0, P) + X0, radius(c)) for c in cs]
    idx = filter(i -> !any(cs[i] ≈ cs[j] for j in 1:i-1), 1:length(cs))
    cs = cs[idx]
    return cs
end

function periodic_density(cs::AbstractVector{C}, bdry::Rectangle{2}) where {C <: Circle{2}}
    # A1 = sum(c->intersect_area(c,bdry), periodic_circles(cs, bdry)) / area(bdry)
    A2 = sum(area, periodic_unique_circles(cs, bdry)) / area(bdry)
    return A2
end

function periodic_circle_repeat(cs::AbstractVector{C}, bdry::Rectangle{2}; Nrepeat = 1) where {C <: Circle{2}}
    cs = periodic_unique_circles(cs, bdry)
    w = widths(bdry)
    cs_out = C[]
    for i in -Nrepeat:Nrepeat, j in -Nrepeat:Nrepeat
        dx = Vec{2}((i*w[1], j*w[2]))
        cs_shifted = (c -> Circle(origin(c) + dx, radius(c))).(cs)
        append!(cs_out, cs_shifted)
    end
    return cs_out
end

function periodic_scale_to_threshold(cs::Vector{C}, bdry::Rectangle{2}, distthresh = zero(floattype(C))) where {C <: Circle{2}}
    # Minimum distance between circles must be positive
    @assert distthresh >= 0

    function expand_geom(α, cs, bdry)
        x0 = minimum(bdry)
        cs = translate_shape.(cs, Ref(x0), α) # circle origins are expanded uniformly by a factor α away from x0
        bdry = scale_shape(bdry, x0, α) # domain is scaled uniformly by a factor α away from x0
        return cs, bdry
    end

    # Compute minimum possible α which satisifes distthresh
    min_alpha(c1, c2) = (radius(c1) + radius(c2) + distthresh) / norm(origin(c1) - origin(c2))
    all_cs = periodic_circles(cs, bdry)
    α_min = maximum(@inbounds(min_alpha(all_cs[i], all_cs[j])) for i in 2:length(all_cs) for j in 1:i-1)

    # Expand initial circles once to α_min to ensure non-overlapping circles
    α = α_min
    cs, bdry = expand_geom(α_min, cs, bdry)
    return (circles = cs, domain = bdry, parameter = α_min)
end

function periodic_scale_to_density(cs::Vector{C}, bdry::Rectangle{2}, goaldensity, distthresh = zero(floattype(C))) where {C <: Circle{2}}
    # Minimum distance between circles must be positive
    @assert distthresh >= 0

    function expand_geom(α, cs, bdry)
        x0 = minimum(bdry)
        cs = translate_shape.(cs, Ref(x0), α) # circle origins are expanded uniformly by a factor α away from x0
        bdry = scale_shape(bdry, x0, α) # domain is scaled uniformly by a factor α away from x0
        return cs, bdry
    end

    # Expand initial circles once to ensure non-overlapping circles
    cs, bdry, α = periodic_scale_to_threshold(cs, bdry, distthresh)
    η = periodic_density(cs, bdry)

    # Check if goal density cannot be reached, otherwise scale
    if η < goaldensity
        msg = "Density cannot be reached without overlapping circles more than distthresh = $distthresh; " *
                "shrinking as much as possible to $η < $goaldensity"
        @warn msg
    else
        # Expand to goaldensity
        α_opt = sqrt(η/goaldensity)
        cs, bdry = expand_geom(α_opt, cs, bdry)
        α *= α_opt
    end

    return (circles = cs, domain = bdry, parameter = α)
end

function periodic_subdomain(
        circles::AbstractVector{Circle{2,T}},
        bdry::Rectangle{2,T}
        # d_scale::Real = T(minimum(radius, circles)/2)
        # λ_relative = T(0.5);
        # λ2_relative = 50 * length(circles)
        # MODE = :scale
    ) where {T}

    function energy(dX)
        r = translate_shape(bdry, dX)
        cs = periodic_circles(circles, r)
        distance_count = 0
        distance_energy = sum(circles) do c
            out = zero(T)
            d = abs(DistMesh.drectangle0(origin(c), r)) #distance btwn circle origin and rect bdry
            if d < radius(c)
                # out += (d/d_scale)^4 # penalize away from zero
                out += d^4 # penalize away from zero
                distance_count += 1 # count number of terms
            end
            return out
        end
        distance_energy /= max(distance_count, 1)
        return distance_energy
    end

    w = widths(bdry)
    xs = xmin(bdry) .+ range(zero(T), w[1]/2, length = 25)
    ys = ymin(bdry) .+ range(zero(T), w[2]/2, length = 25)
    _, i_opt = findmin([energy(Vec{2}((x,y))) for (x,y) in Iterators.product(xs,ys)])
    dX = Vec{2}((xs[i_opt[1]], ys[i_opt[2]]))
    opt_bdry = translate_shape(bdry, dX)

    return (domain = opt_bdry, parameter = dX)
end