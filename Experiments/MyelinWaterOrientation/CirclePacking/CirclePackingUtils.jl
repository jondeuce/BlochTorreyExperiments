# ============================================================================ #
# Tools for circle packing
# ============================================================================ #

module CirclePackingUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using GeometryUtils
using LinearAlgebra, Statistics
using DiffBase, Optim, LineSearches, ForwardDiff, Roots
using Tensors

export estimate_density, scale_to_density
export tocircles, tocircles!, tovectors, tovectors!, initialize_origins
export pairwise_sum, pairwise_grad!
export wrap_gradient, check_density_callback

# ---------------------------------------------------------------------------- #
# Circle packing density tools
# ---------------------------------------------------------------------------- #

function estimate_density(circles::AbstractVector{Circle{2,T}},
                          α = T(0.75)) where {T}
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

function scale_to_density(input_circles, goaldensity)
    # Check that desired desired packing density can be attained
    expand_circles = (α) -> translate_shape.(input_circles, α)
    density = (α) -> estimate_density(expand_circles(α))

    α_min = find_zero(α -> is_any_overlapping(expand_circles(α)) - 0.5, (1.0e-3, 1.0e3), Bisection())
    ϵ = 100 * eps(α_min)

    if density(α_min + ϵ) ≤ goaldensity
        # Goal density can't be reached; shrink as much as possible
        @warn ("Density cannot be reached without overlapping circles; " *
               "can only reach $(density(α_min + ϵ)) < $goaldensity")
        packed_circles = expand_circles(α_min + ϵ)
    else
        # Find α which results in the desired packing density
        α_best = find_zero(α -> density(α) - goaldensity, (α_min + ϵ, 1.0e3), Bisection())
        packed_circles = expand_circles(α_best)
    end

    return packed_circles
end

# ---------------------------------------------------------------------------- #
# General circle tools
# ---------------------------------------------------------------------------- #

function tocircles!(circles::AbstractVector{Circle{DIM,T}},
                    x::AbstractVector,
                    r::AbstractVector) where {DIM,T}
    N = length(circles)
    @assert length(x) == DIM*length(r) == DIM*N
    x = reinterpret(Vec{DIM,eltype(x)}, x)
    @inbounds for i in 1:N
        circles[i] = Circle{DIM,T}(x[i], r[i])
    end
    return circles
end

function tocircles(x::AbstractVector,
                   r::AbstractVector,
                   ::Val{DIM} = Val(2)) where {DIM}
    @assert length(x) == DIM*length(r)
    T = promote_type(eltype(x), eltype(r))
    circles = Vector{Circle{DIM,T}}(undef, length(r))
    tocircles!(circles, x, r)
    return circles
end

function tovectors!(x::AbstractVector{T},
                    r::AbstractVector{T},
                    c::AbstractVector{Circle{DIM,T}}) where {DIM,T}
    # Unpack input vector of circles to radius and x vectors
    N = length(c)
    @assert length(x) == DIM*length(r) == DIM*N
    x = reinterpret(Vec{DIM,T}, x)
    for (i, ci) in enumerate(c)
        x[i], r[i] = origin(ci), radius(ci)
    end
    x = Vector(reinterpret(T, x))
    return x, r
end

function tovectors(c::AbstractVector{Circle{DIM,T}}) where {DIM,T}
    # Unpack input vector of circles to radius and x vectors
    N = length(c)
    x, r = zeros(T,DIM*N), zeros(T,N)
    tovectors!(x, r, c)
    return x, r
end

function initialize_origins(radii::AbstractVector{T};
                            distribution = :uniformsquare) where {T}
    # Initialize with random origins
    Ncircles = length(radii)
    Rmax = maximum(radii)
    mesh_scale = T(2*Rmax*sqrt(Ncircles))

    if distribution == :random
        # Randomly distributed origins
        initial_origins = mesh_scale .* (T(2.0).*rand(T,2*Ncircles).-one(T))
        initial_origins = reinterpret(Vec{2,T}, initial_origins)
        # initial_origins .-= [initial_origins[1]] # shift such that initial_origins[1] is at the origin
        # R = rotmat(initial_origins[2]) # rotation matrix for initial_origins[2]
        # broadcast!(o -> R' ⋅ o, initial_origins, initial_origins) # rotate such that initial_origins[2] is on the x-axis
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

# ---------------------------------------------------------------------------- #
# Symmetric functions on circles
# ---------------------------------------------------------------------------- #

# Symmetric pairwise sum of the four argument function `f` where the first two
# arguments are Vec{2,T}'s and the second two are scalars. The output is the sum
# over i<j of f(o[i], o[j], r[i], r[j]) where o[i] is the Vec{2,T} formed from
# x[2i-1:2i].
# `f` is symmetric in both the first two arguments and the second two, i.e.
# f(o1,o2,r1,r2) == f(o2,o1,r1,r2) and f(o1,o2,r1,r2) == f(o1,o2,r2,r1).
function pairwise_sum(f,
                      x::AbstractVector{T},
                      r::AbstractVector) where {T}
    @assert length(x) == 2*length(r) >= 4
    o = reinterpret(Vec{2,T}, x)

    # Pull first iteration outside of the loop for easy initialization
    Σ = f(o[1], o[2], r[1], r[2])
    @inbounds for j = 3:length(o)
        for i in 1:j-1
            Σ += f(o[i], o[j], r[i], r[j])
        end
    end

    return Σ
end

# Gradient of the pairwise_sum function w.r.t. `x`. Result is stored in `g`.
function pairwise_grad!(g::AbstractVector{T},
                        ∇f,
                        x::AbstractVector{T},
                        r::AbstractVector) where {T}
    @assert length(g) == length(x) == 2*length(r) >= 4
    fill!(g, zero(T))
    G = reinterpret(Vec{2,T}, g)
    o = reinterpret(Vec{2,T}, x)

    @inbounds for i in 1:length(o)
        for j in 1:i-1
            G[i] += ∇f(o[i], o[j], r[i], r[j])
        end
        for j in i+1:length(o)
            G[i] += ∇f(o[i], o[j], r[i], r[j])
        end
    end

    return g
end

# ---------------------------------------------------------------------------- #
# ForwardDiff tools
# ---------------------------------------------------------------------------- #

function wrap_gradient(f, x0::AbstractArray{T},
        ::Type{Val{N}} = Val{min(10,length(x0))},
        ::Type{Val{TF}} = Val{true}) where {T,N,TF}

    # ForwardDiff gradient (pre-recorded config; faster, type stable, but static)
    cfg = ForwardDiff.GradientConfig(f, x0, ForwardDiff.Chunk{min(N,length(x0))}())
    g! = (out, x) -> ForwardDiff.gradient!(out, f, x, cfg)

    # # ForwardDiff gradient (dynamic config; slower, type unstable, but dynamic chunk sizing)
    # g! = (out, x) -> ForwardDiff.gradient!(out, f, x)

    fg! = (out, x) -> begin
        # `DiffResult` is both a light wrapper around gradient `out` and storage for the forward pass
        all_results = DiffBase.DiffResult(zero(T), out)
        ForwardDiff.gradient!(all_results, f, x, cfg, Val{TF}()) # update out == ∇f(x)
        return DiffBase.value(all_results) # return f(x)
    end

    return g!, fg!, cfg
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
    return (estimate_density(circles) > goaldensity) && !is_any_overlapping(circles, <, epsilon)
end

end # module CirclePackingUtils

nothing
