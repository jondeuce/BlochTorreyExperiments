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

export estimate_density, opt_subdomain, scale_to_density, covariance_energy
export tocircles, tocircles!, tovectors, tovectors!, initialize_origins
export pairwise_sum, pairwise_grad!, pairwise_hess!
export wrap_gradient, check_density_callback

# ---------------------------------------------------------------------------- #
# Circle packing density tools
# ---------------------------------------------------------------------------- #

function opt_subdomain(
        circles::AbstractVector{Circle{2,T}},
        alpha_lb = T(0.1),
        alpha_ub = T(1.0)
    ) where {T}

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
    # α = Optim.minimizer(optimize(energy, alpha_lb, alpha_ub, Brent()))
    # opt_rectangle = scale_shape(starting_rectangle, α)

    α = 0.75
    opt_rectangle = scale_shape(inscribed_square(crude_bounding_circle(circles)), α)

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
        circles::AbstractVector{Circle{2,T}},
        α = nothing
    ) where {T}

    domain = if (α == nothing)
        opt_subdomain(circles)[1]
    else
        scale_shape(inscribed_square(crude_bounding_circle(circles)), α)
    end

    A = prod(maximum(domain) - minimum(domain)) # domain area
    Σ = intersect_area(circles, domain) # total circle areas

    return T(Σ/A)
end

function scale_to_density(input_circles, goaldensity)
    # Check that desired desired packing density can be attained
    # _, α_inner_domain = opt_subdomain(input_circles) # fix the opt_subdomain relative size
    expand_circles = (α) -> translate_shape.(input_circles, α)
    density = (α) -> estimate_density(expand_circles(α))#, α_inner_domain)

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

function covariance_energy(circles::Vector{Circle{DIM,T}}) where {DIM,T}
    circlepoints = reshape(reinterpret(T, circles), (DIM+1, length(circles))) # reinterp as DIM+1 x Ncircles array
    @views origins = circlepoints[1:DIM, :] # DIM x Ncircles view of origin points
    Σ = cov(origins; dims = 2) # covariance matrix of origin locations
    σ² = T(tr(Σ)/DIM) # mean variance
    return sum(abs2, Σ - σ²*I) # penalize non-diagonal covariance matrices
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
    N = length(o)

    # Pull first iteration outside of the loop for easy initialization
    @inbounds Σ = f(o[1], o[2], r[1], r[2])
    @inbounds for i = 3:N
        oᵢ, rᵢ = o[i], r[i]
        for j in 1:i-1
            Σ += f(oᵢ, o[j], rᵢ, r[j])
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
    N = length(o)

    @inbounds for i in 1:N
        oᵢ, rᵢ = o[i], r[i]
        for j in 1:i-1
            G[i] += ∇f(oᵢ, o[j], rᵢ, r[j])
        end
        for j in i+1:N
            G[i] += ∇f(oᵢ, o[j], rᵢ, r[j])
        end
    end

    return g
end

# Hessian of the pairwise_sum function w.r.t. `x`. Result is stored in `H`.
function pairwise_hess!(H::AbstractMatrix{T},
                        ∇²f,
                        x::AbstractVector{T},
                        r::AbstractVector) where {T}
    @assert size(H,1) == size(H,2) == length(x) == 2*length(r) >= 4
    fill!(H, zero(T))
    o = reinterpret(Vec{2,T}, x)
    N = length(o)

    # Hoist l=k=1 outside of main loop to get htype for free
    oₖ, rₖ = o[1], r[1]
    @inbounds h = ∇²f(oₖ, o[2], rₖ, r[2])
    @inbounds for j in 3:N
        h += ∇²f(oₖ, o[j], rₖ, r[j])
    end
    unsafe_assignat!(H, h, 1, 1)
    htype = typeof(h)

    @inbounds for k in 2:N
        oₖ, rₖ, K = o[k], r[k], 2k-1

        # lower off-diagonal terms (l<k)
        for l in 1:k-1
            h = -∇²f(oₖ, o[l], rₖ, r[l]) # off-diags have minus sign
            unsafe_assignat!(H, h, K, 2l-1)
        end

        # diagonal terms (l==k): sum over j ≠ k
        h = zero(htype)
        for j in 1:k-1
            h += ∇²f(oₖ, o[j], rₖ, r[j])
        end
        for j in k+1:N
            h += ∇²f(oₖ, o[j], rₖ, r[j])
        end
        unsafe_assignat!(H, h, K, K)
    end

    # upper off-diagonal (l>k) terms
    unsafe_symmetrize_upper!(H)

    return H
end

function unsafe_symmetrize_upper!(H)
    @inbounds for j in 2:size(H,1), i in 1:j-1
        H[i,j] = H[j,i]
    end
    return H
end

const Tensor2D = Tensor{2,2}
const SymTensor2D = SymmetricTensor{2,2}
const MaybeSymTensor2D{T} = Union{<:Tensor2D{T}, <:SymTensor2D{T}} where {T}

@inline function unsafe_assignat!(
        H::AbstractMatrix{T},
        h::Tensor2D{T},
        i::Int,
        j::Int) where {T}
    @inbounds H[i,   j]   = h[1]
    @inbounds H[i+1, j]   = h[2]
    @inbounds H[i,   j+1] = h[3]
    @inbounds H[i+1, j+1] = h[4]
    return H
end

@inline function unsafe_assignat!(
        H::AbstractMatrix{T},
        h::SymTensor2D{T},
        i::Int,
        j::Int) where {T}
    @inbounds H[i,   j]   = h.data[1]
    @inbounds H[i+1, j]   = h.data[2]
    @inbounds H[i,   j+1] = h.data[2]
    @inbounds H[i+1, j+1] = h.data[3]
    return H
end

function unsafe_assignall!(H, h::MaybeSymTensor2D{T}) where {T}
    @inbounds for j in 1:2:size(H,2), i in 1:2:size(H,1)
        unsafe_assignat!(H,h,i,j)
    end
    return H
end

# ---------------------------------------------------------------------------- #
# Example symmetric functions on circles
# ---------------------------------------------------------------------------- #
function Base.:-(J::UniformScaling{Bool}, A::SymmetricTensor{2,2})
    T, d = typeof(A), Tensors.get_data(A)
    @inbounds B = T((J - d[1], -d[2], J - d[3]))
    return B
end

# Symmetric circle distance function on the circles (o1,r1) and (o2,r2)
@inline function d(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1)))
    return norm(o1 - o2) - r1 - r2 - ϵ
end
@inline function ∇d(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1)))
    # Gradient w.r.t `o1`
    return (o1 - o2)/norm(o1 - o2)
end
@inline function ∇²d(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1)))
    # Hessian w.r.t `o1`, i.e. ∂²d/∂o1², given by:
    #   1/|o1-o2|*I - (o1-o2)*(o1-o2)'/|o1-o2|^3
    # Or, writing e12 = o1 - o2 and d12 = |e12|, we have:
    #   (I - e12 * e12')/d12
    # NOTE: ∂²d/∂o1o2 == ∂²d/∂o2o1 == -∂²/∂o1²
    e = o1 - o2
    α = inv(norm(e))
    h = α * (I - otimes(α * e))
    return h
end

# Symmetric squared circle distance function on the circles (o1,r1) and (o2,r2)
@inline function d²(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1)))
    return d(o1,o2,r1,r2,ϵ)^2
end
@inline function ∇d²(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1)))
    # Gradient w.r.t `o1`
    return 2*d(o1,o2,r1,r2,ϵ) * ∇d(o1,o2,r1,r2,ϵ)
end
@inline function ∇²d²(o1::Vec, o2::Vec, r1, r2, ϵ = zero(typeof(r1)))
    # Hessian w.r.t `o1`, i.e. ∂²d/∂o1²
    return 2*d(o1,o2,r1,r2,ϵ) * ∇²d(o1,o2,r1,r2,ϵ) + 2*otimes(∇d(o1,o2,r1,r2,ϵ))
end

# Symmetric squared circle distance function on the circles (o1,r1) and (o2,r2)
@inline function d²_overlap(o1::Vec{dim,T}, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) where {dim,T}
    _d = d(o1,o2,r1,r2,ϵ)
    return _d >= zero(T) ? zero(T) : _d^2
end
@inline function ∇d²_overlap(o1::Vec{dim,T}, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) where {dim,T}
    # Gradient w.r.t `o1`
    _d = d(o1,o2,r1,r2,ϵ)
    return _d >= zero(T) ? zero(Vec{dim,T}) : 2 * _d * ∇d(o1,o2,r1,r2,ϵ)
end
@inline function ∇²d²_overlap(o1::Vec{dim,T}, o2::Vec, r1, r2, ϵ = zero(typeof(r1))) where {dim,T}
    # Hessian w.r.t `o1`, i.e. ∂²d/∂o1²
    _d = d(o1,o2,r1,r2,ϵ)
    return _d >= zero(T) ? zero(SymmetricTensor{2,dim,T}) : 2 * _d * ∇²d(o1,o2,r1,r2,ϵ) + 2 * otimes(∇d(o1,o2,r1,r2,ϵ))
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
