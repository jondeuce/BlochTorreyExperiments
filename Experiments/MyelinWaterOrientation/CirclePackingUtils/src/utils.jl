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
            initial_origins[ix] = Vec{2,T}((2*Rmax*i, 2*Rmax*j))
        end
    else
        error("Unknown initial origins distribution: $distribution.")
    end

    return initial_origins
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
        all_results = DiffResults.DiffResult(zero(T), out)
        ForwardDiff.gradient!(all_results, f, x, cfg, Val{TF}()) # update out == ∇f(x)
        return DiffResults.value(all_results) # return f(x)
    end

    return g!, fg!, cfg
end