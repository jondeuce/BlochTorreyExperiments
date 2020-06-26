####
#### Math utils
####

# Smoothed version of max(x,e) for fixed e > 0
smoothmax(x,e) = e + e * Flux.softplus((x-e) / e)

"""
    log10range(a, b; length = 10)

Returns a `length`-element vector with log-linearly spaced data
between `a` and `b`
"""
log10range(a, b; length = 10) = 10 .^ range(log10(a), log10(b); length = length)

"""
    linspace(x1,x2,y1,y2) = x -> (y2 - y1) / (x2 - x1) * (x - x1) + y1
"""
@inline linspace(x1,x2,y1,y2) = x -> (y2 - y1) / (x2 - x1) * (x - x1) + y1

"""
    logspace(x1,x2,y1,y2) = x -> 10^linspace(x1, x2, log10(y1), log10(y2))(x)
"""
@inline logspace(x1,x2,y1,y2) = x -> 10^linspace(x1, x2, log10(y1), log10(y2))(x)

"""
    unitsum(x; dims = :) = x ./ sum(x; dims = dims)
"""
unitsum(x; dims = :) = x ./ sum(x; dims = dims)
unitsum!(x; dims = :) = x ./= sum(x; dims = dims)

"""
    snr(x, n)

Signal-to-noise ratio of the signal `x` relative to the noise `n`.
"""
snr(x, n; dims = 1) = 10 .* log10.(sum(abs2, x; dims = dims) ./ sum(abs2, n; dims = dims))

"""
    noise_level(z, SNR)

Standard deviation of gaussian noise with a given `SNR` level, proportional to the first time point.
    Note: `SNR` ≤ 0 is special cased to return a noise level of zero.
"""
noise_level(z::AbstractArray{T}, SNR::Number) where {T} =
    SNR ≤ 0 ? 0 .* z[1:1, ..] : abs.(z[1:1, ..]) ./ T(10^(SNR/20)) # Works for both real and complex

gaussian_noise(z::AbstractArray, SNR) = noise_level(z, SNR) .* randn(eltype(z), size(z))

"""
    add_gaussian(z, SNR)

Add gaussian noise with signal-to-noise ratio `SNR` proportional to the first time point.
"""
add_gaussian!(out::AbstractArray, z::AbstractArray, SNR) = out .= z .+ gaussian_noise(z, SNR)
add_gaussian!(z::AbstractArray, SNR) = z .+= gaussian_noise(z, SNR)
add_gaussian(z::AbstractArray, SNR) = z .+ gaussian_noise(z, SNR)

"""
    add_rician(z, SNR)

Add rician noise with signal-to-noise ratio `SNR` proportional to the first time point.
Always returns a real array.
"""
add_rician(m::AbstractArray{<:Real}, SNR) = add_rician(complex.(m), SNR)
add_rician(z::AbstractArray{<:Complex}, SNR) = abs.(add_gaussian(z, SNR))
# add_rician(m::AbstractArray{<:Real}, SNR) = add_rician!(copy(m), SNR)
# add_rician!(m::AbstractArray{<:Real}, SNR) = (gr = inv(√2) * gaussian_noise(m, SNR); gi = inv(√2) * gaussian_noise(m, SNR); m .= sqrt.(abs2.(m.+gr) .+ abs2.(gi)); return m)

# Total variation penalty
function make_tv_penalty(n::Int, T = Float64)
    L = diagm(1 => ones(T, n-1), 0 => -2*ones(T, n), -1 => ones(T, n-1))
    L[[1,end],:] .= 0
    tv_penalty(y) = mean(abs, L * y)
    return tv_penalty
end
make_tv_penalty(x::AbstractVecOrMat{T}) where {T} = make_tv_penalty(size(x,1), T)

# Tikhonov penalty on laplacian
function make_tikh_penalty(n::Int, T = Float64)
    L = diagm(1 => ones(T, n-1), 0 => -2*ones(T, n), -1 => ones(T, n-1))
    L[[1,end],:] .= 0
    tikh_penalty(y) = sqrt(mean(abs2, L * y))
    return tikh_penalty
end
make_tikh_penalty(x::AbstractVecOrMat{T}) where {T} = make_tikh_penalty(size(x,1), T)

function mix_columns(X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)
    m = size(X, 2)
    XY = hcat(X, Y)
    idx = randperm(2m)
    return (XY[:, idx[1:m]], XY[:, idx[m+1:end]])
end

function sample_columns(X::AbstractMatrix, batchsize)
    X[:, sample(1:size(X,2), batchsize; replace = false)]
end

function column_mse(X::AbstractVecOrMat, Y::AbstractVecOrMat, i, j)
    @assert size(X) == size(Y) && 1 <= min(i,j) <= max(i,j) <= size(X,2)
    T = promote_type(eltype(X), eltype(Y))
    Σ = zero(T)
    n, m = size(X,1), size(X,2)
    if !(X === Y && i == j) && n > 0
        @inbounds @simd for k in 1:n
            δxy = X[k,i] - Y[k,j]
            Σ += δxy * δxy
        end
        Σ /= n
    end
    return Σ
end

function column_mse(X::AbstractVector, Y::AbstractVector)
    @assert size(X) == size(Y)
    T = promote_type(eltype(X), eltype(Y))
    Σ = zero(T)
    n = length(X)
    if !(X === Y) && n > 0
        @inbounds @simd for k in 1:n
            δxy = X[k] - Y[k]
            Σ += δxy * δxy
        end
        Σ /= n
    end
    return Σ
end

function sum_columns!(out::AbstractVector{T}, X::AbstractMatrix{T}) where {T}
    @assert length(out) == size(X,1)
    @inbounds @simd for i in 1:size(X,1)
        out[i] = X[i,1]
    end
    @inbounds for j in 2:size(X,2)
        @simd for i in 1:size(X,1)
            out[i] += X[i,j]
        end
    end
    return out
end

function sum_rows!(out::AbstractVector{T}, X::AbstractMatrix{T}) where {T}
    @assert length(out) == size(X,1)
    @inbounds for j in 1:size(X,2)
        Σ = zero(T)
        @simd for i in 1:size(X,1)
            Σ += X[i,j]
        end
        out[j] = Σ
    end
    return out
end

function frob_norm2(X::AbstractArray)
    Σ = zero(eltype(X))
    @inbounds @simd for i in 1:length(X)
        xi = X[i]
        Σ += xi * xi
    end
    return Σ
end

####
#### Gradient testing
####

function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        tmp = x[i]
        δ = cbrt(eps(eltype(x))) # cbrt seems to be slightly better than sqrt; larger step size helps
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
        display.(Δ[i]) #TODO FIXME
    end
    return grads
end

function gradcheck(f, xs::AbstractArray...)
    dx0 = Flux.gradient(f, xs...)
    dx1 = ngradient(f, xs...)
    @show maximum.(abs, dx0)
    @show maximum.(abs, dx1)
    @show maximum.(abs, (dx0 .- dx1) ./ dx0)
    all(isapprox.(dx0, dx1, rtol = 1e-4, atol = 0))
end

function gradcheck(f, m::Flux.Chain, xs::AbstractArray...; onlyfirst = true, seed = 0)
    ps = Flux.params(m)
    !isnothing(seed) && Random.seed!(seed)
    g0 = Flux.gradient(() -> f(m, xs...), ps) |> g -> [g[p] for p in ps]
    onlyfirst && (g0 = first.(g0))
    display.(g0) #TODO FIXME

    m  = Flux.paramtype(BigFloat, m)
    ys = Flux.paramtype(BigFloat, xs)
    ps = Flux.params(m)
    onlyfirst && (ps = [@views(p[1:1]) for p in ps])
    g1 = ngradient(ps...) do (args...)
        !isnothing(seed) && Random.seed!(seed)
        f(m, ys...)
    end
    onlyfirst && (g1 = first.(g1))
    display.(g1) #TODO FIXME

    display.(g0 .- g1) #TODO FIXME
    map(g0, g1) do g0, g1
        display(abs.(g0 .- g1) .< cbrt.(eps.(g0))^2) #TODO FIXME
    end
end;

#=
let
    m = Flux.Dense(2,2,Flux.relu) |> Flux.f32
    x = 100*rand(2) |> Flux.f32
    gradcheck((m,x) -> sum(abs2, m(x)), m, x)
end;
let m = Flux.f32(m), xy = Flux.f32(test_data)
    # H_LIGOCVAE(m, xy...) |> display
    gradcheck((m,xy...) -> H_LIGOCVAE(m, xy...), m, xy...)
end;
let m = Flux.f64(m), xy = Flux.f64(test_data)
    # H_LIGOCVAE(m, xy...) |> display
    gradcheck((m,xy...) -> H_LIGOCVAE(m, xy...), m, xy...)
end;
=#
