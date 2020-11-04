####
#### Math utils
####

# rand_similar and randn_similar
for f in [:rand, :randn]
    f_sim = Symbol(f, :_similar)
    @eval $f_sim(x::AbstractArray, sz...) = $f_sim(typeof(x), sz...)
    @eval $f_sim(::Type{<:AbstractArray{T}}, sz...) where {T} = Base.$f(T, sz...) # fallback
    @eval $f_sim(::Type{<:CUDA.CuArray{T}}, sz...) where {T} = Zygote.ignore(() -> CUDA.$f(T, sz...)) # gpu version
end    

# Apply function `f` along dimension 1 of `x` by first flattening `x` into a matrix
@inline apply_dim1(f, x::AbstractMatrix) = f(x)
@inline apply_dim1(f, x::AbstractArray) = (y = f(reshape(x, size(x,1), :)); return reshape(y, size(y,1), Base.tail(size(x))...))
@inline apply_dim1(f, xs::AbstractArray...) = (ys = f(map(x -> reshape(x, size(x,1), :), xs)...); return map((y, xtail) -> reshape(y, size(y,1), xtail...), ys, Base.tail.(size.(xs))))

# Split `x` in half along first dimension
@inline split_dim1(x::AbstractArray) = (x[1:end÷2, ..], x[end÷2+1:end, ..])

# Split array into mean/standard deviation
@inline std_thresh(::AbstractArray{T}) where {T} = eps(T)
@inline split_mean_std(μ::AbstractArray) = split_dim1(μ)
@inline split_mean_exp_std(μ::AbstractArray) = ((μ0, logσ) = split_dim1(μ); return (μ0, exp.(logσ) .+ std_thresh(logσ)))
@inline split_mean_softplus_std(μ::AbstractArray) = ((μ0, invσ) = split_dim1(μ); return (μ0, Flux.softplus.(invσ) .+ std_thresh(invσ)))

# Sample multivariate normal
@inline sample_mv_normal(μ::Tuple) = sample_mv_normal(μ...)
@inline sample_mv_normal(μ::AbstractMatrix) = sample_mv_normal(split_dim1(μ)...)
@inline sample_mv_normal(μ0::AbstractMatrix{T}, σ::AbstractMatrix{T}) where {T} = μ0 .+ σ .* randn_similar(σ, max.(size(μ0), size(σ)))
@inline sample_mv_normal(μ0::AbstractMatrix{T}, σ::AbstractMatrix{T}, nsamples::Int) where {T} = μ0 .+ σ .* randn_similar(σ, max.(size(μ0), size(σ))..., nsamples)

# Compile time constant log(2π)
@inline log2π(::Type{T}) where {T} = log(2*T(π))

# One element
@inline one_element(x) = one_element(typeof(x))
@inline one_element(::Type{<:AbstractArray{T}}) where {T} = one{T}

# TODO: Tracker was much faster differentiating pow2.(x) than x.^2 - check for Zygote?
@inline pow2(x) = x*x

# Map over dictionary values
map_dict(f, d::Dict{K,V}) where {K,V} = Dict{K,V}(map(((k,v),) -> k => f(v), collect(d)))

# Differentiable summing of dictionary values
sum_dict(d::Dict{K,V}) where {K,V} = sum(values(d))

Zygote.@adjoint function sum_dict(d::Dict{K,V}) where {K,V}
    sum_dict(d), function (Δ)
        grad = Zygote.grad_mut(__context__, d)
        for k in keys(d)
            grad[k] = Zygote.accum(get(grad, k, nothing), Δ)
        end
        return (grad,)
    end
end

@generated function mask_tuple(tup::NamedTuple{keys, NTuple{N,T}}, ::Val{mask}) where {keys,N,T,mask}
    ex = [:(keys[$i] => getproperty(tup, keys[$i])) for i in 1:N if mask[i]]
    return :((; $(ex...)))
end

@generated function mask_tuple(tup::NTuple{N,T}, ::Val{mask}) where {N,T,mask}
    ex = [:(tup[$i]) for i in 1:N if mask[i]]
    return :(($(ex...),))
end


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
    unitmean(x; dims = :) = x ./ mean(x; dims = dims)
"""
unitmean(x; dims = :) = x ./ mean(x; dims = dims)
unitmean!(x; dims = :) = x ./= mean(x; dims = dims)

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

function sample_columns(X::AbstractMatrix, batchsize; replace = false)
    X[:, sample(1:size(X,2), batchsize; replace = replace)]
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
        # display.(Δ[i]) #TODO FIXME
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

####
#### Histogram
####

function fast_hist_1D(y, edges; normalize = nothing)
    @assert !isempty(y) && length(edges) >= 2
    _y = sort!(copy(y))
    @assert _y[1] >= edges[1]
    h = Histogram((edges,), zeros(Int, length(edges)-1), :left)
    j = 1
    @inbounds for i = 1:length(_y)
        _yi = _y[i]
        while _yi >= edges[j+1]
            j += 1
            (j+1 > length(edges)) && return h
        end
        h.weights[j] += 1
    end
    !isnothing(normalize) && (h = Plots.normalize(h, mode = normalize))
    return h
end

function fast_hist_1D(y, edges::AbstractRange; normalize = nothing)
    @assert length(edges) >= 2
    lo, hi, dx, n = first(edges), last(edges), step(edges), length(edges)
    h = Histogram((edges,), zeros(Int, n-1), :left)
    @inbounds for (i, yi) in enumerate(y)
        j = div(yi - lo, dx, RoundDown) + 1 |> Int
        (1 <= j <= n-1) && (h.weights[j] += 1)
    end
    !isnothing(normalize) && (h = Plots.normalize(h, mode = normalize))
    return h
end

function _fast_hist_test()
    _make_hist(x, edges) = fit(Histogram, x, UnitWeights{Float64}(length(x)), edges; closed = :left)
    for _ in 1:100
        n = rand(1:10)
        x = 100 .* rand(n)
        edges = rand(Bool) ? [0.0; sort(100 .* rand(n))] : (rand(1:10) : rand(1:10) : 100-rand(1:10))
        try
            @assert fast_hist_1D(x, edges) == _make_hist(x, edges)
        catch e
            if e isa InterruptException
                break
            else
                fast_hist_1D(x, edges).weights' |> x -> (display(x); display((first(x), last(x), sum(x))))
                _make_hist(x, edges).weights' |> x -> (display(x); display((first(x), last(x), sum(x))))
                rethrow(e)
            end
        end
    end

    x = rand(10^6)
    for ne in 2 .^ (4:10)
        edges = range(0, 1; length = ne)
        @info "range (ne = $ne)"; @btime fast_hist_1D($x, $edges)
        @info "array (ne = $ne)"; @btime fast_hist_1D($x, $(collect(edges)))
        @info "plots (ne = $ne)"; @btime $_make_hist($x, $edges)
    end
end
