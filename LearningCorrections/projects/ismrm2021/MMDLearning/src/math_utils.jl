####
#### Math utils
####

arr_similar(x::AbstractArray, y::AbstractArray) = arr_similar(typeof(x), y)
arr_similar(::Type{<:AbstractArray{T}}, y::AbstractArray) where {T} = convert(Array{T}, y)
arr_similar(::Type{<:AbstractArray{T}}, y::CUDA.CuArray{T}) where {T} = convert(Array{T}, y) #TODO: CuArray -> Array works directly if eltypes are equal
arr_similar(::Type{<:AbstractArray{T1}}, y::CUDA.CuArray{T2}) where {T1,T2} = convert(Array{T1}, y |> Flux.cpu) #TODO: CuArray -> Array falls back to scalar indexing with unequal eltypes
arr_similar(::Type{<:CUDA.CuArray{T1}}, y::CUDA.CuArray{T2}) where {T1,T2} = convert(CUDA.CuArray{T1}, y) #TODO: Needed for disambiguation
arr_similar(::Type{<:CUDA.CuArray{T}}, y::AbstractArray) where {T} = convert(CUDA.CuArray{T}, y)
Zygote.@adjoint arr_similar(::Type{Tx}, y::Ty) where {Tx <: AbstractArray, Ty <: AbstractArray} = arr_similar(Tx, y), Δ -> (nothing, arr_similar(Ty, Δ)) # preserve input type on backward pass

arr32(x::AbstractArray) = arr_similar(Array{Float32}, x)
arr64(x::AbstractArray) = arr_similar(Array{Float64}, x)

# rand_similar and randn_similar
for f in [:zeros, :ones, :rand, :randn]
    f_similar = Symbol(f, :_similar)
    @eval $f_similar(x::AbstractArray, sz...) = $f_similar(typeof(x), sz...)
    @eval $f_similar(::Type{<:AbstractArray{T}}, sz...) where {T} = Zygote.ignore(() -> Base.$f(T, sz...)) # fallback
    @eval $f_similar(::Type{<:CUDA.CuArray{T}}, sz...) where {T} = Zygote.ignore(() -> CUDA.$f(T, sz...)) # CUDA
end

fill_similar(x::AbstractArray, v, sz...) = fill_similar(typeof(x), v, sz...)
fill_similar(::Type{<:AbstractArray{T}}, v, sz...) where {T} = Base.fill(T(v), sz...) # fallback
fill_similar(::Type{<:CUDA.CuArray{T}}, v, sz...) where {T} = CUDA.fill(T(v), sz...) # CUDA

#TODO: `acosd` gets coerced to Float64 by Zygote on reverse pass; file bug?
_acosd_cuda(x::AbstractArray{T}) where {T} = clamp.(T(57.29577951308232) .* acos.(x), T(0.0), T(180.0)) # 180/π ≈ 57.29577951308232

# Soft minimum between `x` and `y` with sharpness `k`
@inline softmin(x,y,k) = (m = min(x,y); return m - log(exp((m-x)/k) + exp((m-y)/k))/k)

# Soft cutoff at `x = x0` with scale `w`. `f` is a sigmoidal function with unit scale which *increases* from 0 to 1 near `x = 0`
soft_cutoff(f, x::AbstractArray, x0, w) = f(@. (x0 - x) / w)
soft_cutoff(x::AbstractArray, x0, w) = soft_cutoff(sigmoid_weights_fun, x, x0, w) # fallback

# Erf scaled such that f(0) = 0.5, f(-1) = k, and f(1) = 1-k
function sigmoid_weights_fun(x::AbstractArray{T}, k::Number = T(0.1)) where {T}
    σ = T(abs(erfinv(1 - 2k)))
    y = @. (1 + erf(σ * x)) / 2
end
function sigmoid_weights_fun(x::CUDA.CuArray{T}, k::Number = T(0.1)) where {T}
    σ = T(abs(erfinv(1 - 2k)))
    y = @. (1 + CUDA.erf(σ * x)) / 2 #TODO: need to explicitly call CUDA.erf here... bug?
end

# Mix two functions
sample_union(f1, f2, p1, x::AbstractMatrix{T}) where {T} = p1 == 1 ? f1(x) : p1 == 0 ? f2(x) : ifelse.(rand_similar(x, 1, size(x,2)) .< T(p1), f1(x), f2(x))

normalized_range(N::Int) = N <= 1 ? zeros(N) : √(3*(N-1)/(N+1)) |> a -> range(-a,a,length=N) |> collect # mean zero and (uncorrected) std one
uniform_range(N::Int) = N <= 1 ? zeros(N) : range(-1,1,length=N) |> collect

@inline clamp_dim1(Y::AbstractArray, X::AbstractArray) = size(X,1) > size(Y,1) ? X[1:size(Y,1), ..] : X
@inline clamp_dim1(Y::AbstractArray, Xs::AbstractArray...) = clamp_dim1(Y, Xs)
@inline clamp_dim1(Y::AbstractArray, Xs) = map(X -> clamp_dim1(Y,X), Xs)

# Apply function `f` along dimension 1 of `x` by first flattening `x` into a matrix
@inline apply_dim1(f, x::AbstractMatrix) = f(x)
@inline apply_dim1(f, x::AbstractArray) = (y = f(reshape(x, size(x,1), :)); return reshape(y, size(y,1), Base.tail(size(x))...))
@inline apply_dim1(f, xs::AbstractArray...) = (ys = f(map(x -> reshape(x, size(x,1), :), xs)...); return map((y, xtail) -> reshape(y, size(y,1), xtail...), ys, Base.tail.(size.(xs))))

# Split `x` in half along first dimension
@inline split_dim1(x::AbstractArray) = (x[1:end÷2, ..], x[end÷2+1:end, ..])

# Split array into mean/standard deviation
@inline std_thresh(::AbstractArray{T}) where {T} = T(1e-6)
@inline split_mean_std(μ::AbstractArray) = split_dim1(μ)
@inline split_mean_exp_std(μ::AbstractArray) = ((μ0, logσ) = split_dim1(μ); return (μ0, exp.(logσ) .+ std_thresh(logσ)))
@inline split_mean_softplus_std(μ::AbstractArray) = ((μ0, invσ) = split_dim1(μ); return (μ0, Flux.softplus.(invσ) .+ std_thresh(invσ)))

# Temporary fix: https://github.com/FluxML/NNlib.jl/issues/254
Zygote.@adjoint Flux.softplus(x::Real) = Flux.softplus(x), Δ -> (Δ * Flux.σ(x),)

# Sample multivariate normal
@inline sample_mv_normal(μ::Union{<:Tuple,<:NamedTuple}) = sample_mv_normal(μ...)
@inline sample_mv_normal(μ::AbstractMatrix) = sample_mv_normal(split_dim1(μ)...)
@inline sample_mv_normal(μ0::AbstractMatrix{T}, σ::AbstractMatrix{T}) where {T} = μ0 .+ σ .* randn_similar(σ, max.(size(μ0), size(σ)))
@inline sample_mv_normal(μ0::AbstractMatrix{T}, σ::AbstractMatrix{T}, nsamples::Int) where {T} = μ0 .+ σ .* randn_similar(σ, max.(size(μ0), size(σ))..., nsamples)

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

# Compute discrete CDF
discrete_cdf(x) = (t = sort(x; dims = 2); c = cumsum(t; dims = 2) ./ sum(t; dims = 2); return permutedims.((t, c))) # return (t[1:12:end, :]', c[1:12:end, :]')

# Unzip array of structs into struct of arrays
unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

function bin_sorted(X, Y; binsize::Int)
    X_sorted, Y_sorted = unzip(sort(collect(zip(X, Y)); by = first))
    X_binned, Y_binned = unzip(map(is -> (mean(X_sorted[is]), mean(Y_sorted[is])), Iterators.partition(1:length(X), binsize)))
end

function bin_edges(X, Y, edges)
    X_binned, Y_binned = map(1:length(edges)-1) do i
        Is = @. edges[i] <= X <= edges[i+1]
        mean(X[Is]), mean(Y[Is])
    end |> unzip
end

function simple_fd_gradient!(g, f, x, lo = nothing, hi = nothing)
    δ = cbrt(eps(float(eltype(x))))
    f₀ = f(x)
    @inbounds for i in 1:length(x)
        x₀ = x[i]
        if (lo !== nothing) && (x₀ - δ/2 <= lo[i]) # near LHS boundary; use second-order forward: (-3 * f(x) + 4 * f(x + δ/2) - f(x + δ)) / δ
            x[i] = x₀ + δ/2
            f₊   = f(x)
            x[i] = x₀ + δ
            f₊₊  = f(x)
            g[i] = (-3f₀ + 4f₊ - f₊₊)/δ
        elseif (hi !== nothing) && (x₀ + δ/2 >= hi[i]) # near RHS boundary; use second-order backward: (3 * f(x) - 4 * f(x - δ/2) + f(x - δ)) / δ
            x[i] = x₀ - δ/2
            f₋   = f(x)
            x[i] = x₀ - δ
            f₋₋  = f(x)
            g[i] = (3f₀ - 4f₋ + f₋₋)/δ
        else # safely within boundary; use second-order central: (f(x + δ/2) - f(x - δ/2)) / δ
            x[i] = x₀ - δ/2
            f₋   = f(x)
            x[i] = x₀ + δ/2
            f₊   = f(x)
            g[i] = (f₊ - f₋)/δ
        end
        x[i] = x₀
    end
    return f₀
end

function fd_gradients(f, xs::Number...; extrapolate = true)
    map(1:length(xs)) do i
        fdm = FiniteDifferences.central_fdm(3, 1) # method
        fᵢ = y -> f(setindex!!(xs, y, i)...) # close over all j != i
        return extrapolate ?
            first(FiniteDifferences.extrapolate_fdm(fdm, fᵢ, xs[i]; breaktol = 2)) : # Richardson extrapolation
            fdm(fᵢ, xs[i])
    end
end

function fd_gradients(f, xs::AbstractArray...; extrapolate = true)
    try
        CUDA.allowscalar(true)
        ∇s = map(zero, xs)
        for (x, ∇) in zip(xs, ∇s), i in 1:length(x)
            fdm = FiniteDifferences.central_fdm(3, 1) # method
            xᵢ = x[i]
            fᵢ = y -> (x[i] = y; out = f(xs...); x[i] = xᵢ; return Float64(out))
            h₀ = 0.1 * Float64(max(abs(xᵢ), one(xᵢ)))
            ∇[i] = extrapolate ?
                first(FiniteDifferences.extrapolate_fdm(fdm, fᵢ, Float64(xᵢ), h₀; breaktol = 2)) : # Richardson extrapolation
                fdm(fᵢ, Float64(xᵢ))
        end
        return ∇s
    finally
        CUDA.allowscalar(false)
    end
end

function gradcheck(f, xs...; extrapolate = true, kwargs...)
    ∇ad = Zygote.gradient(f, xs...)
    ∇fd = fd_gradients(f, xs...; extrapolate)
    @assert all(isapprox.(∇ad, ∇fd; kwargs...))
end

function subset_indices_dict(ps::Flux.Params, subset = nothing)
    (subset === nothing) && return nothing
    Is = IdDict()
    if subset === :first
        foreach(p -> Is[p] = (I = first(CartesianIndices(p)); I:I), ps)
    else # :random
        foreach(p -> Is[p] = (I = rand(CartesianIndices(p)); I:I), ps)
    end
    return Is
end

walk_model(f, m) = mapfoldl((x,y) -> isempty(y) ? x : (x..., y), fieldnames(typeof(m)); init = ()) do k
    x = getfield(m, k)
    out = f(m,k,x)
    x isa AbstractArray ? (out,) : (out, walk_model(f, x)...)
end

function find_model_param(m, x)
    keychain = Any[]
    function find_model_param_inner(c)
        if isempty(c)
            false
        elseif c isa Pair
            c[2] === true
        else
            v = any(find_model_param_inner.(c))
            v && c[1] isa Pair && pushfirst!(keychain, c[1][1])
            return v
        end
    end
    chain = walk_model((_m,_k,_x) -> _k => _x === x, m)
    find_model_param_inner(chain)
    return keychain
end

function fd_modelgradients(f, ps::Flux.Params, Is::Union{<:IdDict, Nothing} = nothing; extrapolate = true)
    (Is !== nothing) && (ps = [view(p, Is[p]) for p in ps])
    fd_gradients((args...,) -> f(), ps...; extrapolate)
end

function fd_modelgradients(f, m; extrapolate = true, subset = nothing)
    ps = Flux.params(m)
    Is = subset_indices_dict(ps, subset)
    fd_modelgradients(f, ps, Is; extrapolate)
end

function modelgradcheck(f, m; extrapolate = true, subset = nothing, verbose = false, seed = nothing, kwargs...)
    (seed !== nothing) && (Random.seed!(seed); CUDA.seed!(0))
    ps = Flux.params(m)
    ℓ, J = Zygote.pullback(f, ps) # compute full gradient with backprop
    ∇ad = J(one(eltype(first(ps))))
    Is = subset_indices_dict(ps, subset)
    ∇fd = fd_modelgradients(f, ps, Is; extrapolate) # compute subset of gradient with finite differences
    ∇pairs = if (Is !== nothing)
        [(Flux.cpu(∇ad[p][Is[p]]), Flux.cpu(∇fd[i])) for (i,p) in enumerate(ps)] # move view of ∇ad to cpu to avoid scalar indexing into view
    else
        [(∇ad[p], ∇fd[i]) for (i,p) in enumerate(ps)]
    end
    verbose && map(zip(ps, ∇pairs)) do (p, ∇pair)
        println("ℓ: $ℓ, AD: $(∇pair[1]), FD: $(∇pair[2]), Δ: $(∇pair[1]-∇pair[2]), ≈: $(isapprox(∇pair...; kwargs...)), p: $(find_model_param(m, p))")
    end
    @assert all([isapprox(∇pair...; kwargs...) for ∇pair in ∇pairs])
end

function _softmax_test()
    c(x) = x # deepcopy(x) # softmax gradient had bug which modified input; wrap in copies to test
    sumabs2(f) = x -> sum(abs2, f(c(x)))
    ∇sumabs2(f) = function(x)
        y = f(c(x)) # f implements softmax
        Δ = 2 .* y # gradient of sum(abs2, y) w.r.t. y
        ∇x = y .* (Δ .- sum(Δ .* y, dims=1)) # softmax gradient, propagating Δ backward
        (∇x,)
    end
    simple_softmax(x; dims=1) = (y = exp.(x); y ./ sum(y; dims))

    let
        Random.seed!(0)
        CUDA.seed!(0)
        x = CUDA.rand(Float32,3,3)
        @assert simple_softmax(c(x)) ≈ Flux.softmax(c(x))

        @info "simple_softmax"
        Zygote.gradient(sumabs2(simple_softmax), c(x)) |> display
        MMDLearning.fd_gradients(sumabs2(simple_softmax), c(x); extrapolate = true) |> display
        ∇sumabs2(simple_softmax)(c(x)) |> display # true answer
        gradcheck(sumabs2(simple_softmax), c(x); rtol = 1e-2, atol = 0, extrapolate = true)
        println("")

        @info "Flux.softmax"
        Zygote.gradient(sumabs2(Flux.softmax), c(x)) |> display
        MMDLearning.fd_gradients(sumabs2(Flux.softmax), c(x); extrapolate = true) |> display
        ∇sumabs2(Flux.softmax)(c(x)) |> display # true answer
        gradcheck(sumabs2(Flux.softmax), c(x); rtol = 1e-2, atol = 0, extrapolate = true)
        println("")
    end
end
