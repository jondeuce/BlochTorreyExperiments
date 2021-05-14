####
#### Math utils
####

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
function sigmoid_weights_fun(x::CuArray{T}, k::Number = T(0.1)) where {T}
    σ = T(abs(erfinv(1 - 2k)))
    y = @. (1 + CUDA.erf(σ * x)) / 2 #TODO: need to explicitly call CUDA.erf here... bug?
end

"smoothstep(x) = x^2 * (3 - 2x) (See: https://en.wikipedia.org/wiki/Smoothstep)"
smoothstep(x) = (x = float(x); x <= 0 ? zero(x) : x >= 1 ? one(x) : x^2 * (3 - 2x))

"smoothstep(x) = x^3 * (6x^2 - 15x + 10) (See: https://en.wikipedia.org/wiki/Smoothstep#Variations)"
smootherstep(x) = (x = float(x); x <= 0 ? zero(x) : x >= 1 ? one(x) : x^3 * evalpoly(x, (10, -15, 6)))

warmup(f, t, τ, δ = zero(t)) = t <= δ ? zero(float(t)) : f((t-δ)/τ)
cooldown(f, t, τ, δ = zero(t)) = t <= δ ? one(float(t)) : f((t-δ)/τ)
exp_warmup(t, τ, δ = zero(t)) = warmup(x -> -expm1(-x), t, τ, δ)
exp_cooldown(t, τ, δ = zero(t)) = cooldown(x -> exp(-x), t, τ, δ)
smoothstep_warmup(t, τ, δ = zero(t)) = warmup(smoothstep, t, τ, δ)
smoothstep_cooldown(t, τ, δ = zero(t)) = cooldown(x -> 1 - smoothstep(x), t, τ, δ)
cos_warmup(t, τ, δ = zero(t)) = warmup(x -> x <= 1 ? (1-cos(π*x))/2 : one(float(x)), t, τ, δ)
cos_cooldown(t, τ, δ = zero(t)) = cooldown(x -> x <= 1 ? (1+cos(π*x))/2 : zero(float(x)), t, τ, δ)

# Mix two functions
sample_union(f1, f2, p1, x::AbstractMatrix) = p1 >= 1 ? f1(x) : p1 <= 0 ? f2(x) : sample_union(f1(x), f2(x), p1)
sample_union(Y1::AbstractMatrix{T}, Y2::AbstractMatrix{T}, p1) where {T} = p1 >= 1 ? Y1 : p1 <= 0 ? Y2 : ifelse.(rand_similar(Y1, 1, size(Y1,2)) .< T(p1), Y1, Y2)

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
@inline split_mean_exp_std(μ::AbstractArray) = ((μ0, logσ) = split_dim1(μ); return (μ0, exp.(logσ))) #TODO add `std_thresh(logσ)`? shouldn't be necessary with properly initialized weights etc...
@inline split_mean_softplus_std(μ::AbstractArray) = ((μ0, log⁺σ) = split_dim1(μ); return (μ0, Flux.softplus.(log⁺σ))) #TODO add `std_thresh(log⁺σ)`? shouldn't be necessary with properly initialized weights etc...

# Temporary fix: https://github.com/FluxML/NNlib.jl/issues/254
Zygote.@adjoint Flux.softplus(x::Real) = Flux.softplus(x), Δ -> (Δ * Flux.σ(x),)

# Temporary fix: softmax broken for CuArray? (fixed on CUDA master; https://github.com/FluxML/Flux.jl/issues/1425, https://github.com/JuliaGPU/CUDA.jl/issues/599)
fast_softmax(x; dims = 1) = Flux.softmax(x; dims)
# function fast_softmax(x::AbstractArray; dims = 1)
#     y = exp.(x .- maximum(x; dims))
#     y ./= sum(y; dims)
#     return y
# end

# Temporary fix: cudnn softmax adjoint is slow?
Zygote.@adjoint function fast_softmax(x; dims = 1)
    y = fast_softmax(x; dims)
    return y, function (Δ)
        ∂A = Δ .* y
        ∂A .-= y .* sum(∂A; dims)
        return (∂A,)
    end
end

# Sample multivariate normal
@inline sample_uniform((lo, hi)) = sample_uniform(lo, hi)
@inline sample_uniform(μ::AbstractArray) = sample_uniform(split_dim1(μ)...)
@inline sample_uniform(lo::AbstractArray, hi::AbstractArray) = sample_uniform(lo, hi, rand_similar(hi, max.(size(lo), size(hi))))
@inline sample_uniform(lo::AbstractArray, hi::AbstractArray, nsamples::Int) = sample_uniform(lo, hi, rand_similar(hi, max.(size(lo), size(hi))..., nsamples))
@inline sample_uniform(lo, hi, u) = @. lo + (hi - lo) * u

# Sample multivariate normal
@inline sample_mv_normal((μ0, σ)) = sample_mv_normal(μ0, σ)
@inline sample_mv_normal(μ::AbstractArray) = sample_mv_normal(split_dim1(μ)...)
@inline sample_mv_normal(μ0::AbstractArray, σ::AbstractArray) = sample_mv_normal(μ0, σ, randn_similar(σ, max.(size(μ0), size(σ))))
@inline sample_mv_normal(μ0::AbstractArray, σ::AbstractArray, nsamples::Int) = sample_mv_normal(μ0, σ, randn_similar(σ, max.(size(μ0), size(σ))..., nsamples))
@inline sample_mv_normal(μ0, σ, z) = @. μ0 + σ * z

# Sample multivariate truncated normal
@inline sample_trunc_mv_normal((μ0, σ, a, b)) = sample_trunc_mv_normal(μ0, σ, a, b)
@inline sample_trunc_mv_normal(μ0::AbstractArray, σ::AbstractArray, a, b) = sample_trunc_mv_normal(μ0, σ, a, b, rand_similar(σ, max.(size(μ0), size(σ))))
@inline sample_trunc_mv_normal(μ0::AbstractArray, σ::AbstractArray, a, b, nsamples::Int) = sample_trunc_mv_normal(μ0, σ, a, b, rand_similar(σ, max.(size(μ0), size(σ))..., nsamples))
@inline function sample_trunc_mv_normal(μ0, σ, a, b, u; ϵ = epseltype(μ0))
    Φlo = Φ.((a.-μ0)./σ) # cdf range lower bound for inverse method
    Φhi = Φ.((b.-μ0)./σ) # cdf range upper bound for inverse method
    p   = clamp.(sample_uniform.(Φlo, Φhi, u), ϵ, 1-ϵ) # random samples within desired cdf range
    z   = Φ⁻¹.(p) # invert cdf to get truncated gaussian samples (standardized, i.e. for μ=0, σ=1)
    return clamp.(μ0 .+ σ .* z, a, b) # scale and shift to desired range
end

# Sample multivariate kumaraswamy
@inline sample_kumaraswamy((α, β)) = sample_kumaraswamy(α, β)
@inline sample_kumaraswamy(α::AbstractArray, β::AbstractArray) = sample_kumaraswamy(α, β, rand_similar(β, max.(size(α), size(β))))
@inline sample_kumaraswamy(α::AbstractArray, β::AbstractArray, nsamples::Int) = sample_kumaraswamy(α, β, rand_similar(β, max.(size(α), size(β))..., nsamples))
@inline sample_kumaraswamy(α, β, u) = @. clamp(exp(softlog(-log(u)*exp(-β)/(1+exp(-β))) * exp(-α)/(1+exp(-α))), 0, 1)

# Mode of multivariate kumaraswamy
@inline mode_kumaraswamy(α, β) = @. clamp(exp(-log((1+exp(-α)) * (1+exp(β)) - exp(-α)) / (1+exp(α))), 0, 1) # equivalent to `((a-1)/(a*b-1))^inv(a)` where `a = 1+exp(α)` and `b = 1+exp(β)`

"""
Compute, between two one-dimensional distributions `u` and `v`,
whose respective CDFs are `U` and `V`, the statistical distance
that is defined as:

        lₚ(u,v) = (∫|U-V|^p)^(1/p)

p is a positive parameter; p = 1 gives the Wasserstein distance,
p = 2 gives the energy distance.

c.f. https://github.com/scipy/scipy/blob/4ec4ab8d6ccc1cdb34b84fdcb66fde2cc0210dbf/scipy/stats/stats.py#L7861
"""
function cdf_distance end

function cdf_distance(p, u_values, v_values, u_weights = nothing, v_weights = nothing)
    u_sorter = sortperm(u_values)
    v_sorter = sortperm(v_values)
    u_sorted = u_values[u_sorter]
    v_sorted = v_values[v_sorter]
    all_values = sort!(vcat(u_values, v_values))

    # Compute the differences between pairs of successive values of u and v.
    deltas = diff(all_values)

    # Get the respective positions of the values of u and v among the values of both distributions.
    u_cdf_indices = [searchsortedlast(u_sorted, all_values[i]) for i in 1:length(all_values)-1]
    v_cdf_indices = [searchsortedlast(v_sorted, all_values[i]) for i in 1:length(all_values)-1]

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights === nothing
        u_cdf = u_cdf_indices ./ length(u_values)
    else
        u_sorted_cumweights = vcat(zero(eltype(u_weights)), cumsum(u_weights[u_sorter]))
        u_cdf = u_sorted_cumweights[1 .+ u_cdf_indices] ./ u_sorted_cumweights[end]
    end

    if v_weights === nothing
        v_cdf = v_cdf_indices ./ length(v_values)
    else
        v_sorted_cumweights = vcat(zero(eltype(v_weights)), cumsum(v_weights[v_sorter]))
        v_cdf = v_sorted_cumweights[1 .+ v_cdf_indices] ./ v_sorted_cumweights[end]
    end
    # @assert (u_cdf[1]   == 0) ⊻ (v_cdf[1]   == 0)
    # @assert (u_cdf[end] == 1) ⊻ (v_cdf[end] == 1)
    # @assert issorted(u_cdf) && issorted(v_cdf)

    # Compute the value of the integral based on the CDFs.
    map(p) do p
        if p == 1
            sum(@. abs(u_cdf - v_cdf) * deltas)
        elseif p == 2
            sqrt(sum(@. (u_cdf - v_cdf)^2 * deltas))
        elseif p == Inf
            maximum(@. abs(u_cdf - v_cdf))
        elseif p > 1
            Δcdf_max = maximum(@. abs(u_cdf - v_cdf))
            Δcdf_max == 0 ? zero(Δcdf_max) :
                Δcdf_max * sum(@. (abs(u_cdf - v_cdf) / Δcdf_max)^p * deltas)^inv(p)
        else # p < 1
            sum(@. abs(u_cdf - v_cdf)^p * deltas)^inv(p)
        end
    end
end

struct CDFDistanceBuffer{T}
    n::Int
    u::Vector{T}
    v::Vector{T}
    u_sorter::Vector{Int}
    v_sorter::Vector{Int}
    all_values::Vector{T}
    deltas::Vector{T}
    u_cdf_indices::Vector{Int}
    v_cdf_indices::Vector{Int}
    u_sorted_cumweights::Vector{T}
    v_sorted_cumweights::Vector{T}
    u_cdf::Vector{T}
    v_cdf::Vector{T}
end
function CDFDistanceBuffer(u::AbstractVector{T}, v::AbstractVector{T}) where {T}
    @assert length(u) == length(v)
    n = length(u)
    CDFDistanceBuffer(n, copy(u), copy(v), zeros(Int, n), zeros(Int, n), zeros(T, 2n), zeros(T, 2n-1), zeros(Int, 2n-1), zeros(Int, 2n-1), zeros(T, n+1), zeros(T, n+1), zeros(T, 2n-1), zeros(T, 2n-1))
end

function cdf_distance!(buf::CDFDistanceBuffer, p, u_values, v_values, u_weights = nothing, v_weights = nothing)
    sortperm!(buf.u_sorter, u_values)
    sortperm!(buf.v_sorter, v_values)
    @inbounds for i in 1:buf.n
        buf.u[i] = u_values[buf.u_sorter[i]]
        buf.v[i] = v_values[buf.v_sorter[i]]
    end

    copyto!(@views(buf.all_values[begin : buf.n]), buf.u)
    copyto!(@views(buf.all_values[buf.n+1 : end]), buf.v)
    sort!(buf.all_values)

    # Compute the differences between pairs of successive values of u and v.
    @inbounds for i in 1:length(buf.all_values)-1
        buf.deltas[i] = buf.all_values[i+1] - buf.all_values[i]
    end

    # Get the respective positions of the values of u and v among the values of both distributions.
    @inbounds for i in 1:length(buf.all_values)-1
        buf.u_cdf_indices[i] = searchsortedlast(buf.u, buf.all_values[i])
        buf.v_cdf_indices[i] = searchsortedlast(buf.v, buf.all_values[i])
    end

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights === nothing
        @inbounds buf.u_cdf .= buf.u_cdf_indices ./ buf.n
    else
        buf.u_sorted_cumweights[1] = 0
        @inbounds for i in 1:length(buf.u_sorted_cumweights)-1
            buf.u_sorted_cumweights[i+1] = buf.u_sorted_cumweights[i] + u_weights[buf.u_sorter[i]]
        end
        @inbounds for i in 1:length(buf.u_cdf_indices)
            buf.u_cdf[i] = buf.u_sorted_cumweights[1+buf.u_cdf_indices[i]] / buf.u_sorted_cumweights[end]
        end
    end

    if v_weights === nothing
        @inbounds buf.v_cdf .= buf.v_cdf_indices ./ buf.n
    else
        buf.v_sorted_cumweights[1] = 0
        @inbounds for i in 1:length(buf.v_sorted_cumweights)-1
            buf.v_sorted_cumweights[i+1] = buf.v_sorted_cumweights[i] + v_weights[buf.v_sorter[i]]
        end
        @inbounds for i in 1:length(buf.v_cdf_indices)
            buf.v_cdf[i] = buf.v_sorted_cumweights[1+buf.v_cdf_indices[i]] / buf.v_sorted_cumweights[end]
        end
    end
    # @assert (buf.u_cdf[1]   == 0) ⊻ (buf.v_cdf[1]   == 0)
    # @assert (buf.u_cdf[end] == 1) ⊻ (buf.v_cdf[end] == 1)
    # @assert issorted(buf.u_cdf) && issorted(buf.v_cdf)

    # Compute the value of the integral based on the CDFs.
    map(p) do p
        if p == 1
            sum(i -> abs(buf.u_cdf[i] - buf.v_cdf[i]) * buf.deltas[i], 1:2*buf.n-1)
        elseif p == 2
            sqrt(sum(i -> (buf.u_cdf[i] - buf.v_cdf[i])^2 * buf.deltas[i], 1:2*buf.n-1))
        elseif p == Inf
            maximum(i -> abs(buf.u_cdf[i] - buf.v_cdf[i]), 1:2*buf.n-1)
        elseif p > 1
            Δcdf_max = maximum(i -> abs(buf.u_cdf[i] - buf.v_cdf[i]), 1:2*buf.n-1)
            Δcdf_max == 0 ? zero(Δcdf_max) :
                Δcdf_max * sum(i -> (abs(buf.u_cdf[i] - buf.v_cdf[i]) / Δcdf_max)^p * buf.deltas[i], 1:2*buf.n-1)^inv(p)
        else # p < 1
            sum(i -> abs(buf.u_cdf[i] - buf.v_cdf[i])^p * buf.deltas[i], 1:2*buf.n-1)^inv(p)
        end
    end
end

"""
Compute the first Wasserstein distance between two 1D distributions.

c.f. https://github.com/scipy/scipy/blob/4ec4ab8d6ccc1cdb34b84fdcb66fde2cc0210dbf/scipy/stats/stats.py#L7701
"""
wasserstein_distance(u_values, v_values, u_weights = nothing, v_weights = nothing) = wasserstein_distance!(CDFDistanceBuffer(u_values, v_values), u_values, v_values, u_weights, v_weights)
wasserstein_distance!(buf::CDFDistanceBuffer, u_values, v_values, u_weights = nothing, v_weights = nothing) = cdf_distance!(buf, 1, u_values, v_values, u_weights, v_weights)

"""
Compute the energy distance between two 1D distributions.

c.f. https://github.com/scipy/scipy/blob/4ec4ab8d6ccc1cdb34b84fdcb66fde2cc0210dbf/scipy/stats/stats.py#L7778
"""
energy_distance(u_values, v_values, u_weights = nothing, v_weights = nothing) = energy_distance!(CDFDistanceBuffer(u_values, v_values), u_values, v_values, u_weights, v_weights)
energy_distance!(buf::CDFDistanceBuffer, u_values, v_values, u_weights = nothing, v_weights = nothing) = sqrt2 * cdf_distance!(buf, 2, u_values, v_values, u_weights, v_weights)

function _test_cdf_distance(; bench = false)
    cnt = 0
    while true
        n = (bench && cnt == 0) ? 100 : rand(1:5)
        p = 10^rand()
        u = randn(n)
        v = randn(n)
        uw = rand(n)
        vw = rand(n)
        buf = CDFDistanceBuffer(u,v)
        @assert wasserstein_distance(u,v) ≈ scipy.stats.wasserstein_distance(u,v)
        @assert wasserstein_distance(u,v,uw,vw) ≈ scipy.stats.wasserstein_distance(u,v,uw,vw)
        @assert wasserstein_distance!(buf,u,v) ≈ scipy.stats.wasserstein_distance(u,v)
        @assert wasserstein_distance!(buf,u,v,uw,vw) ≈ scipy.stats.wasserstein_distance(u,v,uw,vw)
        @assert energy_distance(u,v) ≈ scipy.stats.energy_distance(u,v)
        @assert energy_distance(u,v,uw,vw) ≈ scipy.stats.energy_distance(u,v,uw,vw)
        @assert energy_distance!(buf,u,v) ≈ scipy.stats.energy_distance(u,v)
        @assert energy_distance!(buf,u,v,uw,vw) ≈ scipy.stats.energy_distance(u,v,uw,vw)
        @assert cdf_distance(p,u,v) ≈ cdf_distance!(buf,p,u,v)
        @assert cdf_distance(p,u,v,uw,vw) ≈ cdf_distance!(buf,p,u,v,uw,vw)

        if bench && cnt == 0
            @btime wasserstein_distance!($buf,$u,$v)
            @btime wasserstein_distance($u,$v)
            @btime scipy.stats.wasserstein_distance($u,$v)
            @btime energy_distance!($buf,$u,$v,$uw,$vw)
            @btime energy_distance($u,$v,$uw,$vw)
            @btime scipy.stats.energy_distance($u,$v,$uw,$vw)
        end
        ((cnt += 1) >= 1000) && break
    end
end

"""
    @inline_cufunc f(x) = ...

Create an two definitions of `f`: one as written using `@inline f(x) = ...`,
and one which replaces `Base` functions used in definition of `f` with
equivalent `CUDA` functions using `CUDA.@cufunc f(x) = ...`.
"""
macro inline_cufunc(ex)
    Base.eval(__module__, :(@inline $ex))
    Base.eval(__module__, :(CUDA.@cufunc $ex))
end

# TODO: CUDA.cufunc() only looks to replace Base functions by default; manually specify replacements for replace erf, erfc, ...
for (f, cu_f) in [
        SpecialFunctions.erf => CUDA.erf,
        SpecialFunctions.erfc => CUDA.erfc,
        SpecialFunctions.erfcx => CUDA.erfcx,
        Distributions.normcdf => CUDA.normcdf,
        Distributions.norminvcdf => CUDA.normcdfinv,
    ]
    @eval CUDA.cufunc(::typeof($f)) = $cu_f
    push!(CUDA.cufuncs(), Symbol(f))
    unique!(CUDA.cufuncs())
end

# Sample multivariate truncated normal
@inline_cufunc Φ(x) = normcdf(x)
@inline_cufunc Φ⁻¹(p) = norminvcdf(p)

"""
    softlog(x) = log(1-exp(-x))

Logarithm softly bounded above by zero: softlog(x) → log(x) as x → 0⁺ and -exp(-x) as x → +∞.
"""
function softlog end
function ∇softlog end
@inline_cufunc softlog(x) = ifelse(x > 1, log1p(-exp(-x)), log(-expm1(-x)))
@inline_cufunc ∇softlog(x) = inv(expm1(x))
Zygote.@adjoint softlog(x) = softlog(x), Δ -> Δ * ∇softlog(x)

# One element
@inline one_element(x) = one_element(typeof(x))
@inline one_element(::Type{<:AbstractArray{T}}) where {T} = one{T}

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
    J = sample(1:size(X,2), batchsize; replace)
    return X[:,J], J
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
#### Gradient definitions
####

@inline _propagate_user_derivative(f, df, x::D) where {D <: ForwardDiff.Dual} = D(f(ForwardDiff.value(x)), df(ForwardDiff.value(x)) * ForwardDiff.partials(x))
@inline _propagate_user_derivative(fdf, x::D) where {D <: ForwardDiff.Dual} = ((fx, dfx) = fdf(ForwardDiff.value(x)); D(fx, dfx * ForwardDiff.partials(x)))

macro dual_gradient(f, df)
    return :($(esc(f))(x::ForwardDiff.Dual) = _propagate_user_derivative($(esc(f)), $(esc(df)), x))
end

macro primal_dual_gradient(f, fdf)
    return :($(esc(f))(x::ForwardDiff.Dual) = _propagate_user_derivative($(esc(fdf)), x))
end

#=
function mforwarddiff_from_scalar_rule(fname, fargs, i)
    # Create template function of one argument which takes gradient w.r.t. the i'th input
    f_args = ntuple(j -> j == i ? :($(fargs[i])::D) : :($(fargs[j])::Real), length(fargs))
    frule_args = ntuple(j -> j == i ? :(ForwardDiff.value($(fargs[i]))) : :($(fargs[j])), length(fargs))
    frule_Δargs = ntuple(j -> j == i ? 1 : 0, length(fargs))
    y, ẏ = gensym(), gensym()
    return :(
        function $fname($(f_args...)) where {D <: ForwardDiff.Dual}
            # Use `frule` defined by `@scalar_rule`
            $y, $ẏ = ChainRules.frule((ChainRules.NO_FIELDS, $(frule_Δargs...)), $fname, $(frule_args...))
            return D($y, $ẏ * ForwardDiff.partials($(fargs[i])))
        end
    )
end

macro forwarddiff_from_scalar_rule(f, Δargs = nothing)
    @assert f.head === :call && (Δargs === nothing || (Δargs.head === :tuple && length(f.args) == 1 + length(Δargs.args)))
    fname, fargs = esc(f.args[1]), f.args[2:end]
    Δargs = Δargs === nothing ? ones(Int, length(fargs)) : Δargs.args
    ex = Expr(:block)
    for (i,Δi) in enumerate(Δargs)
        Δi === :nothing && continue
        push!(ex.args, mforwarddiff_from_scalar_rule(fname, fargs, i))
    end
    return ex
end
=#

####
#### Gradient testing
####

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

function fwd_gradients(f, xs...)
    ntuple(length(xs)) do i
        ∇ = xs[i] isa Number ? ForwardDiff.derivative : ForwardDiff.gradient
        ∇(yi -> f(setindex!!(xs, yi, i)...), xs[i])
    end
end

function fd_gradients(f, xs::Number...; extrapolate = true, breaktol = 2)
    ntuple(length(xs)) do i
        fdm = FiniteDifferences.central_fdm(3, 1) # method
        fᵢ = y -> f(setindex!!(xs, y, i)...) # close over all j != i
        return extrapolate ?
            first(FiniteDifferences.extrapolate_fdm(fdm, fᵢ, xs[i]; breaktol)) : # Richardson extrapolation
            fdm(fᵢ, xs[i])
    end
end

function fd_gradients(f, xs::AbstractArray...; extrapolate = true, breaktol = 2)
    ∇s = map(zero, xs)
    CUDA.@allowscalar for (x, ∇) in zip(xs, ∇s), i in 1:length(x)
        fdm = FiniteDifferences.central_fdm(3, 1) # method
        xᵢ = x[i]
        fᵢ = y -> (x[i] = y; out = f(xs...); x[i] = xᵢ; return Float64(out))
        h₀ = 0.1 * Float64(max(abs(xᵢ), one(xᵢ)))
        ∇[i] = extrapolate ?
            first(FiniteDifferences.extrapolate_fdm(fdm, fᵢ, Float64(xᵢ), h₀; breaktol)) : # Richardson extrapolation
            fdm(fᵢ, Float64(xᵢ))
    end
    return ∇s
end

function gradcheck(f, xs...; extrapolate = true, breaktol = 2, backward = true, forward = true, verbose = false, kwargs...)
    @assert backward || forward
    ∇fd = fd_gradients(f, xs...; extrapolate, breaktol)
    bwdpassed = fwdpassed = true
    if backward
        ∇ad = Zygote.gradient(f, xs...)
        bwdpassed &= all(isapprox.(∇ad, ∇fd; kwargs...))
        verbose && !bwdpassed && @info "backward gradient failed", f(xs...), ∇ad, ∇fd, maximum(map((∇1,∇2) -> maximum(abs, ∇1.-∇2), ∇ad, ∇fd))
    end
    if forward
        ∇ad = CUDA.@allowscalar fwd_gradients(f, xs...)
        fwdpassed &= all(isapprox.(∇ad, ∇fd; kwargs...))
        verbose && !fwdpassed && @info "forward gradient failed", f(xs...), ∇ad, ∇fd, maximum(map((∇1,∇2) -> maximum(abs, ∇1.-∇2), ∇ad, ∇fd))
    end
    return bwdpassed && fwdpassed
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

function fd_modelgradients(f, ps::Flux.Params, Is::Union{<:IdDict, Nothing} = nothing; extrapolate = true, breaktol = 2)
    (Is !== nothing) && (ps = [view(p, Is[p]) for p in ps])
    fd_gradients((args...,) -> f(), ps...; extrapolate, breaktol)
end

function fd_modelgradients(f, m; extrapolate = true, breaktol = 2, subset = nothing)
    ps = Flux.params(m)
    Is = subset_indices_dict(ps, subset)
    fd_modelgradients(f, ps, Is; extrapolate, breaktol)
end

function modelgradcheck(f, m; extrapolate = true, breaktol = 2, subset = nothing, verbose = false, seed = nothing, kwargs...)
    (seed !== nothing) && (Random.seed!(seed); CUDA.seed!(0))
    ps = m isa Flux.Params ? m : Flux.params(m)
    ℓ, J = Zygote.pullback(f, ps) # compute full gradient with backprop
    ∇ad = J(one(eltype(first(ps))))
    Is = subset_indices_dict(ps, subset)
    ∇fd = fd_modelgradients(f, ps, Is; extrapolate, breaktol) # compute subset of gradient with finite differences
    ∇pairs = if (Is !== nothing)
        [(cpu(∇ad[p][Is[p]]), cpu(∇fd[i])) for (i,p) in enumerate(ps)] # move view of ∇ad to cpu to avoid scalar indexing into view
    else
        [(∇ad[p], ∇fd[i]) for (i,p) in enumerate(ps)]
    end
    verbose && map(zip(ps, ∇pairs)) do (p, ∇pair)
        println("ℓ: $ℓ, AD: $(∇pair[1]), FD: $(∇pair[2]), Δ: $(∇pair[1]-∇pair[2]), ≈: $(isapprox(∇pair...; kwargs...))" * (m isa Flux.Params ? "" : ", p: $(find_model_param(m, p))"))
    end
    return all([isapprox(∇pair...; kwargs...) for ∇pair in ∇pairs])
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
        fd_gradients(sumabs2(simple_softmax), c(x); extrapolate = true, breaktol = 2) |> display
        ∇sumabs2(simple_softmax)(c(x)) |> display # true answer
        @assert gradcheck(sumabs2(simple_softmax), c(x); rtol = 1e-2, atol = 0, extrapolate = true, breaktol = 2, backward = true, forward = true)
        println("")

        @info "Flux.softmax"
        Zygote.gradient(sumabs2(Flux.softmax), c(x)) |> display
        fd_gradients(sumabs2(Flux.softmax), c(x); extrapolate = true, breaktol = 2) |> display
        ∇sumabs2(Flux.softmax)(c(x)) |> display # true answer
        @assert gradcheck(sumabs2(Flux.softmax), c(x); rtol = 1e-2, atol = 0, extrapolate = true, breaktol = 2, backward = true, forward = false) #TODO cudnn calls fail with duals
        println("")
    end
end
