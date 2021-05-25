####
#### CVAE losses (KL-divergence, ELBO, likelihood, ...)
####

#### Ensemble of gaussian distributions

"Gaussian distribution which matches the mean and variance of an equally weighted ensemble of the input Gaussian distributions along dimension `dims`"
function ensemble_of_gaussians(μ, logσ; dims = :)
    # Fast version
    μ̄ = mean(μ; dims)
    σ̄² = mean(@. exp(2logσ) + (μ - μ̄)^2; dims)
    logσ̄ = @. log(σ̄²)/2
    return μ̄, logσ̄

    #= Numerically stable version
    μ̄, logσ_max = mean(μ), maximum(logσ)
    σ̄²_σ = mean(@. exp(2*(logσ - logσ_max))) # variance due to individual variances
    σ̄²_μ = mean(@. (μ - μ̄)^2) # variance due to variance in means
    logσ̄ = logσ_max > 0 ?
        logσ_max + log(σ̄²_σ + exp(-2logσ_max) * σ̄²_μ)/2 :
        log(exp(2logσ_max) * σ̄²_σ + σ̄²_μ)/2
    return μ̄, logσ̄
    =#
end

#### KL divergence between Gaussian and unit Gaussian

@inline_cufunc kldiv_unitgaussian(μ, logσ) = (expm1(2logσ) + μ^2) / 2  - logσ
Zygote.@adjoint kldiv_unitgaussian(μ, logσ) = kldiv_unitgaussian(μ, logσ), Δ -> Δ .* ∇kldiv_unitgaussian(μ, logσ)
@inline_cufunc ∇kldiv_unitgaussian(μ, logσ) = (μ, expm1(2logσ))

#### KL divergence between Gaussians

@inline_cufunc kldiv_gaussian(μq0, logσq, μr0, logσr) = (expm1(2*(logσq-logσr)) + exp(-2logσr) * (μr0 - μq0)^2) / 2 - (logσq - logσr)
Zygote.@adjoint kldiv_gaussian(μq0, logσq, μr0, logσr) = kldiv_gaussian(μq0, logσq, μr0, logσr), Δ -> Δ .* ∇kldiv_gaussian(μq0, logσq, μr0, logσr)
@inline_cufunc function ∇kldiv_gaussian(μq0, logσq, μr0, logσr)
    ∂μq0 = exp(-2logσr) * (μq0-μr0)
    ∂μr0 = -∂μq0
    ∂logσq = expm1(2*(logσq-logσr))
    ∂logσr = ∂μr0 * (μq0-μr0) - ∂logσq
    return (∂μq0, ∂logσq, ∂μr0, ∂logσr)
end

#### Gaussian negative log-likelihood

@inline_cufunc neglogL_gaussian(x, μ, logσ) = ((exp(-logσ) * (x - μ))^2 + log2π) / 2 + logσ
Zygote.@adjoint neglogL_gaussian(x, μ, logσ) = neglogL_gaussian(x, μ, logσ), Δ -> Δ .* ∇neglogL_gaussian(x, μ, logσ)
@inline_cufunc function ∇neglogL_gaussian(x, μ, logσ)
    ∂x = exp(-2logσ) * (x - μ)
    ∂μ = -∂x
    ∂logσ = 1 - ∂x * (x - μ)
    (∂x, ∂μ, ∂logσ)
end

#### Truncated Gaussian negative log-likelihood

function _test_trunc_gaussian_log_Z(; verbose = false, lo = -30, hi = 30)
    prnt(α, β, logẐ, logZ, verb) = verb && @info "
    α:      $α
    β:      $β
    approx: $logẐ
    true:   $logZ
    error:  $(abs((logZ - logẐ) / logZ))"

    vals = exp.(lo:hi)
    vals = [-reverse(vals); 0; vals]
    for i in 1:length(vals), j in i+1:length(vals), T in [Float64, Float32], gpu in [true, false]
        α, β = T(vals[i]), T(vals[j])
        if gpu
            α′   = CuVector{T}(T[α])
            β′   = CuVector{T}(T[β])
            logẐ = trunc_gaussian_log_Z(α′, β′) |> sum
        else
            logẐ = trunc_gaussian_log_Z(α, β)
        end
        logZ = trunc_gaussian_log_Z(big(α), big(β))
        prnt(α, β, logẐ, logZ, verbose)
        try
            @assert logẐ isa T # correct type
            @assert isapprox(logZ, logẐ; rtol = 1*eps(T), atol = eps(T))
        catch e
            prnt(α, β, logẐ, logZ, true)
            rethrow(e)
        end
    end
end

function trunc_gaussian_log_Z(α::Number, β::Number)
    # Compute logZ = log(Φ(β) - Φ(α)) = log(½(erf(β/√2) - erf(α/√2))) in a numerically stable way. Φ(x) = ½(1+erf(x/√2)) is the standard normal cdf
    α, β = invsqrt2 * α, invsqrt2 * β
    if β < 0
        α, β = -β, -α
    end
    if α < 0
        if β < 1
            log((erf(β) + erf(-α))/2)
        else
            log1p(-(erfc(β) + erfc(-α))/2)
        end
    else
        if β < 1
            log((erf(β) - erf(α))/2)
        else
            log(erfcx(α) - exp((α-β)*(α+β)) * erfcx(β)) - α^2 - logtwo
        end
    end
end
trunc_gaussian_log_Z(α::AbstractArray, β::AbstractArray) = trunc_gaussian_log_Z.(α, β)

function trunc_gaussian_log_Z(α::CuArray, β::CuArray)
    # Compute logZ = log(Φ(β) - Φ(α)) = log(½(erf(β/√2) - erf(α/√2))) in a numerically stable way. Φ(x) = ½(1+erf(x/√2)) is the standard normal cdf
    α′ = @. invsqrt2 * ifelse(β < 0, -β, α)
    β′ = @. invsqrt2 * ifelse(β < 0, -α, β)
    @. ifelse(
        α′ < 0,
        ifelse(
            β′ < 1,
            CUDA.log((CUDA.erf(β′) + CUDA.erf(-α′))/2),
            CUDA.log1p(-(CUDA.erfc(β′) + CUDA.erfc(-α′))/2),
        ),
        ifelse(
            β′ < 1,
            CUDA.log((CUDA.erf(β′) - CUDA.erf(α′))/2),
            CUDA.log(CUDA.erfcx(α′) - CUDA.exp((α′-β′)*(α′+β′)) * CUDA.erfcx(β′)) - α′^2 - logtwo,
        ),
    )
end

@inline unsafe_trunc_gaussian_bound(::Type{Float64}) = 8.12589066470191     # abs(Φ⁻¹(eps(Float64)))
@inline unsafe_trunc_gaussian_bound(::Type{Float32}) = 5.166578f0           # abs(Φ⁻¹(eps(Float32)))
@inline unsafe_trunc_gaussian_bound(::Type{T}) where {T} = abs(Φ⁻¹(eps(T))) # generic fallback
@inline unsafe_trunc_gaussian_bound(::AbstractArray{T}) where {T} = unsafe_trunc_gaussian_bound(float(T))
@inline unsafe_trunc_gaussian_bound(x::Real) = unsafe_trunc_gaussian_bound(float(typeof(x)))
@inline unsafe_trunc_gaussian_bound(x::ForwardDiff.Dual) = unsafe_trunc_gaussian_bound(ForwardDiff.valtype(x))
@inline unsafe_trunc_gaussian_clamp(x) = (bound = unsafe_trunc_gaussian_bound(x); @. clamp(x, -bound, bound))

# NOTE: Likelihood `unsafe_trunc_gaussian_log_Z` assumes a <= μ <= b, and, via `unsafe_trunc_gaussian_bound`,
#       rounds α=(μ-a)/σ and β=(b-μ)/σ values to -Inf/+Inf if they are too small/large, respectively, for numerical stability.
#       Incorrect answers will be given if these assumptions are violated

function unsafe_trunc_gaussian_log_Z(α, β)
    logZ = trunc_gaussian_log_Z(α, β)
    T    = typeof(logZ)
    bd   = unsafe_trunc_gaussian_bound(T)
    if α < -bd
        if β > bd
            zero(T) # Φ(α) ≈ 0, Φ(β) ≈ 1, log(Φ(β) - Φ(α)) ≈ 0
        else
            log1p(erf(invsqrt2 * β)) - logtwo # Φ(α) ≈ 0, log(Φ(β) - Φ(α)) ≈ log(Φ(β)) = log((1+erf(β/√2))/2)
        end
    else
        if β > bd
            log1p(-erf(invsqrt2 * α)) - logtwo # Φ(β) ≈ 1, log(Φ(β) - Φ(α)) ≈ log(1 - Φ(α)) = log((1-erf(α/√2))/2)
        else
            logZ # general case
        end
    end
end
unsafe_trunc_gaussian_log_Z(α::AbstractArray, β::AbstractArray) = unsafe_trunc_gaussian_log_Z.(α, β)

function unsafe_trunc_gaussian_log_Z(α::CuArray, β::CuArray)
    logZ  = trunc_gaussian_log_Z(α, β)
    zero_ = ofeltypefloat(logZ, 0)
    bd    = unsafe_trunc_gaussian_bound(eltype(logZ))
    @. ifelse(
        α < -bd,
        ifelse(
            β > bd,
            zero_, # Φ(α) ≈ 0, Φ(β) ≈ 1, log(Φ(β) - Φ(α)) ≈ 0
            log1p(erf(invsqrt2 * β)) - logtwo, # Φ(α) ≈ 0, log(Φ(β) - Φ(α)) ≈ log(Φ(β)) = log((1+erf(β/√2))/2)
        ),
        ifelse(
            β > bd,
            log1p(-erf(invsqrt2 * α)) - logtwo, # Φ(β) ≈ 1, log(Φ(β) - Φ(α)) ≈ log(1 - Φ(α)) = log((1-erf(α/√2))/2)
            logZ, # general case
        ),
    )
end

Zygote.@adjoint function unsafe_trunc_gaussian_log_Z(α, β)
    logZ = unsafe_trunc_gaussian_log_Z(α, β)
    function ∇unsafe_trunc_gaussian_log_Z_inner(Δ)
        ∇unsafe_trunc_gaussian_log_Z(Δ, α, β, logZ)
    end
    return logZ, ∇unsafe_trunc_gaussian_log_Z_inner
end

function ∇unsafe_trunc_gaussian_log_Z(Δ, α, β, logZ)
    zero_ = ofeltypefloat(logZ, 0)
    nrm_  = ofeltypefloat(logZ, invsqrt2π)
    bd    = unsafe_trunc_gaussian_bound(eltype(logZ))
    ∂α    = @. ifelse(α < -bd, zero_, -nrm_ * exp(-α^2/2 - logZ)) # ∂/∂α log(Φ(β) - Φ(α)) = -ϕ(α) / (Φ(β) - Φ(α)) = -ϕ(α) / exp(logZ)
    ∂β    = @. ifelse(β >  bd, zero_,  nrm_ * exp(-β^2/2 - logZ)) # ∂/∂β log(Φ(β) - Φ(β)) = +ϕ(β) / (Φ(β) - Φ(β)) = +ϕ(β) / exp(logZ)
    (Δ .* ∂α, Δ .* ∂β)
end

function neglogL_trunc_gaussian(x, μ, logσ, a, b)
    logϕ = @. neglogL_gaussian(x, μ, logσ)
    σ⁻¹  = @. exp(-logσ) # inv(σ)
    logZ = unsafe_trunc_gaussian_log_Z(@.(σ⁻¹ * (a - μ)), @.(σ⁻¹ * (b - μ)))
    return @. logϕ + logZ
end

#=
function neglogL_trunc_gaussian(x, μ, logσ, a, b)
    σ⁻¹  = @. exp(-logσ) # inv(σ)
    α    = unsafe_trunc_gaussian_clamp(σ⁻¹ .* (a .- μ))
    β    = unsafe_trunc_gaussian_clamp(σ⁻¹ .* (b .- μ))
    logZ = trunc_gaussian_log_Z(α, β)
    return @. ((σ⁻¹ * (x - μ))^2 + log2π) / 2 + logσ + logZ
end

Zygote.@adjoint neglogL_trunc_gaussian(x, μ, logσ, a, b) = neglogL_trunc_gaussian(x, μ, logσ, a, b), Δ -> ∇neglogL_trunc_gaussian(Δ, x, μ, logσ, a, b)

function ∇neglogL_trunc_gaussian(Δ, x, μ, logσ, a, b)
    σ⁻¹ = @. exp(-logσ) # inv(σ)
    α = unsafe_trunc_gaussian_clamp(@. σ⁻¹ * (a - μ))
    β = unsafe_trunc_gaussian_clamp(@. σ⁻¹ * (b - μ))
    logZ = trunc_gaussian_log_Z(α, β)
    logϕa = @. -(α^2 + log2π) / 2
    logϕb = @. -(β^2 + log2π) / 2
    ∂a = @. -exp(logϕa - logσ - logZ)
    ∂b = @. exp(logϕb - logσ - logZ)
    ∂x = @. σ⁻¹^2 * (x - μ)
    ∂μ = @. -∂x - (∂a + ∂b)
    ∂logσ = @. 1 - ∂x * (x - μ) + (α * exp(logϕa - logZ) - β * exp(logϕb - logZ))
    (Δ .* ∂x, Δ .* ∂μ, Δ .* ∂logσ, Δ .* ∂a, Δ .* ∂b)
end
=#

#### Laplace negative log-likelihood

@inline_cufunc neglogL_laplace(x, μ, logσ) = exp(-logσ) * abs(x - μ) + logσ + logtwo
Zygote.@adjoint neglogL_laplace(x, μ, logσ) = neglogL_laplace(x, μ, logσ), Δ -> Δ .* ∇neglogL_laplace(x, μ, logσ)
@inline_cufunc function ∇neglogL_laplace(x, μ, logσ)
    e⁻ˢ = exp(-logσ)
    ∂x = e⁻ˢ * sign(x - μ)
    ∂μ = -∂x
    ∂logσ = 1 - e⁻ˢ * abs(x - μ)
    (∂x, ∂μ, ∂logσ)
end

#### Kumaraswamy negative log-likelihood

@inline_cufunc neglogL_kumaraswamy(x, α, b) =  -Flux.softplus(α) - Flux.softplus(b) - exp(α)*log(x) - exp(b)*softlog(-(1+exp(α)) * log(x)) # equivalent to `-log(a) - log(b) - (a-1)*log(x) - (b-1)*log1p(-x^a)` where `a = 1+exp(α)` and `b = 1+exp(β)`
Zygote.@adjoint neglogL_kumaraswamy(x, α, β) = neglogL_kumaraswamy(x, α, β), Δ -> Δ .* ∇neglogL_kumaraswamy(x, α, β)
@inline_cufunc function ∇neglogL_kumaraswamy(x, α, β)
    logx, eᵅ, e⁻ᵅ, eᵝ, e⁻ᵝ = log(x), exp(α), exp(-α), exp(β), exp(-β)
    logxᵃ = (1 + eᵅ) * logx # log(x^a) == log(exp(a*log(x))) = (1+exp(α)) * log(x)
    xᵃ = exp(logxᵃ)
    ∂x = -(eᵅ / x) * (1 - (xᵃ / (1 - xᵃ)) * (1 + e⁻ᵅ) * eᵝ)
    ∂α = -(logx * eᵅ * (1 - eᵝ * xᵃ/(1 - xᵃ)) + 1/(1 + e⁻ᵅ))
    ∂β = -(eᵝ * softlog(-logxᵃ) + 1/(1 + e⁻ᵝ))
    (∂x, ∂α, ∂β)
end

#### Rician negative log-likelihood

@inline_cufunc function neglogL_rician_unsafe(x, ν, logσ)
    σ⁻¹ = exp(-logσ)
    x′, ν′ = σ⁻¹ * x, σ⁻¹ * ν
    return ifelse(
        x′ * ν′ < 1000,
        2logσ - log(x) + (x′^2 + ν′^2)/2 - _logbesseli0_cuda_unsafe(x′ * ν′), # accurate for small x′ * ν′
        logσ + ((x′-ν′)^2 + log(ν / x) + log2π)/2 - log1p(inv(8 * x′ * ν′)),  # for large x′ * ν′, use asymptotic form to avoid numerical issues
    )
end
@inline_cufunc neglogL_rician(x, ν, logσ; ϵ = epseltype(x), logϵ = log(epseltype(x))) = neglogL_rician_unsafe(max(x,ϵ), max(ν,ϵ), max(logσ,logϵ)) # abs(⋅)+ϵ retains some gradient informating, whereas max(⋅,ϵ) drops gradient when <ϵ; which is preferred?

ChainRules.@scalar_rule(
    neglogL_rician_unsafe(x, ν, logσ),
    @setup(σ⁻² = exp(-2logσ), z = σ⁻²*x*ν, r = _besselix1_cuda_unsafe(z) / _besselix0_cuda_unsafe(z)),
    (∂x_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r), ∂ν_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r), ∂logσ_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r)), # assumes strictly positive values
)
@inline_cufunc ∂x_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r) = σ⁻² * (x - ν * r) - 1/x
@inline_cufunc ∂ν_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r) = σ⁻² * (ν - x * r)
@inline_cufunc ∂logσ_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r) = 2 - σ⁻² * ((x - ν)^2 + 2 * ν * x * (1 - r))

#### Rician mean

@inline_cufunc mean_rician_unsafe(ν, σ) = sqrthalfπ * σ * _laguerre½_cuda_unsafe(-ν^2 / 2σ^2)
@inline_cufunc mean_rician(ν, σ; ϵ = epseltype(ν)) = mean_rician_unsafe(max(ν,ϵ), max(σ,ϵ)) # abs(⋅)+ϵ instead of e.g. max(⋅,ϵ) to avoid completely dropping gradient when <ϵ

#### Scalar losses (scalar functions of entire arrays)

EnsembleKLDivUnitGaussian(μ, logσ; dims = :) = KLDivUnitGaussian(ensemble_of_gaussians(μ, logσ; dims)...) # fit single Gaussian to (equally weighted) ensemble of Gaussians (arrays μ, logσ), and take KL divergence of this fitted Gaussian with a unit Gaussian

#### Arraywise losses (sum of individual elementwise losses)
@inline_cufunc _cap(f) = min(f, _capval(f))
@inline_cufunc _capval(f) = oftype(float(f), 1000)
@inline_cufunc _expcap(f) = min(f, _expcapval(f))
@inline_cufunc _expcapval(f) = oftype(float(f), 20)
@inline_cufunc _open_clampcap(f, x, a, b) = ifelse(a < x < b, min(f, _capval(f)), _capval(f))
@inline_cufunc _closed_clampcap(f, x, a, b) = ifelse(a <= x <= b, min(f, _capval(f)), _capval(f))

KLDivUnitGaussian(μ, logσ) = sum(_cap.(kldiv_unitgaussian.(μ, logσ))) / size(μ,2) # KL-divergence between approximation posterior and N(0, 1) prior (Note: sum over dims=1, mean over dims=2)
KLDivGaussian(μq0, logσq, μr0, logσr) = sum(_cap.(kldiv_gaussian.(μq0, logσq, μr0, logσr))) / size(μq0,2) # KL-divergence (Note: sum over dims=1, mean over dims=2)
NegLogLGaussian(x, μ, logσ) = sum(_cap.(neglogL_gaussian.(x, μ, logσ))) / size(μ,2) # Negative log-likelihood for Gaussian (Note: sum over dims=1, mean over dims=2)
NegLogLTruncatedGaussian(x, μ, logσ, a, b) = sum(_closed_clampcap.(neglogL_trunc_gaussian(x, μ, logσ, a, b), x, a, b)) / size(μ,2) # Negative log-likelihood for truncated Gaussian (Note: sum over dims=1, mean over dims=2)
NegLogLRician(x, μ, logσ) = sum(_open_clampcap.(neglogL_rician.(x, μ, logσ), x, zero(float(x)), oftype(float(x), Inf))) / size(μ,2) # Negative log-likelihood for Rician (Note: sum over dims=1, mean over dims=2)
NegLogLKumaraswamy(x, α, β) = sum(_open_clampcap.(neglogL_kumaraswamy.(x, _expcap.(α), _expcap.(β)), x, zero(float(x)), one(float(x)))) / size(α,2) # Negative log-likelihood for Kumaraswamy (Note: sum over dims=1, mean over dims=2)

function _crossentropy_gradcheck_test()
    for T in [Float32, Float64]
        @assert gradcheck(kldiv_unitgaussian, randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck(kldiv_gaussian, randn(T), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck(neglogL_gaussian, randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck(unsafe_trunc_gaussian_log_Z, -abs(randn(T)), abs(randn(T)); extrapolate = false, verbose = true)
        @assert gradcheck(neglogL_trunc_gaussian, ((a,μ,b) = sort(randn(T,3)); x = a+(b-a)*rand(T); logσ = randn(T); (x, logσ, μ, a, b))...; extrapolate = false, verbose = true)
        @assert gradcheck(neglogL_kumaraswamy, Flux.σ(randn(T)), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck(neglogL_laplace, randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck(neglogL_rician, exp(randn(T)), exp(randn(T)), randn(T); extrapolate = false, verbose = true)
    end
end

####
#### GANs
####

"Binary cross-entropy loss"
BCE(ŷ, y; kwargs...) = Flux.Losses.binarycrossentropy(ŷ, y; kwargs...)

"Binary cross-entropy loss with respect to labels of all 1s; minimized when ŷ = 1"
BCEOne(ŷ; agg = mean, ϵ = epseltype(ŷ)) = agg(@.(-log(ŷ+ϵ)))

"Binary cross-entropy loss with respect to labels of all 0s; minimized when ŷ = 0"
BCEZero(ŷ; agg = mean, ϵ = epseltype(ŷ)) = agg(@.(-log(1-ŷ+ϵ)))

"Binary cross-entropy loss from logit probabilities"
LogitBCE(σ⁻¹ŷ, y; kwargs...) = Flux.Losses.logitbinarycrossentropy(σ⁻¹ŷ, y; kwargs...)

"Binary cross-entropy loss from logit probabilities with respect to labels of all 1s; minimized when ŷ = 1"
LogitBCEOne(σ⁻¹ŷ; agg = mean) = agg(@.(-Flux.logσ(σ⁻¹ŷ)))

"Binary cross-entropy loss from logit probabilities with respect to labels of all 0s; minimized when ŷ = 0"
LogitBCEZero(σ⁻¹ŷ; agg = mean) = agg(@.(σ⁻¹ŷ - Flux.logσ(σ⁻¹ŷ)))

####
#### Regularization
####

function DepthwiseSmoothReg(; type)
    type = Symbol(type)
    rmse(X) = sqrt(mean(abs2.(X)))
    mae(X) = mean(abs.(X))
    if type === :L2grad
        Stack(@nntopo(X => ∇X => L2grad), ForwardDifference(), rmse)
    elseif type === :L2lap
        Stack(@nntopo(X => ∇²X => L2lap), Laplacian(), rmse)
    elseif type === :L1grad
        Stack(@nntopo(X => ∇X => L1grad), ForwardDifference(), mae)
    elseif type === :L1lap
        Stack(@nntopo(X => ∇²X => L1lap), Laplacian(), mae)
    else
        error("Unknown regularization type: $type")
    end
end

function ChannelwiseSmoothReg(; type)
    type = Symbol(type)
    ΔZ²(Z) = mean(abs2.(Z); dims = 2)
    ΔX²_ΔZ²(ΔX, ΔZ²) = @. abs2(ΔX) / (ΔZ² + 1f-3)
    ΔX_ΔZ(ΔX, ΔZ²) = @. abs(ΔX) / √(ΔZ² + 1f-3)
    rootmean(ΔX²_ΔZ²) = √(mean(ΔX²_ΔZ²))
    transp(X) = permutedims(X, (2,1))
    FD = Flux.Chain(transp, ForwardDifference())
    if type === :L2diff
        Stack(@nntopo((X,Z) : X => ΔX : Z => ΔZ => ΔZ² : (ΔX,ΔZ²) => ΔX²_ΔZ² => L2diff), FD, FD, ΔZ², ΔX²_ΔZ², rootmean)
    elseif type === :L1diff
        Stack(@nntopo((X,Z) : X => ΔX : Z => ΔZ => ΔZ² : (ΔX,ΔZ²) => ΔX_ΔZ => L1diff), FD, FD, ΔZ², ΔX_ΔZ, mean)
    else
        error("Unknown regularization type: $type")
    end
end

function VAEReg(vae_dec; regtype)
    L1(Y, M, Ydec) = sum(@. M * abs(Y - Ydec)) / sum(M) # mean of recon error within mask M
    RiceNegLogL(Y, M, (μYdec, logσYdec)) = sum(@. M * _cap(neglogL_rician(Y, μYdec, logσYdec))) / sum(M) # mean negative Rician log likelihood within mask M
    GaussianNegLogL(Y, M, (μYdec, logσYdec)) = sum(@. M * _cap(neglogL_gaussian(Y, μYdec, logσYdec))) / sum(M) # mean negative Gaussian log likelihood within mask M
    LaplaceNegLogL(Y, M, (μYdec, logσYdec)) = sum(@. M * _cap(neglogL_laplace(Y, μYdec, logσYdec))) / sum(M) # mean negative Gaussian log likelihood within mask M

    regtype = Symbol(regtype)
    (regtype === :None) && return nothing

    # `vae_dec` does not apply nonlinearity to output; enforce positivity of μ
    μ_decoder = Flux.Chain(vae_dec, Base.BroadcastFunction(Flux.softplus))
    μlogσ_decoder = Stack(@nntopo(z => μlogσ => (μ, logσ) : μ => μ⁺ : (μ⁺, logσ) => μ⁺logσ), vae_dec, split_dim1, Base.BroadcastFunction(Flux.softplus), tuple)

    decoder, vae_regloss =
        regtype === :L1 ? (μ_decoder, L1) :
        regtype === :Rician ? (μlogσ_decoder, RiceNegLogL) :
        regtype === :Gaussian ? (μlogσ_decoder, GaussianNegLogL) :
        regtype === :Laplace ? (μlogσ_decoder, LaplaceNegLogL) :
        error("Unknown VAE regularization type: $regtype")

    return Stack(@nntopo((Y, M, z) : z => Ydec : (Y, M, Ydec) => vae_regloss), decoder, vae_regloss)
end
