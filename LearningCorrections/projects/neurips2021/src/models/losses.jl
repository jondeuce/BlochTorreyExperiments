####
#### CVAE losses (KL-divergence, ELBO, likelihood, ...)
####

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

# TODO: update gradients for asymptotic branch
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
