####
#### CVAE losses (KL-divergence, ELBO, likelihood, ...)
####

#### Ensemble of gaussian distributions

"Return Gaussian distbn which matches the mean and variance of an equally weighted ensemble of the input distributions"
function ensemble_of_gaussians(μ, logσ)
    # Fast version
    μ̄ = mean(μ)
    logσ̄ = log(mean(@. exp(2logσ) + (μ - μ̄)^2)) / 2
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

#=
@inline_cufunc kldiv_unitgaussian(μ, σ) = (σ^2 + μ^2 - 1) / 2  - log(σ)
Zygote.@adjoint kldiv_unitgaussian(μ, σ) = kldiv_unitgaussian(μ, σ), Δ -> (Δ * μ, Δ * (σ - inv(σ)))
=#

@inline_cufunc kldiv_unitgaussian(μ, logσ) = (expm1(2logσ) + μ^2) / 2  - logσ
Zygote.@adjoint kldiv_unitgaussian(μ, logσ) = kldiv_unitgaussian(μ, logσ), Δ -> Δ .* ∇kldiv_unitgaussian(μ, logσ)
@inline_cufunc ∇kldiv_unitgaussian(μ, logσ) = (μ, expm1(2logσ))

#### KL divergence between Gaussians

#=
@inline_cufunc kldiv_gaussian(μq0, σq, μr0, σr) = ((σq / σr)^2 + ((μr0 - μq0) / σr)^2 - 1) / 2 - log(σq / σr)
Zygote.@adjoint kldiv_gaussian(μq0, σq, μr0, σr) = kldiv_gaussian(μq0, σq, μr0, σr), Δ -> (Δ * (μq0 - μr0) / σr^2, Δ * (σq / σr - σr / σq) / σr, Δ * (μr0 - μq0) / σr^2, Δ * (σr^2 - (μr0 - μq0)^2 - σq^2) / σr^3)
=#

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

#=
@inline_cufunc neglogL_gaussian(x, μx0, σx) = (((x - μx0) / σx)^2 + log2π) / 2 + log(σx)
Zygote.@adjoint neglogL_gaussian(x, μx0, σx) = neglogL_gaussian(x, μx0, σx), Δ -> (Δ * (x - μx0) / σx^2, Δ * (μx0 - x) / σx^2, Δ * (σx^2 - (x - μx0)^2) / σx^3)
=#

@inline_cufunc neglogL_gaussian(x, μx0, logσx) = (exp(-2logσx) * (x - μx0)^2 + log2π) / 2 + logσx
Zygote.@adjoint neglogL_gaussian(x, μx0, logσx) = neglogL_gaussian(x, μx0, logσx), Δ -> Δ .* ∇neglogL_gaussian(x, μx0, logσx)
@inline_cufunc function ∇neglogL_gaussian(x, μx0, logσx)
    ∂x = exp(-2logσx) * (x - μx0)
    ∂μx0 = -∂x
    ∂logσx = 1 - ∂x * (x - μx0)
    (∂x, ∂μx0, ∂logσx)
end

#### Laplace negative log-likelihood

#=
@inline_cufunc neglogL_laplace(x, μx0, σx) = abs(x - μx0) / σx + log(σx) + logtwo
Zygote.@adjoint neglogL_laplace(x, μx0, σx) = neglogL_laplace(x, μx0, σx), Δ -> (Δ * sign(x - μx0) / σx, Δ * sign(μx0 - x) / σx, Δ * (σx - abs(x - μx0)) / σx^2)
=#

@inline_cufunc neglogL_laplace(x, μx0, logσx) = exp(-logσx) * abs(x - μx0) + logσx + logtwo
Zygote.@adjoint neglogL_laplace(x, μx0, logσx) = neglogL_laplace(x, μx0, logσx), Δ -> Δ .* ∇neglogL_laplace(x, μx0, logσx)
@inline_cufunc function ∇neglogL_laplace(x, μx0, logσx)
    e⁻ˢ = exp(-logσx)
    ∂x = e⁻ˢ * sign(x - μx0)
    ∂μx0 = -∂x
    ∂logσx = 1 - e⁻ˢ * abs(x - μx0)
    (∂x, ∂μx0, ∂logσx)
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

#=
@inline_cufunc neglogL_rician_unsafe(x, ν, σ) = 2*log(σ) - log(x) + (x^2 + ν^2)/(2*σ^2) - _logbesseli0_cuda_unsafe(x*ν/σ^2)
@inline_cufunc neglogL_rician(x, ν, σ; ϵ = epseltype(x)) = neglogL_rician_unsafe(max(x,ϵ), max(ν,ϵ), max(σ,ϵ)) # abs(⋅)+ϵ retains some gradient informating, whereas max(⋅,ϵ) drops gradient when <ϵ; which is preferred?

ChainRules.@scalar_rule(
    neglogL_rician_unsafe(x, ν, σ),
    @setup(σ⁻² = inv(σ^2), z = σ⁻²*x*ν, r = _besselix1_cuda_unsafe(z) / _besselix0_cuda_unsafe(z)),
    (∂x_neglogL_rician_unsafe(x, ν, σ, σ⁻², z, r), ∂ν_neglogL_rician_unsafe(x, ν, σ, σ⁻², z, r), ∂σ_neglogL_rician_unsafe(x, ν, σ, σ⁻², z, r)), # assumes strictly positive values
)
@inline_cufunc ∂x_neglogL_rician_unsafe(x, ν, σ, σ⁻², z, r) = σ⁻² * (x - ν * r) - 1/x
@inline_cufunc ∂ν_neglogL_rician_unsafe(x, ν, σ, σ⁻², z, r) = σ⁻² * (ν - x * r)
@inline_cufunc ∂σ_neglogL_rician_unsafe(x, ν, σ, σ⁻², z, r) = (2 - σ⁻² * ((x - ν)^2 + 2 * ν * x * (1 - r))) / σ
@forwarddiff_from_scalar_rule neglogL_rician_unsafe(x, ν, σ)
=#

@inline_cufunc neglogL_rician_unsafe(x, ν, logσ) = 2logσ - log(x) + exp(-2logσ) * (x^2 + ν^2)/2 - _logbesseli0_cuda_unsafe(exp(-2logσ) * x * ν)
@inline_cufunc neglogL_rician(x, ν, logσ; ϵ = epseltype(x), logϵ = log(epseltype(x))) = neglogL_rician_unsafe(max(x,ϵ), max(ν,ϵ), max(logσ,logϵ)) # abs(⋅)+ϵ retains some gradient informating, whereas max(⋅,ϵ) drops gradient when <ϵ; which is preferred?

ChainRules.@scalar_rule(
    neglogL_rician_unsafe(x, ν, logσ),
    @setup(σ⁻² = exp(-2logσ), z = σ⁻²*x*ν, r = _besselix1_cuda_unsafe(z) / _besselix0_cuda_unsafe(z)),
    (∂x_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r), ∂ν_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r), ∂logσ_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r)), # assumes strictly positive values
)
@inline_cufunc ∂x_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r) = σ⁻² * (x - ν * r) - 1/x
@inline_cufunc ∂ν_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r) = σ⁻² * (ν - x * r)
@inline_cufunc ∂logσ_neglogL_rician_unsafe(x, ν, logσ, σ⁻², z, r) = 2 - σ⁻² * ((x - ν)^2 + 2 * ν * x * (1 - r))
@forwarddiff_from_scalar_rule neglogL_rician_unsafe(x, ν, logσ)

#### Rician mean

@inline_cufunc mean_rician_unsafe(ν, σ) = sqrthalfπ * σ * _laguerre½_cuda_unsafe(-ν^2 / 2σ^2)
@inline_cufunc mean_rician(ν, σ; ϵ = epseltype(ν)) = mean_rician_unsafe(max(ν,ϵ), max(σ,ϵ)) # abs(⋅)+ϵ instead of e.g. max(⋅,ϵ) to avoid completely dropping gradient when <ϵ

#### Scalar losses (scalar functions of entire arrays)
EnsembleKLDivUnitGaussian(μ, logσ) = KLDivUnitGaussian(ensemble_of_gaussians(μ, logσ)...) # fit single Gaussian to (equally weighted) ensemble of Gaussians (arrays μ, logσ), and take KL divergence of this fitted Gaussian with a unit Gaussian

#### Arraywise losses (sum of individual elementwise losses)

@inline_cufunc _cap(x) = min(x, oftype(x, 1000))
@inline_cufunc _expcap(x) = min(x, oftype(x, 20))
# KLDivUnitGaussian(μ, σ) = sum(_cap.(kldiv_unitgaussian.(μ, σ))) / size(μ,2) # KL-divergence between approximation posterior and N(0, 1) prior (Note: sum over dims=1, mean over dims=2)
# KLDivGaussian(μq0, σq, μr0, σr) = sum(_cap.(kldiv_gaussian.(μq0, σq, μr0, σr))) / size(μq0,2) # KL-divergence (Note: sum over dims=1, mean over dims=2)
# NegLogLGaussian(x, μx0, σx) = sum(_cap.(neglogL_gaussian.(x, μx0, σx))) / size(μx0,2) # Negative log-likelihood for Gaussian (Note: sum over dims=1, mean over dims=2)
KLDivUnitGaussian(μ, logσ) = sum(_cap.(kldiv_unitgaussian.(μ, logσ))) / size(μ,2) # KL-divergence between approximation posterior and N(0, 1) prior (Note: sum over dims=1, mean over dims=2)
KLDivGaussian(μq0, logσq, μr0, logσr) = sum(_cap.(kldiv_gaussian.(μq0, logσq, μr0, logσr))) / size(μq0,2) # KL-divergence (Note: sum over dims=1, mean over dims=2)
NegLogLGaussian(x, μx0, logσx) = sum(_cap.(neglogL_gaussian.(x, μx0, logσx))) / size(μx0,2) # Negative log-likelihood for Gaussian (Note: sum over dims=1, mean over dims=2)
NegLogLRician(x, μx0, logσx) = sum(_cap.(neglogL_rician.(x, μx0, logσx))) / size(μx0,2) # Negative log-likelihood for Rician (Note: sum over dims=1, mean over dims=2)
NegLogLKumaraswamy(x, α, β) = sum(_cap.(neglogL_kumaraswamy.(x, _expcap.(α), _expcap.(β)))) / size(α,2) # Negative log-likelihood for Kumaraswamy (Note: sum over dims=1, mean over dims=2)

function _crossentropy_gradcheck_test()
    for T in [Float32, Float64]
        # @assert gradcheck((μ, σ′) -> kldiv_unitgaussian(μ, exp(σ′)), randn(T), randn(T); extrapolate = false, verbose = true)
        # @assert gradcheck((μq0, σq′, μr0, σr′) -> kldiv_gaussian(μq0, exp(σq′), μr0, exp(σr′)), randn(T), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        # @assert gradcheck((x, μx0, σx′) -> neglogL_gaussian(x, μx0, exp(σx′)), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        # @assert gradcheck((x, μx0, σx′) -> neglogL_laplace(x, μx0, exp(σx′)), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        # @assert gradcheck(neglogL_rician, exp(randn(T)), exp(randn(T)), exp(randn(T)); extrapolate = false, verbose = true)
        @assert gradcheck(kldiv_unitgaussian, randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck(kldiv_gaussian, randn(T), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck(neglogL_gaussian, randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
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
