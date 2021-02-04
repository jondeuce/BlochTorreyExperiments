####
#### Gaussian losses (KL-divergence, ELBO, likelihood, ...)
####

@inline kldiv_unitgaussian(μ, σ) = (σ^2 + μ^2 - 1) / 2  - log(σ)
@inline kldiv_gaussian(μq0, σq, μr0, σr) = ((σq / σr)^2 + ((μr0 - μq0) / σr)^2 - 1) / 2 - log(σq / σr)
@inline neglogL_gaussian(x, μx0, σx) = (((x - μx0) / σx)^2 + log2π) / 2 + log(σx)
@inline neglogL_kumaraswamy(x, a, b) = log(a) + log(b) + (a-1)*log(x) + (b-1)*log1p(-x^a)
@inline neglogL_laplace(x, μx0, σx) = abs(x - μx0) / σx + log(σx) + logtwo
@inline neglogL_rician(x, μx0, σx) = -_rician_logpdf_cuda(x, μx0, σx) # adjoint of `_rician_logpdf_cuda` is defined elsewhere

CUDA.@cufunc kldiv_unitgaussian(μ, σ) = (σ^2 + μ^2 - 1) / 2  - log(σ)
CUDA.@cufunc kldiv_gaussian(μq0, σq, μr0, σr) = ((σq / σr)^2 + ((μr0 - μq0) / σr)^2 - 1) / 2 - log(σq / σr)
CUDA.@cufunc neglogL_gaussian(x, μx0, σx) = (((x - μx0) / σx)^2 + log2π) / 2 + log(σx)
CUDA.@cufunc neglogL_kumaraswamy(x, a, b) = log(a) + log(b) + (a-1)*log(x) + (b-1)*log1p(-x^a)
CUDA.@cufunc neglogL_laplace(x, μx0, σx) = abs(x - μx0) / σx + log(σx) + logtwo

Zygote.@adjoint kldiv_unitgaussian(μ, σ) = kldiv_unitgaussian(μ, σ), Δ -> (Δ * μ, Δ * (σ - inv(σ)))
Zygote.@adjoint kldiv_gaussian(μq0, σq, μr0, σr) = kldiv_gaussian(μq0, σq, μr0, σr), Δ -> (Δ * (μq0 - μr0) / σr^2, Δ * (σq / σr - σr / σq) / σr, Δ * (μr0 - μq0) / σr^2, Δ * (σr^2 - (μr0 - μq0)^2 - σq^2) / σr^3)
Zygote.@adjoint neglogL_gaussian(x, μx0, σx) = neglogL_gaussian(x, μx0, σx), Δ -> (Δ * (x - μx0) / σx^2, Δ * (μx0 - x) / σx^2, Δ * (σx^2 - (x - μx0)^2) / σx^3)
Zygote.@adjoint neglogL_kumaraswamy(x, a, b) = neglogL_kumaraswamy(x, a, b), Δ -> (Δ * (a - 1 - (a*b - 1) * x^a) / (x * (1 - x^a)), Δ * (1/a + log(x) * (1 - b * x^a) / (1 - x^a)), Δ * (1/b + log1p(-x^a)))
Zygote.@adjoint neglogL_laplace(x, μx0, σx) = neglogL_laplace(x, μx0, σx), Δ -> (Δ * sign(x - μx0) / σx, Δ * sign(μx0 - x) / σx, Δ * (σx - abs(x - μx0)) / σx^2)

@inline _cap(x) = min(x, oftype(x, 1000))
@inline _epseltype(x) = 10*Flux.epseltype(x)
KLDivUnitGaussian(μ, σ) = sum(_cap.(kldiv_unitgaussian.(μ, σ))) / size(μ,2) # KL-divergence between approximation posterior and N(0, 1) prior (Note: sum over dims=1, mean over dims=2)
KLDivGaussian(μq0, σq, μr0, σr) = sum(_cap.(kldiv_gaussian.(μq0, σq, μr0, σr))) / size(μq0,2) # KL-divergence (Note: sum over dims=1, mean over dims=2)
NegLogLGaussian(x, μx0, σx) = sum(_cap.(neglogL_gaussian.(x, μx0, σx))) / size(μx0,2) # Negative log-likelihood for Gaussian (Note: sum over dims=1, mean over dims=2)
NegLogLKumaraswamy(x, a, b) = sum(_cap.(neglogL_kumaraswamy.(clamp.(x, _epseltype(x), 1-_epseltype(x)), a, b))) / size(a,2) # Negative log-likelihood for Kumaraswamy (Note: sum over dims=1, mean over dims=2)

function _crossentropy_gradcheck_test()
    for T in [Float32, Float64]
        @assert gradcheck((μ, σ′) -> kldiv_unitgaussian(μ, exp(σ′)), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck((μq0, σq′, μr0, σr′) -> kldiv_gaussian(μq0, exp(σq′), μr0, exp(σr′)), randn(T), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck((x, μx0, σx′) -> neglogL_gaussian(x, μx0, exp(σx′)), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck((x′,a′,b′) -> neglogL_kumaraswamy(Flux.σ(x′), 1+exp(a′), 1+exp(b′)), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck((x, μx0, σx′) -> neglogL_laplace(x, μx0, exp(σx′)), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
        @assert gradcheck((x′, μx0′, σx′) -> neglogL_rician(exp(x′), exp(μx0′), exp(σx′)), randn(T), randn(T), randn(T); extrapolate = false, verbose = true)
    end
end

####
#### GANs
####

"Binary cross-entropy loss"
BCE(ŷ, y; kwargs...) = Flux.Losses.binarycrossentropy(ŷ, y; kwargs...)

"Binary cross-entropy loss with respect to labels of all 1s; minimized when ŷ = 1"
BCEOne(ŷ; agg = mean, ϵ = Flux.epseltype(ŷ)) = agg(@.(-log(ŷ+ϵ)))

"Binary cross-entropy loss with respect to labels of all 0s; minimized when ŷ = 0"
BCEZero(ŷ; agg = mean, ϵ = Flux.epseltype(ŷ)) = agg(@.(-log(1-ŷ+ϵ)))

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
    RiceNegLogL(Y, M, (μYdec, σYdec)) = sum(@. M * _cap(neglogL_rician(Y, μYdec, σYdec))) / sum(M) # mean negative Rician log likelihood within mask M
    GaussianNegLogL(Y, M, (μYdec, σYdec)) = sum(@. M * _cap(neglogL_gaussian(Y, μYdec, σYdec))) / sum(M) # mean negative Gaussian log likelihood within mask M
    LaplaceNegLogL(Y, M, (μYdec, σYdec)) = sum(@. M * _cap(neglogL_laplace(Y, μYdec, σYdec))) / sum(M) # mean negative Gaussian log likelihood within mask M

    regtype = Symbol(regtype)
    (regtype === :None) && return nothing

    if regtype === :L1
        decoder = vae_dec
        vae_regloss = L1
    elseif regtype === :Rician
        decoder = Flux.Chain(vae_dec, split_mean_std) # `vae_dec` handles exp/softplus/etc. for mean/std outputs; just split the output in half
        vae_regloss = RiceNegLogL
    elseif regtype === :Gaussian
        decoder = Flux.Chain(vae_dec, split_mean_std) # `vae_dec` handles exp/softplus/etc. for mean/std outputs; just split the output in half
        vae_regloss = GaussianNegLogL
    elseif regtype === :Laplace
        decoder = Flux.Chain(vae_dec, split_mean_std) # `vae_dec` handles exp/softplus/etc. for mean/std outputs; just split the output in half
        vae_regloss = LaplaceNegLogL
    else
        error("Unknown VAE regularization type: $regtype")
    end

    Stack(
        @nntopo((Y, M, z) : z => Ydec : (Y, M, Ydec) => vae_regloss),
        decoder,
        vae_regloss,
    )
end
