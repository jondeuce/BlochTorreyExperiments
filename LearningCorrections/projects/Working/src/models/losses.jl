####
#### Crossentropy
####

kldiv_unitnormal(μ, σ) = (σ^2 + μ^2 - 1) / 2  - log(σ)
kldiv(μq0, σq, μr0, σr) = ((σq / σr)^2 + ((μr0 - μq0) / σr)^2 - 1) / 2 - log(σq / σr)
elbo(x, μx0, σx) = (((x - μx0) / σx)^2 + log2π) / 2 + log(σx)

Zygote.@adjoint kldiv_unitnormal(μ, σ) = kldiv_unitnormal(μ, σ), Δ -> (Δ * μ, Δ * (σ - inv(σ)))
Zygote.@adjoint kldiv(μq0, σq, μr0, σr) = kldiv(μq0, σq, μr0, σr), Δ -> (Δ * (μq0 - μr0) / σr^2, Δ * (σq / σr - σr / σq) / σr, Δ * (μr0 - μq0) / σr^2, Δ * (σr^2 - (μr0 - μq0)^2 - σq^2) / σr^3)
Zygote.@adjoint elbo(x, μx0, σx) = elbo(x, μx0, σx), Δ -> (Δ * (x - μx0) / σx^2, Δ * (μx0 - x) / σx^2, Δ * (σx^2 - (x - μx0)^2) / σx^3)

@inline _cap(x) = min(x, oftype(x, 1000))
KLDivUnitNormal(μ, σ) = sum(_cap.(kldiv_unitnormal.(μ, σ))) / size(μ,2) # KL-divergence between approximation posterior and N(0, 1) prior (Note: sum over dims=1, mean over dims=2)
KLDivergence(μq0, σq, μr0, σr) = sum(_cap.(kldiv.(μq0, σq, μr0, σr))) / size(μq0,2) # KL-divergence contribution to cross-entropy (Note: sum over dims=1, mean over dims=2)
EvidenceLowerBound(x, μx0, σx) = sum(_cap.(elbo.(x, μx0, σx))) / size(μx0,2) # Negative log-likelihood/ELBO contribution to cross-entropy (Note: sum over dims=1, mean over dims=2)

function _crossentropy_gradcheck_test()
    for T in [Float32, Float64]
        _rand() = one(T) + rand(T)
        gradcheck(kldiv_unitnormal, _rand(), _rand(); extrapolate = false)
        gradcheck(kldiv, _rand(), _rand(), _rand(), _rand(); extrapolate = false)
        gradcheck(elbo, _rand(), _rand(), _rand(); extrapolate = false)
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
