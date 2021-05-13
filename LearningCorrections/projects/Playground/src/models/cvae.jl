"""
Conditional variational autoencoder.

Architecture inspired by:
    "Bayesian parameter estimation using conditional variational autoencoders for gravitational-wave astronomy"
    https://arxiv.org/abs/1802.08797
"""
struct CVAE{n,nθ,nθM,k,nz,Dist,E1,E2,D,B1,B2}
    E1 :: E1
    E2 :: E2
    D  :: D
    θbd :: B1
    θ̄bd :: B2
end
CVAE{n,nθ,nθM,k,nz}(enc1::E1, enc2::E2, dec::D, θbd::B1, θ̄bd::B2; posterior_dist::Type = Gaussian) where {n,nθ,nθM,k,nz,E1,E2,D,B1,B2} = CVAE{n,nθ,nθM,k,nz,posterior_dist,E1,E2,D,B1,B2}(enc1, enc2, dec, θbd, θ̄bd)

const CVAEPosteriorDist{Dist} = CVAE{n,nθ,nθM,k,nz,Dist} where {n,nθ,nθM,k,nz}

Flux.functor(::Type{<:CVAE{n,nθ,nθM,k,nz,Dist}}, c) where {n,nθ,nθM,k,nz,Dist} = (E1 = c.E1, E2 = c.E2, D = c.D, θbd = c.θbd, θ̄bd = c.θ̄bd,), fs -> CVAE{n,nθ,nθM,k,nz}(fs...; posterior_dist = Dist)
Base.show(io::IO, ::CVAE{n,nθ,nθM,k,nz,Dist}) where {n,nθ,nθM,k,nz,Dist} = print(io, "CVAE$((;n,nθ,nθM,k,nz,Dist))")

nsignal(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = n
ntheta(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nθ
nmarginalized(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nθM
nlatent(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = k
nembedding(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nz

function θ_linear_xform(::CVAE, θ, bd1, bd2)
    slope, bias = Zygote.@ignore begin
        bounds = bd1[1:size(θ,1)] .=> bd2[1:size(θ,1)]
        slope, bias = unzip(linear_xform_slope_and_bias.(bounds))
        arr_similar(θ, slope), arr_similar(θ, bias)
    end
    return slope .* θ .+ bias
end
θ_linear_normalize(cvae::CVAE, θ) = θ_linear_xform(cvae, θ, cvae.θbd, cvae.θ̄bd)
θ̄_linear_unnormalize(cvae::CVAE, θ) = θ_linear_xform(cvae, θ, cvae.θ̄bd, cvae.θbd)

# Layer which transforms matrix of [μ′; logσ′] ∈ [ℝ^nz; ℝ^nz] to bounded intervals [μ; logσ] ∈ [𝒟μ^nz; 𝒟logσ^nz]:
#      μ bounded: prevent CVAE from "memorizing" inputs via mean latent embedding vectors which are far from zero
#   logσ bounded: similarly, prevent CVAE from "memorizing" inputs via latent embedding vectors which are nearly constant, i.e. have zero variance
CVAELatentTransform(nz, 𝒟μ = (-2,2), 𝒟logσ = (-4,0)) = Flux.Chain(
    Base.BroadcastFunction(tanh),
    CatScale([(-1,1) => 𝒟μ, (-1,1) => 𝒟logσ], [nz, nz]),
)

####
#### CVAE helpers
####

@inline split_at(x::AbstractVecOrMat, n::Int) = n == size(x,1) ? (x, zeros_similar(x, 0, size(x)[2:end]...)) : (x[1:n, ..], x[n+1:end, ..])
split_theta_latent(cvae::CVAE, x::AbstractVecOrMat) = split_at(x, ntheta(cvae))
split_marginal_latent(cvae::CVAE, x::AbstractVecOrMat) = split_at(x, nmarginalized(cvae))

function split_marginal_latent_pairs(cvae::CVAE, x::AbstractVecOrMat)
    μ1θ_μ1Z, μ2θ_μ2Z = split_dim1(x) # x = [μ1θ; μ1Z; μ2θ; μ1Z]
    μ1θ, μ1Z = split_marginal_latent(cvae, μ1θ_μ1Z) # size(μ1θ,1) = nθM, size(μ1Z,1) = nlatent
    μ2θ, μ2Z = split_marginal_latent(cvae, μ2θ_μ2Z) # size(μ2θ,1) = nθM, size(μ2Z,1) = nlatent
    return (μ1θ, μ2θ, μ1Z, μ2Z)
end

####
#### CVAE methods
####

function pad_signal(Y::AbstractVecOrMat, n)
    if size(Y,1) < n
        pad = zeros_similar(Y, n - size(Y,1), size(Y)[2:end]...)
        vcat(Y, pad)
    elseif size(Y,1) > n
        Y[1:n, ..]
    else
        Y
    end
end
pad_signal(::CVAE{n}, Y) where {n} = pad_signal(Y, n)

function signal_mask(Y::AbstractVecOrMat; minkept::Int = firstindex(Y,1), maxkept::Int = lastindex(Y,1))
    # Create mask which randomly zeroes the tails of the columns of Y. That is, for each column `j` the mask satisfies
    # `m[i <= nⱼ, j] = 1` and `m[i > nⱼ, j] = 0`, with `minkept <= nⱼ <= maxkept` chosen randomly per column `j`
    Irows   = arr_similar(Y, collect(1:size(Y,1)))
    Icutoff = arr_similar(Y, collect(rand(minkept:maxkept, 1, size(Y)[2:end]...)))
    mask    = arr_similar(Y, Irows .<= Icutoff)
    return mask
end

struct CVAETrainingState{C <: CVAE, A}
    cvae::C
    Y::A
    θ̄::A
    Z::A
    μr0::A
    logσr::A
    μq0::A
    logσq::A
end

function CVAETrainingState(cvae::CVAE, Y, θ, Z = zeros_similar(θ, 0, size(θ,2)))
    Ypad = pad_signal(cvae, Y)
    θ̄ = θ_linear_normalize(cvae, θ)
    μr = cvae.E1(Ypad)
    μq = cvae.E2(Ypad, θ̄, Z)
    μr0, logσr = split_dim1(μr)
    μq0, logσq = split_dim1(μq)
    return CVAETrainingState(cvae, Ypad, θ̄, Z, μr0, logσr, μq0, logσq)
end
signal(state::CVAETrainingState) = state.Y

struct CVAEInferenceState{C <: CVAE, A}
    cvae::C
    Y::A
    μr0::A
    logσr::A
end

function CVAEInferenceState(cvae::CVAE, Y)
    Ypad = pad_signal(cvae, Y)
    μr = cvae.E1(Ypad)
    μr0, logσr = split_dim1(μr)
    return CVAEInferenceState(cvae, Ypad, μr0, logσr)
end
signal(state::CVAEInferenceState) = state.Y

function KLDivergence(state::CVAETrainingState)
    @unpack μq0, logσq, μr0, logσr = state
    KLDivGaussian(μq0, logσq, μr0, logσr)
end

function EvidenceLowerBound(state::CVAETrainingState{C}; marginalize_Z::Bool = nlatent(state.cvae) == 0) where {C <: CVAEPosteriorDist{Gaussian}}
    @unpack cvae, Y, θ̄, Z, μq0, logσq = state
    nθM = nmarginalized(cvae)
    zq = sample_mv_normal(μq0, exp.(logσq))
    μx0, logσx = split_dim1(cvae.D(Y, zq))
    ELBO = marginalize_Z ?
        NegLogLGaussian(θ̄[1:nθM, ..], μx0[1:nθM, ..], logσx[1:nθM, ..]) :
        NegLogLGaussian(vcat(θ̄[1:nθM, ..], Z), μx0, logσx)
end

function EvidenceLowerBound(state::CVAETrainingState{C}; marginalize_Z::Bool = nlatent(state.cvae) == 0) where {C <: CVAEPosteriorDist{TruncatedGaussian}}
    @unpack cvae, Y, θ̄, Z, μq0, logσq = state
    nθM = nmarginalized(cvae)
    zq = sample_mv_normal(μq0, exp.(logσq))
    μx = cvae.D(Y, zq) # μx = D(Y, zq) = [μθ̄M; μZ; logσθ̄M; logσZ]
    θ̄Mlo = Zygote.@ignore arr_similar(μx, (x->x[1]).(cvae.θ̄bd[1:nθM, ..]))
    θ̄Mhi = Zygote.@ignore arr_similar(μx, (x->x[2]).(cvae.θ̄bd[1:nθM, ..]))
    σ⁻¹μθ̄M, logσθ̄M, μZ, logσZ = split_marginal_latent_pairs(cvae, μx)
    μθ̄M  = clamp.(θ̄Mlo .+ (θ̄Mhi .- θ̄Mlo) .* Flux.σ.(σ⁻¹μθ̄M), θ̄Mlo, θ̄Mhi) # transform from unbounded σ⁻¹μθ̄M ∈ ℝ^nθ to bounded interval [θ̄Mlo, θ̄Mhi]^nθ
    ELBO_θ = NegLogLTruncatedGaussian(θ̄[1:nθM, ..], μθ̄M, logσθ̄M, θ̄Mlo, θ̄Mhi)
    if marginalize_Z
        ELBO = ELBO_θ
    else
        ELBO = ELBO_θ + NegLogLGaussian(Z, μZ, logσZ)
    end
end

function EvidenceLowerBound(state::CVAETrainingState{C}; marginalize_Z::Bool = nlatent(state.cvae) == 0) where {C <: CVAEPosteriorDist{Kumaraswamy}}
    @unpack cvae, Y, θ̄, Z, μq0, logσq = state
    nθM = nmarginalized(cvae)
    zq = sample_mv_normal(μq0, exp.(logσq))
    μx = cvae.D(Y, zq) # μx = D(Y, zq) = [αθ; μZ; βθ; logσZ]
    αθ, βθ, μZ, logσZ = split_marginal_latent_pairs(cvae, μx)
    ELBO_θ = NegLogLKumaraswamy(θ̄[1:nθM, ..], αθ, βθ)
    if marginalize_Z
        ELBO = ELBO_θ
    else
        ELBO = ELBO_θ + NegLogLGaussian(Z, μZ, logσZ)
    end
end

function KL_and_ELBO(state::CVAETrainingState; marginalize_Z::Bool = nlatent(state.cvae) == 0)
    KLDiv = KLDivergence(state)
    ELBO = EvidenceLowerBound(state; marginalize_Z)
    return (; KLDiv, ELBO)
end

KL_and_ELBO(cvae::CVAE, Y, θ, Z = zeros_similar(θ, 0, size(θ,2)); marginalize_Z::Bool = nlatent(cvae) == 0) = KL_and_ELBO(CVAETrainingState(cvae, Y, θ, Z); marginalize_Z)

sampleθZposterior(cvae::CVAE, Y; kwargs...) = sampleθZposterior(CVAEInferenceState(cvae, Y); kwargs...)

function sampleθZposterior(state::CVAEInferenceState{C}; mode = false) where {C <: CVAEPosteriorDist{Gaussian}}
    #TODO: `mode` is probably not strictly the correct term, but in practice it should be something akin to the distribution mode since `μr0` is the most likely value for `zr` and `μx0` is the most likely value for `x` **conditional on `zr`**; likely there are counterexamples to this simple reasoning, though...
    @unpack cvae, Y, μr0, logσr = state
    zr = mode ? μr0 : sample_mv_normal(μr0, exp.(logσr))
    μx = cvae.D(Y, zr)
    μx0, logσx = split_dim1(μx)
    x = mode ? μx0 : sample_mv_normal(μx0, exp.(logσx))
    θ̄M, Z = split_marginal_latent(cvae, x)
    θM = θ̄_linear_unnormalize(cvae, θ̄M)
    return θM, Z
end

function sampleθZposterior(state::CVAEInferenceState{C}; mode = false) where {C <: CVAEPosteriorDist{TruncatedGaussian}}
    #TODO: `mode` is probably not strictly the correct term, but in practice it should be something akin to the distribution mode since `μr0` is the most likely value for `zr` and `μx0` is the most likely value for `x` **conditional on `zr`**; likely there are counterexamples to this simple reasoning, though...
    @unpack cvae, Y, μr0, logσr = state
    nθM = nmarginalized(cvae)
    zr = mode ? μr0 : sample_mv_normal(μr0, exp.(logσr))
    μx = cvae.D(Y, zr)
    θ̄Mlo = Zygote.@ignore arr_similar(μx, (x->x[1]).(cvae.θ̄bd[1:nθM, ..]))
    θ̄Mhi = Zygote.@ignore arr_similar(μx, (x->x[2]).(cvae.θ̄bd[1:nθM, ..]))
    σ⁻¹μθ̄M, logσθ̄M, μZ, logσZ = split_marginal_latent_pairs(cvae, μx)
    μθ̄M  = clamp.(θ̄Mlo .+ (θ̄Mhi .- θ̄Mlo) .* Flux.σ.(σ⁻¹μθ̄M), θ̄Mlo, θ̄Mhi) # transform from unbounded σ⁻¹μθ̄M ∈ ℝ^nθ to bounded interval [θ̄Mlo, θ̄Mhi]^nθ
    θ̄M = mode ? μθ̄M : sample_trunc_mv_normal(μθ̄M, exp.(logσθ̄M), θ̄Mlo, θ̄Mhi)
    Z = mode || nlatent(state.cvae) == 0 ? μZ : sample_mv_normal(μZ, exp.(logσZ))
    θM = θ̄_linear_unnormalize(cvae, θ̄M)
    return θM, Z
end

function sampleθZposterior(state::CVAEInferenceState{C}; mode = false) where {C <: CVAEPosteriorDist{Kumaraswamy}}
    #TODO: `mode` is probably not strictly the correct term, but in practice it should be something akin to the distribution mode since `μr0` is the most likely value for `zr` and `μx0` is the most likely value for `x` **conditional on `zr`**; likely there are counterexamples to this simple reasoning, though...
    @unpack cvae, Y, μr0, logσr = state
    zr = mode ? μr0 : sample_mv_normal(μr0, exp.(logσr))
    μx = cvae.D(Y, zr)
    αθ, βθ, μZ, logσZ = split_marginal_latent_pairs(cvae, μx)
    θ̄M = mode ? mode_kumaraswamy(αθ, βθ) : sample_kumaraswamy(αθ, βθ)
    Z = mode || nlatent(state.cvae) == 0 ? μZ : sample_mv_normal(μZ, exp.(logσZ))
    θM = θ̄_linear_unnormalize(cvae, θ̄M)
    return θM, Z
end

function θZposterior_sampler(cvae::CVAE, Y; kwargs...)
    state = CVAEInferenceState(cvae, Y) # constant over posterior samples
    θZposterior_sampler_inner() = sampleθZposterior(state; kwargs...)
    return θZposterior_sampler_inner
end

####
#### Deep prior
####

"""
Deep prior for learning distribution of some ϕ, wrapping (possibly parameterized) functions `prior` and `noisesource`

    noisesource: ∅ -> R^kϕ
    prior : R^kϕ -> R^nϕ

which generates samples ϕ ~ prior(ϕ) from kϕ-dimensional samples from noisesource.
"""
struct DeepPrior{T,kϕ,FP,FN}
    prior :: FP
    noisesource :: FN
end
DeepPrior{T,kϕ}(prior::FP, noisesource::FN) where {T,kϕ,FP,FN} = DeepPrior{T,kϕ,FP,FN}(prior, noisesource)

Flux.functor(::Type{<:DeepPrior{T,kϕ}}, p) where {T,kϕ} = (prior = p.prior, noisesource = p.noisesource,), fs -> DeepPrior{T,kϕ}(fs...)
Base.show(io::IO, ::DeepPrior{T,kϕ}) where {T,kϕ} = print(io, "DeepPrior$((;T,kϕ))")

(p::DeepPrior{T,kϕ})(x::A) where {T, kϕ, A <: AbstractVecOrMat{T}} = p.prior(p.noisesource(A, kϕ, size(x,2))) # sample from distribution

StatsBase.sample(p::DeepPrior{T}, n::Int) where {T} = StatsBase.sample(p, CuMatrix{T}, n) # default to sampling θ on the gpu
StatsBase.sample(p::DeepPrior, Y::AbstractVecOrMat, n::Int = size(Y,2)) = StatsBase.sample(p, typeof(Y), n) # θ type is similar to Y type
StatsBase.sample(p::DeepPrior{T,kϕ}, ::Type{A}, n::Int) where {T, kϕ, A <: AbstractVecOrMat{T}} = p.prior(p.noisesource(A, kϕ, n)) # sample from distribution

const MaybeDeepPrior = Union{Nothing, <:DeepPrior}

#### CVAE + PhysicsModel + DeepPrior + AbstractMetaDataSignal methods

function sampleθZ(phys::PhysicsModel, cvae::CVAE, θprior::MaybeDeepPrior, Zprior::MaybeDeepPrior, Ymeta::AbstractMetaDataSignal; posterior_θ = true, posterior_Z = true, posterior_mode = false)
    if posterior_θ || posterior_Z
        return sampleθZ(phys, cvae, θprior, Zprior, Ymeta, CVAEInferenceState(cvae, signal(Ymeta)); posterior_θ, posterior_Z, posterior_mode)
    else
        θ = sample(θprior, signal(Ymeta))
        Z = sample(Zprior, signal(Ymeta))
        return θ, Z
    end
end
sampleθZ(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleθZ(phys, cvae, nothing, nothing, Ymeta; kwargs..., posterior_θ = true, posterior_Z = true) # no prior passed -> posterior_θ = posterior_Z = true

function sampleθZ(phys::PhysicsModel, cvae::CVAE, θprior::MaybeDeepPrior, Zprior::MaybeDeepPrior, Ymeta::AbstractMetaDataSignal, state::CVAEInferenceState; posterior_θ = true, posterior_Z = true, posterior_mode = false)
    if posterior_θ || posterior_Z
        θ̂M, Ẑ = sampleθZposterior(state; mode = posterior_mode)
        Z = if posterior_Z
            Ẑ
        else
            sample(Zprior, signal(Ymeta))
        end
        θ = if posterior_θ
            θMlo = arr_similar(θ̂M, θmarginalized(phys, θlower(phys)))
            θMhi = arr_similar(θ̂M, θmarginalized(phys, θupper(phys)))
            vcat(clamp.(θ̂M, θMlo, θMhi), Zygote.@ignore(θnuissance(phys, Ymeta))) #TODO Zygote.@ignore necessary?
        else
            sample(θprior, signal(Ymeta))
        end
    else
        θ = sample(θprior, signal(Ymeta))
        Z = sample(Zprior, signal(Ymeta))
    end
    return θ, Z
end
sampleθZ(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal, state::CVAEInferenceState; kwargs...) = sampleθZ(phys, cvae, nothing, nothing, Ymeta, state; kwargs..., posterior_θ = true, posterior_Z = true) # no prior passed -> posterior_θ = posterior_Z = true

function θZ_sampler(phys::PhysicsModel, cvae::CVAE, θprior::MaybeDeepPrior, Zprior::MaybeDeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...)
    state = CVAEInferenceState(cvae, signal(Ymeta)) # constant over posterior samples
    θZ_sampler_inner() = sampleθZ(phys, cvae, θprior, Zprior, Ymeta, state; kwargs...)
    return θZ_sampler_inner
end
θZ_sampler(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...) = θZ_sampler(phys, cvae, nothing, nothing, Ymeta; kwargs..., posterior_θ = true, posterior_Z = true) # no prior passed -> posterior_θ = posterior_Z = true

function sampleXθZ(phys::PhysicsModel, cvae::CVAE, θprior::MaybeDeepPrior, Zprior::MaybeDeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...)
    #TODO: can't differentiate through @timeit "sampleθZ"
    #TODO: can't differentiate through @timeit "signal_model"
    θ, Z = sampleθZ(phys, cvae, θprior, Zprior, Ymeta; kwargs...)
    X = signal_model(phys, Ymeta, θ)
    (size(X,1) > nsignal(Ymeta)) && (X = X[1:nsignal(Ymeta), ..])
    return X, θ, Z
end
sampleXθZ(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleXθZ(phys, cvae, nothing, nothing, Ymeta; kwargs..., posterior_θ = true, posterior_Z = true) # no prior passed -> posterior_θ = posterior_Z = true

sampleX(phys::PhysicsModel, cvae::CVAE, θprior::MaybeDeepPrior, Zprior::MaybeDeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleXθZ(phys, cvae, θprior, Zprior, Ymeta; kwargs...)[1]
sampleX(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleX(phys, cvae, nothing, nothing, Ymeta; kwargs..., posterior_θ = true, posterior_Z = true) # no prior passed -> posterior_θ = posterior_Z = true

function posterior_state(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; accum_loss = ℓ -> sum(ℓ; dims = 1), kwargs...)
    θ, Z = sampleθZ(phys, cvae, Ymeta; posterior_θ = true, posterior_Z = true, posterior_mode = false, kwargs...)
    X = signal_model(phys, Ymeta, θ)
    X = clamp_dim1(signal(Ymeta), X)
    X̂ = add_noise_instance(phys, X, θ)
    ℓ = loglikelihood(phys, signal(Ymeta), X, θ; accum_loss) #TODO make a loglikelihood
    ϵ = noiselevel(phys, θ)
    return (; Y = signal(Ymeta), X̂, θ, Z, X, ϵ, ν = X, δ = zeros_similar(X, 1, size(X,2)), ℓ)
end

####
#### Rician posterior state
####

sampleX̂(rice::RicianCorrector, X, Z = nothing, ninstances = nothing) = sample_rician_state(rice, X, Z, ninstances).X̂

function sampleX̂θZ(phys::PhysicsModel, rice::RicianCorrector, cvae::CVAE, θprior::MaybeDeepPrior, Zprior::MaybeDeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...)
    #TODO: can't differentiate through @timeit "sampleXθZ"
    #TODO: can't differentiate through @timeit "sampleX̂"
    X, θ, Z = sampleXθZ(phys, cvae, θprior, Zprior, Ymeta; kwargs...)
    X̂ = sampleX̂(rice, X, Z)
    return X̂, θ, Z
end
sampleX̂θZ(phys::PhysicsModel, rice::RicianCorrector, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleX̂θZ(phys, rice, cvae, nothing, nothing, Ymeta; kwargs..., posterior_θ = true, posterior_Z = true) # no prior passed -> posterior_θ = posterior_Z = true

sampleX̂(phys::PhysicsModel, rice::RicianCorrector, cvae::CVAE, θprior::MaybeDeepPrior, Zprior::MaybeDeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleX̂θZ(phys, rice, cvae, θprior, Zprior, Ymeta; kwargs...)[1]
sampleX̂(phys::PhysicsModel, rice::RicianCorrector, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleX̂(phys, rice, cvae, nothing, nothing, Ymeta; kwargs..., posterior_θ = true, posterior_Z = true) # no prior passed -> posterior_θ = posterior_Z = true

function NegLogLikelihood(::PhysicsModel, rice::RicianCorrector, Y::AbstractVecOrMat, μ0, σ)
    if typeof(rice) <: NormalizedRicianCorrector && (rice.normalizer !== nothing)
        # Approximate the normalization factor as the normalization factor of the mean signal.
        # For gaussian noise mean signal = μ0, but for rician noise mean signal ~ sqrt(μ0^2 + σ^2), at least when μ0 >> σ
        s = inv.(rice.normalizer(mean_rician.(μ0, σ)))
        neglogL_rician.(Y, s .* μ0, log.(s .* σ)) # Rician negative log likelihood
    else
        neglogL_rician.(Y, μ0, log.(σ)) # Rician negative log likelihood
    end
end

function NegLogLikelihood(::EPGModel, rice::RicianCorrector, Y::AbstractVecOrMat, μ0, σ)
    # Likelihood is "maximimally generous" w.r.t. normalization factor, i.e. we perform MLE to find optimal scaling factor
    logs = Zygote.@ignore begin
        _, results = mle_biexp_epg_noise_only(μ0, Y, log.(σ); freeze_logϵ = true, freeze_logs = false, verbose = false)
        arr_similar(Y, permutedims(results.logscale))
    end
    neglogL_rician.(Y, exp.(logs) .* μ0, logs .+ log.(σ)) # Rician negative log likelihood
end

function posterior_state(phys::PhysicsModel, rice::RicianCorrector, Y::AbstractVecOrMat, θ::AbstractVecOrMat, Z::AbstractVecOrMat; accum_loss = ℓ -> sum(ℓ; dims = 1))
    X = signal_model(phys, θ)
    @unpack δ, ϵ, ν = rician_state(rice, X, Z)
    X, δ, ϵ, ν = clamp_dim1(Y, (X, δ, ϵ, ν))
    ℓ = NegLogLikelihood(phys, rice, Y, ν, ϵ)
    (accum_loss !== nothing) && (ℓ = accum_loss(ℓ))
    return (; Y, θ, Z, X, δ, ϵ, ν, ℓ)
end

function posterior_state(
        phys::PhysicsModel,
        rice::RicianCorrector,
        cvae::CVAE,
        Ymeta::AbstractMetaDataSignal{T};
        miniter = 5,
        maxiter = 100,
        alpha = 0.01,
        mode = :maxlikelihood,
        verbose = false
    ) where {T}

    (mode === :mode) && (miniter = maxiter = 1)
    θZ_sampler_instance = θZ_sampler(phys, cvae, Ymeta; posterior_mode = mode === :mode)
    new_posterior_state(θnew, Znew) = posterior_state(phys, rice, signal(Ymeta), θnew, Znew; accum_loss = ℓ -> sum(ℓ; dims = 1))

    function update(last_state, i)
        θnew, Znew = θZ_sampler_instance()
        θlast = (last_state === nothing) ? nothing : last_state.θ
        Zlast = (last_state === nothing) ? nothing : last_state.Z

        if mode === :mean
            θnew = (last_state === nothing) ? θnew : T(1/i) .* θnew .+ T(1-1/i) .* θlast
            Znew = (last_state === nothing) ? Znew : T(1/i) .* Znew .+ T(1-1/i) .* Zlast
            new_state = new_posterior_state(θnew, Znew)
        elseif mode === :mode
            new_state = new_posterior_state(θnew, Znew)
        elseif mode === :maxlikelihood
            new_state = new_posterior_state(θnew, Znew)
            if (last_state !== nothing)
                mask = new_state.ℓ .< last_state.ℓ
                new_state = map(new_state, last_state) do new, last
                    new .= ifelse.(mask, new, last)
                end
            end
        else
            error("Unknown mode: $mode")
        end

        # Check for convergence
        p = (last_state === nothing) ? nothing :
            HypothesisTests.pvalue(
                HypothesisTests.UnequalVarianceTTest(
                    map(x -> x |> cpu |> vec |> Vector{Float64}, (new_state.ℓ, last_state.ℓ))...
                )
            )

        return new_state, p
    end

    state, _ = update(nothing, 1)
    verbose && @info 1, mean_and_std(state.ℓ)
    for i in 2:maxiter
        state, p = update(state, i)
        verbose && @info i, mean_and_std(state.ℓ), p
        (i >= miniter) && (p > 1 - alpha) && break
    end

    return state
end

#= TODO: update to return both θ and Z means + stddevs?
function (cvae::CVAE)(Y; nsamples::Int = 1, stddev::Bool = false)
    @assert nsamples ≥ ifelse(stddev, 2, 1)
    smooth(a, b, γ) = a + γ * (b - a)
    θZsampler = θZposterior_sampler(cvae, Y)
    μx = θZsampler()
    μx_last, σx2 = zero(μx), zero(μx)
    for i in 2:nsamples
        x = θZsampler()
        μx_last .= μx
        μx .= smooth.(μx, x, 1//i)
        σx2 .= smooth.(σx2, (x .- μx) .* (x .- μx_last), 1//i)
    end
    return stddev ? vcat(μx, sqrt.(σx2)) : μx
end
=#
