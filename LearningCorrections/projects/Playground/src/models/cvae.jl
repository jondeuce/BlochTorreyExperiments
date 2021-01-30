"""
Conditional variational autoencoder.

Architecture inspired by:
    "Bayesian parameter estimation using conditional variational autoencoders for gravitational-wave astronomy"
    https://arxiv.org/abs/1802.08797
"""
struct CVAE{n,nθ,nθM,k,nz,E1,E2,D}
    E1 :: E1
    E2 :: E2
    D  :: D
    CVAE{n,nθ,nθM,k,nz}(enc1::E1, enc2::E2, dec::D) where {n,nθ,nθM,k,nz,E1,E2,D} = new{n,nθ,nθM,k,nz,E1,E2,D}(enc1, enc2, dec)
end
Flux.@functor CVAE
Base.show(io::IO, ::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = print(io, "CVAE$((;n,nθ,nθM,k,nz))")

nsignal(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = n
ntheta(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nθ
nmarginalized(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nθM
nlatent(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = k
nembedding(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nz

####
#### CVAE helpers
####

@inline _split_at(x::AbstractVecOrMat, n::Int) = n == size(x,1) ? (x, similar(x, 0, size(x)[2:end]...)) : (x[1:n,..], x[n+1:end,..])
split_theta_latent(::CVAE{n,nθ,nθM,k,nz}, x::AbstractVecOrMat) where {n,nθ,nθM,k,nz} = _split_at(x, nθ)
split_marginal_latent(::CVAE{n,nθ,nθM,k,nz}, x::AbstractVecOrMat) where {n,nθ,nθM,k,nz} = _split_at(x, nθM)

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

struct CVAETrainingState{n,nθ,nθM,k,nz,A}; Y::A; θ::A; Z::A; μr0::A; σr::A; μq0::A; σq::A; μx0::A; σx::A; end

function CVAETrainingState(cvae::CVAE{n,nθ,nθM,k,nz}, Y, θ, Z) where {n,nθ,nθM,k,nz}
    Ypad = pad_signal(cvae, Y)
    μr0, σr = split_mean_softplus_std(cvae.E1(Ypad))
    μq0, σq = split_mean_softplus_std(cvae.E2(Ypad,θ,Z))
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_softplus_std(cvae.D(Ypad,zq))
    return CVAETrainingState{n,nθ,nθM,k,nz,typeof(Ypad)}(Ypad, θ, Z, μr0, σr, μq0, σq, μx0, σx)
end
signal(state::CVAETrainingState) = state.Y

struct CVAEInferenceState{n,nθ,nθM,k,nz,A}; Y::A; μr0::A; σr::A; end

function CVAEInferenceState(cvae::CVAE{n,nθ,nθM,k,nz}, Y) where {n,nθ,nθM,k,nz}
    Ypad = pad_signal(cvae, Y)
    μr = cvae.E1(Ypad)
    μr0, σr = split_mean_softplus_std(μr)
    return CVAEInferenceState{n,nθ,nθM,k,nz,typeof(Ypad)}(Ypad, μr0, σr)
end
signal(state::CVAEInferenceState) = state.Y

function KL_and_ELBO(state::CVAETrainingState{n,nθ,nθM,k,nz}; marginalize_Z::Bool) where {n,nθ,nθM,k,nz}
    @unpack Y, θ, Z, μr0, σr, μq0, σq, μx0, σx = state
    KLDiv = KLDivergence(μq0, σq, μr0, σr)
    ELBO = marginalize_Z ?
        EvidenceLowerBound(θ[1:nθM,..], μx0[1:nθM,..], σx[1:nθM,..]) :
        EvidenceLowerBound(vcat(θ[1:nθM,..], Z), μx0, σx)
    return (; KLDiv, ELBO)
end
KL_and_ELBO(cvae::CVAE, Y, θ, Z; marginalize_Z::Bool) = KL_and_ELBO(CVAETrainingState(cvae, Y, θ, Z); marginalize_Z)

sampleθZposterior(cvae::CVAE, Y; kwargs...) = sampleθZposterior(cvae, CVAEInferenceState(cvae, Y); kwargs...)

function sampleθZposterior(cvae::CVAE, state::CVAEInferenceState; mode = false)
    #TODO: `mode` is probably not strictly the correct term, but in practice it should be something akin to the distribution mode since `μr0` is the most likely value for `zr` and `μx0` is the most likely value for `x` **conditional on `zr`**; likely there are counterexamples to this simple reasoning, though...
    @unpack Y, μr0, σr = state
    zr = mode ? μr0 : sample_mv_normal(μr0, σr)
    μx = cvae.D(Y,zr)
    μx0, σx = split_mean_softplus_std(μx)
    x = mode ? μx0 : sample_mv_normal(μx0, σx)
    θM, Z = split_marginal_latent(cvae, x)
    return θM, Z
end

function θZposterior_sampler(cvae::CVAE, Y; kwargs...)
    state = CVAEInferenceState(cvae, Y) # constant over posterior samples
    θZposterior_sampler_inner() = sampleθZposterior(cvae, state; kwargs...)
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
    DeepPrior{T,kϕ}(prior::FP, noisesource::FN) where {T,kϕ,FP,FN} = new{T,kϕ,FP,FN}(prior, noisesource)
end
Flux.@functor DeepPrior
Base.show(io::IO, ::DeepPrior{T,kϕ}) where {T,kϕ} = print(io, "DeepPrior$((;T,kϕ))")

(p::DeepPrior{T,kϕ})(x::A) where {T, kϕ, A <: AbstractVecOrMat{T}} = p.prior(p.noisesource(A, kϕ, size(x,2))) # sample from distribution

StatsBase.sample(p::DeepPrior{T}, n::Int) where {T} = StatsBase.sample(p, CuMatrix{T}, n) # default to sampling θ on the gpu
StatsBase.sample(p::DeepPrior, Y::AbstractVecOrMat, n::Int = size(Y,2)) = StatsBase.sample(p, typeof(Y), n) # θ type is similar to Y type
StatsBase.sample(p::DeepPrior{T,kϕ}, ::Type{A}, n::Int) where {T, kϕ, A <: AbstractVecOrMat{T}} = p.prior(p.noisesource(A, kϕ, n)) # sample from distribution

#### CVAE + DeepPrior + AbstractMetaDataSignal methods

function sampleθZ(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; posterior_θ = true, posterior_Z = true, posterior_mode = false)
    if posterior_θ || posterior_Z
        return sampleθZ(phys, cvae, θprior, Zprior, Ymeta, CVAEInferenceState(cvae, signal(Ymeta)); posterior_θ, posterior_Z, posterior_mode)
    else
        θ = sample(θprior, signal(Ymeta))
        Z = sample(Zprior, signal(Ymeta))
        return θ, Z
    end
end

function sampleθZ(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal, state::CVAEInferenceState; posterior_θ = true, posterior_Z = true, posterior_mode = false)
    if posterior_θ || posterior_Z
        θMhat, Zhat = sampleθZposterior(cvae, state; mode = posterior_mode)
        Z = posterior_Z ? Zhat : sample(Zprior, signal(Ymeta))
        θ = if posterior_θ
            θMlo = arr_similar(θMhat, θmarginalized(phys, θlower(phys)))
            θMhi = arr_similar(θMhat, θmarginalized(phys, θupper(phys)))
            vcat(clamp.(θMhat, θMlo, θMhi), θnuissance(phys, Ymeta))
        else
            sample(θprior, signal(Ymeta))
        end
    else
        θ = sample(θprior, signal(Ymeta))
        Z = sample(Zprior, signal(Ymeta))
    end
    return θ, Z
end

function θZ_sampler(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...)
    state = CVAEInferenceState(cvae, signal(Ymeta)) # constant over posterior samples
    θZ_sampler_inner() = sampleθZ(phys, cvae, θprior, Zprior, Ymeta, state; kwargs...)
    return θZ_sampler_inner
end

function sampleXθZ(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...)
    #TODO: can't differentiate through @timeit "sampleθZ"
    #TODO: can't differentiate through @timeit "signal_model"
    θ, Z = sampleθZ(phys, cvae, θprior, Zprior, Ymeta; kwargs...)
    X = signal_model(phys, θ)
    (size(X,1) > nsignal(Ymeta)) && (X = X[1:nsignal(Ymeta),..])
    return X, θ, Z
end

sampleX(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleXθZ(phys, cvae, θprior, Zprior, Ymeta; kwargs...)[1]

####
#### Rician posterior state
####

sampleX̂(rice::RicianCorrector, X, Z = nothing, ninstances = nothing) = sample_rician_state(rice, X, Z, ninstances).X̂

function sampleX̂θZ(phys::PhysicsModel, rice::RicianCorrector, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...)
    #TODO: can't differentiate through @timeit "sampleXθZ"
    #TODO: can't differentiate through @timeit "sampleX̂"
    X, θ, Z = sampleXθZ(phys, cvae, θprior, Zprior, Ymeta; kwargs...)
    X̂ = sampleX̂(rice, X, Z)
    return X̂, θ, Z
end

sampleX̂(phys::PhysicsModel, rice::RicianCorrector, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleX̂θZ(phys, rice, cvae, θprior, Zprior, Ymeta; kwargs...)[1]

function NegLogLikelihood(rice::RicianCorrector, Y::AbstractVecOrMat, μ0, σ)
    if typeof(rice) <: NormalizedRicianCorrector && (rice.normalizer !== nothing)
        Σμ = rice.normalizer(_rician_mean_cuda.(μ0, σ))
        μ0, σ = (μ0 ./ Σμ), (σ ./ Σμ)
    end
    -sum(_rician_logpdf_cuda.(Y, μ0, σ); dims = 1) # Rician negative log likelihood
end

function posterior_state(phys::PhysicsModel, rice::RicianCorrector, Y::AbstractVecOrMat, θ::AbstractVecOrMat, Z::AbstractVecOrMat)
    X = signal_model(phys, θ)
    @unpack δ, ϵ, ν = rician_state(rice, X, Z)
    X, δ, ϵ, ν = clamp_dim1(Y, (X, δ, ϵ, ν))
    ℓ = reshape(NegLogLikelihood(rice, Y, ν, ϵ), 1, :)
    return (; Y, θ, Z, X, δ, ϵ, ν, ℓ)
end

function posterior_state(
        phys::PhysicsModel,
        rice::RicianCorrector,
        cvae::CVAE,
        θprior::DeepPrior,
        Zprior::DeepPrior,
        Ymeta::AbstractMetaDataSignal{T};
        miniter = 5,
        maxiter = 100,
        alpha = 0.01,
        mode = :maxlikelihood,
        verbose = false
    ) where {T}

    θZ_sampler_instance = θZ_sampler(phys, cvae, θprior, Zprior, Ymeta; posterior_θ = true, posterior_Z = true)

    function update(last_state, i)
        θnew, Znew = θZ_sampler_instance()
        θlast = (last_state === nothing) ? nothing : last_state.θ
        Zlast = (last_state === nothing) ? nothing : last_state.Z

        if mode === :mean
            θnew = (last_state === nothing) ? θnew : T(1/i) .* θnew .+ T(1-1/i) .* θlast
            Znew = (last_state === nothing) ? Znew : T(1/i) .* Znew .+ T(1-1/i) .* Zlast
            new_state = posterior_state(phys, rice, signal(Ymeta), θnew, Znew)
        elseif mode === :maxlikelihood
            new_state = posterior_state(phys, rice, signal(Ymeta), θnew, Znew)
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
