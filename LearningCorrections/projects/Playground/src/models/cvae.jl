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

@inline _split_at(x::AbstractMatrix, n::Int) = n == size(x,1) ? (x, similar(x, 0, size(x,2))) : (x[1:n,:], x[n+1:end,:])
split_theta_latent(::CVAE{n,nθ,nθM,k,nz}, x::AbstractMatrix) where {n,nθ,nθM,k,nz} = _split_at(x, nθ)
split_marginal_latent(::CVAE{n,nθ,nθM,k,nz}, x::AbstractMatrix) where {n,nθ,nθM,k,nz} = _split_at(x, nθM)

####
#### CVAE methods
####

function pad_signal(Y::AbstractArray, n)
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

function signal_mask(Y::AbstractMatrix; minkept::Int = firstindex(Y,1), maxkept::Int = lastindex(Y,1))
    # Create mask which randomly zeroes the tails of the columns of Y. That is, for each column `j` the mask satisfies
    # `m[i <= nⱼ, j] = 1` and `m[i > nⱼ, j] = 0`, with `minkept <= nⱼ <= maxkept` chosen randomly per column `j`
    Irows   = arr_similar(Y, collect(1:size(Y,1)))
    Icutoff = arr_similar(Y, collect(rand(minkept:maxkept, 1, size(Y,2))))
    mask    = arr_similar(Y, Irows .<= Icutoff)
    return mask
end

function mv_normal_parameters(cvae::CVAE, Y)
    Ypad = pad_signal(cvae, Y)
    μr = cvae.E1(Ypad)
    μr0, σr = split_mean_softplus_std(μr)
    return (; μr0, σr)
end

function mv_normal_parameters(cvae::CVAE, Y, θ, Z)
    Ypad = pad_signal(cvae, Y)
    μr0, σr = split_mean_softplus_std(cvae.E1(Ypad))
    μq0, σq = split_mean_softplus_std(cvae.E2(Ypad,θ,Z))
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_softplus_std(cvae.D(Ypad,zq))
    return (; μr0, σr, μq0, σq, μx0, σx)
end

function KL_and_ELBO(cvae::CVAE{n,nθ,nθM,k,nz}, Y, θ, Z; marginalize_Z::Bool) where {n,nθ,nθM,k,nz}
    @unpack μr0, σr, μq0, σq, μx0, σx = mv_normal_parameters(cvae, pad_signal(cvae, Y), θ, Z)
    KLDiv = KLDivergence(μq0, σq, μr0, σr)
    ELBO = marginalize_Z ?
        EvidenceLowerBound(θ[1:nθM,..], μx0[1:nθM,..], σx[1:nθM,..]) :
        EvidenceLowerBound(vcat(θ[1:nθM,..], Z), μx0, σx)
    return (; KLDiv, ELBO)
end

sampleθZposterior(cvae::CVAE, Y) = (Ypad = pad_signal(cvae, Y); sampleθZposterior(cvae, Ypad, mv_normal_parameters(cvae, Ypad)...))

function sampleθZposterior(cvae::CVAE, Y, μr0, σr)
    Ypad = pad_signal(cvae, Y)
    zr = sample_mv_normal(μr0, σr)
    μx = cvae.D(Ypad,zr)
    μx0, σx = split_mean_softplus_std(μx)
    x = sample_mv_normal(μx0, σx)
    θM, Z = split_marginal_latent(cvae, x)
    return θM, Z
end

function θZposterior_sampler(cvae::CVAE, Y)
    Ypad = pad_signal(cvae, Y)
    @unpack μr0, σr = mv_normal_parameters(cvae, Ypad) # constant over posterior samples
    θZposterior_sampler_inner() = sampleθZposterior(cvae, Ypad, μr0, σr)
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

(p::DeepPrior{T,kϕ})(x::A) where {T, kϕ, A <: AbstractArray{T}} = p.prior(p.noisesource(A, kϕ, size(x,2))) # sample from distribution

StatsBase.sample(p::DeepPrior{T}, n::Int) where {T} = StatsBase.sample(p, CuMatrix{T}, n) # default to sampling θ on the gpu
StatsBase.sample(p::DeepPrior, Y::AbstractArray, n::Int = size(Y,2)) = StatsBase.sample(p, typeof(Y), n) # θ type is similar to Y type
StatsBase.sample(p::DeepPrior{T,kϕ}, ::Type{A}, n::Int) where {T, kϕ, A <: AbstractArray{T}} = p.prior(p.noisesource(A, kϕ, n)) # sample from distribution

#### CVAE + DeepPrior + AbstractMetaDataSignal methods

function sampleθZ(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; posterior_θ = true, posterior_Z = true)
    if posterior_θ || posterior_Z
        return sampleθZ(phys, cvae, θprior, Zprior, Ymeta, mv_normal_parameters(cvae, signal(Ymeta))...; posterior_θ, posterior_Z)
    else
        θ = sample(θprior, signal(Ymeta))
        Z = sample(Zprior, signal(Ymeta))
        return θ, Z
    end
end

function sampleθZ(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal, μr0, σr; posterior_θ = true, posterior_Z = true)
    if posterior_θ || posterior_Z
        θMhat, Zhat = sampleθZposterior(cvae, signal(Ymeta), μr0, σr)
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

function θZ_sampler(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; posterior_θ = true, posterior_Z = true)
    @unpack μr0, σr = mv_normal_parameters(cvae, signal(Ymeta)) # constant over posterior samples
    θZ_sampler_inner() = sampleθZ(phys, cvae, θprior, Zprior, Ymeta, μr0, σr; posterior_θ, posterior_Z)
    return θZ_sampler_inner
end

function sampleXθZ(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; posterior_θ = true, posterior_Z = true)
    #TODO: can't differentiate through @timeit "sampleθZ"
    #TODO: can't differentiate through @timeit "signal_model"
    θ, Z = sampleθZ(phys, cvae, θprior, Zprior, Ymeta; posterior_θ, posterior_Z)
    X = signal_model(phys, θ)
    (size(X,1) > nsignal(Ymeta)) && (X = X[1:nsignal(Ymeta),..])
    return X, θ, Z
end

sampleX(phys::PhysicsModel, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; posterior_θ = true, posterior_Z = true) = sampleXθZ(phys, cvae, θprior, Zprior, Ymeta; posterior_θ, posterior_Z)[1]

####
#### Rician posterior state
####

sampleX̂(rice::RicianCorrector, X, Z = nothing, ninstances = nothing) = sample_rician_state(rice, X, Z, ninstances).X̂

function sampleX̂θZ(phys::PhysicsModel, rice::RicianCorrector, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; posterior_θ = true, posterior_Z = true)
    #TODO: can't differentiate through @timeit "sampleXθZ"
    #TODO: can't differentiate through @timeit "sampleX̂"
    X, θ, Z = sampleXθZ(phys, cvae, θprior, Zprior, Ymeta; posterior_θ, posterior_Z)
    X̂ = sampleX̂(rice, X, Z)
    return X̂, θ, Z
end

sampleX̂(phys::PhysicsModel, rice::RicianCorrector, cvae::CVAE, θprior::DeepPrior, Zprior::DeepPrior, Ymeta::AbstractMetaDataSignal; posterior_θ = true, posterior_Z = true) = sampleX̂θZ(phys, rice, cvae, θprior, Zprior, Ymeta; posterior_θ, posterior_Z)[1]

function NegLogLikelihood(rice::RicianCorrector, Y::AbstractVecOrMat, μ0, σ)
    if typeof(rice) <: NormalizedRicianCorrector && (rice.normalizer !== nothing)
        Σμ = rice.normalizer(_rician_mean_cuda.(μ0, σ))
        μ0, σ = (μ0 ./ Σμ), (σ ./ Σμ)
    end
    -sum(_rician_logpdf_cuda.(Y, μ0, σ); dims = 1) # Rician negative log likelihood
end

function posterior_state(phys::PhysicsModel, rice::RicianCorrector, Y::AbstractVecOrMat, θ::AbstractMatrix, Z::AbstractMatrix)
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
