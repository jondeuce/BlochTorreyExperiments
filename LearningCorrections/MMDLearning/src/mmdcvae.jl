function RESCNN(sz::Pair{Int,Int}, Nhid::Int, Dhid::Int, σhid = Flux.relu, σout = identity; skip = false)
    Flux.Chain(
        x::AbstractMatrix -> reshape(x, sz[1], 1, 1, size(x,2)),
        Flux.Conv((3,1), 1=>Dhid, identity; pad = Flux.SamePad()),
        mapreduce(vcat, 1:Nhid÷2) do _
            convlayers = [Flux.Conv((3,1), Dhid=>Dhid, σhid; pad = Flux.SamePad()) for _ in 1:2]
            skip ? [Flux.SkipConnection(Flux.Chain(convlayers...), +)] : convlayers
        end...,
        Flux.Conv((1,1), Dhid=>1, identity; pad = Flux.SamePad()),
        x::AbstractArray{<:Any,4} -> reshape(x, sz[1], size(x,4)),
        Flux.Dense(sz[1], sz[2], σout),
    )
end

function TransformerEncoder(; n = 48, psize = 16, head = 8, hdim = 256, nhidden = 2)
    t = Transformers.Stack(
        Transformers.@nntopo(
            X : # Input (n × b)
            X => X : # Reshape (1 × n × b)
            X => pe : # Positional embedding pe (psize × n)
            (X, pe) => E : # Add positional embedding (psize × n × b)
            E => H : # Transformer encoder (psize × n × b)
            H => H # Flatten output (psize*n × b)
        ),
        X -> reshape(X, 1, size(X)...),
        Transformers.Basic.PositionEmbedding(psize, n; trainable = true),
        (X, pe) -> X .+ pe,
        Flux.Chain([Transformers.Basic.Transformer(psize, head, hdim; future = true, act = Flux.relu, pdrop = 0.0) for i = 1:nhidden]...),
        Flux.flatten,
    )
    Flux.fmap(Flux.testmode!, t) # Force dropout layers inactive
end

"""
Conditional variational autoencoder.

Architecture inspired by:
    "Bayesian parameter estimation using conditional variational autoencoders for gravitational-wave astronomy"
    https://arxiv.org/abs/1802.08797
"""
struct CVAE{n,nθ,k,nz,E1,E2,D}
    E1 :: E1
    E2 :: E2
    D  :: D
    CVAE{n,nθ,k,nz}(enc1::E1, enc2::E2, dec::D) where {n,nθ,k,nz,E1,E2,D} = new{n,nθ,k,nz,E1,E2,D}(enc1, enc2, dec)
end
Flux.@functor CVAE
Base.show(io::IO, m::CVAE) = model_summary(io, Dict("E1" => m.E1, "E2" => m.E2, "D" => m.D))

nsignal(::CVAE{n,nθ,k,nz}) where {n,nθ,k,nz} = n
ntheta(::CVAE{n,nθ,k,nz}) where {n,nθ,k,nz} = nθ
nlatent(::CVAE{n,nθ,k,nz}) where {n,nθ,k,nz} = k
nembedding(::CVAE{n,nθ,k,nz}) where {n,nθ,k,nz} = nz

####
#### CVAE helpers
####

@inline function split_theta_latent(::CVAE{n,nθ,k,nz}, x::AbstractMatrix) where {n,nθ,k,nz}
    @assert size(x,1) == nθ + k
    θ, Z = if k == 0
        x, similar(x, 0, size(x,2))
    else
        x[1:nθ,:], x[nθ+1:end,:]
    end
    return θ, Z
end

####
#### CVAE methods
####

KLDivUnitNormal(μ, σ) = (sum(@. pow2(σ) + pow2(μ) - 2 * log(σ)) - length(μ)) / (2 * size(μ,2)) # KL-divergence between approximation posterior and N(0, 1) prior (Note: sum over dim=1, mean over dim=2)
KLDivergence(μq0, σq, μr0, σr) = (sum(@. pow2(σq / σr) + pow2((μr0 - μq0) / σr) - 2 * log(σq / σr)) - length(μq0)) / (2 * size(μq0,2)) # KL-divergence contribution to cross-entropy (Note: sum over dim=1, mean over dim=2)
EvidenceLowerBound(x, μx0, σx) = (sum(@. pow2((x - μx0) / σx) + 2 * log(σx)) + length(μx0) * log2π(eltype(μx0))) / (2 * size(μx0,2)) # Negative log-likelihood/ELBO contribution to cross-entropy (Note: sum over dim=1, mean over dim=2)

function apply_pad(::CVAE{n}, Y) where {n}
    if size(Y,1) < n
        pad = zeros_similar(Y, n - size(Y,1), size(Y)[2:end]...)
        return vcat(Y, pad)
    else
        return Y
    end
end

function mv_normal_parameters(cvae::CVAE, Y, θ, Z)
    Ypad = apply_pad(cvae, Y)
    μr0, σr = split_mean_softplus_std(cvae.E1(Ypad))
    μq0, σq = split_mean_softplus_std(cvae.E2(vcat(Ypad,θ,Z)))
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_softplus_std(cvae.D(vcat(Ypad,zq)))
    return (; μr0, σr, μq0, σq, μx0, σx)
end

function KL_and_ELBO(cvae::CVAE{n,nθ,k,nz}, Y, θ, Z; marginalize_Z::Bool) where {n,nθ,k,nz}
    @unpack μr0, σr, μq0, σq, μx0, σx = mv_normal_parameters(cvae, apply_pad(cvae, Y), θ, Z)
    KLDiv = KLDivergence(μq0, σq, μr0, σr)
    ELBO = marginalize_Z ?
        EvidenceLowerBound(θ, μx0[1:nθ,..], σx[1:nθ,..]) :
        EvidenceLowerBound(vcat(θ,Z), μx0, σx)
    return (; KLDiv, ELBO)
end

function sampleθZ_setup(cvae::CVAE, Y)
    Ypad = apply_pad(cvae, Y)
    μr = cvae.E1(Ypad)
    μr0, σr = split_mean_softplus_std(μr)
    return μr0, σr
end

sampleθZposterior(cvae::CVAE, Y) = (Ypad = apply_pad(cvae, Y); sampleθZposterior(cvae, Ypad, sampleθZ_setup(cvae, Ypad)...))

function sampleθZposterior(cvae::CVAE, Y, μr0, σr)
    Ypad = apply_pad(cvae, Y)
    zr = sample_mv_normal(μr0, σr)
    μx = cvae.D(vcat(Ypad,zr))
    μx0, σx = split_mean_softplus_std(μx)
    x = sample_mv_normal(μx0, σx)
    θ, Z = split_theta_latent(cvae, x)
    return θ, Z
end

function θZposterior_sampler(cvae::CVAE, Y)
    Ypad = apply_pad(cvae, Y)
    μr0, σr = sampleθZ_setup(cvae, Ypad) # constant over posterior samples
    θZposterior_sampler_inner() = sampleθZposterior(cvae, Ypad, μr0, σr)
    return θZposterior_sampler_inner
end

####
#### Deep prior
####

"""
Deep prior for learning to θ distribution, wrapping (parameterized) functions `θprior` and `Zprior`

    θprior : R^kθ -> R^nθ
    Zprior : R^kZ -> R^nZ

which generates samples θ ~ θprior(θ), Z ~ Zprior(Z) via the transformation of `kθ` and `kZ` implicitly
sampled latent variables, respectively. These θ parameterize physics models, e.g. phys : R^nθ -> R^n,
and Z parameterize latent variable Rician models.
"""
struct DeepPriorRicianPhysicsModel{T,kθ,kZ,P<:PhysicsModel{T},R<:RicianCorrector,Fθ,FZ}
    phys   :: P
    rice   :: R
    θprior :: Fθ
    Zprior :: FZ
    DeepPriorRicianPhysicsModel{T,kθ,kZ}(phys::P, rice::R, θprior::Fθ, Zprior::FZ) where {T,kθ,kZ,P,R,Fθ,FZ} = new{T,kθ,kZ,P,R,Fθ,FZ}(phys, rice, θprior, Zprior)
end
Flux.@functor DeepPriorRicianPhysicsModel
Flux.trainable(prior::DeepPriorRicianPhysicsModel) = (prior.θprior, prior.Zprior)
Base.show(io::IO, prior::DeepPriorRicianPhysicsModel) = model_summary(io, Dict("θprior" => prior.θprior, "θprior" => prior.θprior))

sampleθprior(prior::DeepPriorRicianPhysicsModel{T}, n::Int) where {T} = sampleθprior(prior, CUDA.CuMatrix{T}, n) # default to sampling θ on the gpu
sampleθprior(prior::DeepPriorRicianPhysicsModel, Y::AbstractArray, n::Int = size(Y,2)) = sampleθprior(prior, typeof(Y), n) # θ type is similar to Y type
sampleθprior(prior::DeepPriorRicianPhysicsModel{T,kθ,kZ}, ::Type{A}, n::Int) where {T, kθ, kZ, A <: AbstractArray{T}} = prior.θprior(randn_similar(A, kθ, n)) # sample from distribution

sampleZprior(prior::DeepPriorRicianPhysicsModel{T}, n::Int) where {T} = sampleZprior(prior, CUDA.CuMatrix{T}, n) # default to sampling Z on the gpu
sampleZprior(prior::DeepPriorRicianPhysicsModel, Y::AbstractArray, n::Int = size(Y,2)) = sampleZprior(prior, typeof(Y), n) # Z type is similar to Y type
sampleZprior(prior::DeepPriorRicianPhysicsModel{T,kθ,kZ}, ::Type{A}, n::Int) where {T, kθ, kZ, A <: AbstractArray{T}} = prior.Zprior(randn_similar(A, kZ, n)) # sample from distribution

#### PhysicsModel + CVAE methods

function sampleθZ(cvae::CVAE, prior::DeepPriorRicianPhysicsModel, Y::AbstractVecOrMat; posterior_θ = true, posterior_Z = true)
    if posterior_θ || posterior_Z
        return sampleθZ(cvae, prior, Y, sampleθZ_setup(cvae, Y)...; posterior_θ, posterior_Z)
    else
        θ = sampleθprior(prior, Y, size(Y,2))
        Z = sampleZprior(prior, Y, size(Y,2))
        return θ, Z
    end
end

function sampleθZ(cvae::CVAE, prior::DeepPriorRicianPhysicsModel, Y::AbstractVecOrMat, μr0, σr; posterior_θ = true, posterior_Z = true)
    if posterior_θ || posterior_Z
        θhat, Zhat = sampleθZposterior(cvae, Y, μr0, σr)
        θhat = clamp.(θhat, todevice(θlower(prior.phys)), todevice(θupper(prior.phys)))
        θ = posterior_θ ? θhat : sampleθprior(prior, Y, size(Y,2))
        Z = posterior_Z ? Zhat : sampleZprior(prior, Y, size(Y,2))
        θ, Z
    else
        θ = sampleθprior(prior, Y, size(Y,2))
        Z = sampleZprior(prior, Y, size(Y,2))
        return θ, Z
    end
end

function θZ_sampler(cvae::CVAE, prior::DeepPriorRicianPhysicsModel, Y::AbstractVecOrMat; posterior_θ = true, posterior_Z = true)
    μr0, σr = sampleθZ_setup(cvae, Y) # constant over posterior samples
    θZ_sampler_inner() = sampleθZ(cvae, prior, Y, μr0, σr; posterior_θ, posterior_Z)
    return θZ_sampler_inner
end

function sampleXθZ(cvae::CVAE, prior::DeepPriorRicianPhysicsModel, Y::AbstractVecOrMat; kwargs...)
    #TODO: can't differentiate through @timeit "sampleθZ"
    #TODO: can't differentiate through @timeit "signal_model"
    θ, Z = sampleθZ(cvae, prior, Y; kwargs...)
    X = signal_model(prior.phys, θ)
    return X, θ, Z
end

sampleX(cvae::CVAE, prior::DeepPriorRicianPhysicsModel, Y::AbstractVecOrMat; kwargs...) = sampleXθZ(cvae, prior, Y; kwargs...)[1]

#### RicianCorrector + PhysicsModel + CVAE methods

function sampleX̂θZ(cvae::CVAE, prior::DeepPriorRicianPhysicsModel, Y::AbstractVecOrMat; kwargs...)
    #TODO: can't differentiate through @timeit "sampleXθZ"
    #TODO: can't differentiate through @timeit "sampleX̂"
    X, θ, Z = sampleXθZ(cvae, prior, Y; kwargs...)
    X̂ = sampleX̂(prior.rice, X, Z)
    return X̂, θ, Z
end

sampleX̂(cvae::CVAE, prior::DeepPriorRicianPhysicsModel, Y::AbstractVecOrMat; kwargs...) = sampleX̂θZ(cvae, prior, Y; kwargs...)[1]

####
#### Rician posterior state
####

function sampleX̂(rice::RicianCorrector, X, Z, ninstances = nothing)
    ν, ϵ = rician_params(rice, X, Z)
    return add_noise_instance(rice, ν, ϵ, ninstances)
end

function NegLogLikelihood(rice::RicianCorrector, Y, μ0, σ)
    if typeof(rice) <: NormalizedRicianCorrector && !isnothing(rice.normalizer)
        Σμ = rice.normalizer(MMDLearning._rician_mean_cuda.(μ0, σ))
        μ0, σ = (μ0 ./ Σμ), (σ ./ Σμ)
    end
    -sum(MMDLearning._rician_logpdf_cuda.(Y, μ0, σ); dims = 1) # Rician negative log likelihood
end

function make_state(prior::DeepPriorRicianPhysicsModel, Y::AbstractMatrix, θ::AbstractMatrix, Z::AbstractMatrix)
    X = signal_model(prior.phys, θ)
    δ, ϵ = correction_and_noiselevel(prior.rice, X, Z)
    ν = add_correction(prior.rice, X, δ)
    ℓ = reshape(NegLogLikelihood(prior.rice, Y, ν, ϵ), 1, :)
    return (; Y, θ, Z, X, δ, ϵ, ν, ℓ)
end

function posterior_state(
        cvae::CVAE,
        prior::DeepPriorRicianPhysicsModel,
        Y::AbstractMatrix{T};
        miniter = 5,
        maxiter = 100,
        alpha = 0.01,
        mode = :maxlikelihood,
        verbose = false
    ) where {T}

    θZ_sampler_instance = θZ_sampler(cvae, prior, Y; posterior_θ = true, posterior_Z = true)

    function update(last_state, i)
        θnew, Znew = θZ_sampler_instance()
        θlast = isnothing(last_state) ? nothing : last_state.θ
        Zlast = isnothing(last_state) ? nothing : last_state.Z

        if mode === :mean
            θnew = isnothing(last_state) ? θnew : T(1/i) .* θnew .+ T(1-1/i) .* θlast
            Znew = isnothing(last_state) ? Znew : T(1/i) .* Znew .+ T(1-1/i) .* Zlast
            new_state = make_state(prior, Y, θnew, Znew)
        elseif mode === :maxlikelihood
            new_state = make_state(prior, Y, θnew, Znew)
            if !isnothing(last_state)
                mask = new_state.ℓ .< last_state.ℓ
                new_state = map(new_state, last_state) do new, last
                    new .= ifelse.(mask, new, last)
                end
            end
        else
            error("Unknown mode: $mode")
        end

        # Check for convergence
        p = isnothing(last_state) ? nothing :
            HypothesisTests.pvalue(
                HypothesisTests.UnequalVarianceTTest(
                    map(x -> x |> Flux.cpu |> vec |> Vector{Float64}, (new_state.ℓ, last_state.ℓ))...
                )
            )

        return new_state, p
    end

    @timeit "posterior state" CUDA.@sync begin
        state, _ = update(nothing, 1)
        verbose && @info 1, mean_and_std(state.ℓ)
        for i in 2:maxiter
            state, p = update(state, i)
            verbose && @info i, mean_and_std(state.ℓ), p
            (i >= miniter) && (p > 1 - alpha) && break
        end
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
