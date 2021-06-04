"""
Conditional variational autoencoder.

Architecture inspired by:
    "Bayesian parameter estimation using conditional variational autoencoders for gravitational-wave astronomy"
    https://arxiv.org/abs/1802.08797
"""
struct CVAE{n,nθ,nθM,k,nz,Dist,E1,E2,D,F,F⁻¹}
    E1  :: E1
    E2  :: E2
    D   :: D
    f   :: F
    f⁻¹ :: F⁻¹
end
CVAE{n,nθ,nθM,k,nz}(enc1::E1, enc2::E2, dec::D, f::F, f⁻¹::F⁻¹; posterior_dist::Type = Gaussian) where {n,nθ,nθM,k,nz,E1,E2,D,F,F⁻¹} = CVAE{n,nθ,nθM,k,nz,posterior_dist,E1,E2,D,F,F⁻¹}(enc1, enc2, dec, f, f⁻¹)

const CVAEPosteriorDist{Dist} = CVAE{n,nθ,nθM,k,nz,Dist} where {n,nθ,nθM,k,nz}

Flux.functor(::Type{<:CVAE{n,nθ,nθM,k,nz,Dist}}, c) where {n,nθ,nθM,k,nz,Dist} = (E1 = c.E1, E2 = c.E2, D = c.D, f = c.f, f⁻¹ = c.f⁻¹,), fs -> CVAE{n,nθ,nθM,k,nz}(fs...; posterior_dist = Dist)
Base.show(io::IO, ::CVAE{n,nθ,nθM,k,nz,Dist}) where {n,nθ,nθM,k,nz,Dist} = print(io, "CVAE$((;n,nθ,nθM,k,nz,Dist))")

nsignal(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = n
ntheta(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nθ
nmarginalized(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nθM
nlatent(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = k
nembedding(::CVAE{n,nθ,nθM,k,nz}) where {n,nθ,nθM,k,nz} = nz

struct CVAETrainingState{C <: CVAE, A, S}
    cvae::C
    Ȳ::A
    θ̄::A
    Z̄::A
    μr0::A
    logσr::A
    μq0::A
    logσq::A
    nrm_state::S
end

function CVAETrainingState(cvae::CVAE, Y, θ, Z = zeros_similar(θ, 0, size(θ,2)))
    Ȳ, θ̄, Z̄, nrm_state = normalize_inputs(cvae, Y, θ, Z)
    Ȳpad = pad_signal(cvae, Ȳ)
    μr = cvae.E1(Ȳpad)
    μq = cvae.E2(Ȳpad, θ̄, Z̄)
    μr0, logσr = split_dim1(μr)
    μq0, logσq = split_dim1(μq)
    return CVAETrainingState(cvae, Ȳpad, θ̄, Z̄, μr0, logσr, μq0, logσq, nrm_state)
end
signal(state::CVAETrainingState) = state.Ȳ

struct CVAEInferenceState{C <: CVAE, A, S}
    cvae::C
    Ȳ::A
    μr0::A
    logσr::A
    nrm_state::S
end

function CVAEInferenceState(cvae::CVAE, Y)
    Ȳ, nrm_state = normalize_inputs(cvae, Y)
    Ȳpad = pad_signal(cvae, Ȳ)
    μr = cvae.E1(Ȳpad)
    μr0, logσr = split_dim1(μr)
    return CVAEInferenceState(cvae, Ȳpad, μr0, logσr, nrm_state)
end
signal(state::CVAEInferenceState) = state.Ȳ

normalize_inputs(cvae::CVAE, Y) = cvae.f((Y,)) # returns (Ȳ, nrm_state)
normalize_inputs(cvae::CVAE, Y, θ, Z) = cvae.f((Y, θ, Z)) # returns (Ȳ, θ̄, Z̄, nrm_state)
unnormalize_outputs(state::CVAETrainingState, θ̄M, Z̄) = state.cvae.f⁻¹((θ̄M, Z̄, state.nrm_state)) # returns (θM, Z)
unnormalize_outputs(state::CVAEInferenceState, θ̄M, Z̄) = state.cvae.f⁻¹((θ̄M, Z̄, state.nrm_state)) # returns (θM, Z)

# Layer which transforms matrix of [μ′; logσ′] ∈ [ℝ^nz; ℝ^nz] to bounded intervals [μ; logσ] ∈ [𝒟μ^nz; 𝒟logσ^nz]:
#      μ bounded: prevent CVAE from "memorizing" inputs via mean latent embedding vectors which are far from zero
#   logσ bounded: similarly, prevent CVAE from "memorizing" inputs via latent embedding vectors which are nearly constant, i.e. have zero variance
CVAELatentTransform(nz, 𝒟μ = (-3,3), 𝒟logσ = (-6,0)) = Flux.Chain(
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

function pad_and_mask_signal(Y::AbstractVecOrMat, n; minkept, maxkept)
    Ypad  = pad_signal(Y, n)
    M     = Zygote.@ignore signal_mask(Ypad; minkept, maxkept)
    Ymask = M .* Ypad
    return Ymask, M
end

####
#### CVAE methods
####

function KLDivergence(state::CVAETrainingState)
    @unpack μq0, logσq, μr0, logσr = state
    KLDivGaussian(μq0, logσq, μr0, logσr)
end

function EvidenceLowerBound(state::CVAETrainingState{C}; marginalize_Z::Bool = nlatent(state.cvae) == 0) where {C <: CVAEPosteriorDist{Gaussian}}
    @unpack cvae, Ȳ, θ̄, Z̄, μq0, logσq = state
    nθM = nmarginalized(cvae)
    zq = sample_mv_normal(μq0, exp.(logσq))
    μx0, logσx = split_dim1(cvae.D(Ȳ, zq))
    ELBO = marginalize_Z ?
        NegLogLGaussian(θ̄[1:nθM, ..], μx0[1:nθM, ..], logσx[1:nθM, ..]) :
        NegLogLGaussian(vcat(θ̄[1:nθM, ..], Z̄), μx0, logσx)
end

function EvidenceLowerBound(state::CVAETrainingState{C}; marginalize_Z::Bool = nlatent(state.cvae) == 0) where {C <: CVAEPosteriorDist{TruncatedGaussian}}
    @unpack cvae, Ȳ, θ̄, Z̄, μq0, logσq = state
    nθM = nmarginalized(cvae)
    zq = sample_mv_normal(μq0, exp.(logσq))
    μx = cvae.D(Ȳ, zq) # μx = D(Ȳ, zq) = [μθ̄M; μZ̄; logσθ̄M; logσZ̄]
    σ⁻¹μθ̄M, logσθ̄M, μZ̄, logσZ̄ = split_marginal_latent_pairs(cvae, μx)
    μθ̄M = tanh.(σ⁻¹μθ̄M) # transform from unbounded σ⁻¹μθ̄M ∈ ℝ^nθ to bounded interval [-1, 1]^nθ
    ELBO_θ = NegLogLTruncatedGaussian(θ̄[1:nθM, ..], μθ̄M, logσθ̄M, -1, 1)
    if marginalize_Z
        ELBO = ELBO_θ
    else
        ELBO = ELBO_θ + NegLogLGaussian(Z̄, μZ̄, logσZ̄)
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
    @unpack cvae, Ȳ, μr0, logσr = state
    zr = mode ? μr0 : sample_mv_normal(μr0, exp.(logσr))
    μx = cvae.D(Ȳ, zr)
    μx0, logσx = split_dim1(μx)
    x = mode ? μx0 : sample_mv_normal(μx0, exp.(logσx))
    θ̄M, Z̄ = split_marginal_latent(cvae, x)
    θM, Z = unnormalize_outputs(state, θ̄M, Z̄)
    return θM, Z̄
end

function sampleθZposterior(state::CVAEInferenceState{C}; mode = false) where {C <: CVAEPosteriorDist{TruncatedGaussian}}
    #TODO: `mode` is probably not strictly the correct term, but in practice it should be something akin to the distribution mode since `μr0` is the most likely value for `zr` and `μx0` is the most likely value for `x` **conditional on `zr`**; likely there are counterexamples to this simple reasoning, though...
    @unpack cvae, Ȳ, μr0, logσr = state
    zr = mode ? μr0 : sample_mv_normal(μr0, exp.(logσr))
    μx = cvae.D(Ȳ, zr)
    σ⁻¹μθ̄M, logσθ̄M, μZ̄, logσZ̄ = split_marginal_latent_pairs(cvae, μx)
    μθ̄M = tanh.(σ⁻¹μθ̄M) # transform from unbounded σ⁻¹μθ̄M ∈ ℝ^nθ to bounded interval [-1, 1]^nθ
    θ̄M = mode ? μθ̄M : sample_trunc_mv_normal(μθ̄M, exp.(logσθ̄M), -1, 1)
    Z̄ = mode || nlatent(state.cvae) == 0 ? μZ̄ : sample_mv_normal(μZ̄, exp.(logσZ̄))
    θM, Z = unnormalize_outputs(state, θ̄M, Z̄)
    return θM, Z
end

function θZposterior_sampler(cvae::CVAE, Y)
    state = CVAEInferenceState(cvae, Y) # constant over posterior samples
    θZposterior_sampler_inner(; kwargs...) = sampleθZposterior(state; kwargs...)
    return θZposterior_sampler_inner
end

#### CVAE + PhysicsModel + AbstractMetaDataSignal methods

function sampleθZ(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; posterior_mode = false)
    return sampleθZ(phys, cvae, Ymeta, CVAEInferenceState(cvae, signal(Ymeta)); posterior_mode)
end

function sampleθZ(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal, state::CVAEInferenceState; posterior_mode = false)
    θM, Z = sampleθZposterior(state; mode = posterior_mode)
    θMlo = arr_similar(θ̂M, θmarginalized(phys, θlower(phys)))
    θMhi = arr_similar(θ̂M, θmarginalized(phys, θupper(phys)))
    θ = vcat(clamp.(θM, θMlo, θMhi), Zygote.@ignore(θnuissance(phys, Ymeta)))
    return θ, Z
end

function θZ_sampler(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal)
    state = CVAEInferenceState(cvae, signal(Ymeta)) # constant over posterior samples
    θZ_sampler_inner(; kwargs...) = sampleθZ(phys, cvae, Ymeta, state; kwargs...)
    return θZ_sampler_inner
end

function sampleXθZ(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...)
    θ, Z = sampleθZ(phys, cvae, Ymeta; kwargs...)
    X = signal_model(phys, Ymeta, θ)
    (size(X,1) > nsignal(Ymeta)) && (X = X[1:nsignal(Ymeta), ..])
    return X, θ, Z
end

sampleX(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...) = sampleXθZ(phys, cvae, Ymeta; kwargs...)[1]

function posterior_state(phys::PhysicsModel, cvae::CVAE, Ymeta::AbstractMetaDataSignal; kwargs...)
    θ, Z = sampleθZ(phys, cvae, Ymeta; posterior_mode = false, kwargs...)
    posterior_state(phys, Ymeta, θ, Z)
end

@with_kw_noshow struct OnlineMetropolisSampler{T}
    θ::Array{T,3} # parameter values
    ntheta::Int           = size(θ, 1) # number of parameters per data point
    ndata::Int            = size(θ, 2) # number of data points, i.e. each column of θ represents estimates for a separate datum Y
    nsamples::Int         = size(θ, 3) # length of the Metropolis-Hastings MCMC chain which is recorded
    i::Vector{Int}        = ones(Int, ndata) # current sample index in cyclical chain buffer θ
    accept::Array{Bool,3} = zeros(Bool, 1, ndata, nsamples) # records whether proposal was accepted or not
    neglogPXθ::Array{T,3} = fill(T(Inf), 1, ndata, nsamples) # negative log likelihoods; initialize with Inf to guarantee acceptance of first sample
    neglogPθ::Array{T,3}  = zeros(T, 1, ndata, nsamples) # negative log priors; initialization is moot due to neglogPXθ initialzed to Inf
    neglogQθ::Array{T,3}  = zeros(T, 1, ndata, nsamples) # proposal distribution negative log likelihood; as opposed to standard MH, assumed to be independent of previous sample, i.e. Q(θ|θ′) ≡ Q(θ); initialization is moot due to neglogPXθ initialzed to Inf
end
Base.show(io::IO, s::OnlineMetropolisSampler{T}) where {T} = print(io, "OnlineMetropolisSampler{$(T)}(ntheta = $(s.ntheta), ndata = $(s.ndata), nsamples = $(s.nsamples))")

buffer_index(s::OnlineMetropolisSampler, j::Int) = CartesianIndex(j, mod1(s.i[j], s.nsamples))
random_index(s::OnlineMetropolisSampler, j::Int) = CartesianIndex(j, rand(1:s.nsamples))
buffer_indices(s::OnlineMetropolisSampler, J = 1:s.ndata) = buffer_index.((s,), J)
random_indices(s::OnlineMetropolisSampler, J = 1:s.ndata) = random_index.((s,), J)
function Random.rand(s::OnlineMetropolisSampler, J = 1:s.ndata)
    idx = random_indices(s, J)
    return s.θ[:, idx], s.neglogPXθ[:, idx], s.neglogPθ[:, idx], s.neglogQθ[:, idx]
end

# c.f. https://stats.stackexchange.com/a/163790
function update!(s::OnlineMetropolisSampler, θ′::AbstractMatrix, neglogPXθ′::AbstractMatrix, neglogPθ′::AbstractMatrix, neglogQθ′::AbstractMatrix, J = 1:s.ndata)
    @assert size(θ′, 1) == s.ntheta
    @assert size(θ′, 2) == size(neglogPXθ′, 2) == size(neglogPθ′, 2) == size(neglogQθ′, 2)
    @assert size(neglogPXθ′, 1) == size(neglogPθ′, 1) == size(neglogQθ′, 1) == 1

    # DECAES.tforeach(eachindex(J); blocksize = 16) do j
    Threads.@threads for j in eachindex(J)
        @inbounds begin
            col        = J[j]
            curr       = buffer_index(s, col)
            s.i[col]  += 1
            next       = buffer_index(s, col)

            # Metropolis-Hastings acceptance ratio:
            #        α = min(1, (PXθ′ * Pθ′ * Qθ) / (PXθ * Pθ * Qθ′))
            # ==> logα = min(0, logPXθ′ + logPθ′ + logQθ - logPXθ - logPθ - logQθ′)
            logα       = min(0, s.neglogPXθ[1,curr] + s.neglogPθ[1,curr] + neglogQθ′[1,j] - neglogPXθ′[1,j] - neglogPθ′[1,j] - s.neglogQθ[1,curr])
            accept     = logα > log(rand())

            # Update theta, negative log likelihoods, and negative log priors with accepted points or current points
            s.accept[1,next]    = accept
            @inbounds for i in 1:size(θ′, 1)
                s.θ[i,next]     = accept ? θ′[i,j]         : s.θ[i,curr]
            end
            s.neglogPXθ[1,next] = accept ? neglogPXθ′[1,j] : s.neglogPXθ[1,curr]
            s.neglogPθ[1,next]  = accept ? neglogPθ′[1,j]  : s.neglogPθ[1,curr]
            s.neglogQθ[1,next]  = accept ? neglogQθ′[1,j]  : s.neglogQθ[1,curr]
        end
    end
end

function update!(s::OnlineMetropolisSampler{T}, phys::BiexpEPGModel{T}, cvae::CVAEPosteriorDist{TruncatedGaussian}, img::CPMGImage, Y_gpu, Y_cpu = cpu(T, Y_gpu); img_cols) where {T}
    θ′, _      = sampleθZposterior(cvae, Y_gpu)
    θ′         = cpu(T, θ′)
    X′         = signal_model(phys, img, θ′)
    neglogPXθ′ = negloglikelihood(phys, Y_cpu, X′, θ′)
    neglogPθ′  = neglogprior(phys, θ′)
    neglogQθ′  = zeros_similar(neglogPθ′)
    update!(s, θ′, neglogPXθ′, neglogPθ′, neglogQθ′, img_cols)
end

function update!(s::OnlineMetropolisSampler{T}, phys::BiexpEPGModel{T}, cvae::CVAE, img::CPMGImage; dataset::Symbol, gpu_batch_size::Int) where {T}
    Y         = img.partitions[dataset]
    J_ranges  = collect(Iterators.partition(1:size(Y, 2), gpu_batch_size))
    for (i, (Y_gpu,)) in enumerate(CUDA.CuIterator((Y[:, J],) for J in J_ranges))
        Y_cpu = Y[:, J_ranges[i]]
        update!(s, phys, cvae, img, Y_gpu, Y_cpu; img_cols = J_ranges[i])
    end
end
