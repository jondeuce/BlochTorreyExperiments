# Initialize generator + discriminator + kernel
function make_mmd_cvae_models(phys::PhysicsModel{Float32}, settings::Dict{String,Any}, models = Dict{String, Any}(), derived = Dict{String, Any}())
    n   = nsignal(phys) # input signal length
    nθ  = ntheta(phys) # number of physics variables
    θbd = θbounds(phys)
    k   = settings["arch"]["nlatent"]::Int # number of latent variables Z
    nz  = settings["arch"]["zdim"]::Int # embedding dimension

    RiceGenType = LatentVectorRicianNoiseCorrector{n,k}
    # RiceGenType = LatentVectorRicianCorrector{n,k}
    # RiceGenType = VectorRicianCorrector{n,k}

    # Rician generator. First `n` elements for `δX` scaled to (-δ, δ), second `n` elements for `logϵ` scaled to (noisebounds[1], noisebounds[2])
    get!(models, "genatr") do
        hdim = settings["arch"]["genatr"]["hdim"]::Int
        nhidden = settings["arch"]["genatr"]["nhidden"]::Int
        skip = settings["arch"]["genatr"]["skip"]::Bool
        layernorm = settings["arch"]["genatr"]["layernorm"]::Bool
        leakyslope = settings["arch"]["genatr"]["leakyslope"]::Float64
        maxcorr = settings["arch"]["genatr"]["maxcorr"]::Float64
        noisebounds = settings["arch"]["genatr"]["noisebounds"]::Vector{Float64}
        nin, nout = ninput(RiceGenType), noutput(RiceGenType)
        σinner = leakyslope == 0 ? Flux.relu : eltype(phys)(leakyslope) |> a -> (x -> Flux.leakyrelu(x, a))
        OutputScale =
            RiceGenType <: Union{<:VectorRicianCorrector, <:LatentVectorRicianCorrector} ? MMDLearning.CatScale([(-maxcorr, maxcorr), (noisebounds...,)], [n,n]) :
            RiceGenType <: FixedNoiseVectorRicianCorrector ? MMDLearning.CatScale([(-maxcorr, maxcorr)], [n]) :
            RiceGenType <: LatentVectorRicianNoiseCorrector ? MMDLearning.CatScale([(noisebounds...,)], [n]) :
            error("Unsupported corrector type: $RiceGenType")

        # Generic nin => nout MLP with output scaling
        Flux.Chain(
            MMDLearning.MLP(nin => nout, nhidden, hdim, σinner, tanh; skip, layernorm)...,
            OutputScale,
        ) |> to32

        # #TODO: only works for LatentVectorRicianNoiseCorrector
        # @assert nin == k == nlatent(RiceGenType) && nout == n
        # Flux.Chain(
        #     # position encoding
        #     Z -> vcat(Z, zeros_similar(Z, 1, size(Z,2))),   # [k x b] -> [(k+1) x b]
        #     Z -> repeat(Z, n, 1),                           # [(k+1) x b] -> [(k+1)*n x b]
        #     NotTrainable(Flux.Diagonal(ones((k+1)*n), vec(vcat(zeros(k, n), uniform_range(n)')))),
        #     Z -> reshape(Z, k+1, :),                        # [(k+1)*n x b] -> [(k+1) x n*b]
        #     # position-wise mlp
        #     MMDLearning.MLP(k+1 => 1, nhidden, hdim, σinner, tanh; skip, layernorm)..., # [(k+1) x n*b] -> [1 x n*b]
        #     # output scaling
        #     Z -> reshape(Z, n, :),                          # [1 x n*b] -> [n x b]
        #     OutputScale,
        # ) |> to32
    end

    # Wrapped generator produces 𝐑^2n outputs parameterizing n Rician distributions
    get!(derived, "ricegen") do
        R = RiceGenType(models["genatr"])
        slicefirst(X) = X[1:1,..]
        maxsignal(X) = maximum(X; dims = 1)
        meansignal(X) = mean(X; dims = 1)
        NormalizedRicianCorrector(R, maxsignal, meansignal) #TODO: normalize by mean? sum? maximum? first echo?
    end

    # Encoders
    get!(models, "enc1") do
        hdim = settings["arch"]["enc1"]["hdim"]::Int
        nhidden = settings["arch"]["enc1"]["nhidden"]::Int
        skip = settings["arch"]["enc1"]["skip"]::Bool
        layernorm = settings["arch"]["enc1"]["layernorm"]::Bool
        psize = settings["arch"]["enc1"]["psize"]::Int
        head = settings["arch"]["enc1"]["head"]::Int
        MMDLearning.MLP(n => 2*nz, nhidden, hdim, Flux.relu, identity; skip, layernorm) |> to32
        # RESCNN(n => 2*nz, nhidden, hdim, Flux.relu, identity; skip) |> to32
        # Transformers.Stack(
        #     Transformers.@nntopo( X : X => H : H => μr ),
        #     TransformerEncoder(; n, psize, head, hdim, nhidden),
        #     MMDLearning.MLP(psize*n => 2*nz, 0, hdim, Flux.relu, identity),
        # ) |> to32
    end

    get!(models, "enc2") do
        hdim = settings["arch"]["enc2"]["hdim"]::Int
        nhidden = settings["arch"]["enc2"]["nhidden"]::Int
        skip = settings["arch"]["enc2"]["skip"]::Bool
        layernorm = settings["arch"]["enc2"]["layernorm"]::Bool
        psize = settings["arch"]["enc2"]["psize"]::Int
        head = settings["arch"]["enc2"]["head"]::Int
        MMDLearning.MLP(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity; skip, layernorm) |> to32
        # RESCNN(n + nθ + k => 2*nz, nhidden, hdim, Flux.relu, identity; skip) |> to32
        # Transformers.Stack(
        #     Transformers.@nntopo( (X,θ,Z) : X => H : (H,θ,Z) => HθZ : HθZ => μq ),
        #     TransformerEncoder(; n, psize, head, hdim, nhidden),
        #     vcat,
        #     MMDLearning.MLP(psize*n + nθ + k => 2*nz, 0, hdim, Flux.relu, identity),
        # ) |> to32
    end

    # Decoder
    get!(models, "dec") do
        hdim = settings["arch"]["dec"]["hdim"]::Int
        nhidden = settings["arch"]["dec"]["nhidden"]::Int
        skip = settings["arch"]["dec"]["skip"]::Bool
        layernorm = settings["arch"]["dec"]["layernorm"]::Bool
        Flux.Chain(
            MMDLearning.MLP(n + nz => 2*(nθ + k), nhidden, hdim, Flux.relu, identity; skip, layernorm)...,
            # RESCNN(n + nz => 2*(nθ + k), nhidden, hdim, Flux.relu, identity; skip)...,
            MMDLearning.CatScale(eltype(θbd)[θbd; (-1, 1)], [ones(Int, nθ); k + nθ + k]),
        ) |> to32
    end

    # Discriminator
    get!(models, "discrim") do
        hdim = settings["arch"]["discrim"]["hdim"]::Int
        nhidden = settings["arch"]["discrim"]["nhidden"]::Int
        skip = settings["arch"]["discrim"]["skip"]::Bool
        layernorm = settings["arch"]["discrim"]["layernorm"]::Bool
        dropout = settings["arch"]["discrim"]["dropout"]::Float64
        chunk = settings["train"]["transform"]["chunk"]::Int
        augsizes = Dict{String,Int}(["gradient" => n-1, "laplacian" => n-2, "encoderspace" => nz, "residuals" => n, "fftcat" => 2*(n÷2 + 1), "fftsplit" => 2*(n÷2 + 1)])
        nin = min(n, chunk) + sum((s -> ifelse(settings["train"]["augment"][s]::Bool, min(augsizes[s], chunk), 0)).(keys(augsizes)))
        MMDLearning.MLP(nin => 1, nhidden, hdim, Flux.relu, Flux.sigmoid; skip, layernorm, dropout) |> to32
        # RESCNN(n => 1, nhidden, hdim, Flux.relu, Flux.sigmoid; skip) |> to32
    end

    # CVAE
    get!(derived, "cvae") do; CVAE{n,nθ,k,nz}(models["enc1"], models["enc2"], models["dec"]) end

    # Misc. useful operators
    get!(derived, "forwarddiff") do; MMDLearning.ForwardDifferemce() |> to32 end
    get!(derived, "laplacian") do; MMDLearning.Laplacian() |> to32 end
    get!(derived, "encoderspace") do # non-trainable sampling of encoder signal representations
        NotTrainable(MMDLearning.flattenchain(Flux.Chain(
            models["enc1"],
            MMDLearning.split_mean_softplus_std,
            MMDLearning.sample_mv_normal,
        )))
    end

    return models, derived
end

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

function mv_normal_parameters(cvae::CVAE, Y, θ, Z)
    μr0, σr = split_mean_softplus_std(cvae.E1(Y))
    μq0, σq = split_mean_softplus_std(cvae.E2(vcat(Y,θ,Z)))
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_softplus_std(cvae.D(vcat(Y,zq)))
    return (; μr0, σr, μq0, σq, μx0, σx)
end

function KL_and_ELBO(cvae::CVAE{n,nθ,k,nz}, Y, θ, Z; recover_Z::Bool) where {n,nθ,k,nz}
    @unpack μr0, σr, μq0, σq, μx0, σx = mv_normal_parameters(cvae, Y, θ, Z)
    KLDiv = KLDivergence(μq0, σq, μr0, σr)
    ELBO = recover_Z ?
        EvidenceLowerBound(vcat(θ,Z), μx0, σx) :
        EvidenceLowerBound(θ, μx0[1:nθ,..], σx[1:nθ,..])
    return (; KLDiv, ELBO)
end

function θZposterior_sampler(cvae::CVAE, Y)
    μr = cvae.E1(Y)
    μr0, σr = split_mean_softplus_std(μr) # constant over posterior samples
    function θZposterior_sampler_inner()
        zr = sample_mv_normal(μr0, σr)
        μx = cvae.D(vcat(Y,zr))
        μx0, σx = split_mean_softplus_std(μx)
        x = sample_mv_normal(μx0, σx)
        θ, Z = split_theta_latent(cvae, x)
        return θ, Z
    end
    return θZposterior_sampler_inner
end

sampleθZposterior(cvae::CVAE, Y) = θZposterior_sampler(cvae, Y)()

####
#### PhysicsModel + CVAE methods
####

function θZ_sampler(cvae::CVAE, phys::PhysicsModel, Y; recover_θ = true, recover_Z = true)
    θZposterior_sampler_instance = θZposterior_sampler(cvae, Y)
    θprior() = sampleθprior(phys, Y, size(Y,2))
    Zprior() = randn_similar(Y, nlatent(cvae), size(Y,2))
    θclamp(θ) = clamp.(θ, todevice(θlower(phys)), todevice(θupper(phys)))
    function θZ_sampler_inner()
        if recover_θ || recover_Z
            θhat, Zhat = θZposterior_sampler_instance()
            θhat = θclamp(θhat)
            θ = recover_θ ? θhat : θprior()
            Z = recover_Z ? Zhat : Zprior()
            θ, Z
        else
            θprior(), Zprior()
        end
    end
    return θZ_sampler_inner
end

sampleθZ(cvae::CVAE, phys::PhysicsModel, Y; kwargs...) = θZ_sampler(cvae, phys, Y; kwargs...)()

function sampleXθZ(cvae::CVAE, phys::PhysicsModel, Y; kwargs...)
    @timeit "sampleθZ"     θ, Z = sampleθZ(cvae, phys, Y; kwargs...)
    @timeit "signal_model" X = signal_model(phys, θ)
    return X, θ, Z
end

sampleX(cvae::CVAE, phys::PhysicsModel, Y; kwargs...) = sampleXθZ(cvae, phys, Y; kwargs...)[1]

####
#### RicianCorrector + PhysicsModel + CVAE methods
####

function sampleX̂θZ(rice::RicianCorrector, cvae::CVAE, phys::PhysicsModel, Y; kwargs...)
    @timeit "sampleXθZ" X, θ, Z = sampleXθZ(cvae, phys, Y; kwargs...)
    @timeit "sampleX̂"   X̂ = sampleX̂(rice, X, Z)
    return X̂, θ, Z
end

sampleX̂(rice::RicianCorrector, cvae::CVAE, phys::PhysicsModel, Y; kwargs...) = sampleX̂θZ(rice, cvae, phys, Y; kwargs...)[1]

function sampleX̂(rice::RicianCorrector, X, Z, ninstances = nothing)
    ν, ϵ = rician_params(rice, X, Z)
    return add_noise_instance(rice, ν, ϵ, ninstances)
end

function NegLogLikelihood(rice::RicianCorrector, Y, μ0, σ)
    if rice isa MMDLearning.NormalizedRicianCorrector
        Σμ = rice.normalizer(MMDLearning._rician_mean_cuda.(μ0, σ))
        μ0, σ = (μ0 ./ Σμ), (σ ./ Σμ)
    end
    -sum(MMDLearning._rician_logpdf_cuda.(Y, μ0, σ); dims = 1) # Rician negative log likelihood
end

function make_state(rice::RicianCorrector, phys::PhysicsModel, Y::AbstractMatrix, θ::AbstractMatrix, Z::AbstractMatrix)
    X = signal_model(phys, θ)
    δ, ϵ = correction_and_noiselevel(rice, X, Z)
    ν = add_correction(rice, X, δ)
    ℓ = reshape(NegLogLikelihood(rice, Y, ν, ϵ), 1, :)
    return (; Y, θ, Z, X, δ, ϵ, ν, ℓ)
end

function posterior_state(
        rice::RicianCorrector,
        cvae::CVAE,
        phys::PhysicsModel,
        Y::AbstractMatrix{T};
        miniter = 5,
        maxiter = 100,
        alpha = 0.01,
        mode = :maxlikelihood,
        verbose = false
    ) where {T}

    θZ_sampler_instance = θZ_sampler(cvae, phys, Y; recover_θ = true, recover_Z = true)

    function update(last_state, i)
        θnew, Znew = θZ_sampler_instance()
        θlast = isnothing(last_state) ? nothing : last_state.θ
        Zlast = isnothing(last_state) ? nothing : last_state.Z

        if mode === :mean
            θnew = isnothing(last_state) ? θnew : T(1/i) .* θnew .+ T(1-1/i) .* θlast
            Znew = isnothing(last_state) ? Znew : T(1/i) .* Znew .+ T(1-1/i) .* Zlast
            new_state = make_state(rice, phys, Y, θnew, Znew)
        elseif mode === :maxlikelihood
            new_state = make_state(rice, phys, Y, θnew, Znew)
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
