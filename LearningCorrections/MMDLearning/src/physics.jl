####
#### Rician correctors
####

abstract type RicianCorrector{n,nz} end

corrector(R::RicianCorrector) = R
generator(R::RicianCorrector) = R.G
nsignal(::Type{<:RicianCorrector{n}}) where {n} = n
nsignal(R::RicianCorrector) = nsignal(typeof(corrector(R)))
nlatent(::Type{<:RicianCorrector{n,nz}}) where {n,nz} = nz
nlatent(R::RicianCorrector) = nlatent(typeof(corrector(R)))
ninput(R::RicianCorrector) = ninput(typeof(corrector(R)))
noutput(R::RicianCorrector) = noutput(typeof(corrector(R)))

# Normalized rician corrector
@with_kw struct NormalizedRicianCorrector{n, nz, Rtype <: RicianCorrector{n,nz}, Ntype, Stype} <: RicianCorrector{n,nz}
    R::Rtype
    normalizer::Ntype
    noisescale::Stype
    NormalizedRicianCorrector(R::Rtype, nrm, scal) where {n, nz, Rtype <: RicianCorrector{n,nz}} = new{n, nz, Rtype, typeof(nrm), typeof(scal)}(R, nrm, scal)
end

const MaybeNormalizedRicianCorrector{n,nz} = Union{<:RicianCorrector{n,nz}, NormalizedRicianCorrector{n,nz}}

corrector(R::NormalizedRicianCorrector) = R.R
generator(R::NormalizedRicianCorrector) = R.R.G

# G : [X; Z] ∈ 𝐑^(n+k) -> [δ; logϵ] ∈ 𝐑^2n
@with_kw struct VectorRicianCorrector{n,nz,Gtype} <: RicianCorrector{n,nz}
    G::Gtype
    VectorRicianCorrector{n,nz}(G) where {n,nz} = new{n,nz,typeof(G)}(G)
end
Flux.@functor VectorRicianCorrector
ninput(::Type{R}) where {R<:VectorRicianCorrector} = nsignal(R) + nlatent(R)
noutput(::Type{R}) where {R<:VectorRicianCorrector} = 2 * nsignal(R)

# G : [X; Z] ∈ 𝐑^(n+k) -> δ ∈ 𝐑^n with fixed noise ϵ0 ∈ 𝐑 or ϵ0 ∈ 𝐑^n
@with_kw struct FixedNoiseVectorRicianCorrector{n,nz,T,Gtype} <: RicianCorrector{n,nz}
    G::Gtype
    ϵ0::T
    FixedNoiseVectorRicianCorrector{n,nz}(G,ϵ0) where {n,nz} = new{n,nz,typeof(ϵ0),typeof(G)}(G,ϵ0)
end
Flux.@functor FixedNoiseVectorRicianCorrector
ninput(::Type{R}) where {R<:FixedNoiseVectorRicianCorrector} = nsignal(R) + nlatent(R)
noutput(::Type{R}) where {R<:FixedNoiseVectorRicianCorrector} = nsignal(R)

# G : Z ∈ 𝐑^k -> [δ; logϵ] ∈ 𝐑^2n
@with_kw struct LatentVectorRicianCorrector{n,nz,Gtype} <: RicianCorrector{n,nz}
    G::Gtype
    LatentVectorRicianCorrector{n,nz}(G) where {n,nz} = new{n,nz,typeof(G)}(G)
end
Flux.@functor LatentVectorRicianCorrector
ninput(::Type{R}) where {R<:LatentVectorRicianCorrector} = nlatent(R)
noutput(::Type{R}) where {R<:LatentVectorRicianCorrector} = 2 * nsignal(R)

# G : Z ∈ 𝐑^k -> logϵ ∈ 𝐑^n with fixed δ = 0
@with_kw struct LatentVectorRicianNoiseCorrector{n,nz,Gtype} <: RicianCorrector{n,nz}
    G::Gtype
    LatentVectorRicianNoiseCorrector{n,nz}(G) where {n,nz} = new{n,nz,typeof(G)}(G)
end
Flux.@functor LatentVectorRicianNoiseCorrector
ninput(::Type{R}) where {R<:LatentVectorRicianNoiseCorrector} = nlatent(R)
noutput(::Type{R}) where {R<:LatentVectorRicianNoiseCorrector} = nsignal(R)

# G : Z ∈ 𝐑^k -> logϵ ∈ 𝐑 with fixed δ = 0
@with_kw struct LatentScalarRicianNoiseCorrector{n,nz,Gtype} <: RicianCorrector{n,nz}
    G::Gtype
    LatentScalarRicianNoiseCorrector{n,nz}(G) where {n,nz} = new{n,nz,typeof(G)}(G)
end
Flux.@functor LatentScalarRicianNoiseCorrector
ninput(::Type{R}) where {R<:LatentScalarRicianNoiseCorrector} = nlatent(R)
noutput(::Type{R}) where {R<:LatentScalarRicianNoiseCorrector} = 1

# Helper functions
@inline _maybe_vcat(X, Z = nothing) = isnothing(Z) ? X : vcat(X,Z)
@inline _split_delta_epsilon(δ_logϵ) = δ_logϵ[1:end÷2, :], exp.(δ_logϵ[end÷2+1:end, :]) .+ sqrt(eps(eltype(δ_logϵ)))
@inline function _add_rician_noise_instance(X, ϵ = nothing, ninstances = nothing)
    isnothing(ϵ) && return X
    ϵsize = isnothing(ninstances) ? size(X) : (size(X)..., ninstances)
    ϵR = ϵ .* randn_similar(X, ϵsize)
    ϵI = ϵ .* randn_similar(X, ϵsize)
    X̂ = @. sqrt((X + ϵR)^2 + ϵI^2)
    return X̂
end

# Concrete methods to extract δ and ϵ
correction_and_noiselevel(G::NormalizedRicianCorrector, args...) = correction_and_noiselevel(corrector(G), args...)
correction_and_noiselevel(G::VectorRicianCorrector, X, Z = nothing) = _split_delta_epsilon(generator(G)(_maybe_vcat(X,Z)))
correction_and_noiselevel(G::FixedNoiseVectorRicianCorrector, X, Z = nothing) = generator(G)(_maybe_vcat(X,Z)), G.ϵ0
correction_and_noiselevel(G::LatentVectorRicianCorrector, X, Z) = _split_delta_epsilon(generator(G)(Z))
correction_and_noiselevel(G::LatentVectorRicianNoiseCorrector, X, Z) = zero(X), exp.(generator(G)(Z)) .+ sqrt(eps(eltype(X)))
correction_and_noiselevel(G::LatentScalarRicianNoiseCorrector, X, Z) = zero(X), exp.(generator(G)(Z)) .* ones_similar(X, nsignal(G)) .+ sqrt(eps(eltype(X)))

# Derived convenience functions
correction(G::RicianCorrector, X, Z = nothing) = correction_and_noiselevel(G, X, Z)[1]
noiselevel(G::RicianCorrector, X, Z = nothing) = correction_and_noiselevel(G, X, Z)[2]
corrected_signal_instance(G::RicianCorrector, X, Z = nothing) = corrected_signal_instance(G, X, correction_and_noiselevel(G, X, Z)...)
corrected_signal_instance(G::RicianCorrector, X, δ, ϵ) = add_noise_instance(G, add_correction(G, X, δ), ϵ)
add_correction(G::RicianCorrector, X, δ) = @. abs(X + δ)
add_noise_instance(G::RicianCorrector, X, ϵ, ninstances = nothing) = _add_rician_noise_instance(X, ϵ, ninstances)
function add_noise_instance(G::NormalizedRicianCorrector, X, ϵ, ninstances = nothing)
    # Input data is assumed properly normalized; add noise relative to noise scale, then normalize X̂
    !isnothing(ϵ) && !isnothing(G.noisescale) && (ϵ = ϵ .* G.noisescale(X))
    X̂ = add_noise_instance(corrector(G), X, ϵ, ninstances)
    !isnothing(G.normalizer) && (X̂ = X̂ ./ G.normalizer(X̂))
    return X̂
end
function rician_params(G::RicianCorrector, X, Z = nothing)
    δ, ϵ = correction_and_noiselevel(G, X, Z)
    ν, σ = add_correction(G, X, δ), ϵ
    return ν, σ
end

####
#### Physics model interface
####

abstract type PhysicsModel{T,isfinite} end

struct ClosedForm{P<:PhysicsModel}
    p::P
end

const MaybeClosedForm{T,isfinite} = Union{<:PhysicsModel{T,isfinite}, <:ClosedForm{<:PhysicsModel{T,isfinite}}}

# Abstract interface
hasclosedform(p::PhysicsModel) = false # fallback
physicsmodel(p::PhysicsModel) = p
physicsmodel(c::ClosedForm) = c.p
Base.eltype(c::MaybeClosedForm) = Base.eltype(typeof(c))
Base.isfinite(c::MaybeClosedForm) = Base.isfinite(typeof(c))
Base.eltype(::Type{<:MaybeClosedForm{T}}) where {T} = T
Base.isfinite(::Type{<:MaybeClosedForm{T,isfinite}}) where {T,isfinite} = isfinite
function ntheta end
function nsignal end
function sampleWprior end
function sampleθprior end
function sampleθ end
function θlower end
function θupper end
function signal_model end
function noiselevel end

# Default samplers for models with data stored in `θ`, `X`, `Y` fields
_sample_data(d::Dict, n::Union{Int, Symbol}; dataset::Symbol) = n === :all ? d[dataset] : sample_columns(d[dataset], n)
sampleθ(p::MaybeClosedForm, n::Union{Int, Symbol};              dataset::Symbol) = _sample_data(physicsmodel(p).θ, n; dataset)
sampleX(p::MaybeClosedForm, n::Union{Int, Symbol}, ϵ = nothing; dataset::Symbol) = _sample_data(physicsmodel(p).X, n; dataset)
sampleY(p::MaybeClosedForm, n::Union{Int, Symbol}, ϵ = nothing; dataset::Symbol) = _sample_data(physicsmodel(p).Y, n; dataset)

sampleθ(p::PhysicsModel{T,false}, n::Union{Int, Symbol}; dataset::Symbol) where {T} = sampleθprior(p, n) # default to sampling θ on cpu from prior for infinite models
sampleθprior(p::PhysicsModel{T}, n::Union{Int, Symbol}) where {T} = sampleθprior(p, Matrix{T}, n) # default to sampling θ on cpu
sampleθprior(p::PhysicsModel, Y::AbstractArray, n::Union{Int, Symbol} = size(Y,2)) = sampleθprior(p, typeof(Y), n) # θ type is similar to Y type

θbounds(p::PhysicsModel) = tuple.(θlower(p), θupper(p)) # fallback
θerror(p::PhysicsModel, θtrue, θfit) = 100 .* todevice(θtrue .- θfit) ./ todevice(θupper(p) .- θlower(p)) # default error metric is elementwise percentage relative to prior width

θderived(p::PhysicsModel, θ) = θ # fallback
θderivedunits(p::PhysicsModel, θ) = θunits(θ) # fallback
θderivedlabels(p::PhysicsModel, θ) = θlabels(θ) # fallback
θderivedbounds(p::PhysicsModel, θ) = θbounds(θ) # fallback

####
#### Abstact toy problem interface
####

abstract type AbstractToyModel{T,isfinite} <: PhysicsModel{T,isfinite} end

const ClosedFormAbstractToyModel{T,isfinite} = ClosedForm{<:AbstractToyModel{T,isfinite}}
const MaybeClosedFormAbstractToyModel{T,isfinite} = Union{<:AbstractToyModel{T,isfinite}, <:ClosedFormAbstractToyModel{T,isfinite}}

function initialize!(p::AbstractToyModel{T,isfinite}; ntrain::Int, ntest::Int, nval::Int, seed::Int = 0) where {T,isfinite}
    rng = Random.seed!(seed)
    for (d, n) in [(:train, ntrain), (:test, ntest), (:val, nval)]
        isfinite ? (p.θ[d] = sampleθprior(p, n)) : empty!(p.θ)
        isfinite ? (p.X[d] = signal_model(p, p.θ[d])) : empty!(p.X)
        θ, W = sampleθprior(p, n), sampleWprior(ClosedForm(p), n)
        ν, ϵ = rician_params(ClosedForm(p), θ, W)
        p.Y[d] = add_noise_instance(p, ν, ϵ)
    end
    signal_histograms!(p)
    Random.seed!(rng)
    return p
end

function signal_histograms!(p::PhysicsModel; nbins = 100, normalize = nothing)
    p.meta[:histograms] = Dict{Symbol, Any}()
    p.meta[:histograms][:train] = signal_histograms(p.Y[:train]; edges = nothing, nbins, normalize)
    train_edges = Dict([k => v.edges[1] for (k,v) in p.meta[:histograms][:train]])
    for dataset in (:val, :test)
        p.meta[:histograms][dataset] = signal_histograms(p.Y[dataset]; edges = train_edges, nbins = nothing, normalize)
    end
    return p
end

# X-sampler deliberately does not take W as argument; W is supposed to be hidden from the outside. Use signal_model directly to pass W
sampleX(p::AbstractToyModel{T,false}, n::Union{Int, Symbol}, ϵ = nothing; dataset::Symbol) where {T} = sampleX(p, sampleθ(physicsmodel(p), n; dataset), ϵ)
sampleX(p::MaybeClosedFormAbstractToyModel, θ, ϵ = nothing) = signal_model(p, θ, ϵ)

# Fallback prior samplers
sampleWprior(c::ClosedFormAbstractToyModel{T}, n::Union{Int, Symbol}) where {T} = sampleWprior(c, Matrix{T}, n)
sampleWprior(c::ClosedFormAbstractToyModel{T}, Y::AbstractArray{T}, n::Union{Int, Symbol} = size(Y,2)) where {T} = sampleWprior(c, typeof(Y), n)

####
#### Toy exponentially decaying model with sinusoidally modulated amplitude
####

@with_kw struct ToyModel{T,isfinite} <: AbstractToyModel{T,isfinite}
    ϵ0::T = 0.01
    θ::Dict{Symbol,Matrix{T}} = Dict()
    X::Dict{Symbol,Matrix{T}} = Dict()
    Y::Dict{Symbol,Matrix{T}} = Dict()
    meta::Dict{Symbol,Any} = Dict()
end
const ClosedFormToyModel{T,isfinite} = ClosedForm{ToyModel{T,isfinite}}
const MaybeClosedFormToyModel{T,isfinite} = Union{ToyModel{T,isfinite}, ClosedFormToyModel{T,isfinite}}

ntheta(::ToyModel) = 5
nsignal(::ToyModel) = 128
nlatent(::ToyModel) = 0
hasclosedform(::ToyModel) = true
beta(::ToyModel) = 4
beta(::ClosedFormToyModel) = 2

θlabels(::ToyModel) = [L"f", L"\phi", L"a_0", L"a_1", L"\tau"]
θasciilabels(::ToyModel) = ["freq", "phase", "a0", "a1", "tau"]
θunits(::ToyModel) = ["Hz", "rad", "a.u.", "a.u.", "s"]
θlower(::ToyModel{T}) where {T} = T[1/T(64), T(0),   1/T(4), 1/T(10), T(16) ]
θupper(::ToyModel{T}) where {T} = T[1/T(32), T(π)/2, 1/T(2), 1/T(4),  T(128)]

sampleWprior(c::ClosedFormToyModel{T}, ::Type{A}, n::Union{Int, Symbol}) where {T, A <: AbstractArray{T}} = nothing
sampleθprior(p::ToyModel{T}, ::Type{A}, n::Union{Int, Symbol}) where {T, A <: AbstractArray{T}} = rand_similar(A, ntheta(p), n) .* (arr_similar(A, θupper(p)) .- arr_similar(A, θlower(p))) .+ arr_similar(A, θlower(p))

noiselevel(c::ClosedFormToyModel, θ = nothing, W = nothing) = physicsmodel(c).ϵ0
add_noise_instance(p::MaybeClosedFormToyModel, X, ϵ, ninstances = nothing) = _add_rician_noise_instance(X, ϵ, ninstances)

function signal_model(p::MaybeClosedFormToyModel, θ::AbstractVecOrMat, ϵ = nothing, W = nothing)
    n = nsignal(p)
    β = beta(p)
    t = 0:n-1
    f, ϕ, a₀, a₁, τ = θ[1:1,:], θ[2:2,:], θ[3:3,:], θ[4:4,:], θ[5:5,:]
    X = @. (a₀ + a₁ * sin(2*(π*f)*t - ϕ)^β) * exp(-t/τ)
    X̂ = add_noise_instance(p, X, ϵ)
    return X̂
end

rician_params(c::ClosedFormToyModel, θ, W = nothing) = signal_model(c, θ, nothing, W), noiselevel(c, θ, W)

####
#### Toy cosine model with latent variable controlling noise amplitude
####

@with_kw struct ToyCosineModel{T,isfinite} <: AbstractToyModel{T,isfinite}
    ϵbd::NTuple{2,T} = (0.01, 0.1)
    θ::Dict{Symbol,Matrix{T}} = Dict()
    X::Dict{Symbol,Matrix{T}} = Dict()
    Y::Dict{Symbol,Matrix{T}} = Dict()
    meta::Dict{Symbol,Any} = Dict()
end
const ClosedFormToyCosineModel{T,isfinite} = ClosedForm{ToyCosineModel{T,isfinite}}
const MaybeClosedFormToyCosineModel{T,isfinite} = Union{ToyCosineModel{T,isfinite}, ClosedFormToyCosineModel{T,isfinite}}

ntheta(::ToyCosineModel) = 3
nsignal(::ToyCosineModel) = 128
nlatent(::ToyCosineModel) = 1
hasclosedform(::ToyCosineModel) = true

θlabels(::ToyCosineModel) = [L"f", L"\phi", L"a_0"]
θasciilabels(::ToyCosineModel) = ["freq", "phase", "a0"]
θunits(::ToyCosineModel) = ["Hz", "rad", "a.u."]
θlower(::ToyCosineModel{T}) where {T} = T[T(1/64), T(0),   T(1/2)]
θupper(::ToyCosineModel{T}) where {T} = T[T(1/32), T(π/2), T(1)]

sampleWprior(c::ClosedFormToyCosineModel{T}, ::Type{A}, n::Union{Int, Symbol}) where {T, A <: AbstractArray{T}} = rand_similar(A, nlatent(physicsmodel(c)), n)
sampleθprior(p::ToyCosineModel{T}, ::Type{A}, n::Union{Int, Symbol}) where {T, A <: AbstractArray{T}} = rand_similar(A, ntheta(p), n) .* (arr_similar(A, θupper(p)) .- arr_similar(A, θlower(p))) .+ arr_similar(A, θlower(p))

noiselevel(c::ClosedFormToyCosineModel, θ = nothing, W = nothing) = ((lo,hi) = physicsmodel(c).ϵbd; return @. lo + W * (hi - lo))
add_noise_instance(p::MaybeClosedFormToyCosineModel, X, ϵ, ninstances = nothing) = _add_rician_noise_instance(X, ϵ, ninstances)

function signal_model(p::MaybeClosedFormToyCosineModel, θ::AbstractVecOrMat, ϵ = nothing, W = nothing)
    n = nsignal(p)
    t = 0:n-1
    f, ϕ, a₀ = θ[1:1,:], θ[2:2,:], θ[3:3,:]
    X = @. 1 + a₀ * cos(2*(π*f)*t - ϕ)
    X̂ = add_noise_instance(p, X, ϵ)
    return X̂
end

rician_params(c::ClosedFormToyCosineModel, θ, W = nothing) = signal_model(c, θ, nothing, W), noiselevel(c, θ, W)

####
#### Biexponential EPG signal models
####

abstract type AbstractToyEPGModel{T,isfinite} <: AbstractToyModel{T,isfinite} end

# Toy EPG model with latent variable controlling noise amplitude
@with_kw struct ToyEPGModel{T,isfinite} <: AbstractToyEPGModel{T,isfinite}
    n::Int = 48 # number of echoes
    T1::T = 1.0 # T1 relaxation (s)
    refcon::T = 180.0 # Refocusing pulse control angle (deg)
    TE::T = 8e-3 # T2 echo spacing (s)
    T2bd::NTuple{2,T} = (TE, 1.0) # min/max allowable T2
    ϵbd::NTuple{2,T} = (0.001, 0.01) # noise bound
    θ::Dict{Symbol,Matrix{T}} = Dict()
    X::Dict{Symbol,Matrix{T}} = Dict()
    Y::Dict{Symbol,Matrix{T}} = Dict()
    meta::Dict{Symbol,Any} = Dict()
end

# EPG model using image data
@with_kw struct EPGModel{T,isfinite} <: PhysicsModel{T,isfinite}
    n::Int = 48 # number of echoes
    T1::T = 1.0 # T1 relaxation (s)
    refcon::T = 180.0 # Refocusing pulse control angle (deg)
    TE::T = 8e-3 # T2 echo spacing (s)
    T2bd::NTuple{2,T} = (TE, 1.0) # min/max allowable T2
    image::Dict{Symbol,AbstractArray} = Dict()
    θ::Dict{Symbol,Matrix{T}} = Dict()
    X::Dict{Symbol,Matrix{T}} = Dict()
    Y::Dict{Symbol,Matrix{T}} = Dict()
    meta::Dict{Symbol,Any} = Dict()
end

function initialize!(p::EPGModel{T,isfinite}; imagepath::String, seed::Int = 0) where {T,isfinite}
    rng = Random.seed!(seed)
    p.image[:data] = convert(Array{T}, DECAES.load_image(imagepath, Val(4))) # load 4D MatrixSize x nTE image

    Imask    = findall(>(0), p.image[:data][..,1]) # image is assumed to be masked; filter out non-positive first echo signals
    Inonmask = findall(<=(0), p.image[:data][..,1]) # compliment of mask indices
    Ishuffle = shuffle(MersenneTwister(seed), Imask) # shuffle indices before splitting to train/test/val

    p.image[:data][Inonmask, :] .= NaN # set signals outside of mask to NaN
    p.image[:data] ./= maximum(p.image[:data]; dims = 4) # p.image[:data][1:1,..] #TODO: normalize by mean? sum? maximum? first echo?

    p.image[:mask_indices]  = Imask # non-shuffled mask indices
    p.image[:train_indices] = Ishuffle[            1 : 2*(end÷4)] # first half for training
    p.image[:test_indices]  = Ishuffle[2*(end÷4) + 1 : 3*(end÷4)] # third quarter held out for testing
    p.image[:val_indices]   = Ishuffle[3*(end÷4) + 1 : end] # fourth quarter for validation

    for d in (:train, :test, :val)
        i = p.image[Symbol(d, :_indices)]
        p.Y[d] = convert(Matrix{T}, permutedims(p.image[:data][i,:])) # convert to nTE x nSamples Matrix
        isfinite ? (p.θ[d] = sampleθprior(p, length(i))) : empty!(p.θ)
        isfinite ? (p.X[d] = signal_model(p, p.θ[d])) : empty!(p.X)
    end
    signal_histograms!(p)
    # t2_distributions!(p)

    Random.seed!(rng)
    return p
end

function t2_distributions!(p::EPGModel)
    p.meta[:decaes] = Dict{Symbol, Any}()
    p.meta[:decaes][:t2mapopts] = DECAES.T2mapOptions{Float64}(
        MatrixSize       = size(p.image[:data])[1:3],
        nTE              = size(p.image[:data])[4],
        TE               = p.TE,
        T1               = p.T1,
        T2Range          = p.T2bd,
        nT2              = 40,
        Threshold        = 0.0,
        Chi2Factor       = 1.02,
        RefConAngle      = p.refcon,
        MinRefAngle      = 50.0,
        nRefAnglesMin    = 8,
        nRefAngles       = 8,
        Reg              = "chi2",
        SaveResidualNorm = true,
        SaveDecayCurve   = true,
        SaveRegParam     = true,
        Silent           = true,
    )
    p.meta[:decaes][:t2partopts] = DECAES.T2partOptions{Float64}(
        MatrixSize = size(p.image[:data])[1:3],
        nT2        = p.meta[:decaes][:t2mapopts].nT2,
        T2Range    = p.meta[:decaes][:t2mapopts].T2Range,
        SPWin      = (prevfloat(p.meta[:decaes][:t2mapopts].T2Range[1]), 40e-3),
        MPWin      = (nextfloat(40e-3), nextfloat(p.meta[:decaes][:t2mapopts].T2Range[2])),
        Silent     = true,
    )
    p.meta[:decaes][:t2maps], p.meta[:decaes][:t2dist], p.meta[:decaes][:t2parts] = (Dict{Symbol,Any}() for _ in 1:3)
    p.meta[:decaes][:t2maps][:Y], p.meta[:decaes][:t2dist][:Y] = DECAES.T2mapSEcorr(p.image[:data] |> arr64, p.meta[:decaes][:t2mapopts])
    p.meta[:decaes][:t2parts][:Y] = DECAES.T2partSEcorr(p.meta[:decaes][:t2dist][:Y], p.meta[:decaes][:t2partopts])
    return p
end

function t2_distributions!(p::EPGModel, X::P) where {P <: Pair{Symbol, <:AbstractTensor4D}}
    Xname, Xdata = X
    t2mapopts = DECAES.T2mapOptions(p.meta[:decaes][:t2mapopts], MatrixSize = size(Xdata)[1:3])
    t2partopts = DECAES.T2partOptions(p.meta[:decaes][:t2partopts], MatrixSize = size(Xdata)[1:3])
    p.meta[:decaes][:t2maps][Xname], p.meta[:decaes][:t2dist][Xname] = DECAES.T2mapSEcorr(Xdata |> arr64, t2mapopts)
    p.meta[:decaes][:t2parts][Xname] = DECAES.T2partSEcorr(p.meta[:decaes][:t2dist][Xname], t2partopts)
    return p
end
t2_distributions!(p::EPGModel, X::P) where {P <: Pair{Symbol, <:AbstractMatrix}} = t2_distributions!(p, X[1] => reshape(permutedims(X[2]), size(X[2],2), 1, 1, size(X[2],1)))
t2_distributions!(p::EPGModel, Xs::Dict{Symbol, Any}) = (for (k,v) in Xs; t2_distributions!(p, k => v); end; return p)

const ClosedFormToyEPGModel{T,isfinite} = ClosedForm{ToyEPGModel{T,isfinite}}
const MaybeClosedFormToyEPGModel{T,isfinite} = Union{ToyEPGModel{T,isfinite}, ClosedFormToyEPGModel{T,isfinite}}

const BiexpEPGModel{T,isfinite} = Union{<:ToyEPGModel{T,isfinite}, <:EPGModel{T,isfinite}}
const ClosedFormBiexpEPGModel{T,isfinite} = ClosedFormToyEPGModel{T,isfinite} # EPGModel has no closed form
const MaybeClosedFormBiexpEPGModel{T,isfinite} = Union{<:BiexpEPGModel{T,isfinite}, <:ClosedFormBiexpEPGModel{T,isfinite}}

ntheta(p::BiexpEPGModel) = 5
nsignal(p::BiexpEPGModel) = p.n

nlatent(p::ToyEPGModel) = 1
nlatent(p::EPGModel) = 0
hasclosedform(p::ToyEPGModel) = true
hasclosedform(p::EPGModel) = false

θlabels(::BiexpEPGModel) = [L"\alpha", L"\beta", L"\eta", L"\delta_1", L"\delta_2"]
θasciilabels(::BiexpEPGModel) = ["alpha", "refcon", "eta", "delta1", "delta2"]
θunits(::BiexpEPGModel) = ["deg", "deg", "a.u.", "a.u.", "a.u."]
θlower(::BiexpEPGModel{T}) where {T} = T[T( 90.0), T( 90.0), T(0.0), T(0.0), T(0.0)]
θupper(::BiexpEPGModel{T}) where {T} = T[T(180.0), T(180.0), T(1.0), T(1.0), T(1.0)]

function sampleθprior(p::BiexpEPGModel{T}, ::Type{A}, n::Union{Int, Symbol}) where {T, A <: AbstractArray{T}}
    # # Parameterize by alpha, short amplitude, relative T2 long and T2 short δs
    # αlo, ηlo, δ1lo, δ2lo = θlower(p)
    # αhi, ηhi, δ1hi, δ2hi = θupper(p)
    # # α  = αlo .+ (αhi .- αlo) .* sqrt.(rand_similar(A, 1, n)) # triangular distbn on (αlo, αhi)
    # # η  = rand_similar(A, 1, n) # uniform distbn on (0, 1)
    # # δ1 = rand_similar(A, 1, n) # uniform distbn on (0, 1)
    # # δ2 = rand_similar(A, 1, n) # uniform distbn on (0, 1)
    # α  = αlo .+ (αhi .- αlo) .* (x -> (1-exp(-3x))/(1-exp(T(-3)))).(rand_similar(A, 1, n)) # concave triangular distbn on (αlo, αhi); encourages less near αlo, more near αhi
    # η  = ((x,y) -> y < T(1/2) ? x : sqrt(x)).(rand_similar(A, 1, n), rand_similar(A, 1, n)) # union of uniform and triangular distbns on (0, 1)
    # δ1 = ((x,y) -> y < T(1/2) ? x : sqrt(x/4)).(rand_similar(A, 1, n), rand_similar(A, 1, n)) # union of uniform on (0, 1) and triangular distbns on (0, 1/2)
    # δ2 = rand_similar(A, 1, n) # uniform distbn on (0, 1)
    # cosα = cosαlo .+ (cosαhi .- cosαlo) .* (x -> (exp(3x)-1)/(exp(T(3))-1)).(rand_similar(A, 1, n)) # concave triangular distbn on (cosαlo, cosαhi); encourages less near cosαlo, more near cosαhi
    # cosβ = cosβlo .+ (cosβhi .- cosβlo) .* (x -> (exp(3x)-1)/(exp(T(3))-1)).(rand_similar(A, 1, n)) # concave triangular distbn on (cosβlo, cosβhi); encourages less near cosβlo, more near cosβhi
    # cosα = cosαlo .+ (cosαhi .- cosαlo) .* (x -> 1-sqrt(x)).(rand_similar(A, 1, n)) # flipped triangular distbn on (cosαlo, cosαhi); more near cosαlo, less near cosαhi
    # cosβ = cosβlo .+ (cosβhi .- cosβlo) .* (x -> 1-sqrt(x)).(rand_similar(A, 1, n)) # flipped triangular distbn on (cosβlo, cosβhi); more near cosβlo, less near cosβhi
    # Parameterize by alpha, refcon, short amplitude, relative T2 long and T2 short δs
    αlo, βlo, ηlo, δ1lo, δ2lo = θlower(p)
    αhi, βhi, ηhi, δ1hi, δ2hi = θupper(p)
    α  = αlo .+ (αhi .- αlo) .* (x -> (1-exp(-3x))/(1-exp(T(-3)))).(rand_similar(A, 1, n)) # concave triangular distbn on (αlo, αhi); encourages less near αlo, more near αhi
    β  = βlo .+ (βhi .- βlo) .* (x -> (1-exp(-3x))/(1-exp(T(-3)))).(rand_similar(A, 1, n)) # concave triangular distbn on (βlo, βhi); encourages less near βlo, more near βhi
    η  = ((x,y) -> y < T(1/2) ? x : sqrt(x)).(rand_similar(A, 1, n), rand_similar(A, 1, n)) # union of uniform and triangular distbns on (0, 1)
    δ1 = ((x,y) -> y < T(1/2) ? x : sqrt(x/4)).(rand_similar(A, 1, n), rand_similar(A, 1, n)) # union of uniform on (0, 1) and triangular distbns on (0, 1/2)
    δ2 = rand_similar(A, 1, n) # uniform distbn on (0, 1)
    return vcat(α, β, η, δ1, δ2)
end

θsignalmodel(c::MaybeClosedFormBiexpEPGModel, θ::AbstractVecOrMat) = θsignalmodel(c, ntuple(i -> θ[i,:], ntheta(c))...)

function θsignalmodel(c::MaybeClosedFormBiexpEPGModel, α, β, η, δ1, δ2)
    # Parameterize by alpha, refcon, short amplitude, relative T2 long and T2 short δs
    logT2lo, logT2hi = log.(physicsmodel(c).T2bd)
    alpha, refcon = α, β
    Ashort, Along = η, 1 .- η
    T2short = @. exp(logT2lo + (logT2hi - logT2lo) * δ1)
    T2long = @. exp(logT2lo + (logT2hi - logT2lo) * (δ1 + δ2 * (1 - δ1)))
    return alpha, refcon, T2short, T2long, Ashort, Along
end

function θderived(
        c::MaybeClosedFormBiexpEPGModel,
        θ::AbstractVecOrMat{T};
        SPcutoff::T = T(40e-3),
        SPwidth::T = T(10e-3),
    ) where {T}
    alpha, refcon, eta, delta1, delta2 = θ[1,:], θ[2,:], θ[3,:], θ[4,:], θ[5,:]
    _, _, T2short, T2long, Ashort, Along = θsignalmodel(c, θ)
    logT2short, logT2long = log.(T2short), log.(T2long)
    logT2bar = @. Ashort * logT2short + Along * logT2long
    T2bar = @. exp(logT2bar)
    mwf = @. Ashort * soft_cutoff(T2short, SPcutoff, SPwidth) + Along * soft_cutoff(T2long, SPcutoff, SPwidth)
    return (;
        alpha, refcon, eta, delta1, delta2, # inference domain params
        T2short, T2long, Ashort, Along, # signal model params (without repeated alpha, refcon)
        logT2short, logT2long, logT2bar, T2bar, mwf, # misc. derived params
    )
end

θsignalmodelunits(::BiexpEPGModel) = ["deg", "deg", "s", "s", "a.u.", "a.u."]
θsignalmodellabels(::BiexpEPGModel) = [L"\alpha", L"\beta", L"T_{2,short}", L"T_{2,long}", L"A_{short}", L"A_{long}"]
θsignalmodelbounds(p::BiexpEPGModel) = [[θbounds(p)[i] for i in 1:2]..., (0.0, 1.0), p.T2bd, (0.0, 0.1), (0.0, 1.0)]

θderivedunits(p::BiexpEPGModel) = [θunits(p); θsignalmodelunits(p)[3:end]; "log(s)"; "log(s)"; "log(s)"; "s"; "a.u."]
θderivedlabels(p::BiexpEPGModel) = [θlabels(p); θsignalmodellabels(p)[3:end]; L"\log T_{2,short}"; L"\log T_{2,long}"; L"\log \bar{T}_2"; L"\bar{T}_2"; L"MWF"]
θderivedbounds(p::BiexpEPGModel) = [θbounds(p); θsignalmodelbounds(p)[3:end]; log.(p.T2bd); log.(p.T2bd); log.(p.T2bd); (0.0, 0.25); (0.0, 0.4)]

#### Toy EPG model

sampleWprior(c::ClosedFormToyEPGModel{T}, ::Type{A}, n::Union{Int, Symbol}) where {T, A <: AbstractArray{T}} = rand_similar(A, nlatent(physicsmodel(c)), n)

function signal_model(p::MaybeClosedFormToyEPGModel, θ::AbstractVecOrMat, ϵ = nothing, W = nothing)
    X, _ = rician_params(ClosedForm(p), θ, W)
    X̂ = add_noise_instance(p, X, ϵ)
    return X̂
end

function rician_params(c::ClosedFormToyEPGModel{T}, θ, W = nothing) where {T}
    p = physicsmodel(c)
    X = _signal_model(p, θ)
    ϵ = isnothing(W) ? nothing : X[1:1,:] .* (p.ϵbd[1] .+ W .* (p.ϵbd[2] .- p.ϵbd[1])) # noise with amplitude relative to first echo
    return X, ϵ
end

noiselevel(c::ClosedFormToyEPGModel, θ = nothing, W = nothing) = rician_params(c, θ, W)[2]

#### MRI data EPG model

function signal_model(p::EPGModel, θ::AbstractVecOrMat, ϵ = nothing)
    X = _signal_model(p, θ)
    X̂ = add_noise_instance(p, X, ϵ)
    return X̂
end

#### Common biexponential EPG model methods

function add_noise_instance(p::MaybeClosedFormBiexpEPGModel, X, ϵ, ninstances = nothing)
    X̂ = _add_rician_noise_instance(X, ϵ, ninstances)
    return X̂ ./ maximum(X̂; dims = 1) #X̂[1:1,:] #TODO: normalize by mean? sum? maximum? first echo?
end

function _signal_model(c::MaybeClosedFormBiexpEPGModel, θ::AbstractVecOrMat)
    X = _signal_model(c, θsignalmodel(c, θ)...)
    X = X ./ maximum(X; dims = 1) #X[1:1,:] #TODO: normalize by mean? sum? maximum? first echo?
    return X
end

# Faster to compute forward/reverse pass on the CPU and convert back to GPU after... DECAES is just too fast (for typical batch sizes of ~1024, anyways)

_signal_model(c::MaybeClosedFormBiexpEPGModel{T}, args::AbstractVector{T}...) where {T} = arr_similar(Matrix{T}, _signal_model_f64(c, map(arr64, args)...))
_signal_model(c::MaybeClosedFormBiexpEPGModel{T}, args::CUDA.CuVector{T}...) where {T} = arr_similar(CUDA.CuMatrix{T}, _signal_model_f64(c, map(arr64, args)...))

function _signal_model_f64(c::MaybeClosedFormBiexpEPGModel, alpha::AbstractVector{Float64}, refcon::AbstractVector{Float64}, T2short::AbstractVector{Float64}, T2long::AbstractVector{Float64}, Ashort::AbstractVector{Float64}, Along::AbstractVector{Float64})
    args = (alpha, refcon, T2short, T2long, Ashort, Along)
    @assert all(==(length(args[1])), length.(args))

    nsignals, nsamples = nsignal(c), length(args[1])
    X = zeros(Float64, nsignals, nsamples)
    work = [BiexpEPGModelWork(c) for _ in 1:Threads.nthreads()]
    DECAES.tforeach(1:nsamples; blocksize = 16) do j
        @inbounds begin
            _signal_model_f64!(view(X,:,j), c, work[Threads.threadid()], ntuple(i -> args[i][j], 6))
        end
    end

    return X
end

Zygote.@adjoint function _signal_model_f64(c::MaybeClosedFormBiexpEPGModel, alpha::AbstractVector{Float64}, refcon::AbstractVector{Float64}, T2short::AbstractVector{Float64}, T2long::AbstractVector{Float64}, Ashort::AbstractVector{Float64}, Along::AbstractVector{Float64})
    args = (alpha, refcon, T2short, T2long, Ashort, Along)
    @assert all(==(length(args[1])), length.(args))

    #TODO hard-coded 6
    nsignals, nsamples = nsignal(c), length(args[1])
    X = zeros(Float64, nsignals, nsamples)
    J = zeros(Float64, nsignals, 6, nsamples)
    out = zeros(Float64, 6, 1, nsamples)
    work = [_signal_model_f64_jacobian_setup(c) for _ in 1:Threads.nthreads()]
    DECAES.tforeach(1:nsamples; blocksize = 16) do j
        @inbounds begin
            f!, res, _, x, gx, cfg = work[Threads.threadid()]
            for i in 1:6; x[i] = args[i][j]; end
            @views ForwardDiff.jacobian!(res, f!, X[:,j], x, cfg)
            @views J[:,:,j] .= ForwardDiff.DiffResults.jacobian(res)
        end
    end

    return X, function (Δ)
        NNlib.batched_mul!(out, NNlib.BatchedTranspose(J), reshape(arr64(Δ), nsignals, 1, nsamples))
        return (nothing, view(out,1,1,:), view(out,2,1,:), view(out,3,1,:), view(out,4,1,:), view(out,5,1,:), view(out,6,1,:))
    end
end

function _signal_model_grad_test(phys::MaybeClosedFormBiexpEPGModel)
    alpha, refcon, T2short, T2long, Ashort, Along = θsignalmodel(phys, eachrow(sampleθprior(phys, 10))...)
    args = (alpha, refcon, T2short, T2long, Ashort, Along)

    f = (_args...) -> sum(abs2, _signal_model(phys, _args...))
    g_zygote = Zygote.gradient(f, args...)

    g_finitediff = map(enumerate(args)) do (i,x)
        g = similar(x)
        simple_fd_gradient!(g, _x -> f(ntuple(j -> j == i ? _x : args[j], 6)...), x)
        return g
    end

    map(g_zygote, g_finitediff) do g_zyg, g_fd
        g1, g2 = vec(g_zyg), vec(g_fd)
        norm(g1 - g2) / norm(g1) |> display
    end

    @btime $f($alpha, $refcon, $T2short, $T2long, $Ashort, $Along)
    @btime Zygote.gradient($f, $alpha, $refcon, $T2short, $T2long, $Ashort, $Along)

    return g_zygote, g_finitediff
end

#### CPU DECAES signal model

const MaybeDualF64 = Union{Float64, <:ForwardDiff.Dual{Nothing, Float64}}

struct BiexpEPGModelWork{T <: MaybeDualF64, ETL, A <: AbstractVector{T}, W1 <: DECAES.AbstractEPGWorkspace{T,ETL}, W2 <: DECAES.AbstractEPGWorkspace{T,ETL}}
    TE::T
    T1::T
    dc::A
    short_work::W1
    long_work::W2
end

function BiexpEPGModelWork(c::MaybeClosedFormBiexpEPGModel, ::Val{ETL} = Val(nsignal(c)), ::Type{T} = Float64) where {ETL, T <: MaybeDualF64}
    dc = DECAES.SizedVector{ETL}(zeros(T, ETL))
    short_work = DECAES.EPGdecaycurve_work(T, ETL)
    long_work = DECAES.EPGdecaycurve_work(T, ETL)
    BiexpEPGModelWork(physicsmodel(c).TE |> T, physicsmodel(c).T1 |> T, dc, short_work, long_work)
end

function _signal_model_f64!(dc::AbstractVector{T}, ::MaybeClosedFormBiexpEPGModel, work::BiexpEPGModelWork{T,ETL}, args::NTuple{6,T}) where {T <: MaybeDualF64, ETL}
    @inbounds begin
        alpha, refcon, T2short, T2long, Ashort, Along = args
        o1  = DECAES.EPGOptions{T,ETL}(alpha, work.TE, T2short, work.T1, refcon)
        o2  = DECAES.EPGOptions{T,ETL}(alpha, work.TE, T2long, work.T1, refcon)
        dc1 = DECAES.EPGdecaycurve!(work.short_work, o1) # short component
        dc2 = DECAES.EPGdecaycurve!(work.long_work, o2) # long component
        for i in 1:ETL
            dc[i] = Ashort * dc1[i] + Along * dc2[i]
        end
    end
    return dc
end
_signal_model_f64(c::MaybeClosedFormBiexpEPGModel, work::BiexpEPGModelWork{T,ETL}, args::NTuple{6,<:MaybeDualF64}) where {T, ETL} = _signal_model_f64!(work.dc, c, work, args)

function _signal_model_f64_jacobian_setup(c::MaybeClosedFormBiexpEPGModel)
    #TODO hard-coded 6
    _y, _x, _gx = zeros(Float64, nsignal(c)), zeros(Float64, 6), zeros(Float64, 6)
    res = ForwardDiff.DiffResults.JacobianResult(_y, _x)
    cfg = ForwardDiff.JacobianConfig(nothing, _y, _x, ForwardDiff.Chunk(_x))
    fwd_work = BiexpEPGModelWork(c, Val(nsignal(c)), Float64)
    jac_work = BiexpEPGModelWork(c, Val(nsignal(c)), ForwardDiff.Dual{Nothing,Float64,6})
    function f!(y, x)
        work = eltype(y) == Float64 ? fwd_work : jac_work
        x̄ = ntuple(i -> @inbounds(x[i]), 6)
        return _signal_model_f64!(y, c, work, x̄)
    end
    return f!, res, _y, _x, _gx, cfg
end

#### GPU DECAES signal model; currently slower than cpu version unless batch size is large (~10000 or more)

function _signal_model_cuda(c::MaybeClosedFormBiexpEPGModel, alpha::CUDA.CuVector, T2short::CUDA.CuVector, T2long::CUDA.CuVector, Ashort::CUDA.CuVector, Along::CUDA.CuVector)
    @unpack TE, T1, refcon = physicsmodel(c)
    return Ashort' .* DECAES.EPGdecaycurve(nsignal(c), alpha, TE, T2short, T1, refcon) .+
            Along' .* DECAES.EPGdecaycurve(nsignal(c), alpha, TE, T2long,  T1, refcon)
end

function DECAES.EPGdecaycurve(
        ETL::Int,
        flip_angles::CUDA.CuVector{T},
        TE::T,
        T2times::CUDA.CuVector{T},
        T1::T,
        refcon::T,
    ) where {T}

    @assert length(flip_angles) == length(T2times)
    nsamples = length(flip_angles)

    function epg_kernel!(decay_curve, Mz, flip_angles, T2times)
        J = CUDA.threadIdx().x + (CUDA.blockIdx().x-1) * CUDA.blockDim().x
        @inbounds if J <= nsamples
            @inbounds α, T2 = flip_angles[J], T2times[J]

            # Precompute compute element flip matrices and other intermediate variables
            E2, E1, E2_half = exp(-TE/T2), exp(-TE/T1), exp(-(TE/2)/T2)
            α2 = α * (refcon/180)
            s_α, c_α, s_½α_sq, c_½α_sq, s_α_½ = sind(α2), cosd(α2), sind(α2/2)^2, cosd(α2/2)^2, sind(α2)/2

            # Initialize magnetization phase state vector (MPSV)
            M0 = E2_half * sind(α/2) # initial population
            M1x, M1y, M1z = M0 * cosd(α/2)^2, M0 * sind(α/2)^2, im * (-M0 * sind(α)/2) # first echo population
            @inbounds decay_curve[1,J] = E2_half * M1y # first echo amplitude

            # Apply first relaxation matrix iteration on non-zero states
            @inbounds begin
                Mz[1,J] = E2 * M1y
                Mz[2,J] = 0
                Mz[3,J] = E1 * M1z
                Mz[4,J] = E2 * M1x
                Mz[5,J] = 0
                Mz[6,J] = 0
            end

            @inbounds for i = 2:ETL
                # Perform the flip for all states
                @inbounds for j in 1:3:3i-2
                    Vmx, Vmy, Vmz = Mz[j,J], Mz[j+1,J], Mz[j+2,J]
                    ms_α_Vtmp  = s_α * im * (Vmz)
                    s_α_½_Vtmp = s_α_½ * im * (Vmy - Vmx)
                    Mz[j,J]   = c_½α_sq * Vmx + (s_½α_sq * Vmy - ms_α_Vtmp)
                    Mz[j+1,J] = s_½α_sq * Vmx + (c_½α_sq * Vmy +  ms_α_Vtmp)
                    Mz[j+2,J] = c_α * Vmz + s_α_½_Vtmp
                end

                # Zero out next elements
                @inbounds if i+1 < ETL
                    j = 3i+1
                    Mz[j,J]   = 0
                    Mz[j+1,J] = 0
                    Mz[j+2,J] = 0
                end

                # Record the magnitude of the population of F1* as the echo amplitude, allowing for relaxation
                decay_curve[i,J] = E2_half * sqrt(abs2(Mz[2,J]))

                # Allow time evolution of magnetization between pulses
                @inbounds if i < ETL
                    mprev = Mz[1,J]
                    Mz[1,J] = E2 * Mz[2,J] # F1* --> F1
                    for j in 2:3:3i-1
                        m1, m2, m3 = Mz[j+1,J], Mz[j+2,J], Mz[j+3,J]
                        mtmp  = m2
                        m0    = E2 * m3     # F(n)* --> F(n-1)*
                        m1   *= E1          # Z(n)  --> Z(n)
                        m2    = E2 * mprev  # F(n)  --> F(n+1)
                        mprev = mtmp
                        Mz[j,J], Mz[j+1,J], Mz[j+2,J] = m0, m1, m2
                    end
                end
            end
        end
    end

    function configurator(kernel)
        # See: https://github.com/JuliaGPU/CUDA.jl/blob/463a41295bfede5125c584e6be9c51a4b9074e12/examples/pairwise.jl#L88
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(nextpow(2, nsamples), config.threads)
        blocks = div(nsamples, threads, RoundUp)
        return (threads=threads, blocks=blocks)
    end

    dc = CUDA.zeros(T, ETL, nsamples)
    mz = CUDA.zeros(Complex{T}, 3*ETL, nsamples)
    CUDA.@cuda name="EPGdecaycurve!" config=configurator epg_kernel!(dc, mz, flip_angles, T2times)

    return dc
end

function _EPGdecaycurve_test(; nsamples = 1024, ETL = 48)
    flip_angles = 180f0 .+ (180f0 .- 50f0) .* CUDA.rand(Float32, nsamples)
    T2_times  = 10f-3 .+ (80f-3 .- 10f-3) .* CUDA.rand(Float32, nsamples)
    TE, T1, refcon = 8f-3, 1f0, 180f0

    gpufun(flip_angles, T2_times) = DECAES.EPGdecaycurve(ETL, flip_angles, TE, T2_times, T1, refcon)

    function cpufun(flip_angles::Vector{T}, T2_times::Vector{T}, ::Val{ETL}) where {T,ETL}
        S = zeros(Float64, ETL, nsamples)
        epg_works = [DECAES.EPGdecaycurve_work(Float64, ETL) for _ in 1:Threads.nthreads()]
        Threads.@threads for j in 1:nsamples
            @inbounds epg_work = epg_works[Threads.threadid()]
            @inbounds decay_curve = DECAES.EPGdecaycurve!(epg_work, Float64(flip_angles[j]), Float64(TE), Float64(T2_times[j]), Float64(T1), Float64(refcon))
            @inbounds for i in 1:ETL
                S[i,j] = decay_curve[i]
            end
        end
        return convert(Matrix{T}, S)
    end

    S1 = gpufun(flip_angles, T2_times)
    S2 = cpufun(map(Flux.cpu, (flip_angles, T2_times))..., Val(ETL))
    @assert Flux.cpu(S1) ≈ S2

    flip_angles_cpu, T2_times_cpu = map(Flux.cpu, (flip_angles, T2_times))
    @btime CUDA.@sync $gpufun($flip_angles, $T2_times) # gpu timing
    @btime $cpufun($flip_angles_cpu, $T2_times_cpu, Val($ETL)) # cpu timing
    @btime CUDA.@sync Flux.gpu($cpufun(map(Flux.cpu, ($flip_angles, $T2_times))..., Val($ETL))) # gpu-to-cpu-to-gpu timing

    nothing
end

function _signal_model_test(; nsamples = 1024)
    p        = initialize!(ToyEPGModel{Float32,true}(); ntrain = 1, ntest = 1, nval = 1)
    alpha    = 180.0f0 .+ (180f0 .- 50f0) .* CUDA.rand(Float32, nsamples)
    T2short  = 10f-3 .+ (80f-3 .- 10f-3) .* CUDA.rand(Float32, nsamples)
    T2long   = 10f-3 .+ (80f-3 .- 10f-3) .* CUDA.rand(Float32, nsamples)
    Ashort   = 0.1f0 .+ (0.9f0 .- 0.1f0) .* CUDA.rand(Float32, nsamples)
    Along    = 0.1f0 .+ (0.9f0 .- 0.1f0) .* CUDA.rand(Float32, nsamples)

    S1 = _signal_model_cuda(p, alpha, T2short, T2long, Ashort, Along)
    S2 = _signal_model(p, map(Flux.cpu, (alpha, T2short, T2long, Ashort, Along))...)
    @assert Flux.cpu(S1) ≈ S2

    cpu_args = map(Flux.cpu, (alpha, T2short, T2long, Ashort, Along))
    @btime CUDA.@sync _signal_model_cuda($p, $alpha, $T2short, $T2long, $Ashort, $Along)
    @btime _signal_model($p, $(cpu_args)...)
    @btime CUDA.@sync Flux.gpu(_signal_model($p, map(Flux.cpu, ($alpha, $T2short, $T2long, $Ashort, $Along))...))

    nothing
end

####
#### Signal model
####

function signal_data(
        image::Array{T,4},
        batchsize = nothing;
        threshold# = T(opts.Threshold)
    ) where {T}
    first_echo = filter!(>(threshold), image[:,:,:,1][:])
    q1 = quantile(first_echo, 0.30)
    q2 = quantile(first_echo, 0.99)
    Is = findall(I -> q1 <= image[I,1] <= q2, CartesianIndices(size(image)[1:3]))
    Y  = image[Is, :]' |> copy
    # map(_ -> display(plot(image[rand(Is),:])), 1:5)
    # histogram(first_echo) |> p -> vline!(p, [q1, q2]) |> display
    Y ./= sum(Y; dims=1)
    return batchsize === nothing ? Y : sample_columns(Y, batchsize)
end

function theta_bounds(T = Float64; ntheta::Int)
    if ntheta == 4
        # theta_labels = ["alpha", "T2short", "dT2", "Ashort"]
        theta_lb = T[ 50.0,    8.0,    8.0, 0.0]
        theta_ub = T[180.0, 1000.0, 1000.0, 1.0]
    elseif ntheta == 5
        # theta_labels = ["alpha", "T2short", "dT2", "Ashort", "Along"]
        theta_lb = T[ 50.0,    8.0,    8.0, 0.0, 0.0]
        theta_ub = T[180.0, 1000.0, 1000.0, 1.0, 1.0]
    else
        error("Number of labels must be 4 or 5")
    end
    theta_bd = collect(zip(theta_lb, theta_ub))
    @assert ntheta == length(theta_bd)
    return theta_bd
end
theta_sampler(args...; kwargs...) = broadcast(bound -> bound[1] + (bound[2]-bound[1]) * rand(typeof(bound[1])), theta_bounds(args...; kwargs...))

function signal_theta_error(theta, thetahat)
    dtheta = (x -> x[2] - x[1]).(theta_bounds(eltype(theta); ntheta = size(theta,1)))
    return abs.((theta .- thetahat)) ./ dtheta
end

noise_model!(buffer::AbstractVector{T}, signal::AbstractVector{T}, ϵ::AbstractVector{T}) where {T} = (randn!(buffer); buffer .*= ϵ .* signal[1]; buffer)
noise_model(signal::AbstractVector, ϵ::AbstractVector) = ϵ .* signal[1] .* randn(eltype(signal), length(signal))

function mutate_signal(Y::AbstractVecOrMat; meanmutations::Int = 0)
    if meanmutations <= 0
        return Y
    end
    nrow = size(Y, 1)
    p = meanmutations / nrow
    return Y .* (rand(size(Y)...) .> p)
end

####
#### Direct data samplers from prior derived from MLE fitted signals
####

function make_mle_data_samplers(
        imagepath,
        thetaspath;
        ntheta::Int,
        plothist = false,
        padtrain = false,
        normalizesignals = true,
        filteroutliers = false,
    )
    @assert !(padtrain && !normalizesignals) "unnormalized padded training data is not implemented"

    # Set random seed for consistent train/test/val sets
    rng = Random.seed!(0)

    # Load + preprocess fit results (~25% of voxels dropped)
    fits = deepcopy(BSON.load(thetaspath)["results"])

    if filteroutliers
        println("before filter: $(nrow(fits))")
        filter!(r -> !(999.99 <= r.dT2 && 999.99 <= r.T2short), fits) # drop boundary failures
        filter!(r -> r.dT2 <= 999.99, fits) # drop boundary failures
        filter!(r -> r.T2short <= 100, fits) # drop long T2short (very few points)
        filter!(r -> 8.01 <= r.T2short, fits) # drop boundary failures
        filter!(r -> 8.01 <= r.dT2, fits) # drop boundary failures
        if ntheta == 5
            filter!(r -> 0.005 <= r.Ashort <= 0.15, fits) # drop outlier fits (very few points)
            filter!(r -> 0.005 <= r.Along <= 0.15, fits) # drop outlier fits (very few points)
        end
        filter!(r -> r.loss <= -250, fits) # drop poor fits (very few points)
        println("after filter:  $(nrow(fits))")
    end

    # Shuffle data + collect thetas
    fits = fits[shuffle(MersenneTwister(0), 1:nrow(fits)), :]
    thetas = ntheta == 4 ? # Create ntheta x nSamples matrix
        permutedims(convert(Matrix{Float64}, fits[:, [:alpha, :T2short, :dT2, :Ashort]])) :
        permutedims(convert(Matrix{Float64}, fits[:, [:alpha, :T2short, :dT2, :Ashort, :Along]]))

    # Load image, keeping signals which correspond to thetas
    image = DECAES.load_image(imagepath) # load 4D MatrixSize x nTE image
    Y = convert(Matrix{Float64}, permutedims(image[CartesianIndex.(fits[!, :index]), :])) # convert to nTE x nSamples Matrix

    if normalizesignals
        # Normalize signals individually such that each signal has unit sum
        Y ./= sum(Y; dims = 1)
    else
        # Don't normalize scales individually, but nevertheless scale the signals uniformly down to avoid numerical difficulties
        Y ./= 1e6
        Ysum = sum(Y; dims = 1)

        # Scale thetas and fit results for using unnormalized Y
        thetas[4:4, :] .*= Ysum # scale Ashort
        fits.Ashort .*= vec(Ysum)
        if ntheta == 5
            thetas[5:5, :] .*= Ysum # scale Along
            fits.Along .*= vec(Ysum)
        end
        fits.logsigma .= log.(exp.(fits.logsigma) .* vec(Ysum)) # scale sigma from fit results
        fits.rmse .*= vec(Ysum) # scale rmse from fit results
        fits.loss .+= size(Y,1) .* log.(vec(Ysum)) # scale mle loss from fit results
    end

    # Forward simulation params
    signal_work = signal_model_work(Float64; nTE = 48)
    signal_fun(θ::AbstractMatrix{Float64}, noise::Union{AbstractVector{Float64}, Nothing} = nothing; kwargs...) =
        signal_model!(signal_work, θ, noise; TE = 8e-3, normalize = normalizesignals, kwargs...)

    # Pad training data with thetas sampled uniformly randomly over the prior space
    local θtrain_pad
    if padtrain
        @assert normalizesignals "unnormalized padded training data is not implemented"
        θ_pad_lo, θ_pad_hi = minimum(thetas; dims = 2), maximum(thetas; dims = 2)
        θtrain_pad = θ_pad_lo .+ (θ_pad_hi .- θ_pad_lo) .* rand(MersenneTwister(0), ntheta, nrow(fits))
        Xtrain_pad = signal_fun(θtrain_pad; normalize = false)
        if ntheta == 5
            θtrain_pad[4:5, :] ./= sum(Xtrain_pad; dims = 1) # normalize Ashort, Along
            train_pad_filter   = map(Ashort -> 0.005 <= Ashort <= 0.15, θtrain_pad[4,:]) # drop outlier samples (very few points)
            train_pad_filter .&= map(Along  -> 0.005 <= Along  <= 0.15, θtrain_pad[5,:]) # drop outlier samples (very few points)
            θtrain_pad = θtrain_pad[:, train_pad_filter]
        end
        println("num padded:    $(size(θtrain_pad,2))")
    end

    # Plot prior distribution histograms
    if plothist
        theta_cols = ntheta == 4 ? [:alpha, :T2short, :dT2, :Ashort] : [:alpha, :T2short, :dT2, :Ashort, :Along]
        display(plot([histogram(fits[!,c]; lab = c, nbins = 75) for c in [theta_cols; :logsigma; :loss]]...))
    end

    # Generate data samplers
    itrain =                   1 : 2*(size(Y,2)÷4)
    itest  = 2*(size(Y,2)÷4) + 1 : 3*(size(Y,2)÷4)
    ival   = 3*(size(Y,2)÷4) + 1 : size(Y,2)

    # True data (Y) samplers
    Ytrain, Ytest, Yval = Y[:,itrain], Y[:,itest], Y[:,ival]
    function sampleY(batchsize; dataset = :train)
        dataset === :train ? (batchsize === nothing ? Ytrain : sample_columns(Ytrain, batchsize)) :
        dataset === :test  ? (batchsize === nothing ? Ytest  : sample_columns(Ytest,  batchsize)) :
        dataset === :val   ? (batchsize === nothing ? Yval   : sample_columns(Yval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Fit parameters (θ) samplers
    θtrain, θtest, θval = thetas[:,itrain], thetas[:,itest], thetas[:,ival]
    if padtrain
        θtrain = hcat(θtrain, θtrain_pad)
        θtrain = θtrain[:,shuffle(MersenneTwister(0), 1:size(θtrain,2))] # mix training + padded thetas
    end
    function sampleθ(batchsize; dataset = :train)
        dataset === :train ? (batchsize === nothing ? θtrain : sample_columns(θtrain, batchsize)) :
        dataset === :test  ? (batchsize === nothing ? θtest  : sample_columns(θtest,  batchsize)) :
        dataset === :val   ? (batchsize === nothing ? θval   : sample_columns(θval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Model data (X) samplers
    function _sampleX_model(batchsize; dataset = :train, kwargs...)
        signal_fun(sampleθ(batchsize; dataset = dataset); kwargs...)
    end

    # Direct model data (X) samplers
    Xtrain = _sampleX_model(nothing; dataset = :train)
    Xtest  = _sampleX_model(nothing; dataset = :test)
    Xval   = _sampleX_model(nothing; dataset = :val)
    function _sampleX_direct(batchsize; dataset = :train)
        dataset === :train ? (batchsize === nothing ? Xtrain : sample_columns(Xtrain, batchsize)) :
        dataset === :test  ? (batchsize === nothing ? Xtest  : sample_columns(Xtest,  batchsize)) :
        dataset === :val   ? (batchsize === nothing ? Xval   : sample_columns(Xval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Model data (X) samplers
    function sampleX(batchsize; kwargs...)
        if batchsize === nothing
            _sampleX_direct(batchsize; kwargs...)
        else
            _sampleX_model(batchsize; kwargs...)
        end
    end

    # Output train/test/val dataframe partitions
    fits_train, fits_test, fits_val = fits[itrain,:], fits[itest,:], fits[ival,:]

    # Reset random seed
    Random.seed!(rng)

    return @ntuple(sampleX, sampleY, sampleθ, fits_train, fits_test, fits_val)
end

####
#### Maximum likelihood estimation inference
####

function signal_loglikelihood_inference(
        y::AbstractVector,
        initial_guess::Union{<:AbstractVector, Nothing},
        model, # x -> (x, zero(x)),
        signal_fun; # θ -> toy_signal_model(θ, nothing, 4);
        bounds::AbstractVector{<:Tuple},
        objective::Symbol = :mle,
        bbopt_kwargs = Dict(:MaxTime => 1.0),
    )

    # Deterministic loss function, suitable for Optim
    function mle_loss(θ)
        ȳhat, ϵhat = model(signal_fun(θ))
        return -sum(logpdf.(Rician.(ȳhat, ϵhat), y))
    end

    # Stochastic loss function, only suitable for BlackBoxOptim
    function rmse_loss(θ)
        ȳhat, ϵhat = model(signal_fun(θ))
        yhat = rand.(Rician.(ȳhat, ϵhat))
        return sqrt(mean(abs2, y .- yhat))
    end

    loss = objective === :mle ? mle_loss : rmse_loss

    bbres = nothing
    if objective !== :mle || (objective === :mle && isnothing(initial_guess))
        loss_f64(θ) = convert(Vector{eltype(y)}, θ) |> loss |> Float64 # TODO bbopt expects Float64
        bbres = BlackBoxOptim.bboptimize(loss_f64;
            SearchRange = NTuple{2,Float64}.(bounds), #TODO bbopt expects Float64
            TraceMode = :silent,
            bbopt_kwargs...
        )
    end

    optres = nothing
    if objective === :mle
        θ0 = isnothing(initial_guess) ? BlackBoxOptim.best_candidate(bbres) : initial_guess #TODO bbopt returns Float64
        θ0 = convert(Vector{Float64}, θ0) #TODO fails with Float32?
        lo = convert(Vector{Float64}, (x -> x[1]).(bounds)) #TODO fails with Float32?
        hi = convert(Vector{Float64}, (x -> x[2]).(bounds)) #TODO fails with Float32?
        df = Optim.OnceDifferentiable(loss, θ0; autodiff = :forward)
        optres = Optim.optimize(df, lo, hi, θ0, Optim.Fminbox(Optim.LBFGS()))
        # optres = Optim.optimize(df, lo, hi, θ0, Optim.Fminbox(Optim.BFGS()))
        # dfc = Optim.TwiceDifferentiableConstraints(lo, hi)
        # df = Optim.TwiceDifferentiable(loss, θ0; autodiff = :forward)
        # optres = Optim.optimize(df, dfc, θ0, Optim.IPNewton())
    end

    return @ntuple(bbres, optres)
end
function signal_loglikelihood_inference(Y::AbstractMatrix, θ0::Union{<:AbstractMatrix, Nothing} = nothing, args...; kwargs...)
    _args = [deepcopy(args) for _ in 1:Threads.nthreads()]
    _kwargs = [deepcopy(kwargs) for _ in 1:Threads.nthreads()]
    tasks = map(1:size(Y,2)) do j
        Threads.@spawn begin
            tid = Threads.threadid()
            initial_guess = !isnothing(θ0) ? θ0[:,j] : nothing
            signal_loglikelihood_inference(Y[:,j], initial_guess, _args[tid]...; _kwargs[tid]...)
        end
    end
    return map(Threads.fetch, tasks)
end

#=
for _ in 1:1
    noise_level = 1e-2
    θ = toy_theta_sampler(1);
    x = toy_signal_model(θ, nothing, 4);
    y = toy_signal_model(θ, nothing, 2);
    xϵ = toy_signal_model(θ, noise_level, 4);
    yϵ = toy_signal_model(θ, noise_level, 2);

    m = x -> ((dx, ϵ) = correction_and_noiselevel(x); return (abs.(x.+dx), ϵ));

    @time bbres1, _ = signal_loglikelihood_inference(yϵ, nothing, m; objective = :rmse)[1];
    θhat1 = BlackBoxOptim.best_candidate(bbres1);
    xhat1 = toy_signal_model(θhat1, nothing, 4);
    dxhat1, ϵhat1 = correction_and_noiselevel(xhat1);
    yhat1 = corrected_signal_instance(xhat1, dxhat1, ϵhat1);

    @time bbres2, optres2 = signal_loglikelihood_inference(yϵ, nothing, m; objective = :mle)[1];
    θhat2 = Optim.minimizer(optres2); #BlackBoxOptim.best_candidate(bbres2);
    xhat2 = toy_signal_model(θhat2, nothing, 4);
    dxhat2, ϵhat2 = correction_and_noiselevel(xhat2);
    yhat2 = corrected_signal_instance(xhat2, dxhat2, ϵhat2);

    p1 = plot([y[:,1] x[:,1]]; label = ["Yθ" "Xθ"], line = (2,));
    p2 = plot([yϵ[:,1] xϵ[:,1]]; label = ["Yθϵ" "Xθϵ"], line = (2,));
    p3 = plot([yϵ[:,1] yhat1]; label = ["Yθϵ" "Ȳθϵ₁"], line = (2,));
    p4 = plot([yϵ[:,1] yhat2]; label = ["Yθϵ" "Ȳθϵ₂"], line = (2,));
    plot(p1,p2,p3,p4) |> display;

    @show toy_theta_error(θ[:,1], θhat1)';
    @show toy_theta_error(θ[:,1], θhat2)';
    @show √mean(abs2, y[:,1] .- (xhat1 .+ dxhat1));
    @show √mean(abs2, y[:,1] .- (xhat2 .+ dxhat2));
    @show √mean([mean(abs2, yϵ[:,1] .- corrected_signal_instance(xhat1, dxhat1, ϵhat1)) for _ in 1:1000]);
    @show √mean([mean(abs2, yϵ[:,1] .- corrected_signal_instance(xhat2, dxhat2, ϵhat2)) for _ in 1:1000]);
end;
=#

####
#### Toy problem MCMC inference
####

#=
Turing.@model toy_model_rician_noise(
        y,
        correction_and_noiselevel,
    ) = begin
    freq   ~ Uniform(1/64,  1/32)
    phase  ~ Uniform( 0.0,  pi/2)
    offset ~ Uniform( 0.25,  0.5)
    amp    ~ Uniform( 0.1,  0.25)
    tconst ~ Uniform(16.0, 128.0)
    # logeps ~ Uniform(-4.0,  -2.0)
    # epsilon = 10^logeps

    # Compute toy signal model without noise
    x = toy_signal_model([freq, phase, offset, amp, tconst], nothing, 4)
    yhat, ϵhat = correction_and_noiselevel(x)

    # Model noise as Rician
    for i in 1:length(y)
        # ν, σ = x[i], epsilon
        ν, σ = yhat[i], ϵhat[i]
        y[i] ~ Rician(ν, σ)
    end
end
=#

function toy_theta_mcmc_inference(
        y::AbstractVector,
        correction_and_noiselevel,
        callback = (y, chain) -> true,
    )
    model = function (x)
        xhat, ϵhat = correction_and_noiselevel(x)
        yhat = rand.(Rician.(xhat, ϵhat))
        return yhat
    end
    res = signal_loglikelihood_inference(y, nothing, model)
    theta0 = best_candidate(res)
    while true
        chain = sample(toy_model_rician_noise(y, correction_and_noiselevel), NUTS(), 1000; verbose = true, init_theta = theta0)
        # chain = psample(toy_model_rician_noise(y, correction_and_noiselevel), NUTS(), 1000, 3; verbose = true, init_theta = theta0)
        callback(y, chain) && return chain
    end
end
function toy_theta_mcmc_inference(Y::AbstractMatrix, args...; kwargs...)
    tasks = map(1:size(Y,2)) do j
        Threads.@spawn signal_loglikelihood_inference(Y[:,j], initial_guess, args...; kwargs...)
    end
    return map(Threads.fetch, tasks)
end

function find_cutoff(x; initfrac = 0.25, pthresh = 1e-4)
    cutoff = clamp(round(Int, initfrac * length(x)), 2, length(x))
    for i = cutoff+1:length(x)
        mu = mean(x[1:i-1])
        sig = std(x[1:i-1])
        p = ccdf(Normal(mu, sig), x[i])
        (p < pthresh) && break
        cutoff += 1
    end
    return cutoff
end

#=
for _ in 1:1
    correction_and_noiselevel = let _model = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/MMDLearning/output/2020-02-20T15:43:48.506/best-model.bson")["model"]) #deepcopy(model)
        function(x)
            out = _model(x)
            dx, logϵ = out[1:end÷2], out[end÷2+1:end]
            return abs.(x .+ dx), exp.(logϵ)
        end
    end
    signal_model = function(θhat)
        x = toy_signal_model(θhat, nothing, 4)
        xhat, ϵhat = correction_and_noiselevel(x)
        # zR = ϵhat .* randn(size(x)...)
        # zI = ϵhat .* randn(size(x)...)
        # yhat = @. sqrt((xhat + zR)^2 + zI^2)
        yhat = rand.(Rician.(xhat, ϵhat))
    end
    fitresults = function(y, c)
        θhat = map(k -> mean(c[k])[1,:mean], [:freq, :phase, :offset, :amp, :tconst])
        # ϵhat = 10^map(k -> mean(c[k])[1,:mean], [:logeps])[1]
        # yhat, ϵhat = correction_and_noiselevel(toy_signal_model(θhat, nothing, 4))
        yhat = signal_model(θhat)
        yerr = sqrt(mean(abs2, y - yhat))
        @ntuple(θhat, yhat, yerr)
    end
    plotresults = function(y, c)
        @unpack θhat, yhat, yerr = fitresults(y, c)
        display(plot(c))
        display(plot([y yhat]))
        return nothing
        # return plot(c) #|> display
        # return plot([y yhat]) #|> display
    end

    # θ = [freq, phase, offset, amp, tconst]
    # Random.seed!(0);
    noise_level = 1e-2;
    θ = toy_theta_sampler(16);
    Y = toy_signal_model(θ, noise_level, 2);

    # @time cs = toy_theta_mcmc_inference(Y, correction_and_noiselevel);
    # res = map(j -> fitresults(Y[:,j], cs[j]), 1:size(Y,2))
    # ps = map(j -> plotresults(Y[:,j], cs[j]), 1:size(Y,2))
    # θhat = reduce(hcat, map(k -> mean(c[k])[1,:mean], [:freq, :phase, :offset, :amp, :tconst]) for c in cs)
    # Yerr = sort(getfield.(res, :yerr))

    @time bbres = signal_loglikelihood_inference(Y, nothing, signal_model);
    Yerr = sort(best_fitness.(bbres))
    θhat = best_candidate.(bbres)
    Yhat = signal_model.(θhat)
    θhat = reduce(hcat, θhat)
    Yhat = reduce(hcat, Yhat)
    map(j -> display(plot([Y[:,j] Yhat[:,j]])), 1:size(Y,2))

    let
        p = plot()
        sticks!(p, Yerr; m = (:circle,4), lab = "Yerr")
        # sticks!(p, [0; diff(Yerr)]; m = (:circle,4), lab = "dYerr")
        hline!(p, [2noise_level]; lab = "2ϵ")
        vline!(p, [find_cutoff(Yerr; pthresh = 1e-4)]; lab = "cutoff", line = (:black, :dash))
        display(p)
    end

    display(θhat)
    display(θ)
    display((θ.-θhat)./θ)
end
=#

#=
for _ in 1:100
    let seed = rand(0:1000_000)
        rng = Random.seed!(seed)
        p = plot();
        Random.seed!(seed); plot!(toy_signal_model(3, nothing, 2); ylim = (0, 1.5));
        Random.seed!(seed); plot!(toy_signal_model(3, nothing, 2.5); ylim = (0, 1.5));
        display(p);
        Random.seed!(rng);
    end;
end;
=#

nothing
