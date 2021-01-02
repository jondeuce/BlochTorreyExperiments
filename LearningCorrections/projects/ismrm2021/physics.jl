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
@inline _maybe_vcat(X, Z = nothing) = (Z === nothing) ? X : vcat(X,Z)
@inline _split_delta_epsilon(δ_logϵ) = δ_logϵ[1:end÷2, :], exp.(δ_logϵ[end÷2+1:end, :]) .+ sqrt(eps(eltype(δ_logϵ)))
@inline function _add_rician_noise_instance(X, ϵ = nothing, ninstances = nothing)
    (ϵ === nothing) && return X
    ϵsize = (ninstances === nothing) ? size(X) : (size(X)..., ninstances)
    ϵR = ϵ .* randn_similar(X, ϵsize)
    ϵI = ϵ .* randn_similar(X, ϵsize)
    X̂ = @. sqrt((X + ϵR)^2 + ϵI^2)
    return X̂
end

# Concrete methods to extract δ and ϵ
correction_and_noiselevel(G::VectorRicianCorrector, X, Z = nothing) = _split_delta_epsilon(generator(G)(_maybe_vcat(X,Z)))
correction_and_noiselevel(G::FixedNoiseVectorRicianCorrector, X, Z = nothing) = generator(G)(_maybe_vcat(X,Z)), G.ϵ0
correction_and_noiselevel(G::LatentVectorRicianCorrector, X, Z) = _split_delta_epsilon(generator(G)(Z))
correction_and_noiselevel(G::LatentVectorRicianNoiseCorrector, X, Z) = zero(X), exp.(generator(G)(Z)) .+ sqrt(eps(eltype(X)))
correction_and_noiselevel(G::LatentScalarRicianNoiseCorrector, X, Z) = zero(X), exp.(generator(G)(Z)) .* ones_similar(X, nsignal(G)) .+ sqrt(eps(eltype(X)))
function correction_and_noiselevel(G::NormalizedRicianCorrector, X, Z = nothing)
    δ, ϵ = correction_and_noiselevel(corrector(G), X, Z)
    (ϵ !== nothing) && (G.noisescale !== nothing) && (ϵ = ϵ .* G.noisescale(X))
    return δ, ϵ
end

# Derived convenience functions
correction(G::RicianCorrector, X, Z = nothing) = correction_and_noiselevel(G, X, Z)[1]
noiselevel(G::RicianCorrector, X, Z = nothing) = correction_and_noiselevel(G, X, Z)[2]
corrected_signal_instance(G::RicianCorrector, X, Z = nothing) = corrected_signal_instance(G, X, correction_and_noiselevel(G, X, Z)...)
corrected_signal_instance(G::RicianCorrector, X, δ, ϵ) = add_noise_instance(G, add_correction(G, X, δ), ϵ)
add_correction(::RicianCorrector, X, δ) = @. abs(X + δ)
add_noise_instance(::RicianCorrector, X, ϵ, ninstances = nothing) = _add_rician_noise_instance(X, ϵ, ninstances)
function add_noise_instance(G::NormalizedRicianCorrector, X, ϵ, ninstances = nothing)
    # X is assumed properly normalized, and ϵ is assumed relative to G.noisescale (i.e. output from correction_and_noiselevel); just add noise, then normalize X̂
    X̂ = add_noise_instance(corrector(G), X, ϵ, ninstances)
    (G.normalizer !== nothing) && (X̂ = X̂ ./ G.normalizer(X̂))
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
function signal_model end
function noiselevel end

####
#### Abstract interface with marginalized parameters
####

# Required
sampleθprior(p::PhysicsModel, ::Type{A}, ::Union{Int, Symbol}) where {T, A <: AbstractArray{T}} = error("sampleθprior not implemented for $(typeof(p))")
θlabels(p::PhysicsModel) = error("θlabels not implemented for $(typeof(p))")
θasciilabels(p::PhysicsModel) = error("θasciilabels not implemented for $(typeof(p))")
θunits(p::PhysicsModel) = error("θunits not implemented for $(typeof(p))")
θlower(p::PhysicsModel) = error("θlower not implemented for $(typeof(p))")
θupper(p::PhysicsModel) = error("θupper not implemented for $(typeof(p))")

# Fallbacks: assume no nuissance parameters by default
nnuissance(::PhysicsModel) = 0
nmarginalized(p::PhysicsModel) = ntheta(p) - nnuissance(p)
nmodel(p::PhysicsModel) = ntheta(p)
θmarginalized(::PhysicsModel, θ) = θ
θnuissance(::PhysicsModel, θ) = similar(θ, 0, size(θ)[2:end]...)
θderived(::PhysicsModel, θ) = θ
θmodel(p::PhysicsModel, θ) = θmodel(p, θmarginalized(p, θ), θnuissance(p, θ))
θmodel(::PhysicsModel, θM, θN) = θM
θsufficient(::PhysicsModel, θ0) = θ0

for m in [:model, :derived]
    for f in [:θlabels, :θasciilabels, :θunits, :θlower, :θupper]
        fnew = Symbol(replace(String(f), "θ" => "θ$m"))
        @eval $fnew(p::PhysicsModel, args...) = $f(p, args...)
    end
end

# Defaults
sampleθprior(p::PhysicsModel{T}, n::Union{Int, Symbol}) where {T} = sampleθprior(p, Matrix{T}, n) # default to sampling θ on cpu
sampleθprior(p::PhysicsModel, Y::AbstractArray, n::Union{Int, Symbol} = size(Y,2)) = sampleθprior(p, typeof(Y), n) # θ type is similar to Y type

for m in [Symbol(), :model, :derived]
    fbounds, ferror, flower, fupper = Symbol(:θ, m, :bounds), Symbol(:θ, m, :error), Symbol(:θ, m, :lower), Symbol(:θ, m, :upper)
    @eval $fbounds(p::PhysicsModel) = tuple.($flower(p), $fupper(p)) # bounds are tuples of (lower, upper)
    @eval $ferror(p::PhysicsModel, θtrue, θfit) = 100 .* (θtrue .- θfit) ./ arr_similar(θtrue, $fupper(p) .- $flower(p)) # default error metric is elementwise percentage relative to prior width
end

####
#### Signal wrapped in meta data
####

abstract type AbstractMetaDataSignal{T} end

signal(Ymeta::AbstractMetaDataSignal) = Ymeta.Y # fallback
nsignal(Ymeta::AbstractMetaDataSignal) = size(signal(Ymeta), 1) # fallback
θnuissance(Ymeta::AbstractMetaDataSignal) = zeros_similar(signal(Ymeta), 0, size(signal(Ymeta))[2:end]...) # fallback

# Default wrapper type
struct MetaSignal{T, P <: PhysicsModel{T}, A <: AbstractArray{T}} <: AbstractMetaDataSignal{T}
    Y::A
end
MetaSignal(::P, Y::A) where {T, P <: PhysicsModel{T}, A <: AbstractArray{T}} = MetaSignal{T,P,A}(Y)
Base.getindex(Ymeta::MetaSignal{T,P}, i...) where {T,P} = (Ynew = getindex(Ymeta.Y, i...); MetaSignal{T,P,typeof(Ynew)}(Ynew))

####
#### Abstact toy problem interface
####

abstract type AbstractToyModel{T,isfinite} <: PhysicsModel{T,isfinite} end

const ClosedFormAbstractToyModel{T,isfinite} = ClosedForm{<:AbstractToyModel{T,isfinite}}
const MaybeClosedFormAbstractToyModel{T,isfinite} = Union{<:AbstractToyModel{T,isfinite}, <:ClosedFormAbstractToyModel{T,isfinite}}

# Default samplers for models with data stored in `θ`, `X`, `Y` fields
_sample_data(d::Dict, n::Union{Int, Symbol}; dataset::Symbol) = n === :all ? d[dataset] : MMDLearning.sample_columns(d[dataset], n)
sampleθ(p::MaybeClosedForm, n::Union{Int, Symbol};              dataset::Symbol) = _sample_data(physicsmodel(p).θ, n; dataset)
sampleX(p::MaybeClosedForm, n::Union{Int, Symbol}, ϵ = nothing; dataset::Symbol) = _sample_data(physicsmodel(p).X, n; dataset)
sampleY(p::MaybeClosedForm, n::Union{Int, Symbol}, ϵ = nothing; dataset::Symbol) = _sample_data(physicsmodel(p).Y, n; dataset)

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
    !hasfield(typeof(p), :images) && return p
    for img in p.images
        img.meta[:histograms] = Dict{Symbol, Any}()
        img.meta[:histograms][:train] = signal_histograms(img.partitions[:train]; edges = nothing, nbins, normalize)
        train_edges = Dict([k => v.edges[1] for (k,v) in img.meta[:histograms][:train]])
        for dataset in (:val, :test)
            img.meta[:histograms][dataset] = signal_histograms(img.partitions[dataset]; edges = train_edges, nbins = nothing, normalize)
        end
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

# CPMG image with default params for DECAES
@with_kw struct CPMGImage{T}
    data::Array{T,4} # Image data as (row, col, slice, echo) (Required)
    t2mapopts::DECAES.T2mapOptions{Float64} # Image properties (Required)
    t2partopts::DECAES.T2partOptions{Float64} # Analysis properties (Required)
    partitions::Dict{Symbol,Matrix{T}} = Dict()
    indices::Dict{Symbol,Vector{CartesianIndex{3}}} = Dict()
    meta::Dict{Symbol,Any} = Dict()
end

nsignal(img::CPMGImage) = size(img.data, 4)
echotime(img::CPMGImage{T}) where {T} = T(img.t2mapopts.TE)
T2range(img::CPMGImage{T}) where {T} = T.(img.t2mapopts.T2Range)
T1time(img::CPMGImage{T}) where {T} = T(img.t2mapopts.T1)
refcon(img::CPMGImage{T}) where {T} = T(img.t2mapopts.RefConAngle)

function CPMGImage(info::NamedTuple; seed::Int)
    rng = Random.seed!(seed)

    data     = convert(Array{Float32}, DECAES.load_image(info.path, Val(4)))
    Imask    = findall(dropdims(all((x -> !isnan(x) && !iszero(x)).(data); dims = 4); dims = 4)) # image is masked, keep signals without zero or NaN entries
    Inonmask = findall(dropdims(any((x ->  isnan(x) ||  iszero(x)).(data); dims = 4); dims = 4)) # compliment of mask indices
    ishuffle = randperm(MersenneTwister(seed), length(Imask)) # shuffle indices before splitting to train/test/val

    data[Inonmask, :] .= NaN # set signals outside of mask to NaN
    data ./= maximum(data; dims = 4) # data[1:1,..] #TODO: normalize by mean? sum? maximum? first echo?

    indices = Dict{Symbol,Vector{CartesianIndex{3}}}(
        :mask  => Imask, # non-shuffled mask indices
        :train => Imask[ishuffle[            1 : 2*(end÷4)]], # first half for training
        :test  => Imask[ishuffle[2*(end÷4) + 1 : 3*(end÷4)]], # third quarter held out for testing
        :val   => Imask[ishuffle[3*(end÷4) + 1 : end]], # fourth quarter for validation
    )

    partitions = Dict{Symbol,Matrix{Float32}}(
        :mask  => permutedims(data[indices[:mask], :]),
        :train => permutedims(data[indices[:train], :]),
        :test  => permutedims(data[indices[:test], :]),
        :val   => permutedims(data[indices[:val], :]),
    )

    t2mapopts = DECAES.T2mapOptions{Float64}(
        MatrixSize       = size(data)[1:3],
        nTE              = size(data)[4],
        TE               = info.TE,
        T1               = 1.0,
        T2Range          = (info.TE, 1.0),
        nT2              = 40,
        Threshold        = 0.0,
        Chi2Factor       = 1.02,
        RefConAngle      = info.refcon,
        MinRefAngle      = 90.0, #TODO
        nRefAnglesMin    = 8,
        nRefAngles       = 8,
        Reg              = "chi2",
        SaveResidualNorm = true,
        SaveDecayCurve   = true,
        SaveRegParam     = true,
        Silent           = true,
    )

    t2partopts = DECAES.T2partOptions{Float64}(
        MatrixSize = t2mapopts.MatrixSize,
        nT2        = t2mapopts.nT2,
        T2Range    = t2mapopts.T2Range,
        SPWin      = (prevfloat(t2mapopts.T2Range[1]), 40e-3),
        MPWin      = (nextfloat(40e-3), nextfloat(t2mapopts.T2Range[2])),
        Sigmoid    = 20e-3, # soft cut-off
        Silent     = true,
    )

    Random.seed!(rng)

    return CPMGImage{Float32}(; data, t2mapopts, t2partopts, partitions, indices)
end

function t2_distributions!(img::CPMGImage)
    img.meta[:decaes] = Dict{Symbol, Any}()
    img.meta[:decaes][:t2maps], img.meta[:decaes][:t2dist], img.meta[:decaes][:t2parts] = (Dict{Symbol,Any}() for _ in 1:3)
    img.meta[:decaes][:t2maps][:Y], img.meta[:decaes][:t2dist][:Y] = DECAES.T2mapSEcorr(img.data |> arr64, img.t2mapopts)
    img.meta[:decaes][:t2parts][:Y] = DECAES.T2partSEcorr(img.meta[:decaes][:t2dist][:Y], img.t2partopts)
    return img
end
function t2_distributions!(img::CPMGImage, X::P) where {P <: Pair{Symbol, <:AbstractTensor4D}}
    Xname, Xdata = X
    t2mapopts = DECAES.T2mapOptions(img.t2mapopts, MatrixSize = size(Xdata)[1:3])
    t2partopts = DECAES.T2partOptions(img.t2partopts, MatrixSize = size(Xdata)[1:3])
    img.meta[:decaes][:t2maps][Xname], img.meta[:decaes][:t2dist][Xname] = DECAES.T2mapSEcorr(Xdata |> arr64, t2mapopts)
    img.meta[:decaes][:t2parts][Xname] = DECAES.T2partSEcorr(img.meta[:decaes][:t2dist][Xname], t2partopts)
    return img
end
t2_distributions!(img::CPMGImage, X::P) where {P <: Pair{Symbol, <:AbstractMatrix}} = t2_distributions!(img, X[1] => reshape(permutedims(X[2]), size(X[2],2), 1, 1, size(X[2],1)))
t2_distributions!(img::CPMGImage, Xs::Dict{Symbol, Any}) = (for (k,v) in Xs; t2_distributions!(img, k => v); end; return img)

abstract type AbstractToyEPGModel{T,isfinite} <: AbstractToyModel{T,isfinite} end

# Toy EPG model with latent variable controlling noise amplitude
@with_kw struct ToyEPGModel{T,isfinite} <: AbstractToyEPGModel{T,isfinite}
    n::Int # maximum number of echoes (Required)
    T1bd::NTuple{2,T} = (0.8, 1.2) # T1 relaxation bounds (s)
    TEbd::NTuple{2,T} = (7e-3, 10e-3) # T2 echo spacing (s)
    T2bd::NTuple{2,T} = TEbd .* (1, 200) # min/max allowable T2 (s)
    ϵbd::NTuple{2,T} = (0.001, 0.01) # noise bound
    θ::Dict{Symbol,Matrix{T}} = Dict()
    X::Dict{Symbol,Matrix{T}} = Dict()
    Y::Dict{Symbol,Matrix{T}} = Dict()
end

# EPG model using image data
@with_kw struct EPGModel{T,isfinite} <: PhysicsModel{T,isfinite}
    n::Int # maximum number of echoes (Required)
    T1bd::NTuple{2,T} = (0.8, 1.2) # T1 relaxation bounds (s)
    TEbd::NTuple{2,T} = (7e-3, 10e-3) # T2 echo spacing (s)
    T2bd::NTuple{2,T} = (10e-3, 1000e-3) # min/max allowable T2 (s)
    θ::Dict{Symbol,Matrix{T}} = Dict()
    X::Dict{Symbol,Matrix{T}} = Dict()
    Y::Dict{Symbol,Matrix{T}} = Dict()
    images::Vector{CPMGImage{T}} = CPMGImage{T}[]
end

sampleθ(p::EPGModel, n::Union{Int, Symbol};              dataset::Symbol) = error("sampleθ not supported for EPGmodel")
sampleX(p::EPGModel, n::Union{Int, Symbol}, ϵ = nothing; dataset::Symbol) = error("sampleX not supported for EPGmodel")
sampleY(p::EPGModel, n::Union{Int, Symbol}, ϵ = nothing; dataset::Symbol) = _sample_data(rand(p.images).partitions, n; dataset)

function initialize!(p::EPGModel{T,isfinite}; image_infos::AbstractVector{<:NamedTuple}, seed::Int) where {T,isfinite}
    @assert !isfinite #TODO
    for info in image_infos
        image = CPMGImage(info; seed)
        # t2_distributions!(image) #TODO
        push!(p.images, image)
    end
    #= TODO @assert !isfinite
    for d in (:train, :test, :val)
        p.Y[d] = mapreduce(image -> image.partitions[d], hcat, p.images)
        isfinite ? (p.θ[d] = sampleθprior(p, size(p.Y[d], 2))) : empty!(p.θ)
        isfinite ? (p.X[d] = signal_model(p, p.θ[d])) : empty!(p.X)
    end
    =#
    empty!.((p.θ, p.X, p.Y))
    signal_histograms!(p)
    return p
end

const ClosedFormToyEPGModel{T,isfinite} = ClosedForm{ToyEPGModel{T,isfinite}}
const MaybeClosedFormToyEPGModel{T,isfinite} = Union{ToyEPGModel{T,isfinite}, ClosedFormToyEPGModel{T,isfinite}}

const BiexpEPGModel{T,isfinite} = Union{<:ToyEPGModel{T,isfinite}, <:EPGModel{T,isfinite}}
const ClosedFormBiexpEPGModel{T,isfinite} = ClosedFormToyEPGModel{T,isfinite} # EPGModel has no closed form
const MaybeClosedFormBiexpEPGModel{T,isfinite} = Union{<:BiexpEPGModel{T,isfinite}, <:ClosedFormBiexpEPGModel{T,isfinite}}

# CPMG signal wrapper type
struct MetaCPMGSignal{T, P <: BiexpEPGModel{T}, I <: CPMGImage{T}, A <: AbstractArray{T}} <: AbstractMetaDataSignal{T}
    img::I
    Y::A
end
MetaCPMGSignal(::P, img::I, Y::A) where {T, P <: BiexpEPGModel{T}, I <: CPMGImage{T}, A <: AbstractArray{T}} = MetaCPMGSignal{T,P,I,A}(img, Y)
Base.getindex(Ymeta::MetaCPMGSignal{T,P,I}, i...) where {T,P,I} = (Ynew = getindex(Ymeta.Y, i...); MetaCPMGSignal{T,P,I,typeof(Ynew)}(Ymeta.img, Ynew))

function θnuissance(p::BiexpEPGModel, Ymeta::MetaCPMGSignal)
    @unpack T1bd, TEbd = p
    logτ1lo, logτ1hi = log(T1bd[1] / TEbd[2]), log(T1bd[2] / TEbd[1])
    TE, T1 = echotime(Ymeta.img), T1time(Ymeta.img)
    δ0 = (log(T1 / TE) - logτ1lo) / (logτ1hi - logτ1lo) |> eltype(signal(Ymeta))
    fill_similar(signal(Ymeta), δ0, 1, size(signal(Ymeta))[2:end]...)
end

nsignal(p::BiexpEPGModel) = p.n
ntheta(p::BiexpEPGModel) = 6
nnuissance(::BiexpEPGModel) = 1
nmodel(::BiexpEPGModel) = 8
θmarginalized(::BiexpEPGModel, θ) = θ[1:end-1,..]
θnuissance(::BiexpEPGModel, θ) = θ[end:end,..]

nlatent(p::ToyEPGModel) = 1
nlatent(p::EPGModel) = 0
hasclosedform(p::ToyEPGModel) = true
hasclosedform(p::EPGModel) = false

θlabels(::BiexpEPGModel) = [L"\alpha", L"\beta", L"\eta", L"\delta_1", L"\delta_2", L"\delta_0"]
θasciilabels(::BiexpEPGModel) = ["alpha", "refcon", "eta", "delta1", "delta2", "delta0"]
θunits(::BiexpEPGModel) = ["deg", "deg", "a.u.", "a.u.", "a.u.", "a.u."]
θlower(p::BiexpEPGModel{T}) where {T} = T[T( 90.0), T( 90.0), T(0.0), T(0.0), T(0.0), T(0.0)]
θupper(p::BiexpEPGModel{T}) where {T} = T[T(180.0), T(180.0), T(1.0), T(1.0), T(1.0), T(1.0)]

θmodel(c::MaybeClosedFormBiexpEPGModel, θM::AbstractVecOrMat, θN::AbstractVecOrMat) = θmodel(c, ntuple(i -> θM[i,:], nmarginalized(c))..., ntuple(i -> θN[i,:], nnuissance(c))...)
θmodel(c::MaybeClosedFormBiexpEPGModel, θ::AbstractVecOrMat) = θmodel(c, ntuple(i -> θ[i,:], ntheta(physicsmodel(c)))...)

function θmodel(c::MaybeClosedFormBiexpEPGModel, α, β, η, δ1, δ2, δ0)
    # Parameterize by alpha, refcon, short amplitude, relative T2 long and T2 short δs
    @unpack T1bd, TEbd, T2bd = physicsmodel(c)
    logτ1lo, logτ1hi = log(T1bd[1] / TEbd[2]), log(T1bd[2] / TEbd[1])
    logτ2lo, logτ2hi = log(T2bd[1] / TEbd[2]), log(T2bd[2] / TEbd[1])
    alpha, refcon = α, β
    Ashort, Along = η, 1 .- η
    T2short = @. exp(logτ2lo + (logτ2hi - logτ2lo) * δ1)
    T2long = @. exp(logτ2lo + (logτ2hi - logτ2lo) * (δ1 + δ2 * (1 - δ1)))
    T1 = @. exp(logτ1lo + (logτ1hi - logτ1lo) * δ0)
    TE = one.(T1) # Parameters are relative to TE; set TE=1
    return alpha, refcon, T2short, T2long, Ashort, Along, T1, TE
end

θsufficient(c::MaybeClosedFormBiexpEPGModel, θ0::AbstractVecOrMat) = θsufficient(c, ntuple(i -> θ0[i,:], nmodel(physicsmodel(c)))...)

function θsufficient(c::MaybeClosedFormBiexpEPGModel{T}, alpha, refcon, T2short, T2long, Ashort, Along, T1, TE) where {T}
    logτ1lo, logτ1hi = log(T1bd[1] / TEbd[2]), log(T1bd[2] / TEbd[1])
    logτ2lo, logτ2hi = log(T2bd[1] / TEbd[2]), log(T2bd[2] / TEbd[1])
    α, β = alpha, refcon
    η  = @. Ashort / (Ashort + Along)
    δ1 = @. clamp((log(T2short / TE) - logτ2lo) / (logτ2hi - logτ2lo), zero(T), one(T))
    δ2 = @. clamp(((log(T2long / TE) - logτ2lo) / (logτ2hi - logτ2lo) - δ1) / max(1 - δ1, eps(T)), zero(T), one(T))
    δ0 = @. clamp((log(T1 / TE) - logτ1lo) / (logτ1hi - logτ1lo), zero(T), one(T))
    return α, β, η, δ1, δ2, δ0
end

function sampleθprior(p::BiexpEPGModel{T}, ::Type{A}, n::Union{Int, Symbol}) where {T, A <: AbstractArray{T}}
    # Parameterize by alpha, refcon, short amplitude, relative T2 long and T2 short δs
    αlo, βlo, ηlo, δ1lo, δ2lo, δ0lo = θlower(p)
    αhi, βhi, ηhi, δ1hi, δ2hi, δ0hi = θupper(p)
    α  = αlo .+ (αhi .- αlo) .* (x -> (1-exp(-3x))/(1-exp(T(-3)))).(rand_similar(A, 1, n)) # concave triangular distbn on (αlo, αhi); encourages less near αlo, more near αhi
    β  = βlo .+ (βhi .- βlo) .* (x -> (1-exp(-3x))/(1-exp(T(-3)))).(rand_similar(A, 1, n)) # concave triangular distbn on (βlo, βhi); encourages less near βlo, more near βhi
    η  = ((x,y) -> y < T(1/2) ? x : sqrt(x)).(rand_similar(A, 1, n), rand_similar(A, 1, n)) # union of uniform and triangular distbns on (0, 1)
    δ1 = ((x,y) -> y < T(1/2) ? x : sqrt(x/4)).(rand_similar(A, 1, n), rand_similar(A, 1, n)) # union of uniform on (0, 1) and triangular distbns on (0, 1/2)
    δ2 = rand_similar(A, 1, n) # uniform distbn on (0, 1)
    δ0 = rand_similar(A, 1, n) # uniform distbn on (0, 1)
    return vcat(α, β, η, δ1, δ2, δ0)
end

function θderived(
        c::MaybeClosedFormBiexpEPGModel,
        img::CPMGImage{T},
        θ::AbstractVecOrMat{T},
        SPcutoff::T = T(40e-3),
        SPwidth::T = T(20e-3),
    ) where {T}
    alpha, refcon, eta, delta1, delta2, delta0 = ntuple(i -> θ[i,:], ntheta(physicsmodel(c)))
    _, _, T2short, T2long, Ashort, Along, _, _ = θmodel(c, θ)
    T2short, T2long = echotime(img) .* T2short, echotime(img) .* T2long # model params are unitless; convert to img timescale
    logT2short, logT2long = log.(T2short), log.(T2long)
    logT2bar = @. (Ashort * logT2short + Along * logT2long) / (Ashort + Along) # log of geometric mean weighted by Ashort, Along
    T2bar = @. exp(logT2bar) # geometric mean weighted by Ashort, Along
    wshort, wlong = MMDLearning.soft_cutoff(T2short, SPcutoff, SPwidth), MMDLearning.soft_cutoff(T2long, SPcutoff, SPwidth)
    mwf = @. 100 * (wshort * Ashort + wlong * Along) / (Ashort + Along)
    T2sgm = @. exp(wshort * Ashort * logT2short + wlong * Along * logT2long) * exp(-(wshort * Ashort + wlong * Along)) # geometric mean weighted by wshort * Ashort, wlong * Along
    T2mgm = @. exp((1 - wshort) * Ashort * logT2short + (1 - wlong) * Along * logT2long) * exp(-((1 - wshort) * Ashort + (1 - wlong) * Along)) # geometric mean weighted by (1 - wshort) * Ashort, (1 - wlong) * Along
    return (;
        alpha, refcon, eta, delta1, delta2, delta0, # inference domain params
        T2short, T2long, Ashort, Along, # signal model params (without repeated alpha, refcon)
        logT2short, logT2long, logT2bar, T2bar, T2sgm, T2mgm, mwf, # misc. derived params
    )
end

θmodelunits(::BiexpEPGModel) = ["deg", "deg", "s", "s", "a.u.", "a.u.", "s", "s"]
θmodellabels(::BiexpEPGModel) = [L"\alpha", L"\beta", L"T_{2,short}", L"T_{2,long}", L"A_{short}", L"A_{long}", L"T_1", L"TE"]
θmodelbounds(p::BiexpEPGModel{T}) where {T} = NTuple{2,T}[θbounds(p)[1], θbounds(p)[2], (p.T2bd[1], T(0.1)), p.T2bd, (T(0.0), T(1.0)), (T(0.0), T(1.0)), p.T1bd, p.TEbd]

θderivedunits(p::BiexpEPGModel) = [θunits(p); θmodelunits(p)[3:end-2]; "log(s)"; "log(s)"; "log(s)"; "s"; "s"; "s"; "%"]
θderivedlabels(p::BiexpEPGModel) = [θlabels(p); θmodellabels(p)[3:end-2]; L"\log T_{2,short}"; L"\log T_{2,long}"; L"\log \bar{T}_2"; L"\bar{T}_2"; L"T_{2,SGM}"; L"T_{2,MGM}"; L"MWF"]
θderivedbounds(p::BiexpEPGModel{T}) where {T} = NTuple{2,T}[θbounds(p); θmodelbounds(p)[3:end-2]; log.(p.T2bd); log.(p.T2bd); log.(p.T2bd); (p.T2bd[1], T(0.25)); (p.T2bd[1], T(0.1)); p.T2bd; (T(0.0), T(40.0))]

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
    ϵ = (W === nothing) ? nothing : X[1:1,:] .* (p.ϵbd[1] .+ W .* (p.ϵbd[2] .- p.ϵbd[1])) # noise with amplitude relative to first echo
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
    X = _signal_model(c, θmodel(c, θ)...)
    X = X ./ maximum(X; dims = 1) #X[1:1,:] #TODO: normalize by mean? sum? maximum? first echo?
    return X
end

# Faster to compute forward/reverse pass on the CPU and convert back to GPU after... DECAES is just too fast (for typical batch sizes of ~1024, anyways)

_signal_model(c::MaybeClosedFormBiexpEPGModel{T}, args::AbstractVector{T}...) where {T} = arr_similar(Matrix{T}, _signal_model_f64(c, map(arr64, args)...))
_signal_model(c::MaybeClosedFormBiexpEPGModel{T}, args::CUDA.CuVector{T}...) where {T} = arr_similar(CUDA.CuMatrix{T}, _signal_model_f64(c, map(arr64, args)...))

function _signal_model_f64(c::MaybeClosedFormBiexpEPGModel, alpha::AbstractVector{Float64}, refcon::AbstractVector{Float64}, T2short::AbstractVector{Float64}, T2long::AbstractVector{Float64}, Ashort::AbstractVector{Float64}, Along::AbstractVector{Float64}, T1::AbstractVector{Float64}, TE::AbstractVector{Float64})
    args = (alpha, refcon, T2short, T2long, Ashort, Along, T1, TE)
    @assert length(args) == nmodel(physicsmodel(c))
    @assert all(==(length(args[1])), length.(args))

    nsignals, nsamples = nsignal(physicsmodel(c)), length(args[1])
    X = zeros(Float64, nsignals, nsamples)
    work = [BiexpEPGModelWork(c) for _ in 1:Threads.nthreads()]
    DECAES.tforeach(1:nsamples; blocksize = 16) do j
        @inbounds begin
            _signal_model_f64!(view(X,:,j), c, work[Threads.threadid()], ntuple(i -> args[i][j], length(args)))
        end
    end

    return X
end

Zygote.@adjoint function _signal_model_f64(c::MaybeClosedFormBiexpEPGModel, alpha::AbstractVector{Float64}, refcon::AbstractVector{Float64}, T2short::AbstractVector{Float64}, T2long::AbstractVector{Float64}, Ashort::AbstractVector{Float64}, Along::AbstractVector{Float64}, T1::AbstractVector{Float64}, TE::AbstractVector{Float64})
    args = (alpha, refcon, T2short, T2long, Ashort, Along, T1, TE)
    @assert length(args) == nmodel(physicsmodel(c))
    @assert all(==(length(args[1])), length.(args))

    nsignals, nsamples, nargs = nsignal(c), length(args[1]), length(args)
    X = zeros(Float64, nsignals, nsamples)
    J = zeros(Float64, nsignals, nargs, nsamples)
    out = zeros(Float64, nargs, 1, nsamples)
    work = [_signal_model_f64_jacobian_setup(c) for _ in 1:Threads.nthreads()]
    DECAES.tforeach(1:nsamples; blocksize = 16) do j
        @inbounds begin
            f!, res, _, x, gx, cfg = work[Threads.threadid()]
            for i in 1:nargs; x[i] = args[i][j]; end
            @views ForwardDiff.jacobian!(res, f!, X[:,j], x, cfg)
            @views J[:,:,j] .= ForwardDiff.DiffResults.jacobian(res)
        end
    end

    return X, function (Δ)
        NNlib.batched_mul!(out, NNlib.BatchedTranspose(J), reshape(arr64(Δ), nsignals, 1, nsamples))
        return (nothing, view(out,1,1,:), view(out,2,1,:), view(out,3,1,:), view(out,4,1,:), view(out,5,1,:), view(out,6,1,:), view(out,7,1,:), view(out,8,1,:))
    end
end

function _signal_model_grad_test(phys::MaybeClosedFormBiexpEPGModel)
    alpha, refcon, T2short, T2long, Ashort, Along, T1, TE = θmodel(phys, eachrow(sampleθprior(phys, 10))...)
    args = (alpha, refcon, T2short, T2long, Ashort, Along, T1, TE)

    f = (_args...) -> sum(abs2, _signal_model(phys, _args...))
    g_zygote = Zygote.gradient(f, args...)

    g_finitediff = map(enumerate(args)) do (i,x)
        g = similar(x)
        simple_fd_gradient!(g, _x -> f(ntuple(j -> j == i ? _x : args[j], length(args))...), x)
        return g
    end

    map(g_zygote, g_finitediff) do g_zyg, g_fd
        g1, g2 = vec(g_zyg), vec(g_fd)
        norm(g1 - g2) / norm(g1) |> display
    end

    @btime $f($alpha, $refcon, $T2short, $T2long, $Ashort, $Along, $T1, $TE)
    @btime Zygote.gradient($f, $alpha, $refcon, $T2short, $T2long, $Ashort, $Along, $T1, $TE)

    return g_zygote, g_finitediff
end

#### CPU DECAES signal model

const MaybeDualF64 = Union{Float64, <:ForwardDiff.Dual{Nothing, Float64}}

struct BiexpEPGModelWork{T <: MaybeDualF64, ETL, A <: AbstractVector{T}, W1 <: DECAES.AbstractEPGWorkspace{T,ETL}, W2 <: DECAES.AbstractEPGWorkspace{T,ETL}}
    dc::A
    short_work::W1
    long_work::W2
end

function BiexpEPGModelWork(c::MaybeClosedFormBiexpEPGModel, ::Val{ETL} = Val(nsignal(c)), ::Type{T} = Float64) where {ETL, T <: MaybeDualF64}
    dc = DECAES.SizedVector{ETL}(zeros(T, ETL))
    short_work = DECAES.EPGdecaycurve_work(T, ETL)
    long_work = DECAES.EPGdecaycurve_work(T, ETL)
    BiexpEPGModelWork(dc, short_work, long_work)
end

function _signal_model_f64!(dc::AbstractVector{T}, c::MaybeClosedFormBiexpEPGModel, work::BiexpEPGModelWork{T,ETL}, args::NTuple{NARGS,T}) where {T <: MaybeDualF64, ETL, NARGS}
    @assert NARGS == nmodel(physicsmodel(c))
    @inbounds begin
        alpha, refcon, T2short, T2long, Ashort, Along, T1, TE = args
        o1  = DECAES.EPGOptions{T,ETL}(alpha, TE, T2short, T1, refcon)
        o2  = DECAES.EPGOptions{T,ETL}(alpha, TE, T2long, T1, refcon)
        dc1 = DECAES.EPGdecaycurve!(work.short_work, o1) # short component
        dc2 = DECAES.EPGdecaycurve!(work.long_work, o2) # long component
        for i in 1:ETL
            dc[i] = Ashort * dc1[i] + Along * dc2[i]
        end
    end
    return dc
end
_signal_model_f64(c::MaybeClosedFormBiexpEPGModel, work::BiexpEPGModelWork{T,ETL}, args::NTuple{NARGS,<:MaybeDualF64}) where {T, ETL, NARGS} = _signal_model_f64!(work.dc, c, work, args)

function _signal_model_f64_jacobian_setup(c::MaybeClosedFormBiexpEPGModel)
    nargs = nmodel(physicsmodel(c))
    _y, _x, _gx = zeros(Float64, nsignal(c)), zeros(Float64, nargs), zeros(Float64, nargs)
    res = ForwardDiff.DiffResults.JacobianResult(_y, _x)
    cfg = ForwardDiff.JacobianConfig(nothing, _y, _x, ForwardDiff.Chunk(_x))
    fwd_work = BiexpEPGModelWork(c, Val(nsignal(c)), Float64)
    jac_work = BiexpEPGModelWork(c, Val(nsignal(c)), ForwardDiff.Dual{Nothing,Float64,nargs})
    function f!(y, x)
        work = eltype(y) == Float64 ? fwd_work : jac_work
        x̄ = ntuple(i -> @inbounds(x[i]), nargs)
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

nothing
