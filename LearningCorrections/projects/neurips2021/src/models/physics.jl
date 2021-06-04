####
#### Physics model interface
####

abstract type PhysicsModel{T} end

# Abstract interface
physicsmodel(p::PhysicsModel) = p
Base.eltype(::PhysicsModel{T}) where {T} = T
Base.eltype(::Type{<:PhysicsModel{T}}) where {T} = T
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

Base.eltype(::AbstractMetaDataSignal{T}) where {T} = T
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
#### Biexponential EPG signal models
####

# CPMG image with default params for DECAES
@with_kw_noshow struct CPMGImage{T}
    data::Array{T,4} # Image data as (row, col, slice, echo) (Required)
    t2mapopts::DECAES.T2mapOptions{Float64} # Image properties (Required)
    t2partopts::DECAES.T2partOptions{Float64} # Analysis properties (Required)
    partitions::Dict{Symbol,Matrix{T}} = Dict()
    indices::Dict{Symbol,Vector{CartesianIndex{3}}} = Dict()
    meta::Dict{Symbol,Any} = Dict()
end
function Base.show(io::IO, img::CPMGImage{T}) where {T}
    print(io, "CPMGImage{$T}(")
    print(io, "data = "); summary(io, img.data); print(io, ", ")
    print(io, join(("$k = $v" for (k,v) in Dict(:nTE => img.t2mapopts.nTE, :TE => img.t2mapopts.TE)), ", ")); print(io, ", ")
    print(io, join(("n$k = $(size(v,2))" for (k,v) in img.partitions), ", "))
    # print(io, join(("partitions[:$k] = $(summary(v))" for (k,v) in img.partitions), ", "))
    print(io, ")")
end

nsignal(img::CPMGImage) = size(img.data, 4)
echotime(img::CPMGImage{T}) where {T} = T(img.t2mapopts.TE)
T2range(img::CPMGImage{T}) where {T} = T.(img.t2mapopts.T2Range)
T1time(img::CPMGImage{T}) where {T} = T(img.t2mapopts.T1)
refcon(img::CPMGImage{T}) where {T} = T(img.t2mapopts.RefConAngle)

function CPMGImage(info::Dict; seed::Int)
    rng = Random.seed!(seed)

    meta     = Dict{Symbol,Any}(:info => info)
    data     = convert(Array{Float32}, DECAES.load_image(joinpath(info["folder_path"], info["image_data_path"]), Val(4)))
    Imask    = findall(dropdims(all((x -> !isnan(x) && !iszero(x)).(data); dims = 4); dims = 4)) # image is masked, keep signals without zero or NaN entries
    Inonmask = findall(dropdims(any((x ->  isnan(x) ||  iszero(x)).(data); dims = 4); dims = 4)) # compliment of mask indices
    ishuffle = randperm(MersenneTwister(seed), length(Imask)) # shuffle indices before splitting to train/test/val

    res = data[Imask, :] |> vec |> sort |> unique! |> diff |> minimum # minimum difference between all sorted datapoints
    if res > 10 * eps(eltype(data)) && all(is_approx_multiple_of.(data, res)) # check for quantized images
        @assert all(is_approx_multiple_of.(data, res)) # assert quantization condition holds
        pepper_noise_level = res/10 # add noise below the resolution limit to avoid quantization issues
        data[Imask, :] .+= pepper_noise_level .* (2 .* rand(MersenneTwister(seed), Float32, length(Imask), size(data, 4)) .- 1) # pepper data with (fixed) zero-mean random noise
        @assert all(data[Imask, :] .>= res - pepper_noise_level) # signals within the mask contain no zeroes by definition, so this is just a sanity check
        @assert !all(is_approx_multiple_of.(data, res)) # assert quantization condition holds
    end
    data[Inonmask, :] .= NaN # set signals outside of mask to NaN
    data ./= maximum(data; dims = 4) # data[1:1,..]

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
        TE               = info["echotime"],
        T1               = 1.0,
        T2Range          = (info["echotime"], 1.0),
        nT2              = 40,
        Threshold        = 0.0,
        Chi2Factor       = 1.02,
        RefConAngle      = info["refcon"],
        MinRefAngle      = 90.0,
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

    return CPMGImage{Float32}(; data, t2mapopts, t2partopts, partitions, indices, meta)
end

function t2_distributions!(img::CPMGImage)
    img.meta[:decaes] = Dict{Symbol, Any}(:t2maps => Dict{Symbol,Any}(), :t2dist => Dict{Symbol,Any}(), :t2parts => Dict{Symbol,Any}())
    t2_distributions!(img, :Y => img.data)
    return img
end
function t2_distributions!(img::CPMGImage, X::P) where {P <: Pair{Symbol, <:AbstractTensor4D}}
    Xname, Xdata = X
    t2mapopts = DECAES.T2mapOptions(img.t2mapopts, MatrixSize = size(Xdata)[1:3])
    t2partopts = DECAES.T2partOptions(img.t2partopts, MatrixSize = size(Xdata)[1:3])
    res = img.meta[:decaes]
    res[:t2maps][Xname], res[:t2dist][Xname] = DECAES.T2mapSEcorr(Xdata |> cpu64, t2mapopts)
    res[:t2parts][Xname] = DECAES.T2partSEcorr(res[:t2dist][Xname], t2partopts)
    return img
end
t2_distributions!(img::CPMGImage, X::P) where {P <: Pair{Symbol, <:AbstractMatrix}} = t2_distributions!(img, X[1] => reshape(permutedims(X[2]), size(X[2],2), 1, 1, size(X[2],1)))
t2_distributions!(img::CPMGImage, Xs::Dict{Symbol, Any}) = (for (k,v) in Xs; t2_distributions!(img, k => v); end; return img)

# EPG model using image data
@with_kw struct BiexpEPGModel{T} <: PhysicsModel{T}
    n::Int # maximum number of echoes (Required)
    τ1::Union{Nothing,T} = 1.0 / 10e-3   # fixed relative T1 time (Optional; if `nothing`, will be sampled uniformly from `T1bd`)
    TEbd::NTuple{2,T} = (7e-3, 10e-3)    # T2 echo spacing (s) (only used for setting bounds on T2/TE and T1/TE)
    T2bd::NTuple{2,T} = (10e-3, 1000e-3) # min/max allowable T2 (s) (only used for setting bounds on T2/TE)
    T1bd::NTuple{2,T} = (0.8, 1.2)       # T1 relaxation bounds (s) (only used for setting bounds on T1/TE)
    θ::Dict{Symbol,Matrix{T}} = Dict()
    X::Dict{Symbol,Matrix{T}} = Dict()
    Y::Dict{Symbol,Matrix{T}} = Dict()
    images::Vector{CPMGImage{T}} = CPMGImage{T}[]
end

sampleθ(p::BiexpEPGModel, n::Union{Int, Symbol}; dataset::Symbol) = error("sampleθ not supported for EPGmodel")
sampleX(p::BiexpEPGModel, n::Union{Int, Symbol}; dataset::Symbol) = error("sampleX not supported for EPGmodel")
sampleY(p::BiexpEPGModel, n::Union{Int, Symbol}; dataset::Symbol) = _sample_data(rand(p.images).partitions, n; dataset)

function initialize!(p::BiexpEPGModel{T}; image_folders::AbstractVector{String}, seed::Int) where {T}
    image_infos = lib.load_cpmg_info.(image_folders)
    for info in image_infos
        image = CPMGImage(info; seed)
        push!(p.images, image)
    end
    return p
end

function t2_distributions!(p::BiexpEPGModel)
    for img in p.images
        t2_distributions!(img)
    end
    return p
end

# CPMG signal wrapper type
struct MetaCPMGSignal{T, P <: BiexpEPGModel{T}, I <: CPMGImage{T}, A <: AbstractArray{T}} <: AbstractMetaDataSignal{T}
    img::I
    Y::A
end
MetaCPMGSignal(::P, img::I, Y::A) where {T, P <: BiexpEPGModel{T}, I <: CPMGImage{T}, A <: AbstractArray{T}} = MetaCPMGSignal{T,P,I,A}(img, Y)
MetaCPMGSignal(Ymeta::MetaCPMGSignal{T,P,I}, Y::A) where {T, P <: BiexpEPGModel{T}, I <: CPMGImage{T}, A <: AbstractArray{T}} = MetaCPMGSignal{T,P,I,A}(Ymeta.img, Y)
Base.getindex(Ymeta::MetaCPMGSignal{T,P,I}, i...) where {T,P,I} = MetaCPMGSignal(Ymeta, getindex(Ymeta.Y, i...))

function θnuissance(p::BiexpEPGModel{T}, τ1::T = p.τ1) where {T}
    @assert τ1 !== nothing
    @unpack T1bd, TEbd = p
    logτ1lo, logτ1hi = log(T1bd[1] / TEbd[2]), log(T1bd[2] / TEbd[1])
    δ0 = (log(τ1) - logτ1lo) / (logτ1hi - logτ1lo)
    return T(δ0)
end
θnuissance(p::BiexpEPGModel{T}, img::CPMGImage{T}) where {T} = θnuissance(p, T1time(img) / echotime(img))

function θnuissance(p::BiexpEPGModel{T}, Ymeta::MetaCPMGSignal{T}) where {T}
    if nnuissance(p) == 0
        return zeros_similar(signal(Ymeta), 0, size(signal(Ymeta))[2:end]...)
    else
        return fill_similar(signal(Ymeta), θnuissance(p, Ymeta.img), 1, size(signal(Ymeta))[2:end]...)
    end
end

nsignal(p::BiexpEPGModel) = p.n
nnuissance(p::BiexpEPGModel) = Int(p.τ1 === nothing)
ntheta(p::BiexpEPGModel) = 7 + nnuissance(p)
nmodel(::BiexpEPGModel) = 10
θmarginalized(p::BiexpEPGModel, θ) = nnuissance(p) == 0 ? θ : θ[1:nmarginalized(p), ..]
θnuissance(p::BiexpEPGModel, θ) = nnuissance(p) == 0 ? zeros_similar(θ, 0, size(θ)[2:end]...) : θ[nmarginalized(p).+(1:nnuissance(p)), ..]

nlatent(p::BiexpEPGModel) = 0
hasclosedform(p::BiexpEPGModel) = false

θlabels(p::BiexpEPGModel) = [L"\alpha", L"\beta", L"\eta", L"\delta_1", L"\delta_2", L"\log\epsilon", L"\log{s}", L"\delta_0"][1:ntheta(p)]
θasciilabels(p::BiexpEPGModel) = ["alpha", "refcon", "eta", "delta1", "delta2", "logeps", "logscale", "delta0"][1:ntheta(p)]
θunits(p::BiexpEPGModel) = ["deg", "deg", "a.u.", "a.u.", "a.u.", "a.u.", "a.u.", "a.u."][1:ntheta(p)]
θlower(p::BiexpEPGModel{T}) where {T} = T[T( 90.0), T( 90.0), T(0.0), T(0.0), T(0.0), log(T(1e-5)), T(-2.5), T(0.0)][1:ntheta(p)]
θupper(p::BiexpEPGModel{T}) where {T} = T[T(180.0), T(180.0), T(1.0), T(1.0), T(1.0), log(T(1e-1)), T(+2.5), T(1.0)][1:ntheta(p)]

function sampleθprior(p::BiexpEPGModel{T}, ::Type{A}, n::Union{Int, Symbol}) where {T, A <: AbstractArray{T}}
    # Parameterize by alpha, refcon, short amplitude, relative T2 long and T2 short δs
    α    = sample_trunc_mv_normal(T(180.0), T(45.0), T(90.0), T(180.0), rand_similar(A, 1, n)) # truncated gaussian with mean at RHS, i.e. flip angle likely near 180 deg
    β    = sample_trunc_mv_normal(T(180.0), T(45.0), T(90.0), T(180.0), rand_similar(A, 1, n)) # truncated gaussian with mean at RHS, i.e. refocusing control angle likely near 180 deg
    η    = sample_trunc_mv_normal(  T(0.0),  T(0.5),  T(0.0),   T(1.0), rand_similar(A, 1, n)) # truncated gaussian with mean at LHS, i.e. short fraction likely to be small
    δ1   = sample_trunc_mv_normal(  T(0.0),  T(0.5),  T(0.0),   T(1.0), rand_similar(A, 1, n)) # truncated gaussian with mean at LHS, i.e. short T2 more likely to be small
    δ2   = sample_trunc_mv_normal(  T(1.0),  T(0.5),  T(0.0),   T(1.0), rand_similar(A, 1, n)) # truncated gaussian with mean at RHS, i.e. long T2 more likely to be large
    logϵ = log(T(1e-5)) .+ (log(T(1e-1)) .- log(T(1e-5))) .* rand_similar(A, 1, n)             # log noise level uniformly between SNR = 20 and SNR = 100
    logs = sample_trunc_mv_normal(  T(0.0),  T(0.5), T(-2.5),   T(2.5), rand_similar(A, 1, n)) # log scale factor uniformly between exp(-2.5) and exp(2.5); somewhat arbitrary
    if nnuissance(p) == 0
        return vcat(α, β, η, δ1, δ2, logϵ, logs)
    else
        δ0 = rand_similar(A, 1, n) # log relative T1 uniform (0, 1), i.e. log(T1) uniformly in [log(T1bd[1]), log(T1bd[2])]
        return vcat(α, β, η, δ1, δ2, logϵ, logs, δ0)
    end
end

function neglogpriors(::BiexpEPGModel, θ::AbstractArray{T}) where {T}
    α, β, η, δ1, δ2, logϵ, logs = ntuple(i -> θ[i:i, ..], 7)
    ℓ = [
        neglogL_trunc_gaussian(α,  T(180.0), log(T(45.0)), T(90.0), T(180.0)) ;  # truncated gaussian prior log likelihood for α
        neglogL_trunc_gaussian(β,  T(180.0), log(T(45.0)), T(90.0), T(180.0)) ;  # truncated gaussian prior log likelihood for β
        neglogL_trunc_gaussian(η,    T(0.0),  log(T(0.5)),  T(0.0),   T(1.0)) ;  # truncated gaussian prior log likelihood for η
        neglogL_trunc_gaussian(δ1,   T(0.0),  log(T(0.5)),  T(0.0),   T(1.0)) ;  # truncated gaussian prior log likelihood for δ1
        neglogL_trunc_gaussian(δ2,   T(1.0),  log(T(0.5)),  T(0.0),   T(1.0)) ;  # truncated gaussian prior log likelihood for δ2
        fill_similar(logϵ, log(log(T(1e-1)) - log(T(1e-5))))                  ;  # uniform prior log likelihood for logϵ
        neglogL_trunc_gaussian(logs, T(0.0),  log(T(0.5)), T(-2.5),   T(2.5)) ;  # truncated gaussian prior log likelihood for logs
    ]                                                                            # uniform prior log likelihood for δ0 equals log(1-0) = 0 and need not be explicitly included
    return ℓ
end

neglogprior(p::BiexpEPGModel, θ::CuArray) = sum(neglogpriors(p, θ); dims = 1)

function neglogprior(::BiexpEPGModel, θ::AbstractArray{T}) where {T}
    ℓ = zeros(T, size(θ)[2:end])
    Threads.@threads for J in eachindex(ℓ)
        @inbounds ℓ[J] =
            neglogL_trunc_gaussian(θ[1,J], T(180.0), log(T(45.0)), T(90.0), T(180.0)) + # truncated gaussian prior log likelihood for α
            neglogL_trunc_gaussian(θ[2,J], T(180.0), log(T(45.0)), T(90.0), T(180.0)) + # truncated gaussian prior log likelihood for β
            neglogL_trunc_gaussian(θ[3,J],   T(0.0),  log(T(0.5)),  T(0.0),   T(1.0)) + # truncated gaussian prior log likelihood for η
            neglogL_trunc_gaussian(θ[4,J],   T(0.0),  log(T(0.5)),  T(0.0),   T(1.0)) + # truncated gaussian prior log likelihood for δ1
            neglogL_trunc_gaussian(θ[5,J],   T(1.0),  log(T(0.5)),  T(0.0),   T(1.0)) + # truncated gaussian prior log likelihood for δ2
            log(log(T(1e-1)) - log(T(1e-5)))                                          + # uniform prior log likelihood for logϵ
            neglogL_trunc_gaussian(θ[7,J], T(0.0),  log(T(0.5)), T(-2.5),   T(2.5))     # truncated gaussian prior log likelihood for logs
                                                                                        # uniform prior log likelihood for δ0 equals log(1-0) = 0 and need not be explicitly included
    end
    return reshape(ℓ, 1, :)
end

θmodel(c::BiexpEPGModel, θM::AbstractVecOrMat, θN::AbstractVecOrMat) = θmodel(c, ntuple(i -> θM[i,:], size(θM,1))..., ntuple(i -> θN[i,:], size(θN,1))...)
θmodel(c::BiexpEPGModel, θ::AbstractVecOrMat) = θmodel(c, ntuple(i -> θ[i,:], size(θ,1))...)

function θmodel(c::BiexpEPGModel, α, β, η, δ1, δ2, logϵ, logs, δ0 = nothing)
    # Parameterize by alpha, refcon, short amplitude, relative T2 long and T2 short δs
    @unpack T1bd, TEbd, T2bd = physicsmodel(c)
    if δ0 === nothing
        # δ0 is not a nuissance variable, i.e. T1 is fixed
        @assert nnuissance(physicsmodel(c)) == 0
        δ0 = fill_similar(α, θnuissance(physicsmodel(c)))
    end
    logτ1lo, logτ1hi = log(T1bd[1] / TEbd[2]), log(T1bd[2] / TEbd[1])
    logτ2lo, logτ2hi = log(T2bd[1] / TEbd[2]), log(T2bd[2] / TEbd[1])
    alpha   = α
    refcon  = β
    Ashort  = η
    Along   = @. (1 - η)
    T2short = @. exp(logτ2lo + (logτ2hi - logτ2lo) * δ1)
    T2long  = @. exp(logτ2lo + (logτ2hi - logτ2lo) * (δ1 + δ2 * (1 - δ1)))
    T1      = @. exp(logτ1lo + (logτ1hi - logτ1lo) * δ0)
    TE      = one.(T1) # Parameters are relative to TE; set TE=1
    return alpha, refcon, T2short, T2long, Ashort, Along, T1, TE
end

function θderived(
        c::BiexpEPGModel,
        img::CPMGImage{T},
        θ::AbstractVecOrMat{T},
        SPcutoff::T = T(40e-3),
        SPwidth::T = T(20e-3),
    ) where {T}
    alpha, refcon, eta, delta1, delta2, logeps, logscale = ntuple(i -> θ[i,:], ntheta(physicsmodel(c)))
    delta0 = nnuissance(c) == 0 ? fill_similar(θ[1:1,:], θnuissance(physicsmodel(c))) : θ[end,:]
    _, _, T2short, T2long, Ashort, Along, _, _, _, _ = θmodel(c, θ)
    T2short, T2long = echotime(img) .* T2short, echotime(img) .* T2long # model params are unitless; convert to img timescale
    logT2short, logT2long = log.(T2short), log.(T2long)
    logT2bar = @. (Ashort * logT2short + Along * logT2long) / (Ashort + Along) # log of geometric mean weighted by Ashort, Along
    T2bar = @. exp(logT2bar) # geometric mean weighted by Ashort, Along
    wshort, wlong = soft_cutoff(T2short, SPcutoff, SPwidth), soft_cutoff(T2long, SPcutoff, SPwidth)
    mwf = @. 100 * (wshort * Ashort + wlong * Along) / (Ashort + Along)
    T2sgm = @. exp(wshort * Ashort * logT2short + wlong * Along * logT2long) * exp(-(wshort * Ashort + wlong * Along)) # geometric mean weighted by wshort * Ashort, wlong * Along
    T2mgm = @. exp((1 - wshort) * Ashort * logT2short + (1 - wlong) * Along * logT2long) * exp(-((1 - wshort) * Ashort + (1 - wlong) * Along)) # geometric mean weighted by (1 - wshort) * Ashort, (1 - wlong) * Along
    return (;
        alpha, refcon, eta, delta1, delta2, logeps, logscale, delta0, # inference domain params
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

#### MRI data EPG model

function signal_model(p::BiexpEPGModel, θ::AbstractVecOrMat)
    X = _signal_model(p, θ)
    return X
end

function signal_model(p::BiexpEPGModel{T}, img::CPMGImage{T}, θ::AbstractVecOrMat) where {T}
    if nnuissance(p) == 0
        θN = fill_similar(θ, θnuissance(p, img), 1, size(θ)[2:end]...)
        return signal_model(p, vcat(θ, θN))[1:nsignal(img), ..]
    else
        return signal_model(p, θ)[1:nsignal(img), ..]
    end
end
signal_model(p::BiexpEPGModel{T}, Ymeta::MetaCPMGSignal{T}, θ::AbstractVecOrMat) where {T} = signal_model(p, Ymeta.img, θ)

function noiselevel(::BiexpEPGModel, θ)
    logϵ, logs = θ[6:6, ..], θ[7:7, ..]
    return @. exp.(logϵ .+ logs)
end

function negloglikelihood(::BiexpEPGModel, Y::AbstractMatrix{T}, X::AbstractMatrix{T}, θ::AbstractMatrix{T}) where {T}
    ℓ = zeros(T, size(Y,2))
    Threads.@threads for j in 1:size(Y, 2)
        Σ = zero(T)
        logϵs = θ[6,j] + θ[7,j]
        for i in 1:size(Y,1)
            Σ += neglogL_rician(Y[i,j], X[i,j], logϵs)
        end
        ℓ[j] = Σ
    end
    return reshape(ℓ, 1, :)
end

function negloglikelihood(::BiexpEPGModel, Y::CuArray, X::CuArray, θ::CuArray)
    logϵs = θ[6:6, ..] .+ θ[7:7, ..]
    ℓ = sum(neglogL_rician.(Y, X, logϵs); dims = 1)
    return ℓ
end

function negloglikelihood(p::BiexpEPGModel, Ymeta::MetaCPMGSignal, θ::A; kwargs...) where {A <: AbstractArray}
    θ  = arr_similar(signal(Ymeta), θ)
    X  = signal_model(p, Ymeta, θ)
    X  = clamp_dim1(signal(Ymeta), X)
    ℓ  = negloglikelihood(p, signal(Ymeta), X, θ; kwargs...)
    return arr_similar(A, ℓ)
end

function posterior_state(p::BiexpEPGModel, Ymeta::MetaCPMGSignal, θ, Z)
    X = signal_model(p, Ymeta, θ)
    X = clamp_dim1(signal(Ymeta), X)
    X̂ = add_noise_instance(p, X, θ)
    ℓ = negloglikelihood(p, signal(Ymeta), X, θ)
    return (; Y = signal(Ymeta), X̂, X, θ, Z, ℓ)
end

#### Common biexponential EPG model methods

function _add_rician_noise_instance(X, ϵ = nothing, ninstances = nothing)
    (ϵ === nothing) && return X
    ϵsize = (ninstances === nothing) ? size(X) : (size(X)..., ninstances)
    ϵR = ϵ .* randn_similar(X, ϵsize)
    ϵI = ϵ .* randn_similar(X, ϵsize)
    X̂ = @. sqrt((X + ϵR)^2 + ϵI^2)
    return X̂
end

function add_noise_instance(c::BiexpEPGModel, X, θ, ninstances = nothing)
    X̂ = _add_rician_noise_instance(X, noiselevel(physicsmodel(c), θ), ninstances)
    return X̂
end

function _signal_model(c::BiexpEPGModel, θ::CuArray)
    X    = _signal_model(c, θmodel(c, θ)...)
    logs = θ[7:7, ..]
    X    = exp.(logs) .* X ./ maximum(X; dims = 1)
    return X
end

function _signal_model(c::BiexpEPGModel, θ::AbstractArray)
    X    = _signal_model(c, θmodel(c, θ)...) # multithreaded internally
    Threads.@threads for J in CartesianIndices(size(X)[2:end])
        Xmax = zero(eltype(X))
        for i in 1:size(X, 1)
            Xmax = max(X[i,J], Xmax)
        end
        logs   = θ[7,J]
        Xscale = exp(logs) / Xmax
        for i in 1:size(X, 1)
            X[i,J] *= Xscale
        end
    end
    return X
end

# Faster to compute forward/reverse pass on the CPU and convert back to GPU after... DECAES is just too fast (for typical batch sizes of ~1024, anyways)

_signal_model(c::BiexpEPGModel{T}, args::AbstractVector{T}...) where {T} = arr_similar(Matrix{T}, _signal_model_f64(c, map(cpu64, args)...))
_signal_model(c::BiexpEPGModel{T}, args::CuVector{T}...) where {T} = arr_similar(CuMatrix{T}, _signal_model_f64(c, map(cpu64, args)...))

function _signal_model_f64(c::BiexpEPGModel, alpha::AbstractVector{Float64}, refcon::AbstractVector{Float64}, T2short::AbstractVector{Float64}, T2long::AbstractVector{Float64}, Ashort::AbstractVector{Float64}, Along::AbstractVector{Float64}, T1::AbstractVector{Float64}, TE::AbstractVector{Float64})
    args = (alpha, refcon, T2short, T2long, Ashort, Along, T1, TE)
    @assert all(length.(args) .== length(args[1]))

    nsignals, nsamples = nsignal(physicsmodel(c)), length(args[1])
    X = zeros(Float64, nsignals, nsamples)
    work = [BiexpEPGModelWork(c) for _ in 1:Threads.nthreads()]
    # DECAES.tforeach(1:nsamples; blocksize = 16) do j
    Threads.@threads for j in 1:nsamples
        @inbounds begin
            _signal_model_f64!(view(X,:,j), c, work[Threads.threadid()], ntuple(i -> args[i][j], length(args)))
        end
    end

    return X
end

Zygote.@adjoint function _signal_model_f64(c::BiexpEPGModel, alpha::AbstractVector{Float64}, refcon::AbstractVector{Float64}, T2short::AbstractVector{Float64}, T2long::AbstractVector{Float64}, Ashort::AbstractVector{Float64}, Along::AbstractVector{Float64}, T1::AbstractVector{Float64}, TE::AbstractVector{Float64})
    args = (alpha, refcon, T2short, T2long, Ashort, Along, T1, TE)
    @assert all(length.(args) .== length(args[1]))

    nsignals, nsamples, nargs = nsignal(c), length(args[1]), length(args)
    X = zeros(Float64, nsignals, nsamples)
    J = zeros(Float64, nsignals, nargs, nsamples)
    out = zeros(Float64, nargs, 1, nsamples)
    work = [_signal_model_f64_jacobian_setup(c) for _ in 1:Threads.nthreads()]
    # DECAES.tforeach(1:nsamples; blocksize = 16) do j
    Threads.@threads for j in 1:nsamples
        @inbounds begin
            f!, res, _, x, gx, cfg = work[Threads.threadid()]
            for i in 1:nargs; x[i] = args[i][j]; end
            @views ForwardDiff.jacobian!(res, f!, X[:,j], x, cfg)
            @views J[:,:,j] .= ForwardDiff.DiffResults.jacobian(res)
        end
    end

    return X, function (Δ)
        NNlib.batched_mul!(out, NNlib.BatchedTranspose(J), reshape(cpu64(Δ), nsignals, 1, nsamples))
        return (nothing, view(out,1,1,:), view(out,2,1,:), view(out,3,1,:), view(out,4,1,:), view(out,5,1,:), view(out,6,1,:), view(out,7,1,:), view(out,8,1,:))
    end
end

#### CPU DECAES signal model

struct BiexpEPGModelWork{T <: MaybeDualF64, ETL, A <: AbstractVector{T}, W1 <: DECAES.AbstractEPGWorkspace{T,ETL}, W2 <: DECAES.AbstractEPGWorkspace{T,ETL}}
    dc::A
    short_work::W1
    long_work::W2
end

function BiexpEPGModelWork(
        ::Type{T},
        ::Val{ETL},
        EPGWorkFactory = DECAES.EPGdecaycurve_work,
    ) where {ETL, T <: MaybeDualF64}
    short_work = EPGWorkFactory(T, ETL)
    long_work = EPGWorkFactory(T, ETL)
    dc = copy(short_work.decay_curve)
    BiexpEPGModelWork(dc, short_work, long_work)
end
BiexpEPGModelWork(::Type{T}, ETL::Int, args...) where {T} = BiexpEPGModelWork(T, Val(ETL), args...)
BiexpEPGModelWork(c::BiexpEPGModel, ::Type{T} = Float64, args...) where {T} = BiexpEPGModelWork(T, Val(nsignal(c)), args...)

function EPGVectorWorkFactory(T, ETL::Int)
    mpsv₁ = zeros(DECAES.SVector{3,T}, ETL)
    mpsv₂ = zeros(DECAES.SVector{3,T}, ETL)
    dc = zeros(T, ETL)
    DECAES.EPGWork_ReIm_DualMVector_Split{T,ETL,typeof(mpsv₁),typeof(dc)}(mpsv₁, mpsv₂, dc)
end

function _biexp_epg_model_f64!(dc::AbstractVector{T}, work::BiexpEPGModelWork{T,ETL}, args::NTuple{8,MaybeDualF64}) where {T <: MaybeDualF64, ETL}
    @inbounds begin
        alpha, refcon, T2short, T2long, Ashort, Along, T1, TE = args
        o1  = DECAES.EPGOptions(ETL, alpha, TE, T2short, T1, refcon)
        o2  = DECAES.EPGOptions(ETL, alpha, TE, T2long, T1, refcon)
        dc1 = DECAES.EPGdecaycurve!(work.short_work, o1) # short component
        dc2 = DECAES.EPGdecaycurve!(work.long_work, o2) # long component
        @simd for i in 1:ETL
            dc[i] = Ashort * dc1[i] + Along * dc2[i]
        end
    end
    return dc
end
@inline _signal_model_f64!(dc::AbstractVector{T}, ::BiexpEPGModel, work::BiexpEPGModelWork{T,ETL}, args::NTuple{N,MaybeDualF64}) where {T <: MaybeDualF64, ETL, N} = _biexp_epg_model_f64!(dc, work, args)
@inline _signal_model_f64(c::BiexpEPGModel, work::BiexpEPGModelWork{T,ETL}, args::NTuple{N,MaybeDualF64}) where {T, ETL, N} = _signal_model_f64!(work.dc, c, work, args)

function _signal_model_f64_jacobian_setup(c::BiexpEPGModel)
    nargs = nmodel(physicsmodel(c))
    _y, _x, _gx = zeros(Float64, nsignal(c)), zeros(Float64, nargs), zeros(Float64, nargs)
    res = ForwardDiff.DiffResults.JacobianResult(_y, _x)
    cfg = ForwardDiff.JacobianConfig(nothing, _y, _x, ForwardDiff.Chunk(_x))
    fwd_work = BiexpEPGModelWork(c, Float64)
    jac_work = BiexpEPGModelWork(c, ForwardDiff.Dual{Nothing,Float64,nargs})
    function f!(y, x)
        work = eltype(y) == Float64 ? fwd_work : jac_work
        x̄ = ntuple(i -> @inbounds(x[i]), nargs)
        return _signal_model_f64!(y, c, work, x̄)
    end
    return f!, res, _y, _x, _gx, cfg
end

#### GPU DECAES signal model; currently slower than cpu version unless batch size is large (~10000 or more)

function _signal_model_cuda(c::BiexpEPGModel, alpha::CuVector, T2short::CuVector, T2long::CuVector, Ashort::CuVector, Along::CuVector)
    @unpack TE, T1, refcon = physicsmodel(c)
    return Ashort' .* DECAES.EPGdecaycurve(nsignal(c), alpha, TE, T2short, T1, refcon) .+
            Along' .* DECAES.EPGdecaycurve(nsignal(c), alpha, TE, T2long,  T1, refcon)
end

function DECAES.EPGdecaycurve(
        ETL::Int,
        flip_angles::CuVector{T},
        TE::T,
        T2times::CuVector{T},
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
