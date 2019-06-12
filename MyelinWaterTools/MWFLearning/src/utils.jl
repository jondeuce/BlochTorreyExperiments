# ---------------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------------- #

"""
    batchsize(x::AbstractArray)

Returns the length of the last dimension of the data `x`, unless `x` is a `Vector`
in which case 1 is returned.
"""
batchsize(x::AbstractVector) = 1
batchsize(x::AbstractArray{T,N}) where {T,N} = size(x, N)

"""
    channelsize(x::AbstractArray)

Returns the length of the second-last dimension of the data `x`.
If `x` is a `Matrix`, 1 is returned.
If `x` is a `Vector`, an error is thrown.
"""
channelsize(x::AbstractVector) = error("Channel size undefined for AbstractVector's")
channelsize(x::AbstractMatrix) = 1
channelsize(x::AbstractArray{T,N}) where {T,N} = size(x, N-1)

"""
    DenseResize()

Non-learnable layer which resizes input arguments `x` to be a matrix with batchsize(x) columns.
"""
struct DenseResize end
Flux.@treelike DenseResize
(l::DenseResize)(x::AbstractArray) = reshape(x, :, batchsize(x))
Base.show(io::IO, l::DenseResize) = print(io, "DenseResize()")

"""
    DenseResize(s::AbstractArray)

Non-learnable layer which scales input `x` by array `s`
"""
struct Scale{V}
    s::V
end
Scale(s::Flux.TrackedArray) = Scale(Flux.data(s)) # Layer is not learnable
Flux.@treelike Scale
(l::Scale)(x::AbstractArray) = x .* l.s
Base.show(io::IO, l::Scale) = print(io, "Scale(", length(l.s), ")")

"""
    log10range(a, b; length = 10)

Returns a `length`-element vector with log-linearly spaced data
between `a` and `b`
"""
log10range(a, b; length = 10) = 10 .^ range(log10(a), log10(b); length = length)

"""
    to_float_type_T(T, x)

Convert a number or collection `x` to have floating point type `T`.
"""
to_float_type_T(T, x) = map(T, x) # fallback
to_float_type_T(T, x::Number) = T(x)
to_float_type_T(T, x::AbstractVector) = convert(Vector{T}, x)
to_float_type_T(T, x::AbstractMatrix) = convert(Matrix{T}, x)
to_float_type_T(T, x::AbstractVector{C}) where {C <: Complex} = convert(Vector{Complex{T}}, x)
to_float_type_T(T, x::AbstractMatrix{C}) where {C <: Complex} = convert(Matrix{Complex{T}}, x)

function prepare_data(settings::Dict, model_settings = settings["model"])
    training_data_dicts = BSON.load.(joinpath.(settings["data"]["train_data"], readdir(settings["data"]["train_data"])))
    testing_data_dicts = BSON.load.(joinpath.(settings["data"]["test_data"], readdir(settings["data"]["test_data"])))

    training_labels = init_labels(settings, training_data_dicts)
    testing_labels = init_labels(settings, testing_data_dicts)
    @assert size(training_labels, 1) == size(testing_labels, 1)
    
    training_data = init_data(settings, training_data_dicts)
    testing_data = init_data(settings, testing_data_dicts)
    
    if settings["data"]["PCA"] == true
        training_data = reshape(training_data, :, batchsize(training_data))
        testing_data = reshape(testing_data, :, batchsize(testing_data))
        
        MVS = MultivariateStats
        M = MVS.fit(MVS.PCA, training_data; maxoutdim = size(training_data, 1))
        training_data = MVS.transform(M, training_data)
        testing_data = MVS.transform(M, testing_data)
        
        training_data = reshape(training_data, :, 1, batchsize(training_data))
        testing_data = reshape(testing_data, :, 1, batchsize(testing_data))
    end

    @assert size(training_data, 1) == size(testing_data, 1)

    labels_scale = init_labels_scale(settings, hcat(training_labels, testing_labels))
    # training_labels .*= labels_scale
    # testing_labels .*= labels_scale

    T = settings["prec"] == 32 ? Float32 : Float64
    training_data, testing_data, training_labels, testing_labels = map(
        x -> to_float_type_T(T, x),
        (training_data, testing_data, training_labels, testing_labels))
    
    if settings["data"]["height"] == "auto"
        settings["data"]["height"] = size(training_data, 1) :: Int
    end

    if model_settings["scale"] == "auto"
        model_settings["scale"] = convert(Vector{T}, labels_scale)
    end

    return @dict(
        training_data_dicts, testing_data_dicts,
        training_data, testing_data,
        training_labels, testing_labels,
        labels_scale)
end
prepare_data(settings_file::String) = prepare_data(TOML.parsefile(settings_file))

function init_data(settings::Dict, ds::AbstractVector{<:Dict})
    T, VT, VC = Float64, Vector{Float64}, Vector{ComplexF64}
    alpha   = settings["data"]["alpha"] :: T
    T2Range = settings["data"]["T2Range"] :: VT
    nT2     = settings["data"]["nT2"] :: Int
    nTEs    = unique(d[:sweepparams][:nTE] for d in ds) :: Vector{Int}
    bufs    = [(A = zeros(T, nTE, nT2), B = zeros(T, nT2, nT2), y = zeros(T, nT2)) for nTE in nTEs]
    bufdict = Dict(nTEs .=> bufs)

    out = reduce(hcat, begin
        signals = d[:signals] :: VC
        TE      = d[:sweepparams][:TE] :: T
        nTE     = d[:sweepparams][:nTE] :: Int
        x       = init_signal(signals) :: VT
        T2      = log10range(T2Range...; length = nT2) :: VT
        y       = project_onto_exp!(bufdict[nTE], x, T2, TE, alpha) :: VT
        copy(y)
    end for d in ds)
    
    return reshape(out, :, 1, size(out, 2))
end

function label_fun(s::String, d::Dict)::Float64
    if s == "mwf" # myelin (small pool) water fraction
        d[:mwfvalues][:exact]
    elseif s == "iewf" # intra/extra-cellular water fraction
        1 - d[:mwfvalues][:exact]
    elseif s == "ewf" # extra-cellular (tissue) water fraction
        1 - d[:btparams_dict][:AxonPDensity]
    elseif s == "iwf" # intra-cellular (large pool) water fraction
        d[:btparams_dict][:AxonPDensity] - d[:mwfvalues][:exact]
    elseif s == "T2iew" # inverse of area-averaged R2 for intra/extra-cellular water
        iwf, ewf = label_fun("iwf", d), label_fun("ewf", d) # area fractions
        R2iew = (iwf * d[:btparams_dict][:R2_lp] + ewf * d[:btparams_dict][:R2_Tissue]) / (iwf + ewf) # area-weighted average
        inv(R2iew) # R2iew -> T2iew
    elseif s == "T2mw" # myelin-water T2
        inv(d[:btparams_dict][:R2_sp])
    else
        k = Symbol(s)
        if k ∈ keys(d[:sweepparams]) # From sweep parameters
            d[:sweepparams][k]
        elseif k ∈ keys(d[:btparams_dict]) # From BlochTorreyParameters
            d[:btparams_dict][k]
        else
            error("Unknown label: $s")
        end
    end
end

function init_labels(settings::Dict, ds::AbstractVector{<:Dict})
    label_names = settings["data"]["labels"] :: Vector{String}
    labels = zeros(Float64, length(label_names), length(ds))
    for j in 1:length(ds)
        d = ds[j] :: Dict
        for i in 1:length(label_names)
            labels[i,j] = label_fun(label_names[i], d) :: Float64
        end
    end
    return labels
end

function init_labels_scale(settings::Dict, labels::AbstractMatrix)
    if settings["model"]["scale"] == "auto"
        inv.(vec(maximum(labels; dims = 2))) :: Vector{Float64}
    else
        settings["model"]["scale"] :: Vector{Float64}
    end
end

"""
Normalize input complex signal data `z` is sampled at (0, TE, ..., nTE*TE).
The magnitude sampled at (TE, 2*TE, ..., nTE*TE), is returned, normalized
to have the first point equal to 1.
"""
function init_signal(z::AbstractVector{C}) where {C <: Complex}
    x = abs.(z[2:end])
    x ./= x[1]
    return x
end
function init_signal(z::AbstractMatrix{C}) where {C <: Complex}
    x = abs.(z[2:end, :])
    x ./= x[1, :]
    return x
end

"""
Projection onto exponentials with tikhonov regularization.
For data `b` sampled at a rate η > 0, we write `b` as a sum of exponentials
with time constants `τ`:

    b_i = Σ_j exp(-η*i/τ_j) * x_j

where x_j are the unknown weights.
Since the problem is fundamentally ill-posed, Tikhonov regularization with
parameter α is performed, i.e. the output `x` is the minimizer of

    ||Ax - b||_2^2 + α ||x||_2^2

and is given by

    x = (A'A + α^2 * I)^{-1} A'b
"""
function project_onto_exp(b::AbstractVecOrMat, τ::AbstractVector, η::Number, α::Number = 1)
    T = promote_type(eltype(b), eltype(τ), eltype(η), eltype(α))
    M, P = size(b)
    N = length(τ)
    t = η.*(1:M)
    # A = [exp(-ti/τj) for ti in t, τj in τ]
    # x = (A'A + α^2*I)\(A'b)
    bufs = (A = zeros(T, M, N), B = zeros(T, N, N), y = zeros(T, N, P))
    y = project_onto_exp!(bufs, b, τ, η, α)
    return copy(y)
end

function project_onto_exp!(bufs, b::AbstractVecOrMat, τ::AbstractVector, η::Number, α::Number = 1)
    M, P = size(b,1), size(b,2)
    N = length(τ)
    t = η.*(1:M)

    @unpack A, B, y = bufs
    @assert size(A) == (M, N) && size(B) == (N, N) &&
            size(y,1) == N && size(y,2) == size(b,2)

    @inbounds for j in 1:N
        for i in 1:M
            A[i,j] = exp(-t[i]/τ[j]) # LHS matrix
        end
    end
    mul!(y, A', b) # RHS vector

    mul!(B, A', A)
    @inbounds for j in 1:N
        B[j,j] += α^2 # Tikhonov regularization
    end

    Bf = cholesky!(B)
    ldiv!(Bf, y) # Solve inverse

    return y
end

"""
Discrete Laplace transform. For a stepsize η > 0, the discrete Laplace transform
of `x` is given by:
    L_η[x](s) = η * Σ_{k} exp(-s*k*η) * x[k]
"""
function dlt(x::AbstractVector, s::AbstractVector, η::Number)
    T = promote_type(eltype(x), eltype(s), typeof(η))
    y = zeros(T, length(s))
    @inbounds for j in eachindex(y,s)
        Σ = zero(T)
        ωj = exp(-η * s[j])
        ω = ωj
        for k in eachindex(x)
            Σ += ω * x[k]
            ω *= ωj
            # Σ += ω^k * x[k] # equivalent to above
        end
        y[j] = η * Σ
    end
    return y
end
