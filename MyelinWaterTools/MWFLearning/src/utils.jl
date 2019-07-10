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

Returns the length of the second-last dimension of the data `x`, unless:
    `x` is a `Matrix`, in which case 1 is returned.
    `x` is a `Vector`, in which case an error is thrown.
"""
channelsize(x::AbstractVector) = error("Channel size undefined for AbstractVector's")
channelsize(x::AbstractMatrix) = 1
channelsize(x::AbstractArray{T,N}) where {T,N} = size(x, N-1)

"""
    heightsize(x::AbstractArray)

Returns the length of the first dimension of the data `x`.
"""
heightsize(x::AbstractVector) = error("TODO: Called here")
heightsize(x::AbstractArray) = size(x, 1)

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

"""
    verify_settings
"""
function verify_settings(settings::Dict)
    # Expand vector properties in model
    for (k,v) in settings["model"]
        if v isa AbstractVector
            for i in 1:length(v)
                settings["model"][k * string(i)] = v[i]
            end
        end
    end
    return settings
end

# ---------------------------------------------------------------------------- #
# Preparing data
# ---------------------------------------------------------------------------- #

abstract type AbstractDataProcessing end
struct PCAProcessing <: AbstractDataProcessing end
struct iLaplaceProcessing <: AbstractDataProcessing end
struct WaveletProcessing <: AbstractDataProcessing end

function prepare_data(settings::Dict)
    training_data_dicts = BSON.load.(realpath.(joinpath.(settings["data"]["train_data"], readdir(settings["data"]["train_data"]))))
    testing_data_dicts = BSON.load.(realpath.(joinpath.(settings["data"]["test_data"], readdir(settings["data"]["test_data"]))))

    training_labels = init_labels(settings, training_data_dicts)
    testing_labels = init_labels(settings, testing_data_dicts)
    @assert size(training_labels, 1) == size(testing_labels, 1)
    
    processing_type =
        settings["data"]["preprocess"]["ilaplace"]["apply"] ? iLaplaceProcessing() :
        settings["data"]["preprocess"]["wavelet"]["apply"] ? WaveletProcessing() :
        error("No processing selected")
        
    training_data = init_data(processing_type, settings, training_data_dicts)
    testing_data = init_data(processing_type, settings, testing_data_dicts)
    
    if settings["data"]["preprocess"]["PCA"]["apply"] == true
        @unpack training_data, testing_data =
            init_data(PCAProcessing(), training_data, testing_data)
    end

    # Redundancy check
    @assert size(training_data, 1) == size(testing_data, 1)

    # Compute numerical properties of labels
    labels_props = init_labels_props(settings, hcat(training_labels, testing_labels))

    T  = settings["prec"] == 32 ? Float32 : Float64
    VT = Vector{T}
    training_data, testing_data, training_labels, testing_labels = map(
        x -> to_float_type_T(T, x),
        (training_data, testing_data, training_labels, testing_labels))
    
    # Set "auto" fields
    (settings["data"]["height"]    == "auto") && (settings["data"]["height"]    = heightsize(training_data) :: Int)
    (settings["data"]["test_size"] == "auto") && (settings["data"]["test_size"] = batchsize(testing_data) :: Int)
    (settings["model"]["scale"]    == "auto") && (settings["model"]["scale"]    = convert(VT, labels_props[:width]) :: VT)
    (settings["model"]["offset"]   == "auto") && (settings["model"]["offset"]   = convert(VT, labels_props[:mean]) :: VT)

    return @dict(
        training_data_dicts, testing_data_dicts,
        training_data, testing_data,
        training_labels, testing_labels,
        labels_props)
end
prepare_data(settings_file::String) = prepare_data(TOML.parsefile(settings_file))

# ---------------------------------------------------------------------------- #
# Initialize data
# ---------------------------------------------------------------------------- #

function init_data(::iLaplaceProcessing, settings::Dict, ds::AbstractVector{<:Dict})
    T, VT, VC = Float64, Vector{Float64}, Vector{ComplexF64}
    alpha   = settings["data"]["preprocess"]["ilaplace"]["alpha"] :: T
    T2Range = settings["data"]["preprocess"]["ilaplace"]["T2Range"] :: VT
    nT2     = settings["data"]["preprocess"]["ilaplace"]["nT2"] :: Int
    nTEs    = unique(d[:sweepparams][:nTE] for d in ds) :: Vector{Int}
    bufs    = [(A = zeros(T, nTE, nT2), B = zeros(T, nT2, nT2), x = zeros(T, nT2)) for nTE in nTEs]
    bufdict = Dict(nTEs .=> bufs)

    out = reduce(hcat, begin
        signals = complex.(transverse.(d[:signals])) :: VC
        TE      = d[:sweepparams][:TE] :: T
        nTE     = d[:sweepparams][:nTE] :: Int
        b       = init_signal(signals, nTE) :: VT
        T2      = log10range(T2Range...; length = nT2) :: VT
        x       = ilaplace!(bufdict[nTE], b, T2, TE, alpha) :: VT
        copy(x)
    end for d in ds)
    
    return reshape(out, :, 1, size(out, 2))
end

function init_data(::WaveletProcessing, settings::Dict, ds::AbstractVector{<:Dict})
    T, VT, VC = Float64, Vector{Float64}, Vector{ComplexF64}
    nTEs     = unique(d[:sweepparams][:nTE] for d in ds) :: Vector{Int}
    bufs     = [ntuple(_ -> zeros(T, nTE), 3) for nTE in nTEs]
    bufdict  = Dict(nTEs .=> bufs)

    nterms   = settings["data"]["preprocess"]["wavelet"]["nterms"] :: Int
    TEfast   = settings["data"]["preprocess"]["peel"]["TEfast"] :: T
    peelbi   = settings["data"]["preprocess"]["peel"]["biexp"] :: Bool
    makefrac = settings["data"]["preprocess"]["peel"]["makefrac"] :: Bool
    peelper  = settings["data"]["preprocess"]["peel"]["periodic"] :: Bool
    
    PLOT_COUNT = 0
    PLOT_LIMIT = 3
    PLOT_FUN = (b, nTE; kwargs...) -> begin
        if PLOT_COUNT < PLOT_LIMIT
            plotdwt(;
                x = b,
                nchopterms = nterms,
                thresh = round(0.01 * norm(b); sigdigits = 3),
                kwargs...)
            PLOT_COUNT += 1
        end
    end

    out = reduce(hcat, begin
        signals = d[:signals] :: VC
        TE      = d[:sweepparams][:TE] :: T
        nTE     = d[:sweepparams][:nTE] :: Int
        Ncutoff = ceil(Int, -TEfast/TE * log(1e-3)) :: Int
        b       = init_signal(signals, nTE) :: VT
        
        x = T[] # Output vector
        sizehint!(x, nterms + 4 * peelbi + 2 * peelper)

        # Peel off slow/fast exponentials
        if peelbi
            p = peel!(bufdict[nTE], b, Ncutoff, Ncutoff)
            b = copy(bufdict[nTE][1]) # Peeled signal
            Aslow, Afast = exp(p[1].α), exp(p[2].α)
            if makefrac
                push!(x, Afast / (Aslow + Afast)) # Fast fraction
                push!(x, Aslow / (Aslow + Afast)) # Slow fraction
            else
                push!(x, Afast) # Fast magnitude
                push!(x, Aslow) # Slow magnitude
            end
            push!(x, TE * inv(-p[1].β)) # Slow decay rate
            push!(x, TE * inv(-p[2].β)) # Fast decay rate
        end

        # Peel off linear term to force periodicity
        if peelper
            b1, bn, n = b[1], b[end], length(b)
            β = (bn - b1) / (nb - 1) # slope forces equal endpoints
            α = b1 - β # TODO subtract mean?
            f = x -> α + β*x
            b .-= (x -> α + β*x).(1:n)
            push!(x, α) # Linear term mean
            push!(x, β) # Linear term slope
        end

        # Apply wavelet transform to peeled signal
        if maxtransformlevels(b) < 2
            npad = 4 - mod(length(b), 4) # pad to a multiple of 4
            append!(b, fill(b[end], npad))
        end
        th = BiggestTH()
        w, _ = chopdwt(b, nterms, th)
        append!(x, w :: VT)

        # Plot final signal
        PLOT_FUN(b, nTE; th = th,
            title = "final padded, nTE = $nTE, norm(err) = " * string(
                norm(b - ichopdwt(chopdwt(b, nterms, th)..., th)) |> x -> round(x; sigdigits=3)
            ))
        
        # Return vector of transformed data
        x :: VT
    end for d in ds)
    
    return reshape(out, :, 1, size(out, 2))
end

function init_data(::PCAProcessing, training_data, testing_data)
    training_data = reshape(training_data, :, batchsize(training_data))
    testing_data = reshape(testing_data, :, batchsize(testing_data))

    MVS = MultivariateStats
    M = MVS.fit(MVS.PCA, training_data; maxoutdim = size(training_data, 1))
    training_data = MVS.transform(M, training_data)
    testing_data = MVS.transform(M, testing_data)

    training_data = reshape(training_data, :, 1, batchsize(training_data))
    testing_data = reshape(testing_data, :, 1, batchsize(testing_data))

    return @dict(training_data, testing_data)
end

"""
Normalize input complex signal data `z`.
Assume that `z` is sampled every `TE/n` seconds for some positive integer `n`.
The output is the magnitude of the last `nTE` points sampled at a multiple of `TE`.
to have the first point equal to 1.
"""
function init_signal(z::AbstractVecOrMat{C}, nTE::Int = size(z,1) - 1) where {C <: Complex}
    n = size(z,1)
    dt = (n-1) ÷ nTE
    @assert n == 1 + dt * nTE
    x = abs.(z[n - dt*(nTE-1) : dt : n, ..])
    x ./= x[1:1, ..]
    return x
end

# ---------------------------------------------------------------------------- #
# Initialize labels
# ---------------------------------------------------------------------------- #

function label_fun(s::String, d::Dict)::Float64
    if s == "mwf" # myelin (small pool) water fraction
        d[:mwfvalues][:exact]
    elseif s == "iewf" # intra/extra-cellular (large pool/axonal + tissue) water fraction
        1 - d[:mwfvalues][:exact]
    elseif s == "ewf" # extra-cellular (tissue) water fraction
        1 - d[:btparams_dict][:AxonPDensity]
    elseif s == "iwf" # intra-cellular (large pool/axonal) water fraction
        d[:btparams_dict][:AxonPDensity] - d[:mwfvalues][:exact]
    elseif s == "T2mw" || s == "T2sp" # myelin-water (small pool) T2
        inv(d[:btparams_dict][:R2_sp])
    elseif s == "T2iw" || s == "T2lp" || s == "T2ax" # intra-cellular (large pool/axonal) T2
        inv(d[:btparams_dict][:R2_lp])
    elseif s == "T2ew" # extra-cellular (tissue) T2
        inv(d[:btparams_dict][:R2_Tissue])
    elseif s == "T2iew" # inverse of area-averaged R2 for intra/extra-cellular (large pool/axonal + tissue) water
        @unpack R2_lp, R2_Tissue = d[:btparams_dict] # R2 values
        iwf, ewf = label_fun("iwf", d), label_fun("ewf", d) # area fractions
        R2iew = (iwf * R2_lp + ewf * R2_Tissue) / (iwf + ewf) # area-weighted average
        inv(R2iew) # R2iew -> T2iew
    elseif s == "T2av" # inverse of area-averaged R2 for whole domain
        @unpack R2_lp, R2_sp, R2_Tissue = d[:btparams_dict] # R2 values
        iwf, mwf, ewf = label_fun("iwf", d), label_fun("mwf", d), label_fun("ewf", d) # area fractions
        R2av = (iwf * R2_lp + mwf * R2_sp + ewf * R2_Tissue) # area-weighted average
        inv(R2av) # R2av -> T2av
    elseif s == "Dav" # area-averaged D-coeff for whole domain
        @unpack D_Axon, D_Sheath, D_Tissue = d[:btparams_dict] # D values
        iwf, mwf, ewf = label_fun("iwf", d), label_fun("mwf", d), label_fun("ewf", d) # area fractions
        Dav = (iwf * D_Axon + mwf * D_Sheath + ewf * D_Tissue) # area-weighted average
    elseif s == "logK" # Logarithm of permeability coefficient
        @unpack K_perm = d[:btparams_dict] # K value
        logK = log10(K_perm)
    else
        k = Symbol(s)
        if k ∈ keys(d[:btparams_dict]) # from BlochTorreyParameters
            d[:btparams_dict][k]
        elseif k ∈ keys(d[:sweepparams]) # from sweep parameters
            d[:sweepparams][k]
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

function init_labels_props(settings::Dict, labels::AbstractMatrix)
    props = Dict{Symbol, Vector{Float64}}(
        :max => vec(maximum(labels; dims = 2)),
        :min => vec(minimum(labels; dims = 2)),
        :mean => vec(mean(labels; dims = 2)),
        :med => vec(median(labels; dims = 2)),
    )
    props[:width] = props[:max] - props[:min]
    return props
end
