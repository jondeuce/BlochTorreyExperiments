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
heightsize(x::AbstractVector) = error("heightsize undefined for vectors")
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
struct SignalChunkingProcessing <: AbstractDataProcessing end
struct PCAProcessing <: AbstractDataProcessing end
struct iLaplaceProcessing <: AbstractDataProcessing end
struct WaveletProcessing <: AbstractDataProcessing end

function prepare_data(settings::Dict)
    training_data_dicts = BSON.load.(realpath.(joinpath.(settings["data"]["train_data"], readdir(settings["data"]["train_data"]))))
    testing_data_dicts = BSON.load.(realpath.(joinpath.(settings["data"]["test_data"], readdir(settings["data"]["test_data"]))))
    
    # TODO: Filtering out bad data
    filter_high_K = (d) -> d[:btparams_dict][:K_perm] < 0.1
    filter!(filter_high_K, training_data_dicts)
    filter!(filter_high_K, testing_data_dicts)

    SNR = settings["data"]["preprocess"]["SNR"]
    training_labels = init_labels(settings, training_data_dicts)
    testing_labels = init_labels(settings, testing_data_dicts)
    @assert size(training_labels, 1) == size(testing_labels, 1)
    
    processing_type =
        settings["data"]["preprocess"]["chunk"]["apply"]    ? SignalChunkingProcessing() :
        settings["data"]["preprocess"]["ilaplace"]["apply"] ? iLaplaceProcessing() :
        settings["data"]["preprocess"]["wavelet"]["apply"]  ? WaveletProcessing() :
        error("No processing selected")
        
    training_data = init_data(processing_type, settings, training_data_dicts)
    testing_data = init_data(processing_type, settings, testing_data_dicts)
    
    if settings["data"]["preprocess"]["PCA"]["apply"] == true
        @unpack training_data, testing_data =
            init_data(PCAProcessing(), training_data, testing_data)
    end

    # Duplicate labels, if data has been duplicated
    if batchsize(testing_data) > batchsize(testing_labels)
        duplicate_data(x::AbstractMatrix, rep) = x |> z -> repeat(z, length(SNR), 1) |> z -> reshape(z, heightsize(x), :)
        duplicate_data(x::AbstractVector, rep) = x |> permutedims |> z -> repeat(z, length(SNR), 1) |> vec
        rep = batchsize(testing_data) ÷ batchsize(testing_labels)
        training_labels, testing_labels, training_data_dicts, testing_data_dicts = map(
            data -> duplicate_data(data, rep),
            (training_labels, testing_labels, training_data_dicts, testing_data_dicts))
    end

    # Shuffle data and labels
    if settings["data"]["preprocess"]["shuffle"] == true
        i_train, i_test = Random.shuffle(1:batchsize(training_data)), Random.shuffle(1:batchsize(testing_data))
        training_data, training_labels, training_data_dicts = training_data[:,:,i_train], training_labels[:,i_train], training_data_dicts[i_train]
        testing_data, testing_labels, testing_data_dicts = testing_data[:,:,i_test], testing_labels[:,i_test], testing_data_dicts[i_test]
    end
    
    # Redundancy check
    @assert heightsize(training_data) == heightsize(testing_data)
    @assert batchsize(training_data) == batchsize(training_labels)
    @assert batchsize(testing_data) == batchsize(testing_labels)

    # Compute numerical properties of labels
    labels_props = init_labels_props(settings, hcat(training_labels, testing_labels))

    # Set output type
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

function init_data(::SignalChunkingProcessing, settings::Dict, ds::AbstractVector{<:Dict})
    T, TC = Float64, ComplexF64
    VT, VC, MT, MC = Vector{T}, Vector{TC}, Matrix{T}, Matrix{TC}
    SNR       = settings["data"]["preprocess"]["SNR"] :: VT
    chunksize = settings["data"]["preprocess"]["chunk"]["size"] :: Int

    PLOT_COUNT, PLOT_LIMIT = 0, 3
    PLOT_FUN = (b, TE, nTE) -> begin
        if PLOT_COUNT < PLOT_LIMIT
            p = plot(0:TE:(nTE-1)*TE, b; line = (2,), m = (:c, :black, 3), label = "SNR = " .* string.(permutedims(SNR)))
            display(p)
            PLOT_COUNT += 1
        end
    end

    out = reduce(hcat, begin
        TE      = d[:sweepparams][:TE] :: T
        nTE     = d[:sweepparams][:nTE] :: Int
        signals = complex.(transverse.(d[:signals])) :: VC
        z       = cplx_signal(signals, nTE)[1:chunksize] :: VC
        Z       = repeat(z, 1, length(SNR))
        for j in 1:length(SNR)
            add_noise!(@views(Z[:,j]), z, SNR[j])
        end
        b       = abs.(Z ./ Z[1:1, ..]) :: MT
        PLOT_FUN(b, TE, chunksize)
        b
    end for d in ds)
    
    return reshape(out, :, 1, size(out, 2))
end

function init_data(::iLaplaceProcessing, settings::Dict, ds::AbstractVector{<:Dict})
    T, TC = Float64, ComplexF64
    VT, VC, MT, MC = Vector{T}, Vector{TC}, Matrix{T}, Matrix{TC}
    SNR     = settings["data"]["preprocess"]["SNR"] :: VT
    alpha   = settings["data"]["preprocess"]["ilaplace"]["alpha"] :: T
    T2Range = settings["data"]["preprocess"]["ilaplace"]["T2Range"] :: VT
    nT2     = settings["data"]["preprocess"]["ilaplace"]["nT2"] :: Int
    nTEs    = unique(d[:sweepparams][:nTE] for d in ds) :: Vector{Int}
    bufs    = [(A = zeros(T, nTE, nT2), B = zeros(T, nT2, nT2), x = zeros(T, nT2, length(SNR)), Z = zeros(TC, nTE, length(SNR))) for nTE in nTEs]
    bufdict = Dict(nTEs .=> bufs)

    PLOT_COUNT, PLOT_LIMIT = 0, 3
    PLOT_FUN = (b, x, TE, nTE, T2) -> begin
        if PLOT_COUNT < PLOT_LIMIT
            plot(
                plot(0:TE:(nTE-1)*TE, b; line = (2,), m = (:c, :black, 3), label = "SNR = " .* string.(permutedims(SNR))),
                plot(T2, x; xaxis = (:log10,), line = (2,), m = (:c, :black, 3), label = "SNR = " .* string.(permutedims(SNR)));
                layout = (2,1)
            ) |> display
            PLOT_COUNT += 1
        end
    end

    out = reduce(hcat, begin
        signals = complex.(transverse.(d[:signals])) :: VC
        TE      = d[:sweepparams][:TE] :: T
        nTE     = d[:sweepparams][:nTE] :: Int
        z       = cplx_signal(signals, nTE) :: VC
        Z       = bufdict[nTE].Z :: MC
        for j in 1:length(SNR)
            add_noise!(@views(Z[:,j]), z, SNR[j])
        end
        b       = abs.(Z ./ Z[1:1, ..]) :: MT
        T2      = log10range(T2Range...; length = nT2) :: VT
        x       = ilaplace!(bufdict[nTE], b, T2, TE, alpha) :: MT
        PLOT_FUN(b, x, TE, nTE, T2)
        copy(x)
    end for d in ds)
    
    return reshape(out, :, 1, size(out, 2))
end

function init_data(::WaveletProcessing, settings::Dict, ds::AbstractVector{<:Dict})
    T, TC = Float64, ComplexF64
    VT, VC, MT, MC = Vector{T}, Vector{TC}, Matrix{T}, Matrix{TC}
    nTEs     = unique(d[:sweepparams][:nTE] for d in ds) :: Vector{Int}
    bufs     = [ntuple(_ -> zeros(T, nTE), 3) for nTE in nTEs]
    bufdict  = Dict(nTEs .=> bufs)
    
    SNR      = settings["data"]["preprocess"]["SNR"] :: VT
    nterms   = settings["data"]["preprocess"]["wavelet"]["nterms"] :: Int
    TEfast   = settings["data"]["preprocess"]["peel"]["TEfast"] :: T
    TEslow   = settings["data"]["preprocess"]["peel"]["TEslow"] :: T
    peelbi   = settings["data"]["preprocess"]["peel"]["biexp"] :: Bool
    makefrac = settings["data"]["preprocess"]["peel"]["makefrac"] :: Bool
    peelper  = settings["data"]["preprocess"]["peel"]["periodic"] :: Bool
    th       = BiggestTH() # Wavelet thresholding type
    
    PLOT_COUNT, PLOT_LIMIT= 0, 1 * length(SNR)
    PLOT_FUN = (x, b, nTE; kwargs...) -> begin
        if PLOT_COUNT < PLOT_LIMIT
            plotdwt(;x = b, nchopterms = nterms, disp = true, thresh = round(0.01 * norm(b); sigdigits = 3), kwargs...)
            if peelbi && !peelper
                plot(
                    plot(x[1:2]; ylim = (0,1), leg = :none, xticks = (1:2, ["iewf", "mwf"]), line = (2, :dash), m = (4, :c, :black)),
                    plot(x[3:4]; ylim = (0,10), leg = :none, xticks = (1:2, ["T2iew/TE", "T2mw/TE"]), line = (2, :dash), m = (4, :c, :black));
                    layout = (1,2)
                ) |> display
            end
            PLOT_COUNT += 1
        end
    end

    out = reduce(hcat, begin
        signals = complex.(transverse.(d[:signals])) :: VC
        TE      = d[:sweepparams][:TE] :: T
        nTE     = d[:sweepparams][:nTE] :: Int
        Nfast = ceil(Int, -TEfast/TE * log(0.1)) :: Int
        Nslow = ceil(Int, -TEslow/TE * log(0.001)) :: Int
        slowrange = Nfast:min((3*Nslow)÷4, nTE)
        fastrange = 1:Nfast

        function process_signal(b,SNR)
            x = T[] # Output vector
            sizehint!(x, nterms + 4 * peelbi + 2 * peelper)

            # Peel off slow/fast exponentials
            if peelbi
                p = peel!(bufdict[nTE], b, slowrange, fastrange)
                b = copy(bufdict[nTE][1]) # Peeled signal
                Aslow, Afast = exp(p[1].α), exp(p[2].α)
                if makefrac
                    push!(x, Afast / (Aslow + Afast)) # Fast fraction
                    push!(x, Aslow / (Aslow + Afast)) # Slow fraction
                else
                    push!(x, Afast) # Fast magnitude
                    push!(x, Aslow) # Slow magnitude
                end
                push!(x, inv(-p[1].β)) # Slow decay rate TODO
                push!(x, inv(-p[2].β)) # Fast decay rate
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

            # Plot final signal
            PLOT_FUN(x, b, nTE; th = th, title = "final padded, SNR = $(SNR), nTE = $nTE, norm(err) = " *
                string(norm(b - ichopdwt(chopdwt(b, nterms, th)..., th)) |> x -> round(x; sigdigits=3)))
            
            w, _ = chopdwt(b, nterms, th)
            append!(x, w :: VT)

            # Return vector of transformed data
            x :: VT
        end

        z = cplx_signal(signals, nTE) :: VC
        Z = repeat(z, 1, length(SNR)) :: MC
        for j in 1:length(SNR)
            add_noise!(@views(Z[:,j]), z, SNR[j])
        end
        b = abs.(Z ./ Z[1:1, ..]) :: MT

        reduce(hcat, process_signal(b[:,j], SNR[j]) for j in 1:size(b,2))
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
function cplx_signal(z::AbstractVecOrMat{C}, nTE::Int = size(z,1) - 1) where {C <: Complex}
    n = size(z,1)
    dt = (n-1) ÷ nTE
    @assert n == 1 + dt * nTE
    return z[n - dt * (nTE-1) : dt : n, ..]
end

"""
    snr(x, n)

Signal-to-noise ratio of the signal `x` relative to the noise `n`.
"""
snr(x, n; dims = 1) = 10 .* log10.(sum(abs2, x; dims = dims) ./ sum(abs2, n; dims = dims))

"""
    noise_level(z, SNR)

Standard deviation of gaussian noise with a given `SNR` level, proportional to the first time point.
    Note: `SNR = 0` is special cased to return a noise level of zero.
"""
noise_level(z::AbstractVecOrMat{T}, SNR::Number) where {T} =
    SNR == 0 ? 0 .* z[1:1,..] : # Special case zero noise
               sqrt.(abs2.(z[1:1,..]) ./ T(10^(SNR/10))) # Same for both real and complex

"""
    add_noise(z, SNR)

Add gaussian noise with signal-to-noise ratio `SNR` proportional to the first time point.
"""
add_noise!(out::AbstractVecOrMat, z::AbstractVecOrMat, SNR) = out .= z .+ noise_level(z, SNR) .* randn(eltype(z), size(z))
add_noise!(z::AbstractVecOrMat, SNR) = add_noise!(z, z, SNR)
add_noise(z::AbstractVecOrMat, SNR) = add_noise!(copy(z), z, SNR)

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
    elseif startswith(s, "log") # Logarithm of parameter
        log10(label_fun(s[4:end], d))
    elseif startswith(s, "TE*") # Parameter relative to TE
        label_fun("TE", d) * label_fun(s[4:end], d)
    elseif endswith(s, "/TE") # Parameter relative to TE
        label_fun(s[1:end-3], d) / label_fun("TE", d)
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
