# ---------------------------------------------------------------------------- #
# Preparing data
# ---------------------------------------------------------------------------- #

abstract type AbstractDataProcessing end
struct SignalChunkingProcessing <: AbstractDataProcessing end
struct SignalZipperProcessing <: AbstractDataProcessing end
struct PCAProcessing <: AbstractDataProcessing end
struct iLaplaceProcessing <: AbstractDataProcessing end
struct WaveletProcessing <: AbstractDataProcessing end

function prepare_data(settings::Dict)
    training_data_dicts = BSON.load.(realpath.(joinpath.(settings["data"]["train_data"], readdir(settings["data"]["train_data"]))))
    testing_data_dicts = BSON.load.(realpath.(joinpath.(settings["data"]["test_data"], readdir(settings["data"]["test_data"]))))
    
    # Filtering out undesired data
    filter_bad_data = (d) -> all(map(
        (label, lower, upper) -> lower <= label_fun(label, d) :: Float64 <= upper,
        settings["data"]["filter"]["labels"] :: Vector{String},
        settings["data"]["filter"]["lower"]  :: Vector{Float64},
        settings["data"]["filter"]["upper"]  :: Vector{Float64}))
    filter!(filter_bad_data, training_data_dicts)
    filter!(filter_bad_data, testing_data_dicts)

    SNR = settings["data"]["preprocess"]["SNR"]
    training_thetas = init_labels(settings, training_data_dicts)
    testing_thetas = init_labels(settings, testing_data_dicts)
    @assert size(training_thetas, 1) == size(testing_thetas, 1)
    
    processing_types = AbstractDataProcessing[]
    settings["data"]["preprocess"]["chunk"]["apply"]    && push!(processing_types, SignalChunkingProcessing())
    settings["data"]["preprocess"]["zipper"]["apply"]   && push!(processing_types, SignalZipperProcessing())
    settings["data"]["preprocess"]["ilaplace"]["apply"] && push!(processing_types, iLaplaceProcessing())
    settings["data"]["preprocess"]["wavelet"]["apply"]  && push!(processing_types, WaveletProcessing())
    isempty(processing_types) && error("No processing selected")

    init_all_data(d) = cat(init_data.(processing_types, Ref(settings), Ref(d))...; dims = 3)
    training_data = init_all_data(training_data_dicts)
    testing_data = init_all_data(testing_data_dicts)
    
    if settings["data"]["preprocess"]["PCA"]["apply"] == true
        @unpack training_data, testing_data =
            init_data(PCAProcessing(), training_data, testing_data)
    end

    # Duplicate labels, if data has been duplicated
    if batchsize(testing_data) > batchsize(testing_thetas)
        duplicate_data(x::AbstractMatrix, rep) = x |> z -> repeat(z, rep, 1) |> z -> reshape(z, size(x, 1), :)
        duplicate_data(x::AbstractVector, rep) = x |> permutedims |> z -> repeat(z, rep, 1) |> vec
        rep = batchsize(testing_data) ÷ batchsize(testing_thetas)
        training_thetas, testing_thetas, training_data_dicts, testing_data_dicts = map(
            data -> duplicate_data(data, rep),
            (training_thetas, testing_thetas, training_data_dicts, testing_data_dicts))
    end

    # Shuffle data and labels
    if settings["data"]["preprocess"]["shuffle"] == true
        i_train, i_test = Random.shuffle(1:batchsize(training_data)), Random.shuffle(1:batchsize(testing_data))
        training_data, training_thetas, training_data_dicts = training_data[:,:,:,i_train], training_thetas[:,i_train], training_data_dicts[i_train]
        testing_data, testing_thetas, testing_data_dicts = testing_data[:,:,:,i_test], testing_thetas[:,i_test], testing_data_dicts[i_test]
    end
    
    # Redundancy check
    @assert heightsize(training_data) == heightsize(testing_data)
    @assert batchsize(training_data) == batchsize(training_thetas)
    @assert batchsize(testing_data) == batchsize(testing_thetas)

    # Compute numerical properties of labels
    labels_props = init_labels_props(settings, hcat(training_thetas, testing_thetas))

    # Set output type
    T  = settings["prec"] == 64 ? Float64 : Float32
    VT = Vector{T}
    training_data, testing_data, training_thetas, testing_thetas = map(
        x -> to_float_type_T(T, x),
        (training_data, testing_data, training_thetas, testing_thetas))
    
    # Set "auto" fields
    (settings["data"]["height"]    == "auto") && (settings["data"]["height"]    = heightsize(training_data) :: Int)
    (settings["data"]["test_size"] == "auto") && (settings["data"]["test_size"] = batchsize(testing_data) :: Int)
    (settings["model"]["scale"]    == "auto") && (settings["model"]["scale"]    = convert(VT, labels_props[:width]) :: VT)
    (settings["model"]["offset"]   == "auto") && (settings["model"]["offset"]   = convert(VT, labels_props[:mean]) :: VT)

    return @dict(
        training_data_dicts, testing_data_dicts,
        training_data, testing_data,
        training_thetas, testing_thetas,
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
    chunk = settings["data"]["preprocess"]["chunk"]["size"] :: Int

    PLOT_COUNT, PLOT_LIMIT = 0, 3
    PLOT_FUN = (b, TE, SNR, chunk) -> begin
        if PLOT_COUNT < PLOT_LIMIT
            p = plot(
                0:chunk-1, [b[:,cols] for cols in partition(1:size(b,2), length(SNR))];
                title = "SNR = " * string(SNR), line = (2,), m = (:c, :black, 3), leg = :none)
            display(p)
            PLOT_COUNT += 1
        end
    end

    # N_CHUNKS = 6
    # function makechunks(b, chunk)
    #     @assert 2*chunk <= length(b)
    #     n = length(b)
    #     r = round.(Int, range(1, n-chunk+1, length=4))
    #     return hcat(
    #         b[r[1]:r[1]+chunk-1], b[r[2]:r[2]+chunk-1], b[r[3]:r[3]+chunk-1], b[r[4]:r[4]+chunk-1],
    #         b[1:2:2*chunk], b[2:2:2*chunk],
    #     )
    # end

    N_CHUNKS = 1
    makechunks(b, chunk) = b[1:chunk]

    out = reduce(hcat, begin
        TE      = d[:sweepparams][:TE] :: T
        nTE     = d[:sweepparams][:nTE] :: Int
        signals = complex.(transverse.(d[:signals])) :: VC
        z       = cplx_signal(signals, nTE) :: VC
        Z       = repeat(z, 1, length(SNR))
        for j in 1:length(SNR)
            add_noise!(@views(Z[:,j]), z, SNR[j])
        end
        # b       = abs.(Z ./ Z[1:1, ..]) :: MT
        b       = abs.(Z) :: MT
        b       = reduce(hcat, makechunks(b[:,j], chunk) for j in 1:size(b,2))
        PLOT_FUN(b, TE, SNR, chunk)
        b
    end for d in ds)
    
    return reshape(out, size(out, 1), 1, N_CHUNKS, :)
end

function init_data(::SignalZipperProcessing, settings::Dict, ds::AbstractVector{<:Dict})
    T, TC = Float64, ComplexF64
    VT, VC, MT, MC, AT = Vector{T}, Vector{TC}, Matrix{T}, Matrix{TC}, Array{T,3}
    SNR   = settings["data"]["preprocess"]["SNR"] :: VT
    chunk = settings["data"]["preprocess"]["zipper"]["size"] :: Int

    PLOT_COUNT, PLOT_LIMIT = 0, 3
    PLOT_FUN = (b, TE, SNR, chunk) -> begin
        if PLOT_COUNT < PLOT_LIMIT
            p1 = plot(0:chunk-1, [b[:,1,1,j] for j in partition(1:size(b,2), length(SNR))]; line = (2,), m = (:c, :black, 3), leg = :none)
            p2 = plot(0:chunk-1, [b[:,1,2,j] for j in partition(1:size(b,2), length(SNR))]; line = (2,), m = (:r, :black, 3), leg = :none)
            p = plot(p1, p2; layout = (2,1), title = "SNR = " * string(SNR))
            display(p)
            PLOT_COUNT += 1
        end
    end

    out = reduce((x,y) -> cat(x, y; dims = 4), begin
        TE  = d[:sweepparams][:TE] :: T
        nTE = d[:sweepparams][:nTE] :: Int
        z   = cplx_signal(complex.(transverse.(d[:signals])) :: VC, nTE) :: VC
        Z   = repeat(z, 1, length(SNR)) :: MC
        for j in 1:length(SNR)
            add_noise!(@views(Z[:,j]), z, SNR[j])
        end

        # mag = abs.(Z ./ Z[1:1,..]) :: MT
        mag = abs.(Z) :: MT
        mag = mag[1:chunk,   ..] :: MT
        top = mag[1:2:chunk, ..] :: MT
        bot = mag[2:2:chunk, ..] :: MT
        b   = similar(mag, chunk, 1, 2, length(SNR))
        for j in 1:size(mag, 2)
            itp_top = Interpolations.CubicSplineInterpolation(1:2:chunk, @views(top[:,j]), extrapolation_bc = Interpolations.Line())
            itp_bot = Interpolations.CubicSplineInterpolation(2:2:chunk, @views(bot[:,j]), extrapolation_bc = Interpolations.Line())
            μ = (itp_top.(1:chunk) .+ itp_bot.(1:chunk))./2
            @views b[:,1,1,j] .= mag[:,j] .- μ # Mean-subtracted magnitude
            # @views b[:,1,1,j] .= abs.(FFTW.fft(b[:,1,1,j])) # Apply fft
            @views b[:,1,2,j] .= μ # Magnitude mean
            # @views b[:,1,2,j] .= [top[end:-1:1,j]; bot[:,j]] # Reordered magnitude
        end

        PLOT_FUN(b, TE, SNR, chunk)
        b
    end for d in ds)
    
    return out
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
        # b       = abs.(Z ./ Z[1:1, ..]) :: MT
        b       = abs.(Z) :: MT
        T2      = log10range(T2Range...; length = nT2) :: VT
        x       = ilaplace!(bufdict[nTE], b, T2, TE, alpha) :: MT
        PLOT_FUN(b, x, TE, nTE, T2)
        copy(x)
    end for d in ds)
    
    return reshape(out, :, 1, 1, size(out, 2))
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
        # b = abs.(Z ./ Z[1:1, ..]) :: MT
        b = abs.(Z) :: MT

        reduce(hcat, process_signal(b[:,j], SNR[j]) for j in 1:size(b,2))
    end for d in ds)
    
    return reshape(out, :, 1, 1, size(out, 2))
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

# ---------------------------------------------------------------------------- #
# Data loading utils
# ---------------------------------------------------------------------------- #

"""
    verify_settings(settings::Dict)
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
# Initialize labels
# ---------------------------------------------------------------------------- #

function label_fun(s::String, d::Dict)::Float64
    if s == "mwf" # myelin (small pool) water fraction
        d[:mwfvalues][:exact]
    elseif s == "iewf" # intra/extra-cellular (large pool/axonal + tissue) water fraction
        1 - d[:mwfvalues][:exact]
    elseif s == "iwf" # intra-cellular (large pool/axonal) water fraction
        d[:btparams_dict][:AxonPDensity] - d[:mwfvalues][:exact]
    elseif s == "ewf" # extra-cellular (tissue) water fraction
        1 - d[:btparams_dict][:AxonPDensity]
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
    elseif s == "T1mw" || s == "T1sp" # myelin-water (small pool) T1
        inv(d[:btparams_dict][:R1_sp])
    elseif s == "T1iew" # inverse of area-averaged R1 for intra/extra-cellular (large pool/axonal + tissue) water
        @unpack R1_lp, R1_Tissue = d[:btparams_dict] # R1 values
        iwf, ewf = label_fun("iwf", d), label_fun("ewf", d) # area fractions
        R1iew = (iwf * R1_lp + ewf * R1_Tissue) / (iwf + ewf) # area-weighted average
        inv(R1iew) # R1iew -> T1iew
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
