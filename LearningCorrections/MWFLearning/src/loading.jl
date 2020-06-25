# ---------------------------------------------------------------------------- #
# Preparing data
# ---------------------------------------------------------------------------- #

abstract type AbstractDataProcessing end
struct SignalMagnitudeProcessing <: AbstractDataProcessing end
struct SignalZipperProcessing <: AbstractDataProcessing end
struct PCAProcessing <: AbstractDataProcessing end
struct iLaplaceProcessing <: AbstractDataProcessing end
struct WaveletProcessing <: AbstractDataProcessing end

function prepare_data(settings::Dict)
    function load_data(path)
        if isfile(path)
            # Measurements are stored in a single .bson file
            data = BSON.load(realpath(path))
            first(values(data)) :: Vector{Dict{Symbol,Any}} # ignore key name
        else
            mapreduce(vcat, readdir(path)) do file
                data = BSON.load(realpath(joinpath(path, file)))
                if length(data) == 1 && first(values(data)) isa Vector{Dict{Symbol,Any}}
                    # Measurements are chunked into separate larger .bson files
                    first(values(data)) :: Vector{Dict{Symbol,Any}} # ignore key name
                else
                    # Measurements are stored in separate .bson files
                    data :: Dict{Symbol,Any}
                end
            end
        end
    end
    training_data_dicts = load_data(settings["data"]["train_data"])
    testing_data_dicts = load_data(settings["data"]["test_data"])

    # Filtering out undesired data
    function filter_bad_data(d)
        if isempty(settings["data"]["filter"]["labnames"])
            true
        else
            @unpack labnames, lower, upper = settings["data"]["filter"]
            all(map(labnames::Vector{String}, lower::Vector{Float64}, upper::Vector{Float64}) do label, lo, up
                lo <= label_fun(label, d)::Float64 <= up
            end)
        end
    end
    filter!(filter_bad_data, training_data_dicts)
    filter!(filter_bad_data, testing_data_dicts)

    training_thetas = init_labels(settings, training_data_dicts)
    testing_thetas = init_labels(settings, testing_data_dicts)
    @assert size(training_thetas, 1) == size(testing_thetas, 1)

    processing_types = AbstractDataProcessing[]
    checkapply(settings, key) = haskey(settings["data"]["preprocess"], key) && settings["data"]["preprocess"][key]["apply"] :: Bool
    checkapply(settings, "magnitude") && push!(processing_types, SignalMagnitudeProcessing())
    checkapply(settings, "zipper")    && push!(processing_types, SignalZipperProcessing())
    checkapply(settings, "ilaplace")  && push!(processing_types, iLaplaceProcessing())
    checkapply(settings, "wavelet")   && push!(processing_types, WaveletProcessing())
    isempty(processing_types) && error("No processing selected")

    init_all_data(d) = cat(init_data.(processing_types, Ref(settings), Ref(d))...; dims = 3)
    training_data = init_all_data(training_data_dicts)
    testing_data = init_all_data(testing_data_dicts)

    if checkapply(settings, "PCA")
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
    if settings["data"]["preprocess"]["shuffle"] :: Bool
        i_train, i_test = Random.shuffle(1:batchsize(training_data)), Random.shuffle(1:batchsize(testing_data))
        training_data, training_thetas, training_data_dicts = training_data[:,:,:,i_train], training_thetas[:,i_train], training_data_dicts[i_train]
        testing_data, testing_thetas, testing_data_dicts = testing_data[:,:,:,i_test], testing_thetas[:,i_test], testing_data_dicts[i_test]
    end

    # Redundancy check
    @assert heightsize(training_data) == heightsize(testing_data)
    @assert channelsize(training_data) == channelsize(testing_data)
    @assert batchsize(training_data) == batchsize(training_thetas)
    @assert batchsize(testing_data) == batchsize(testing_thetas)

    # Compute numerical properties of labels
    labels_props = init_labels_props(settings, hcat(training_thetas, testing_thetas))

    # Set output type
    T = settings["prec"] == 64 ? Float64 : Float32
    VT = Vector{T}
    training_data, testing_data, training_thetas, testing_thetas = map(
        x -> to_float_type_T(T, x),
        (training_data, testing_data, training_thetas, testing_thetas))
    
    # Set "auto" fields
    setauto!(d, field, val) = d[field] == "auto" ? (d[field] = val) : d[field]
    setauto!(settings["data"], "test_batch", batchsize(testing_data)) :: Int
    setauto!(settings["data"]["info"], "nfeatures", heightsize(training_data)) :: Int
    setauto!(settings["data"]["info"], "nchannels", channelsize(training_data)) :: Int
    setauto!(settings["data"]["info"], "nlabels", size(training_thetas, 1)) :: Int
    setauto!(settings["data"]["info"], "labwidth", convert(VT, labels_props[:width])) :: VT
    setauto!(settings["data"]["info"], "labmean", convert(VT, labels_props[:mean])) :: VT

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

function preprocessed_magnitude(settings::Dict, d::Dict)
    T, TC = Float64, ComplexF64
    VT, VC, MT, MC = Vector{T}, Vector{TC}, Matrix{T}, Matrix{TC}
    SNR       = settings["data"]["preprocess"]["SNR"] :: VT
    normalize = settings["data"]["preprocess"]["normalize"] :: String
    chunk     = settings["data"]["preprocess"]["chunk"] :: Int

    # Load complex signal
    nTE = d[:sweepparams][:nTE]
    signals = d[:signals]
    S = complex.(transverse.(signals)) :: VC
    z = cplx_signal(S, nTE) :: VC

    # Add complex gaussian noise
    Z = repeat(z, 1, length(SNR)) :: MC
    for j in 1:length(SNR)
        SNR[j] ≥ 0 && add_gaussian!(@views(Z[:,j]), z, SNR[j])
    end

    # Keep signal chunk
    if chunk < nTE
        Z = Z[1:chunk, ..]
    end

    # Normalize signal magnitude
    mag = abs.(Z) :: MT
    if normalize == "unitsum"
        unitsum!(mag; dims = 1)
    end

    return mag
end

function init_data(::SignalMagnitudeProcessing, settings::Dict, ds::AbstractVector{<:Dict})
    T, TC = Float64, ComplexF64
    VT, VC, MT, MC = Vector{T}, Vector{TC}, Matrix{T}, Matrix{TC}
    SNR = settings["data"]["preprocess"]["SNR"] :: VT

    PLOT_COUNT, PLOT_LIMIT = 0, 0
    PLOT_FUN = function (b, TE, SNR)
        if PLOT_COUNT < PLOT_LIMIT
            p = plot(
                1e3 .* TE .* (1:size(b,1)),
                reduce(hcat, [b[:,cols] for cols in partition(1:size(b,2), length(SNR))]);
                title = "SNR = " * string(SNR), line = (2,), m = (:c, :black, 3), leg = :none
            )
            display(p)
            PLOT_COUNT += 1
        end
    end

    out = mapreduce(hcat, ds) do d
        b = preprocessed_magnitude(settings, d)
        PLOT_FUN(b, d[:sweepparams][:TE], SNR)
        return b
    end

    return reshape(out, size(out, 1), 1, 1, :)
end

function init_data(::SignalZipperProcessing, settings::Dict, ds::AbstractVector{<:Dict})
    T, TC = Float64, ComplexF64
    VT, VC, MT, MC, AT = Vector{T}, Vector{TC}, Matrix{T}, Matrix{TC}, Array{T,3}
    SNR   = settings["data"]["preprocess"]["SNR"] :: VT

    PLOT_COUNT, PLOT_LIMIT = 0, 0
    PLOT_FUN = (b, TE, SNR) -> begin
        if PLOT_COUNT < PLOT_LIMIT
            p1 = plot(0:chunk-1, [b[:,1,1,j] for j in partition(1:size(b,2), length(SNR))]; line = (2,), m = (:c, :black, 3), leg = :none)
            p2 = plot(0:chunk-1, [b[:,1,2,j] for j in partition(1:size(b,2), length(SNR))]; line = (2,), m = (:r, :black, 3), leg = :none)
            p = plot(p1, p2; layout = (2,1), title = "SNR = " * string(SNR))
            display(p)
            PLOT_COUNT += 1
        end
    end

    out = mapreduce((x,y) -> cat(x, y; dims = 4), ds) do d
        TE  = d[:sweepparams][:TE] :: T
        mag = preprocessed_magnitude(settings, d)
        top = mag[1:2:end, ..] :: MT
        bot = mag[2:2:end, ..] :: MT
        b   = similar(mag, size(mag,1), 1, 2, length(SNR))
        for j in 1:size(mag, 2)
            itp_top = Interpolations.CubicSplineInterpolation(1:2:size(mag,1), @views(top[:,j]), extrapolation_bc = Interpolations.Line())
            itp_bot = Interpolations.CubicSplineInterpolation(2:2:size(mag,1), @views(bot[:,j]), extrapolation_bc = Interpolations.Line())
            μ = (itp_top.(1:size(mag,1)) .+ itp_bot.(1:size(mag,1)))./2
            @views b[:,1,1,j] .= mag[:,j] .- μ # Mean-subtracted magnitude
            # @views b[:,1,1,j] .= abs.(FFTW.fft(b[:,1,1,j])) # Apply fft
            @views b[:,1,2,j] .= μ # Magnitude mean
            # @views b[:,1,2,j] .= [top[end:-1:1,j]; bot[:,j]] # Reordered magnitude
        end

        PLOT_FUN(b, TE, SNR)
        return b
    end

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

    PLOT_COUNT, PLOT_LIMIT = 0, 0
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

    out = mapreduce(hcat, ds) do d
        signals = complex.(transverse.(d[:signals])) :: VC
        TE  = d[:sweepparams][:TE] :: T
        nTE = d[:sweepparams][:nTE] :: Int
        z   = cplx_signal(signals, nTE) :: VC
        Z   = bufdict[nTE].Z :: MC
        for j in 1:length(SNR)
            add_gaussian!(@views(Z[:,j]), z, SNR[j])
        end
        b   = abs.(Z) :: MT
        T2  = log10range(T2Range...; length = nT2) :: VT
        x   = ilaplace!(bufdict[nTE], b, T2, TE, alpha) :: MT
        PLOT_FUN(b, x, TE, nTE, T2)
        copy(x)
    end
    
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
    
    PLOT_COUNT, PLOT_LIMIT= 0, 0 # length(SNR)
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

    out = mapreduce(hcat, ds) do d
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

        b = preprocessed_magnitude(settings, d)
        mapreduce(hcat, 1:size(b,2)) do j
            process_signal(b[:,j], SNR[j])
        end
    end

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
# Initialize labels
# ---------------------------------------------------------------------------- #

function label_fun(s::String, d::Dict)::Float64
    if s == "mvf" # myelin (small pool) volume fraction
        g, eta = d[:btparams_dict][:g_ratio], d[:btparams_dict][:AxonPDensity]
        (1 - g^2) * eta
    elseif s == "ivf" # intra-cellular (large pool/axonal) volume fraction
        g, eta = d[:btparams_dict][:g_ratio], d[:btparams_dict][:AxonPDensity]
        g^2 * eta
    elseif s == "evf" # extra-cellular (tissue) volume fraction
        1 - d[:btparams_dict][:AxonPDensity]
    elseif s == "ievf" # intra/extra-cellular (large pool/axonal + tissue) volume fraction
        1 - label_fun("mvf", d)
    elseif s == "mwf" # myelin (small pool) water fraction
        d[:mwfvalues][:exact]
    elseif s == "iewf" # intra/extra-cellular (large pool/axonal + tissue) water fraction
        1 - d[:mwfvalues][:exact]
    elseif s == "iwf" # intra-cellular (large pool/axonal) water fraction
        p_r = d[:btparams_dict][:PD_sp] / d[:btparams_dict][:PD_lp] # relative proton density
        d[:mwfvalues][:exact] * label_fun("ivf", d) / (p_r * label_fun("mvf", d))
    elseif s == "ewf" # extra-cellular (tissue) water fraction
        p_r = d[:btparams_dict][:PD_sp] / d[:btparams_dict][:PD_lp] # relative proton density
        d[:mwfvalues][:exact] * label_fun("evf", d) / (p_r * label_fun("mvf", d))
    elseif s == "g" || s == "gratio" # g-ratio of fibre
        d[:btparams_dict][:g_ratio]
    elseif s == "T2mw" || s == "T2sp" # myelin-water (small pool) T2
        inv(d[:btparams_dict][:R2_sp])
    elseif s == "T2iw" || s == "T2lp" || s == "T2ax" # intra-cellular (large pool/axonal) T2
        inv(d[:btparams_dict][:R2_lp])
    elseif s == "T2ew" # extra-cellular (tissue) T2
        inv(d[:btparams_dict][:R2_Tissue])
    elseif s == "T2iew" # inverse of area-averaged R2 for intra/extra-cellular (large pool/axonal + tissue) water
        @unpack R2_lp, R2_Tissue = d[:btparams_dict] # R2 values
        ivf, evf = label_fun("ivf", d), label_fun("evf", d) # area fractions
        R2iew = (ivf * R2_lp + evf * R2_Tissue) / (ivf + evf) # area-weighted average
        inv(R2iew) # R2iew -> T2iew
    elseif s == "T2av" # inverse of area-averaged R2 for whole domain
        @unpack R2_lp, R2_sp, R2_Tissue = d[:btparams_dict] # R2 values
        ivf, mvf, evf = label_fun("ivf", d), label_fun("mvf", d), label_fun("evf", d) # area fractions
        R2av = (ivf * R2_lp + mvf * R2_sp + evf * R2_Tissue) # area-weighted average
        inv(R2av) # R2av -> T2av
    elseif s == "T1mw" || s == "T1sp" # myelin-water (small pool) T1
        inv(d[:btparams_dict][:R1_sp])
    elseif s == "T1iw" || s == "T1lp" || s == "T1ax" # intra-cellular (large pool/axonal) T1
        inv(d[:btparams_dict][:R2_lp])
    elseif s == "T1ew" # extra-cellular (tissue) T1
        inv(d[:btparams_dict][:R2_Tissue])
    elseif s == "T1iew" # inverse of area-averaged R1 for intra/extra-cellular (large pool/axonal + tissue) water
        @unpack R1_lp, R1_Tissue = d[:btparams_dict] # R1 values
        ivf, evf = label_fun("ivf", d), label_fun("evf", d) # area fractions
        R1iew = (ivf * R1_lp + evf * R1_Tissue) / (ivf + evf) # area-weighted average
        inv(R1iew) # R1iew -> T1iew
    elseif s == "Dav" # area-averaged D-coeff for whole domain
        @unpack D_Axon, D_Sheath, D_Tissue = d[:btparams_dict] # D values
        ivf, mvf, evf = label_fun("ivf", d), label_fun("mvf", d), label_fun("evf", d) # area fractions
        Dav = (ivf * D_Axon + mvf * D_Sheath + evf * D_Tissue) # area-weighted average
    elseif startswith(s, "log(") # Logarithm of parameter
        log10(label_fun(s[5:end-1], d))
    elseif startswith(s, "sind(") # Sine of angle (in degrees)
        sind(label_fun(s[6:end-1], d))
    elseif startswith(s, "cosd(") # Cosine of angle (in degrees)
        cosd(label_fun(s[6:end-1], d))
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
    label_names = settings["data"]["info"]["labnames"] :: Vector{String}
    label_infer = get!(settings["data"]["info"], "labinfer", copy(label_names)) :: Vector{String}
    @assert label_infer == label_names[1:length(label_infer)]
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

# ---------------------------------------------------------------------------- #
# Physics models
# ---------------------------------------------------------------------------- #

"""
    Hard-coded forward physics model given matrix of parameters
"""
function forward_physics_8arg(x::Matrix{T}) where {T}
    nTE  = 32 # Number of echoes (fixed)
    Smw  = zeros(Vec{3,T}, nTE) # Buffer for myelin signal
    Siew = zeros(Vec{3,T}, nTE) # Buffer for IE water signal
    M    = zeros(T, nTE, size(x,2)) # Total signal magnitude
    for j in 1:size(x,2)
        mwf, iewf, rT2iew, rT2mw, alpha, _, rT1iew, rT1mw = x[1,j], x[2,j], x[3,j], x[4,j], x[5,j], x[6,j], x[7,j], x[8,j]
        forward_prop!(Smw,  rT2mw,  rT1mw,  alpha, nTE)
        forward_prop!(Siew, rT2iew, rT1iew, alpha, nTE)
        @views M[:,j] .= norm.(transverse.(mwf .* Smw .+ iewf .* Siew))
    end
    return M
end
function forward_physics_14arg(x::Matrix{T}) where {T}
    nTE  = 32 # Number of echoes (fixed)
    Smw  = zeros(Vec{3,T}, nTE) # Buffer for myelin signal
    Siw  = zeros(Vec{3,T}, nTE) # Buffer for intra-axonal water signal
    Sew  = zeros(Vec{3,T}, nTE) # Buffer for extra-axonal water signal
    M    = zeros(T, nTE, size(x,2)) # Total signal magnitude
    for j in 1:size(x,2)
        alpha    = acosd(x[1,j])
        gratio   = x[2,j]
        mwf      = x[3,j]
        rT2mw    = x[4,j]
        rT2iew   = x[5,j]
        iwf      = x[6,j]
        ewf      = x[7,j]
        iewf     = x[8,j]
        rT2iw    = x[9,j]
        rT2ew    = x[10,j]
        rT1mw    = x[11,j]
        rT1iw    = x[12,j]
        rT1ew    = x[13,j]
        rT1iew   = x[14,j]
        forward_prop!(Smw, rT2mw, rT1mw, alpha, nTE)
        forward_prop!(Siw, rT2iw, rT1iw, alpha, nTE)
        forward_prop!(Sew, rT2ew, rT1ew, alpha, nTE)
        @views M[:,j] .= norm.(transverse.(mwf .* Smw .+ iwf .* Siw .+ ewf .* Sew))
    end
    return M
end
function forward_physics_15arg(x::Matrix{T}) where {T}
    nTE  = 32 # Number of echoes (fixed)
    Smw  = zeros(Vec{3,T}, nTE) # Buffer for myelin signal
    Siw  = zeros(Vec{3,T}, nTE) # Buffer for intra-axonal water signal
    Sew  = zeros(Vec{3,T}, nTE) # Buffer for extra-axonal water signal
    M    = zeros(T, nTE, size(x,2)) # Total signal magnitude
    for j in 1:size(x,2)
        alpha    = acosd(x[1,j])
        gratio   = x[2,j]
        mwf      = x[3,j]
        rT2mw    = x[4,j]
        rT2iew   = x[5,j]
        Kperm    = x[6,j]
        iwf      = x[7,j]
        ewf      = x[8,j]
        iewf     = x[9,j]
        rT2iw    = x[10,j]
        rT2ew    = x[11,j]
        rT1mw    = x[12,j]
        rT1iw    = x[13,j]
        rT1ew    = x[14,j]
        rT1iew   = x[15,j]
        forward_prop!(Smw, rT2mw, rT1mw, alpha, nTE)
        forward_prop!(Siw, rT2iw, rT1iw, alpha, nTE)
        forward_prop!(Sew, rT2ew, rT1ew, alpha, nTE)
        @views M[:,j] .= norm.(transverse.(mwf .* Smw .+ iwf .* Siw .+ ewf .* Sew))
    end
    return M
end
function forward_physics_15arg_Kperm(x::Matrix{T}) where {T}
    nTE  = 32 # Number of echoes (fixed)
    Smw  = zeros(Vec{3,T}, nTE) # Buffer for myelin signal
    Siw  = zeros(Vec{3,T}, nTE) # Buffer for intra-axonal water signal
    Sew  = zeros(Vec{3,T}, nTE) # Buffer for extra-axonal water signal
    M    = zeros(T, nTE, size(x,2)) # Total signal magnitude
    for j in 1:size(x,2)
        Kperm    = x[1,j]
        alpha    = acosd(x[2,j])
        gratio   = x[3,j]
        mwf      = x[4,j]
        rT2mw    = x[5,j]
        rT2iew   = x[6,j]
        iwf      = x[7,j]
        ewf      = x[8,j]
        iewf     = x[9,j]
        rT2iw    = x[10,j]
        rT2ew    = x[11,j]
        rT1mw    = x[12,j]
        rT1iw    = x[13,j]
        rT1ew    = x[14,j]
        rT1iew   = x[15,j]
        forward_prop!(Smw, rT2mw, rT1mw, alpha, nTE)
        forward_prop!(Siw, rT2iw, rT1iw, alpha, nTE)
        forward_prop!(Sew, rT2ew, rT1ew, alpha, nTE)
        @views M[:,j] .= norm.(transverse.(mwf .* Smw .+ iwf .* Siw .+ ewf .* Sew))
    end
    return M
end
ForwardProp8Arg() = @λ(x -> forward_physics_8arg(x))

function theta_sampler_8arg(Nbatch)
    out = zeros(8, Nbatch)
    for j in 1:size(out,2)
        g_bounds, η_bounds = (0.60, 0.92), (0.15, 0.82)
        mvf    = MWFLearning.linearsampler(0.025, 0.4)
        η      = min(mvf / (1 - g_bounds[2]^2), η_bounds[2]) # Maximum density for given bounds
        g      = sqrt(1 - mvf/η) # g-ratio corresponding to maximum density
        # g    = MWFLearning.linearsampler(g_bounds...) # Uniformly random g-ratio
        # η    = mvf / (1 - g^2) # MWFLearning.linearsampler(η_bounds...)
        evf    = 1 - η
        ivf    = 1 - (mvf + evf)
        mwf    =  mvf / (2evf + 2ivf + mvf)
        ewf    = 2evf / (2evf + 2ivf + mvf)
        iwf    = 2ivf / (2evf + 2ivf + mvf)
        alpha  = MWFLearning.linearsampler(120.0, 180.0)
        K      = 1e-3 # Fix mock signals at constant near-zero permeability (lower bound of MWFLearning.log10sampler(1e-3, 10.0))
        T2sp   = MWFLearning.linearsampler(10e-3, 70e-3)
        T2lp   = MWFLearning.linearsampler(50e-3, 180e-3)
        while !(T2lp ≥ 1.5*T2sp) # Enforce constraint
            T2sp = MWFLearning.linearsampler(10e-3, 70e-3)
            T2lp = MWFLearning.linearsampler(50e-3, 180e-3)
        end
        T1sp   = MWFLearning.linearsampler(150e-3, 250e-3)
        T1lp   = MWFLearning.linearsampler(949e-3, 1219e-3)
        T2tiss = T2lp # MWFLearning.linearsampler(50e-3, 180e-3)
        T1tiss = T1lp # MWFLearning.linearsampler(949e-3, 1219e-3)
        TE     = 10e-3
        out[1,j] = mwf
        out[2,j] = 1 - mwf # iewf
        out[3,j] = inv((ivf * inv(T2lp) + evf * inv(T2tiss)) / (ivf + evf)) / TE # T2iew/TE
        out[4,j] = T2sp / TE # T2mw/TE
        out[5,j] = alpha
        out[6,j] = log10(TE*K)
        out[7,j] = inv((ivf * inv(T1lp) + evf * inv(T1tiss)) / (ivf + evf)) / TE # T1iew/TE
        out[8,j] = T1sp / TE # T1mw/TE
    end
    return out
end
function theta_sampler_14arg(Nbatch)
    out = zeros(14, Nbatch)
    for j in 1:size(out,2)
        g_bounds, η_bounds = (0.60, 0.92), (0.15, 0.82)
        mvf    = MWFLearning.linearsampler(0.025, 0.4)
        η      = min(mvf / (1 - g_bounds[2]^2), η_bounds[2]) # Maximum density for given bounds
        g      = sqrt(1 - mvf/η) # g-ratio corresponding to maximum density
        # g    = MWFLearning.linearsampler(g_bounds...) # Uniformly random g-ratio
        # η    = mvf / (1 - g^2) # MWFLearning.linearsampler(η_bounds...)
        evf    = 1 - η
        ivf    = 1 - (mvf + evf)
        mwf    =  mvf / (2evf + 2ivf + mvf)
        ewf    = 2evf / (2evf + 2ivf + mvf)
        iwf    = 2ivf / (2evf + 2ivf + mvf)
        alpha  = MWFLearning.linearsampler(120.0, 180.0)
        K      = 1e-3 # Fix mock signals at constant near-zero permeability (lower bound of MWFLearning.log10sampler(1e-3, 10.0))
        T2sp   = MWFLearning.linearsampler(10e-3, 70e-3)
        T2lp   = MWFLearning.linearsampler(50e-3, 180e-3)
        while !(T2lp ≥ 1.5*T2sp) # Enforce constraint
            T2sp = MWFLearning.linearsampler(10e-3, 70e-3)
            T2lp = MWFLearning.linearsampler(50e-3, 180e-3)
        end
        T1sp   = MWFLearning.linearsampler(150e-3, 250e-3)
        T1lp   = MWFLearning.linearsampler(949e-3, 1219e-3)
        T2tiss = T2lp # MWFLearning.linearsampler(50e-3, 180e-3)
        T1tiss = T1lp # MWFLearning.linearsampler(949e-3, 1219e-3)
        TE     = 10e-3
        out[1,j]  = cosd(alpha) # cosd(alpha)
        out[2,j]  = g # gratio
        out[3,j]  = mwf # mwf
        out[4,j]  = T2sp / TE # T2mw/TE
        out[5,j]  = inv((ivf * inv(T2lp) + evf * inv(T2tiss)) / (ivf + evf)) / TE # T2iew/TE
        out[6,j]  = iwf # iwf
        out[7,j]  = ewf # ewf
        out[8,j]  = iwf + ewf # iewf
        out[9,j]  = T2lp / TE # T2iw/TE
        out[10,j] = T2tiss / TE # T2ew/TE
        out[11,j] = T1sp / TE # T1mw/TE
        out[12,j] = T1lp / TE # T1iw/TE
        out[13,j] = T1tiss / TE # T1ew/TE
        out[14,j] = inv((ivf * inv(T1lp) + evf * inv(T1tiss)) / (ivf + evf)) / TE # T1iew/TE
    end
    return out
end
function theta_sampler_15arg(Nbatch)
    out = zeros(15, Nbatch)
    for j in 1:size(out,2)
        g_bounds, η_bounds = (0.60, 0.92), (0.15, 0.82)
        mvf    = MWFLearning.linearsampler(0.025, 0.4)
        η      = min(mvf / (1 - g_bounds[2]^2), η_bounds[2]) # Maximum density for given bounds
        g      = sqrt(1 - mvf/η) # g-ratio corresponding to maximum density
        # g    = MWFLearning.linearsampler(g_bounds...) # Uniformly random g-ratio
        # η    = mvf / (1 - g^2) # MWFLearning.linearsampler(η_bounds...)
        evf    = 1 - η
        ivf    = 1 - (mvf + evf)
        mwf    =  mvf / (2evf + 2ivf + mvf)
        ewf    = 2evf / (2evf + 2ivf + mvf)
        iwf    = 2ivf / (2evf + 2ivf + mvf)
        alpha  = MWFLearning.linearsampler(120.0, 180.0)
        K      = 1e-3 # Fix mock signals at constant near-zero permeability (lower bound of MWFLearning.log10sampler(1e-3, 10.0))
        T2sp   = MWFLearning.linearsampler(10e-3, 70e-3)
        T2lp   = MWFLearning.linearsampler(50e-3, 180e-3)
        while !(T2lp ≥ 1.5*T2sp) # Enforce constraint
            T2sp = MWFLearning.linearsampler(10e-3, 70e-3)
            T2lp = MWFLearning.linearsampler(50e-3, 180e-3)
        end
        T1sp   = MWFLearning.linearsampler(150e-3, 250e-3)
        T1lp   = MWFLearning.linearsampler(949e-3, 1219e-3)
        T2tiss = T2lp # MWFLearning.linearsampler(50e-3, 180e-3)
        T1tiss = T1lp # MWFLearning.linearsampler(949e-3, 1219e-3)
        TE     = 10e-3
        out[1,j]  = cosd(alpha) # cosd(alpha)
        out[2,j]  = g # gratio
        out[3,j]  = mwf # mwf
        out[4,j]  = T2sp / TE # T2mw/TE
        out[5,j]  = inv((ivf * inv(T2lp) + evf * inv(T2tiss)) / (ivf + evf)) / TE # T2iew/TE
        out[6,j]  = log10(TE*K) # log(TE*Kperm)
        out[7,j]  = iwf # iwf
        out[8,j]  = ewf # ewf
        out[9,j]  = iwf + ewf # iewf
        out[10,j] = T2lp / TE # T2iw/TE
        out[11,j] = T2tiss / TE # T2ew/TE
        out[12,j] = T1sp / TE # T1mw/TE
        out[13,j] = T1lp / TE # T1iw/TE
        out[14,j] = T1tiss / TE # T1ew/TE
        out[15,j] = inv((ivf * inv(T1lp) + evf * inv(T1tiss)) / (ivf + evf)) / TE # T1iew/TE
    end
    return out
end

"""
    myelin_prop(...)
"""
function myelin_prop(
        mwf::T    = T(0.2),
        iewf::T   = T(1-mwf),
        rT2iew::T = T(63e-3/10e-3),
        rT2mw::T  = T(15e-3/10e-3),
        alpha::T  = T(170.0),
        rT1iew::T = T(10_000e-3/10e-3), # By default, assume T1 effects are negligeable
        rT1mw::T  = T(10_000e-3/10e-3), # By default, assume T1 effects are negligeable
        nTE::Int  = 32,
    ) where {T}

    M = mwf  .* forward_prop(rT2mw, rT1mw, alpha, nTE) .+
        iewf .* forward_prop(rT2iew, rT1iew, alpha, nTE)
    
    return (m -> √(m[1]^2 + m[2]^2)).(M)
end

"""
    forward_prop(...)
"""
function forward_prop!(
        M::AbstractVector{Vec{3,T}},
        rT2::T   = T(65e-3 / 10e-3),
        rT1::T   = T(10_000e-3 / 10e-3), # By default, assume T1 effects are negligeable
        alpha::T = T(170.0),
        nTE::Int = 32
    ) where {T}

    @assert length(M) == nTE

    m₀ = Vec{3,T}((0, -1, 0))
    m∞ = Vec{3,T}((0, 0, 1))

    # By Tensors.jl convention, this specifies the transpose rotation matrix
    At = Tensor{2,3,T}((
        one(T),  zero(T),      zero(T),
        zero(T), cosd(alpha), -sind(alpha),
        zero(T), sind(alpha),  cosd(alpha)))
    A  = transpose(At)
    R  = Vec{3,T}((exp(-inv(2*rT2)), exp(-inv(2*rT2)), exp(-inv(2*rT1))))

    step1 = (m) -> m∞ - R ⊙ (m∞ - A  ⋅ (m∞ - R ⊙ (m∞ - m)))
    step2 = (m) -> m∞ - R ⊙ (m∞ - A' ⋅ (m∞ - R ⊙ (m∞ - m)))

    M[1] = step1(m₀)
    M[2] = step2(M[1])
    for ii = 3:2:nTE
        M[ii  ] = step1(M[ii-1])
        M[ii+1] = step2(M[ii  ])
    end

    return M
end
forward_prop(rT2::T, rT1::T, alpha::T, nTE::Int) where {T} =
    forward_prop!(zeros(Vec{3,T}, nTE), rT2, rT1, alpha, nTE)
