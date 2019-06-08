# ---------------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------------- #

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

function prepare_data(settings::Dict)
    training_data_dicts = BSON.load.(joinpath.(settings["data"]["train_data"], readdir(settings["data"]["train_data"])))
    testing_data_dicts = BSON.load.(joinpath.(settings["data"]["test_data"], readdir(settings["data"]["test_data"])))

    label_fun(d) = [d[:mwfvalues][:exact], d[:sweepparams][:K]]
    training_labels = [label_fun(d) for d in training_data_dicts]
    testing_labels = [label_fun(d) for d in testing_data_dicts]
    training_labels = reshape(reduce(vcat, training_labels), :, 1, length(training_labels))
    testing_labels = reshape(reduce(vcat, testing_labels), :, 1, length(testing_labels))
    
    training_data = reduce(hcat, init_data(settings, d) for d in training_data_dicts)
    testing_data = reduce(hcat, init_data(settings, d) for d in testing_data_dicts)
    training_data = reshape(training_data, :, 1, size(training_data, 2))
    testing_data = reshape(testing_data, :, 1, size(testing_data, 2))
    
    T = settings["prec"] == 32 ? Float32 : Float64
    training_data, testing_data, training_labels, testing_labels = map(
        x -> to_float_type_T(T, x),
        (training_data, testing_data, training_labels, testing_labels))

    return @dict(
        training_data_dicts, testing_data_dicts,
        training_data, testing_data,
        training_labels, testing_labels)
end
prepare_data(settings_file::String) = prepare_data(TOML.parsefile(settings_file))

function init_data(settings::Dict, d::Dict)
    @unpack signals = d
    @unpack TE = d[:sweepparams]
    @unpack alpha, T2Range, nT2 = settings["data"]
    
    x = init_signal(signals)
    T2 = log10range(T2Range...; length = nT2);
    y = project_onto_exp(x, T2, TE, alpha)

    return y
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

log10range(a, b; length = 10) = 10 .^ range(log10(a); stop = log10(b), length = length)

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
    M = size(b,1)
    t = η.*(1:M)
    A = [exp(-ti/τj) for ti in t, τj in τ]
    x = (A'A + α^2*I)\(A'b)
    return x
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
