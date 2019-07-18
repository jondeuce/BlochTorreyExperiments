# ---------------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------------- #

"""
    batchsize(x::AbstractArray)

Returns the length of the last dimension of the data `x`.
`x` must have dimension of at least 2, otherwise an error is thrown.
"""
# batchsize(x::AbstractVector) = 1
batchsize(x::AbstractVector) = error("x must have dimension of at least 2, but x is a $(typeof(x))")
# batchsize(x::AbstractVecOrMat) = error("x must have dimension of at least 3, but x is a $(typeof(x))")
batchsize(x::AbstractArray{T,N}) where {T,N} = size(x, N)

"""
    channelsize(x::AbstractArray)

Returns the length of the second-last dimension of the data `x`.
`x` must have dimension of at least 3, otherwise an error is thrown.
"""
# Old docstring:
# Returns the length of the second-last dimension of the data `x`, unless:
#     `x` is a `Matrix`, in which case 1 is returned.
#     `x` is a `Vector`, in which case an error is thrown.
# channelsize(x::AbstractVector) = error("Channel size undefined for AbstractVector's")
# channelsize(x::AbstractMatrix) = 1
channelsize(x::AbstractVecOrMat) = error("x must have dimension of at least 3, but x is a $(typeof(x))")
channelsize(x::AbstractArray{T,N}) where {T,N} = size(x, N-1)

"""
    heightsize(x::AbstractArray)

Returns the length of the first dimension of the data `x`.
`x` must have dimension of at least 3, otherwise an error is thrown.
"""
# heightsize(x::AbstractVector) = error("heightsize undefined for vectors")
heightsize(x::AbstractVecOrMat) = error("x must have dimension of at least 3, but x is a $(typeof(x))")
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
    SNR == 0 ? 0 .* z[1:1, ..] : sqrt.(abs2.(z[1:1, ..]) ./ T(10^(SNR/10))) # Same for both real and complex

"""
    add_noise(z, SNR)

Add gaussian noise with signal-to-noise ratio `SNR` proportional to the first time point.
"""
add_noise!(out::AbstractVecOrMat, z::AbstractVecOrMat, SNR) = out .= z .+ noise_level(z, SNR) .* randn(eltype(z), size(z))
add_noise!(z::AbstractVecOrMat, SNR) = add_noise!(z, z, SNR)
add_noise(z::AbstractVecOrMat, SNR) = add_noise!(copy(z), z, SNR)

"""
 Kaiming uniform initialization.
 """
 function kaiming_uniform(T::Type, dims; gain = 1)
    fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
    bound = sqrt(3) * gain / sqrt(fan_in)
    return rand(Uniform(-bound, bound), dims) |> Array{T}
 end
 kaiming_uniform(T::Type, dims...; kwargs...) = kaiming_uniform(T::Type, dims; kwargs...)
 kaiming_uniform(args...; kwargs...) = kaiming_uniform(Float32, args...; kwargs...)
 
 """
 Kaiming normal initialization.
 """
 function kaiming_normal(T::Type, dims; gain = 1)
    fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
    std = gain / sqrt(fan_in)
    return rand(Normal(0, std), dims) |> Array{T}
 end
 kaiming_normal(T::Type, dims...; kwargs...) = kaiming_normal(T::Type, dims; kwargs...)
 kaiming_normal(args...; kwargs...) = kaiming_normal(Float32, args...; kwargs...)
 
 """
 Xavier uniform initialization.
 """
 function xavier_uniform(T::Type, dims; gain = 1)
    fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
    fan_out = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end-1]
    bound = sqrt(3) * gain * sqrt(2 / (fan_in + fan_out))
    return rand(Uniform(-bound, bound), dims) |> Array{T}
 end
 xavier_uniform(T::Type, dims...; kwargs...) = xavier_uniform(T::Type, dims; kwargs...)
 xavier_uniform(args...; kwargs...) = xavier_uniform(Float32, args...; kwargs...)
 
 """
 Xavier normal initialization.
 """
 function xavier_normal(T::Type, dims; gain = 1)
    fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
    fan_out = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end-1]
    std = gain * sqrt(2 / (fan_in + fan_out))
    return rand(Normal(0, std), dims) |> Array{T}
 end
 xavier_normal(T::Type, dims...; kwargs...) = xavier_normal(T::Type, dims; kwargs...)
 xavier_normal(args...; kwargs...) = xavier_normal(Float32, args...; kwargs...)
 
 # Override flux defaults
 Flux.glorot_uniform(dims...) = xavier_uniform(Float32, dims...)
 Flux.glorot_uniform(T::Type, dims...) = xavier_uniform(T, dims...)
 Flux.glorot_normal(dims...) = xavier_normal(Float32, dims...)
 Flux.glorot_normal(T::Type, dims...) = xavier_normal(T, dims...)
 