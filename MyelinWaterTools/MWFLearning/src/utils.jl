# ---------------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------------- #

# Batching
make_minibatch(features, labels, idxs) = (features[.., idxs], labels[.., idxs])
function training_batches(features, labels, minibatchsize; overtrain = false)
    @assert batchsize(features) == batchsize(labels)
    batches = partition(1:batchsize(features), minibatchsize)
    if overtrain
        train_set = [make_minibatch(features, labels, batches[1])]
    else
        train_set = [make_minibatch(features, labels, b) for b in batches]
    end
end
testing_batches(features, labels) = make_minibatch(features, labels, :)
features(batch) = batch[1]
labels(batch) = batch[2]

function param_summary(model, train_set, test_set)
    test_dofs = length(test_set[2])
    train_dofs = sum(batch -> length(batch[2]), train_set)
    param_dofs = sum(length, Flux.params(model))
    test_param_density = param_dofs / test_dofs
    train_param_density = param_dofs / train_dofs
    @info @sprintf(" Testing parameter density: %d/%d (%.2f %%)", param_dofs, test_dofs, 100 * test_param_density)
    @info @sprintf("Training parameter density: %d/%d (%.2f %%)", param_dofs, train_dofs, 100 * train_param_density)
end

# Losses
function makelosses(model, losstype, weights = nothing)
    l1 = weights == nothing ? @λ((x,y) -> sum(abs, model(x) .- y))  : @λ((x,y) -> sum(abs, weights .* (model(x) .- y)))
    l2 = weights == nothing ? @λ((x,y) -> sum(abs2, model(x) .- y)) : @λ((x,y) -> sum(abs2, weights .* (model(x) .- y)))
    crossent = @λ((x,y) -> Flux.crossentropy(model(x), y))
    mae = @λ((x,y) -> l1(x,y) * 1 // length(y))
    mse = @λ((x,y) -> l2(x,y) * 1 // length(y))
    rmse = @λ((x,y) -> sqrt(mse(x,y)))
    mincrossent = @λ (y) -> -sum(y .* log.(y))
    
    lossdict = Dict("l1" => l1, "l2" => l2, "crossent" => crossent, "mae" => mae, "mse" => mse, "rmse" => rmse, "mincrossent" => mincrossent)
    if losstype ∉ keys(lossdict)
        @warn "Unknown loss $(losstype); defaulting to mse"
        losstype = "mse"
    end

    loss = lossdict[losstype]
    accloss = losstype == "crossent" ? @λ((x,y) -> loss(x,y) - mincrossent(y)) : rmse # default
    accuracy = @λ((x,y) -> 100 - 100 * accloss(x,y))
    labelacc = @λ((x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ (maximum(abs.(y); dims = 2) .- minimum(abs.(y); dims = 2))))
    # labelacc = @λ((x,y) -> 100 .* vec(mean(abs.((model(x) .- y) ./ y); dims = 2)))
    # labelacc = @λ((x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ maximum(abs.(y); dims = 2)))
    
    return @ntuple(loss, accuracy, labelacc)
end

# Optimizer
lr(opt) = opt.eta
lr!(opt, α) = (opt.eta = α; opt.eta)
lr(opt::Flux.Optimiser) = lr(opt[1])
lr!(opt::Flux.Optimiser, α) = lr!(opt[1], α)

fixedlr(e,opt) = lr(opt) # Fixed learning rate
geometriclr(e,opt,rate=100,factor=10^(1/4)) = mod(e, rate) == 0 ? lr(opt) / factor : lr(opt) # Drop lr every `rate` epochs
findlr(e,opt,epochs=100,minlr=1e-6,maxlr=0.5) = e <= epochs ? logspace(1,epochs,minlr,maxlr)(e) : maxlr # Learning rate finder
cyclelr(e,opt,lrstart=1e-5,lrmin=1e-6,lrmax=1e-2,lrwidth=50,lrtail=5) = # Learning rate cycling
                     e <=   lrwidth          ? linspace(        1,            lrwidth, lrstart,   lrmax)(e) :
      lrwidth + 1 <= e <= 2*lrwidth          ? linspace(  lrwidth,          2*lrwidth,   lrmax, lrstart)(e) :
    2*lrwidth + 1 <= e <= 2*lrwidth + lrtail ? linspace(2*lrwidth, 2*lrwidth + lrtail, lrstart,   lrmin)(e) :
    lrmin

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
    linspace(x1,x2,y1,y2) = x -> (y2 - y1) / (x2 - x1) * (x - x1) + y1
"""
@inline linspace(x1,x2,y1,y2) = x -> (y2 - y1) / (x2 - x1) * (x - x1) + y1

"""
    logspace(x1,x2,y1,y2) = x -> 10^linspace(x1, x2, log10(y1), log10(y2))(x)
"""
@inline logspace(x1,x2,y1,y2) = x -> 10^linspace(x1, x2, log10(y1), log10(y2))(x)

"""
    unitsum(x) = x ./ sum(x)
"""
unitsum(x) = x ./ sum(x)

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
    # Extract last nTE echoes, and normalize by the first point |S0| = |S(t=0)|.
    # This sets |S(t=0)| = 1 for any domain, but allows all measurable points,
    # i.e. S(t=TE), S(T=2TE), ..., to be unnormalized.
    Z = z[n - dt * (nTE-1) : dt : n, ..]
    Z ./= abs.(z[1:1, ..])
    return Z
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
    myelin_prop(...)
"""
function myelin_prop(
        mwf::T    = T(0.25),
        iewf::T   = T(1 - mwf),
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
 