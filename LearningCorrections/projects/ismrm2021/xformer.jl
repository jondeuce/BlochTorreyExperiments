"""
    LinearProjection(in::Int, out::Int)
"""
struct LinearProjection{W}
    W :: W
end
Flux.@functor LinearProjection
LinearProjection(in::Int, out::Int) = LinearProjection(Flux.glorot_uniform(out, in))
(e::LinearProjection)(x) = e.W * x

"""
    SignalProjector(; psize::Int, chunksize::Int, overlap::Int, nfeatures::Int, nsignals::Int)
"""
struct SignalProjector{PX, PY, E0, S}
    PX :: PX
    PY :: PY
    E0 :: E0
    shape :: S
end
Flux.@functor SignalProjector
Flux.trainable(e::SignalProjector) = (e.PX, e.PY, e.E0)
Base.show(io::IO, ::SignalProjector) = print(io, "SignalProjector()")

function SignalProjector(; psize::Int, nshards::Int, nfeatures::Int, nsignals::Int)
    PX = nfeatures <= 0 ? nothing : Flux.Chain(LinearProjection(nfeatures, psize), X -> reshape(X, psize, 1, :))
    PY = Flux.Chain(LinearProjection(nsignals, psize * nshards), Y -> reshape(Y, psize, nshards, :))
    E0 = Flux.glorot_uniform(psize, 1 + nshards + ifelse(nfeatures > 0, 1, 0))
    shape = (; psize, nshards, nfeatures, nsignals)
    SignalProjector(PX, PY, E0, shape)
end

function (e::SignalProjector)(Y::AbstractMatrix, X::Union{Nothing, <:AbstractMatrix} = nothing)
    Zclass = zeros_similar(Y, e.shape.psize, 1, size(Y,2))
    Yenc = e.PY(Y) # [nY x b] -> [p x mY x b]
    Z0 = if (X !== nothing)
        Xenc = e.PX(X) # [nX x b] -> [p x 1 x b]
        hcat(Zclass, Yenc, Xenc) # [p x (1 + mY + 1) x b]
    else
        hcat(Zclass, Yenc) # [p x (1 + mY) x b]
    end
    return Z0 .+ e.E0
end
(e::SignalProjector)(Y::AbstractVector) = dropdims(e(reshape(Y, length(Y), 1), nothing); dims = 3)
(e::SignalProjector)(Y::AbstractVector, X::AbstractVector) = dropdims(e(reshape(Y, length(Y), 1), reshape(X, length(X), 1)); dims = 3)

"""
    SignalEncoder(; psize::Int, chunksize::Int, overlap::Int, nfeatures::Int, nsignals::Int)
"""
struct SignalEncoder{DX <: AbstractArray, EY <: AbstractArray, E0<: AbstractArray, S}
    DX :: DX
    EY :: EY
    E0 :: E0
    shape :: S
end
Flux.@functor SignalEncoder
Flux.trainable(e::SignalEncoder) = (e.DX, e.EY, e.E0)
Base.show(io::IO, ::SignalEncoder) = print(io, "SignalEncoder()")

function SignalEncoder(; psize::Int, chunksize::Int, overlap::Int, nfeatures::Int, nsignals::Int)
    nshards = (nsignals - chunksize) ÷ (chunksize - overlap) + 1
    @assert nsignals == nshards * (chunksize - overlap) + overlap
    DX = Flux.glorot_uniform(psize, nfeatures)
    EY = Flux.glorot_uniform(psize, chunksize)
    E0 = Flux.glorot_uniform(psize, 1 + nshards + nfeatures)
    shape = (; psize, nshards, nfeatures, chunksize, overlap)
    SignalEncoder(DX, EY, E0, shape)
end

function (e::SignalEncoder)(Y::AbstractMatrix, X::Union{Nothing, <:AbstractMatrix} = nothing)
    @unpack psize, nshards, nfeatures, chunksize, overlap = e.shape
    nbatches = size(Y,2)
    Yshard = shard_array(Y, chunksize, overlap) # [nY x b] -> [cY x mY x b]
    Yenc = e.EY * reshape(Yshard, chunksize, nshards * nbatches) # [cY x mY x b] -> [cY x mY*b] -> [p x mY*b]
    Yenc = reshape(Yenc, psize, nshards, nbatches) # [p x mY*b] -> [p x mY x b]
    Zclass = zeros_similar(Y, psize, 1, nbatches)
    Z0 = if (X !== nothing)
        Xenc = e.DX .* reshape(X, 1, nfeatures, nbatches) # [mX x b] -> [1 x mX x b] -> [p x mX x b]
        hcat(Zclass, Yenc, Xenc) # [p x (1 + mY + mX) x b]
    else
        hcat(Zclass, Yenc) # [p x (1 + mY + mX) x b] =  # [p x (1 + mY) x b]
    end
    return Z0 .+ e.E0
end
(e::SignalEncoder)(Y::AbstractVector) = dropdims(e(reshape(Y, length(Y), 1), nothing); dims = 3)
(e::SignalEncoder)(Y::AbstractVector, X::AbstractVector) = dropdims(e(reshape(Y, length(Y), 1), reshape(X, length(X), 1)); dims = 3)

"""
Basic Transformer encoder with learned positional embedding
"""
function TransformerEncoder(
        MLPHead = nothing;
        head::Int, hsize::Int, hdim::Int, nhidden::Int,
        psize::Int, nshards::Int, chunksize::Int, overlap::Int,
        nsignals::Int, ntheta::Int, nlatent::Int
    )

    topology = if ntheta == 0 && nlatent == 0
        Transformers.@nntopo(
            Y : # Input (nY × b)
            Y => E : # SignalProjector (psize × _ × b)
            E => H : # Transformer encoder (psize × _ × b)
            H => A : # Extract "class" token activations (psize × b)
            A => C # MLP head mapping "class" embedding to output (nout x b)
        )
    elseif ntheta == 0 || nlatent == 0
        Transformers.@nntopo(
            (Y,X) : # Inputs (nY × b), (mX x b)
            (Y,X) => E : # SignalProjector (psize × _ × b)
            E => H : # Transformer encoder (psize × _ × b)
            H => A : # Extract "class" token activations (psize × b)
            A => C # MLP head mapping "class" embedding to output (nout x b)
        )
    else
        Transformers.@nntopo(
            (Y,θ,Z) : # Inputs (nY × b), (nθ x b), (nZ x b)
            (θ,Z) => X : # Concatenate θ,Z inputs (nY × b), (mX x b)
            (Y,X) => E : # SignalProjector (psize × _ × b)
            E => H : # Transformer encoder (psize × _ × b)
            H => A : # Extract "class" token activations (psize × b)
            A => C # MLP head mapping "class" embedding to output (nout x b)
        )
    end

    xf = Transformers.Stack(
        topology,
        (ntheta != 0 && nlatent != 0 ? [vcat] : [])...,
        # SignalEncoder(; psize, chunksize, overlap, nsignals, nfeatures = ntheta + nlatent),
        SignalProjector(; psize, nshards, nsignals, nfeatures = ntheta + nlatent),
        Flux.Chain(map(1:nhidden) do _
            Transformers.Basic.Transformer(psize, head, hsize, hdim; future = true, act = Flux.relu, pdrop = 0.0)
        end...),
        H -> H[:, 1, :],
        (MLPHead === nothing) ? identity : MLPHead,
    )
    Flux.fmap(Flux.testmode!, xf) # Force dropout layers inactive
end

"""
    shard_array(Y::AbstractMatrix; chunksize::Int)

Transform `n x b` matrix into `c x m x b` array (`c = chunksize`, `m = n - c + 1`)
such that each `n x 1` column

    [Y_1, Y_2, ..., Y_n]

of the original array is transform into an `c x m` matrix of consecutive shards

    [ Y_1 Y_2   Y_3   ... Y_n-c+1 ]
    [ Y_2 Y_3   Y_4   ... Y_n-c+2 ]
    [ ... ...   ...   ... ...     ]
    [ Y_c Y_c+1 Y_c+2 ... Y_n     ]
"""
function shard_array(Y::AbstractVecOrMat, chunksize::Int, overlap::Int)
    nsignals = size(Y,1)
    chunk = 1:chunksize
    offset = 0:chunksize-overlap:nsignals-chunksize
    chunk .+ offset'
    I = CartesianIndex.(chunk .+ offset')
    return Y[I,..]
end

#=
let
    img_idx, img, Ytrain, Ytrainmeta = (sample_batch(:train)...,) # .|> deepcopy
    ps = Flux.params(models["enc1"], models["enc2"], models["dec"]) # |> deepcopy
    ℓ, back = Zygote.pullback(() -> sum(CVAElosses(Ytrainmeta; marginalize_Z = false)), ps)

    # @btime sum(CVAElosses($Ytrainmeta; marginalize_Z = false)) #TODO CUDA.@sync
    # @btime Zygote.pullback(() -> sum(CVAElosses($Ytrainmeta; marginalize_Z = false)), $ps) #TODO CUDA.@sync
    @btime $back(one(eltype($phys))) #TODO CUDA.@sync
end
=#
