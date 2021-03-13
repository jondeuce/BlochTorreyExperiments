# Multihead Attention

"""
    MultiheadAttention(insize::Int, args...) = MultiheadAttention(insize => insize, args...)
    MultiheadAttention(sizes::Pair{Int,Int}, nheads::Int, headsize::Int = sizes[1] ÷ nheads)

Multihead dot product attention layer. `nheads` is the number of attention heads,
`insize` is the input embedding size, `headsize` is the size of the input embedding
following the projection layer for each head, `outsize` is the output embedding size.
"""
struct MultiheadAttention{Q<:Flux.Dense, K<:Flux.Dense, V<:Flux.Dense, O<:Flux.Dense}
    insize::Int
    outsize::Int
    nheads::Int
    headsize::Int
    iqproj::Q
    ikproj::K
    ivproj::V
    oproj::O
end
Base.show(io::IO, mh::MultiheadAttention) = print(io, "MultiheadAttention($(mh.insize)=>$(mh.outsize), nheads=$(mh.nheads), headsize=$(mh.headsize))")

Flux.functor(mh::MultiheadAttention) = (mh.iqproj, mh.ikproj, mh.ivproj, mh.oproj), w -> MultiheadAttention(mh.insize, mh.outsize, mh.nheads, mh.headsize, w...)

MultiheadAttention(insize::Int, args...) = MultiheadAttention(insize => insize, args...)
MultiheadAttention(sizes::Pair{Int,Int}, nheads::Int, headsize::Int = sizes[1] ÷ nheads) =
    MultiheadAttention(
        sizes..., nheads, headsize,
        Flux.Dense(sizes[1], headsize * nheads),
        Flux.Dense(sizes[1], headsize * nheads),
        Flux.Dense(sizes[1], headsize * nheads),
        Flux.Dense(headsize * nheads, sizes[2]),
    )

(mh::MultiheadAttention)(query, key, value) = attention(mh, query, key, value)
(mh::MultiheadAttention)(x) = attention(mh, x, x, x) # self-attention

function attention(mh::MultiheadAttention, Q::AbstractMatrix{T}, K::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T}
    seqlength = size(Q,2)
    QP = reshape(mh.iqproj(Q), :, mh.nheads, seqlength) # size(QP) == (headsize, nheads, seqlength)
    KP = reshape(mh.ikproj(K), :, mh.nheads, seqlength)
    VP = reshape(mh.ivproj(V), :, mh.nheads, seqlength)

    @ein A[i,j,h] := KP[k,h,i] * QP[k,h,j]
    scale = 1/sqrt(T(mh.headsize))
    A = fast_softmax(scale .* A; dims = 1)
    @ein score[i,h,j] := VP[i,h,k] * A[k,j,h]

    mh.oproj(reshape(score, mh.headsize * mh.nheads, seqlength))
end

function attention(mh::MultiheadAttention, Q::AbstractTensor3D{T}, K::AbstractTensor3D{T}, V::AbstractTensor3D{T}) where {T}
    seqlength, batchsize = size(Q,2), size(Q,3)
    QP = reshape(mh.iqproj(Q), :, mh.nheads, seqlength, batchsize) # size(QP) == (headsize, nheads, seqlength, batchsize)
    KP = reshape(mh.ikproj(K), :, mh.nheads, seqlength, batchsize)
    VP = reshape(mh.ivproj(V), :, mh.nheads, seqlength, batchsize)

    @ein A[i,j,h,b] := KP[k,h,i,b] * QP[k,h,j,b]
    scale = 1/sqrt(T(mh.headsize))
    A = fast_softmax(scale .* A; dims = 1)
    @ein score[i,h,j,b] := VP[i,h,k,b] * A[k,j,h,b]

    mh.oproj(reshape(score, mh.headsize * mh.nheads, seqlength, batchsize))
end

function _attention_test(;
        nheads = 3, # number of attention heads
        insize = 7, # input size, i.e. embedding dimension size
        headsize = 4, # projection size, i.e. length of vectors for attention inner products
        outsize = 6, # output embedding size
        seqlength = 11,
        batchsize = 64,
    )
    mh = MultiheadAttention(insize => outsize, nheads, headsize) |> gpu
    mh_xf = TransformersMHA.MultiheadAttention(mh.nheads, mh.iqproj, mh.ikproj, mh.ivproj, mh.oproj)
    as3D(f, xs::AbstractMatrix...) = dropdims(f(map(x -> reshape(x, size(x)..., 1), xs)...); dims = 3)

    Q,K,V = [CUDA.rand(insize, seqlength) for _ in 1:3]
    @assert mh(Q,K,V) ≈ mh_xf(Q,K,V) ≈ as3D(mh,Q,K,V) ≈ as3D(mh_xf,Q,K,V)
    @btime CUDA.@sync $mh($Q,$K,$V)
    @btime CUDA.@sync $mh_xf($Q,$K,$V)
    @btime CUDA.@sync Zygote.gradient(() -> sum($mh($Q,$K,$V)), $(Flux.params(mh)))
    # @btime CUDA.@sync Zygote.gradient(() -> sum($mh_xf($Q,$K,$V)), $(Flux.params(mh))) #TODO Zygote errors, something about PermutedDimsArray?

    Q,K,V = [CUDA.rand(insize, seqlength, batchsize) for _ in 1:3]
    @assert mh(Q,K,V) ≈ mh_xf(Q,K,V)
    @btime CUDA.@sync $mh($Q,$K,$V)
    @btime CUDA.@sync $mh_xf($Q,$K,$V)
    @btime CUDA.@sync Zygote.gradient(() -> sum($mh($Q,$K,$V)), $(Flux.params(mh)))
    # @btime CUDA.@sync Zygote.gradient(() -> sum($mh_xf($Q,$K,$V)), $(Flux.params(mh))) #TODO Zygote errors, something about PermutedDimsArray?
end

"""
    Transformer(insize::Int, nheads::Int, hiddensize::Int; act = relu)
    Transformer(nheads::Int, insize::Int, headsize::Int, hiddensize::Int; act = relu)

Transformer encoder layer. `insize` is the input embedding size, `nheads` is the number of attention heads,
`headsize` is the size of the input projection for each head, with default value `div(insize, nheads)` if unspecified,
`hiddensize` is the number of hidden nodes in the positionwise feedforward layer, and `act` the corresponding activation function.
"""
struct Transformer{MH<:MultiheadAttention, MHN, FF, FFN}
    mh::MH
    mhn::MHN
    ff::FF
    ffn::FFN
end
Flux.@functor Transformer

function Transformer(insize::Int, nheads::Int, hiddensize::Int; kwargs...)
    @assert rem(insize, nheads) == 0
    Transformer(insize, nheads, div(insize, nheads), hiddensize; kwargs...)
end

Transformer(insize::Int, nheads::Int, headsize::Int, hiddensize::Int; act = Flux.relu) = Transformer(
    MultiheadAttention(insize, nheads, headsize),
    Flux.LayerNorm(insize),
    MLP(insize => insize, 0, hiddensize, act, identity),
    Flux.LayerNorm(insize),
)

function (xf::Transformer)(x::AbstractArray)
    y = xf.mhn(x + xf.mh(x))
    return xf.ffn(y + xf.ff(y))
end

"""
Basic Transformer encoder with learned positional embedding
"""
function TransformerEncoder(;
        esize::Int, nheads::Int, headsize::Int, hdim::Int, seqlength::Int, nhidden::Int,
        insizes::Tuple{Vararg{Int}}, outsize::Int,
    )
    @assert esize * seqlength >= sum(insizes) # positional encoding output should retain at least as many datapoints as input
    @assert esize <= nheads * headsize # projection head layers should conserve datapoints

    if length(insizes) == 1
        topology = @nntopo(
            Y :        # input signals
            Y  => E  : # positional embedding
            E  => R1 : # resize
            R1 => T  : # transformer layers
            T  => R2 : # resize
            R2 => Z    # dense reduction
        )
        top = ()
    else
        inputs = "(" * join(["Y$i" for i in 1:length(insizes)], ", ") * ")"
        topology = NNTopo("$(inputs) => V => E => R1 => T => R2 => Z")
        top = (vcat,)
    end

    xf = Stack(
        topology,
        top..., # Collect input arguments
        # Flux.Dense(sum(insizes), esize * seqlength), # Positional encoding
        MLP(sum(insizes) => esize * seqlength, 0, hdim, Flux.relu, identity), # Positional encoding
        Resize(esize, seqlength, :), # Resize
        Flux.Chain(
            [Transformer(esize, nheads, headsize, hdim) for _ in 1:nhidden]...
        ),
        Resize(esize * seqlength, :), # Resize
        MLP(esize * seqlength => outsize, 0, hdim, Flux.relu, identity), # Dense reduction
    )
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
