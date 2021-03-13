module TransformersMHA

import Flux, CUDA

"""
    MultiheadAttention(nHeads::Int, inputSize::Int, headSize::Int, outputSize::Int)

Multihead dot product attention layer. `nHeads` is the number of attention heads,
`inputSize` is the input embedding size, `headSize` is the hidden size of input
projection layer of each head, `outputSize` is the output embedding size.
"""
struct MultiheadAttention{Q<:Flux.Dense, K<:Flux.Dense, V<:Flux.Dense, O<:Flux.Dense}
    nHeads::Int
    iqproj::Q
    ikproj::K
    ivproj::V
    oproj::O
end

Flux.functor(mh::MultiheadAttention) = (mh.iqproj, mh.ikproj, mh.ivproj, mh.oproj), w -> MultiheadAttention(mh.nHeads, w...)

MultiheadAttention(nHeads::Int, inputSize::Int, headSize::Int, outputSize::Int) =
    MultiheadAttention(
        nHeads,
        Flux.Dense(inputSize, headSize * nHeads),
        Flux.Dense(inputSize, headSize * nHeads),
        Flux.Dense(inputSize, headSize * nHeads),
        Flux.Dense(headSize * nHeads, outputSize),
    )

function Base.show(io::IO, mh::MultiheadAttention)
    headSize = size(mh.iqproj.weight, 1) ÷ mh.nHeads
    inputSize = size(mh.iqproj.weight, 2)
    outputSize = size(mh.oproj.weight, 1)
    print(io, "MultiheadAttention(nHeads=$(mh.nHeads), headSize=$(headSize), $(inputSize)=>$(outputSize))")
end

function (mh::MultiheadAttention)(query::A1, key::A2, value::A3) where {T, A1 <: AbstractArray{T,3}, A2 <: AbstractArray{T,3}, A3 <: AbstractArray{T,3}}
    qs = size(query)
    ks = size(key)
    vs = size(value)

    #size(ipq) == (h, seqLength, batch)
    ipq = mh.iqproj(query)
    ipk = mh.ikproj(key)
    ipv = mh.ivproj(value)

    h = size(ipq, 1)
    headSize = div(h, mh.nHeads)

    #size(ipq) == (headSize, seqLength, nHeads, batch)
    ipq = permutedims(reshape(ipq, headSize, mh.nHeads, qs[2], qs[3]), (1, 3, 2, 4))
    ipk = permutedims(reshape(ipk, headSize, mh.nHeads, ks[2], ks[3]), (1, 3, 2, 4))
    ipv = permutedims(reshape(ipv, headSize, mh.nHeads, vs[2], vs[3]), (1, 3, 2, 4))

    #size(ipq) == (headSize, seqLength, nHeads * batch)
    ipq = reshape(ipq, headSize, qs[2], :)
    ipk = reshape(ipk, headSize, ks[2], :)
    ipv = reshape(ipv, headSize, vs[2], :)

    atten = attention(ipq,ipk,ipv)

    atten = permutedims(reshape(atten, headSize, qs[2], mh.nHeads, qs[3]), (1, 3, 2, 4)) #size(atten) == (headSize, nHeads, ql, b)
    atten = reshape(atten, h, qs[2], qs[3]) #size(atten) == (h, ql, b)

    out = mh.oproj(atten)
    out #size(out) == (h, seqLength, batch)
end

function (mh::MultiheadAttention)(query::A1, key::A2, value::A3) where {T, A1 <: AbstractMatrix{T}, A2 <: AbstractMatrix{T}, A3 <: AbstractMatrix{T}}
    # size(query) == (inputSize, seqLength)
    ipq = mh.iqproj(query)
    ipk = mh.ikproj(key)
    ipv = mh.ivproj(value)

    #size(QP) == (headSize, seqLength, nHeads)
    totalProjSize, seqLength = size(ipq)
    headSize = div(totalProjSize, mh.nHeads) # totalProjSize == headSize * nHeads

    QP = permutedims(reshape(ipq, headSize, mh.nHeads, :), (1, 3, 2))
    KP = permutedims(reshape(ipk, headSize, mh.nHeads, :), (1, 3, 2))
    VP = permutedims(reshape(ipv, headSize, mh.nHeads, :), (1, 3, 2))

    atten = attention(QP, KP, VP)

    # size(atten) == (totalProjSize, seqLength)
    atten = reshape(permutedims(atten, (1, 3, 2)), totalProjSize, :)

    mh.oproj(atten)
end

function attention(query::A1, key::A2, value::A3) where {T, A1 <: AbstractArray{T,3}, A2 <: AbstractArray{T,3}, A3 <: AbstractArray{T,3}}
    #size(query) == size(key) == size(value) == (inputSize, seqLength, batch)
    #size(score) == (seqLength, seqLength, batch)
    dk = size(key, 1)
    score = Flux.batched_mul(Flux.batched_transpose(key), query)
    score = Flux.softmax(score ./ T(sqrt(dk)); dims=1)
    Flux.batched_mul(value, score) #size(return) == (inputSize, seqLength, batch)
end

#### Multihead attention via cudnn call

Base.@kwdef struct cudnnAttentionDims
    nHeads::Int
    kProjSize::Int
    qProjSize::Int
    vProjSize::Int
    oProjSize::Int
end

struct cudnnMultiheadAttention{W <: AbstractArray}
    weights::W
    dims::cudnnAttentionDims
end
Flux.functor(mh::cudnnMultiheadAttention) = (mh.weights,), w -> cudnnMultiheadAttention(w..., mh.dims)

function cudnnMultiheadAttention(;
        nHeads::Int,
        inputSize::Int,
        projSize::Int,
        outputSize::Int = inputSize,
        init = Flux.glorot_uniform
    )
    # Translation from Transformers.jl: head => nHeads, is => inputSize, hs => projSize, os => outputSize
    d = cudnnAttentionDims(; nHeads, kProjSize = projSize, qProjSize = projSize, vProjSize = projSize, oProjSize = outputSize)
    w = init(d.nHeads * (d.qProjSize * inputSize + d.kProjSize * inputSize + d.vProjSize * inputSize + d.oProjSize * d.vProjSize))
    cudnnMultiheadAttention(w, d)
end

function attention(mh::cudnnMultiheadAttention, queries, keys, values, residuals = nothing)
    nHeads, qProjSize, kProjSize, vProjSize, oProjSize = mh.dims.nHeads, mh.dims.qProjSize, mh.dims.kProjSize, mh.dims.vProjSize, mh.dims.oProjSize
    axes = [CUDA.CUDNN.CUDNN_SEQDATA_VECT_DIM, CUDA.CUDNN.CUDNN_SEQDATA_TIME_DIM, CUDA.CUDNN.CUDNN_SEQDATA_BATCH_DIM, CUDA.CUDNN.CUDNN_SEQDATA_BEAM_DIM]
    out = CUDA.CUDNN.cudnnMultiHeadAttnForward(
        mh.weights, queries, keys, values;
        axes, nHeads, qProjSize, kProjSize, vProjSize, oProjSize,
        residuals, smScaler = inv(sqrt(kProjSize)),
    )
end
attention(mh::cudnnMultiheadAttention, x, residuals = nothing) = attention(mh, x, x, x, residuals)

#=
Zygote.@adjoint function attention(mh::cudnnMultiheadAttention, queries, keys, values, residuals = nothing)
    #TODO
    cudnnMultiHeadAttnForwardAD(
        weights, queries, keys, values, residuals;
        dready, dweights, dqueries, dkeys, dvalues,
        attnDesc, currIdx, loWinIdx, hiWinIdx,
        devSeqLengthsQO, devSeqLengthsKV,
        qDesc, kDesc, vDesc, oDesc,
        out, workspace, reserveSpace
    )
end
=#

function _test_mha()
    nHeads = 3 # number of attention heads
    inputSize = 7 # input size
    headSize = 4 # projection size, i.e. length of vectors for attention inner products
    outputSize = 6 # output size
    mh = MultiheadAttention(nHeads, inputSize, headSize, outputSize) |> gpu

    T = Float32
    bs = 10 # batch size
    seq = 7 # sequence length
    Q, K, V = [CUDA.rand(T, inputSize, seq, bs) for _ in 1:3]
    out = mh(Q, K, V)
    @assert size(out) == (outputSize, seq, bs)

    weights = vcat(vec.((mh.iqproj.weight, mh.ikproj.weight, mh.ivproj.weight, mh.oproj.weight))...)
    queries, keys, values = Q, K, V
    axes = [CUDA.CUDNN.CUDNN_SEQDATA_VECT_DIM, CUDA.CUDNN.CUDNN_SEQDATA_TIME_DIM, CUDA.CUDNN.CUDNN_SEQDATA_BATCH_DIM, CUDA.CUDNN.CUDNN_SEQDATA_BEAM_DIM]
    nHeads = nHeads
    kProjSize = qProjSize = vProjSize = headSize
    oProjSize = outputSize
    residuals = nothing
    @assert sizeof(weights) == sizeof(T) * nHeads * (qProjSize * size(queries,1) + kProjSize * size(keys,1) + vProjSize * size(values,1) + oProjSize * vProjSize)

    dims = cudnnAttentionDims(; nHeads, kProjSize, qProjSize, vProjSize, oProjSize)
    cudnn_mh = cudnnMultiheadAttention(weights, dims)
    cudnn_out = attention(cudnn_mh, queries, keys, values, residuals)
    @assert size(cudnn_out) == (outputSize, seq, bs)

    return sum(abs.(out .- cudnn_out) ./ max.(abs.(out), abs.(cudnn_out))) / length(out)
end

end # TransformersMHA
