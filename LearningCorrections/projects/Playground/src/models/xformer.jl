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
    # @assert size(Q,1) == size(K,1) == size(V,1)
    qseqlength, kvseqlength = size(Q,2), size(K,2)
    QP = reshape(mh.iqproj(Q), :, mh.nheads, qseqlength)  # size(QP) == (headsize, nheads, qseqlength)
    KP = reshape(mh.ikproj(K), :, mh.nheads, kvseqlength) # size(KP) == (headsize, nheads, kvseqlength)
    VP = reshape(mh.ivproj(V), :, mh.nheads, kvseqlength) # size(VP) == (headsize, nheads, kvseqlength)

    @ein A[i,j,h] := KP[k,h,i] * QP[k,h,j]
    scale = 1/sqrt(T(mh.headsize))
    A = fast_softmax(scale .* A; dims = 1)
    # A = Flux.softmax(scale .* A; dims = 1)
    @ein score[i,h,j] := VP[i,h,k] * A[k,j,h]

    mh.oproj(reshape(score, mh.headsize * mh.nheads, qseqlength))
end

function attention(mh::MultiheadAttention, Q::AbstractTensor3D{T}, K::AbstractTensor3D{T}, V::AbstractTensor3D{T}) where {T}
    # @assert size(Q,1) == size(K,1) == size(V,1) && size(Q,3) == size(K,3) == size(V,3)
    qseqlength, kvseqlength, batchsize = size(Q,2), size(K,2), size(K,3)
    QP = reshape(mh.iqproj(Q), :, mh.nheads, qseqlength, batchsize)  # size(QP) == (headsize, nheads, qseqlength, batchsize)
    KP = reshape(mh.ikproj(K), :, mh.nheads, kvseqlength, batchsize) # size(KP) == (headsize, nheads, kvseqlength, batchsize)
    VP = reshape(mh.ivproj(V), :, mh.nheads, kvseqlength, batchsize) # size(VP) == (headsize, nheads, kvseqlength, batchsize)

    @ein A[i,j,h,b] := KP[k,h,i,b] * QP[k,h,j,b]
    scale = 1/sqrt(T(mh.headsize))
    A = fast_softmax(scale .* A; dims = 1)
    # A = Flux.softmax(scale .* A; dims = 1)
    @ein score[i,h,j,b] := VP[i,h,k,b] * A[k,j,h,b]

    mh.oproj(reshape(score, mh.headsize * mh.nheads, qseqlength, batchsize))
end

# Manual "broadcasting" of single batch of queries
function attention(mh::MultiheadAttention, Q::AbstractMatrix{T}, K::AbstractTensor3D{T}, V::AbstractTensor3D{T}) where {T}
    # @assert size(Q,1) == size(K,1) == size(V,1) && size(K,3) == size(V,3)
    # Q′ = repeat(Q, 1, 1, size(K,3)) #TODO fails for CuArrays?
    Q′ = Q .* ones_similar(Q, 1, 1, size(K,3)) #MATLAB-esque repeat
    attention(mh, Q′, K, V)
end

function _attention_test(;
        nheads      = 12,  # 3, # number of attention heads
        insize      = 64,  # 7, # input size, i.e. embedding dimension size
        headsize    = 32,  # 4, # projection size, i.e. length of vectors for attention inner products
        outsize     = 128, # 6, # output embedding size
        kvseqlength = 20,  # 11,
        qseqlength  = 10,  # 11,
        batchsize   = 256, # 64,
    )
    mh = MultiheadAttention(insize => outsize, nheads, headsize) |> gpu
    mh_xf = TransformersMHA.MultiheadAttention(mh.nheads, mh.iqproj, mh.ikproj, mh.ivproj, mh.oproj)
    as3D(f, xs::AbstractMatrix...) = dropdims(f((x -> reshape(x, size(x)..., 1)).(xs)...); dims = 3)

    Q,K,V = [CUDA.rand(insize, seq) for seq in (qseqlength, kvseqlength, kvseqlength)]
    # @assert disp_ret(mh(Q,K,V)) ≈ disp_ret(mh_xf(Q,K,V))
    # @assert disp_ret(as3D(mh,Q,K,V)) ≈ disp_ret(as3D(mh_xf,Q,K,V))
    # @assert disp_ret(mh_xf(Q,K,V)) ≈ disp_ret(as3D(mh_xf,Q,K,V))
    # @assert disp_ret(mh(Q,K,V)) ≈ disp_ret(as3D(mh,Q,K,V))
    @assert mh(Q,K,V) ≈ mh_xf(Q,K,V) ≈ as3D(mh,Q,K,V) ≈ as3D(mh_xf,Q,K,V)

    @btime CUDA.@sync $mh($Q,$K,$V)
    @btime CUDA.@sync $mh_xf($Q,$K,$V)
    @btime CUDA.@sync Zygote.gradient(() -> sum($mh($Q,$K,$V)), $(Flux.params(mh)))
    # @btime CUDA.@sync Zygote.gradient(() -> sum($mh_xf($Q,$K,$V)), $(Flux.params(mh))) #TODO Zygote errors, something about PermutedDimsArray?

    Q,K,V = [CUDA.rand(insize, seq, batchsize) for seq in (qseqlength, kvseqlength, kvseqlength)]
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
function Transformer(insize::Int, nheads::Int, hiddensize::Int; kwargs...)
    @assert rem(insize, nheads) == 0
    Transformer(insize, nheads, div(insize, nheads), hiddensize; kwargs...)
end

function Transformer(insize::Int, nheads::Int, headsize::Int, hiddensize::Int; act = Flux.relu, norm = :pre, self = true)
    @assert norm ∈ (:pre, :post)
    mh   = MultiheadAttention(insize, nheads, headsize)
    mhn  = Flux.LayerNorm(insize)
    mhnq = Flux.LayerNorm(insize)
    mhnk = Flux.LayerNorm(insize)
    mhnv = Flux.LayerNorm(insize)
    ff   = MLP(insize => insize, 0, hiddensize, act, identity)
    ffn  = Flux.LayerNorm(insize)
    plus = Base.BroadcastFunction(+)

    # Re: pre/post layer normalization, see: "On Layer Normalization in the Transformer Architecture" (Xiong et al., PMLR 2020)
    #   http://proceedings.mlr.press/v119/xiong20b.html

    if self
        if norm === :pre
            # Equivalent to: tmp = X + mh(mhn(X)); out = tmp + ff(ffn(tmp))
            topo = @nntopo X => MHN => MH : (X,MH) => MH⁺ => FFN => FF : (MH⁺,FF) => FF⁺
            Stack(topo, mhn, mh, plus, ffn, ff, plus)
        else # norm === :post
            # Equivalent to: tmp = mhn(X + mh(X)); out = ffn(tmp + ff(tmp))
            topo = @nntopo X => MH : (X,MH) => MH⁺ => MHN => FF : (MHN,FF) => FF⁺ => FFN
            Stack(topo, mh, plus, mhn, ff, plus, ffn)
        end
    else
        if norm === :pre
            # Equivalent to: tmp = Q + mh(mhnq(Q),mhnk(K),mhnv(V)); out = tmp + ff(ffn(tmp))
            topo = @nntopo (Q,K,V) : Q => QN : K => KN : V => VN : (QN,KN,VN) => MH : (Q,MH) => MH⁺ => FFN => FF : (MH⁺,FF) => FF⁺
            Stack(topo, mhnq, mhnk, mhnv, mh, plus, ffn, ff, plus)
        else # norm === :post
            # Equivalent to: tmp = mhn(Q + mh(Q,K,V)); out = ffn(tmp + ff(tmp))
            topo = @nntopo (Q,K,V) => MH : (Q,MH) => MH⁺ => MHN => FF : (MHN,FF) => FF⁺ => FFN
            Stack(topo, mh, plus, mhn, ff, plus, ffn)
        end
    end
end

function Perceiver(
        insize::Int, nheads::Int, headsize::Int, hiddensize::Int, qseqlength::Int, nhidden::Int;
        act = Flux.relu, norm = :pre, share = false,
    )
    attn(; self) = Transformer(insize, nheads, headsize, hiddensize; act, norm, self)
    latent = HiddenState(insize, qseqlength)
    cross₀ = attn(self = false)
    self₀  = attn(self = true)
    layers = [latent, cross₀, self₀]
    topo   = NNTopo("X => H : " * join(["(H,X,X) => H => H" for _ in 1:nhidden], " : "))
    for _ in 2:nhidden
        if share
            append!(layers, [cross₀, self₀])
        else
            append!(layers, [attn(self = false), attn(self = true)])
        end
    end
    Stack(topo, layers...)
end

function _test_transformer(;
        insize      = 7,
        nheads      = 4,
        headsize    = 8,
        hiddensize  = 11,
        batchsize   = 15,
        qseqlength  = 5,
        kvseqlength = 7,
    )
    Q,K,V = [CUDA.rand(insize, seq, batchsize) for seq in (qseqlength, kvseqlength, kvseqlength)]
    let
        xf = Transformer(insize, nheads, headsize, hiddensize; norm = :pre, self = true) |> gpu
        mhn, mh, _, ffn, ff, _ = xf.models
        tmp = Q + mh(mhn(Q)); out = tmp + ff(ffn(tmp)); @assert out ≈ xf(Q)
    end
    let
        xf = Transformer(insize, nheads, headsize, hiddensize; norm = :post, self = true) |> gpu
        mh, _, mhn, ff, _, ffn = xf.models
        tmp = mhn(Q + mh(Q)); out = ffn(tmp + ff(tmp)); @assert out ≈ xf(Q)
    end
    let
        xf = Transformer(insize, nheads, headsize, hiddensize; norm = :pre, self = false) |> gpu
        mhnq, mhnk, mhnv, mh, _, ffn, ff, _ = xf.models
        tmp = Q + mh(mhnq(Q),mhnk(K),mhnv(V)); out = tmp + ff(ffn(tmp)); @assert out ≈ xf(Q,K,V)
    end
    let
        xf = Transformer(insize, nheads, headsize, hiddensize; norm = :post, self = false) |> gpu
        mh, _, mhn, ff, _, ffn = xf.models
        tmp = mhn(Q + mh(Q,K,V)); out = ffn(tmp + ff(tmp)); @assert out ≈ xf(Q,K,V)
    end
end

"""
Basic Transformer encoder with learned positional embedding
"""
function TransformerEncoder(
        tail_network = nothing;
        esize::Int, nheads::Int, headsize::Int, hdim::Int, seqlength::Int, nhidden::Int, qseqlength::Int = 0, share::Bool = false,
        insizes::Tuple{Vararg{Int}}, outsize::Int,
    )
    @assert esize * seqlength >= sum(insizes) # positional encoding output should retain at least as many datapoints as input
    # @assert esize <= nheads * headsize # projection head layers should conserve datapoints

    # Positional encoding
    if true
        # Full (non-)linear mapping from `sum(insizes)` -> `esize * seqlength` dimensions w/ positional encoding
        pos_encode = Flux.Chain(
            Flux.Dense(sum(insizes), esize * seqlength), # Linear positional encoding
            # MLP(sum(insizes) => esize * seqlength, 0, hdim, Flux.relu, identity), # Non-linear positional encoding
            Resize(esize, seqlength, :), # Resize
        )
    else
        # Factorized (non-)linear mapping from `sum(insizes)` -> `esize` -> `esize * seqlength` dimensions w/ positional encoding
        pos_encode = Flux.Chain(
            Resize(sum(insizes), 1, :), # Resize input to represent single channel/sequence token
            ChannelwiseDense(sum(insizes), 1=>seqlength, identity), # linearly expand channel dimension/number of tokens + add positional encoding
            # Flux.Dense(sum(insizes), esize), # Linear dimension reduction to token dimension size
            MLP(sum(insizes) => esize, 0, hdim, Flux.relu, identity), # Non-linear dimension reduction to token dimension size
        )
    end

    # Number of output tokens: qseqlength if using Perceiver, else just seqlength
    is_perceiver = qseqlength > 0
    if is_perceiver
        xf_layers = Perceiver(esize, nheads, headsize, hdim, qseqlength, nhidden; share) # Perceiver builds `nhidden` layers internally
    else
        xf_layers = Flux.Chain([Transformer(esize, nheads, headsize, hdim) for _ in 1:nhidden]...) # Transformer layers
    end

    # MLP reducer to encoder space
    outseqlength = is_perceiver ? qseqlength : seqlength
    reducer = Flux.Chain(
        Resize(esize * outseqlength, :), # Resize
        MLP(esize * outseqlength => outsize, 0, hdim, Flux.relu, identity), # Dense reduction
        (tail_network === nothing ? () : (tail_network,))..., # Append `tail_network` following reduction
    )

    # Build transformer encoder
    xf = Flux.Chain(pos_encode, xf_layers, reducer) |> flattenchain
    if length(insizes) == 1
        return xf
    else
        input_names = join(["X$i" for i in 1:length(insizes)], ", ") # input arg names = (X1, X2, ..., XN)
        topology = NNTopo("($input_names) => V => Y") # vcat input args and forward to transformer
        return Stack(topology, vcat, xf)
    end
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
