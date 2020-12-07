"""
    SignalEmbedder(p::Int, cY::Int, nX::Int, nY::Int)
"""
struct SignalEmbedder{p,cY,mY,nX,nY,DX,EY,E0}
    DX :: DX
    EY :: EY
    E0 :: E0
end
Base.show(io::IO, m::SignalEmbedder) = print(io, "SignalEmbedder()")

function Flux.functor(e::SignalEmbedder{p,cY,mY,nX,nY}) where {p,cY,mY,nX,nY}
    return (e.DX, e.EY, e.E0), function ((DX, EY, E0),)
        SignalEmbedder{p,cY,mY,nX,nY,typeof(DX),typeof(EY),typeof(E0)}(DX, EY, E0)
    end
end

function SignalEmbedder(; p::Int, cY::Int, nX::Int, nY::Int)
    mY = nY - cY + 1
    # DX = nX > 0 ? Flux.Diagonal(nX) : nothing
    DX = nX > 0 ? identity : nothing
    EY = Flux.Dense(cY, p; initb = Flux.Zeros)
    E0 = randn(Float32, p, 1 + mY + nX)
    SignalEmbedder(DX, EY, E0; p, cY, nX, nY)
end

function SignalEmbedder(DX, EY, E0; p::Int, cY::Int, nX::Int, nY::Int)
    mY = nY - cY + 1
    return SignalEmbedder{p,cY,mY,nX,nY,typeof(DX),typeof(EY),typeof(E0)}(DX, EY, E0)
end

function (e::SignalEmbedder{p,cY,mY,nX,nY})(Y::AbstractMatrix, X = nothing) where {p,cY,mY,nX,nY}
    @assert (nX == 0 && isnothing(X)) || (nX != 0 && !isnothing(X))
    b = size(Y,2)
    Yshard = shard_array(Y, cY) # [nY x b] -> [cY x mY x b]
    Yenc = e.EY(reshape(Yshard, cY, mY*b)) # [cY x mY x b] -> [cY x mY*b] -> [p x mY*b]
    Yenc = reshape(Yenc, p, mY, b) # [p x mY*b] -> [p x mY x b]
    Zclass = zeros_similar(Y, p, 1, b)
    Z0 = if !isnothing(X)
        Xenc = e.DX(X) # [nX x b] -> [nX x b]
        # Xenc = repeat(reshape(Xenc, 1, nX, b), p, 1, 1) # [nX x b] -> [1 x nX x b] -> [p x nX x b]
        Xenc = ones_similar(Y, p, 1, 1) .* reshape(Xenc, 1, nX, b) #TODO [nX x b] -> [1 x nX x b] -> [p x nX x b]
        hcat(Zclass, Yenc, Xenc) # [p x (1 + mY + nX) x b]
    else
        hcat(Zclass, Yenc) # [p x (1 + mY + nX) x b] =  # [p x (1 + mY) x b]
    end
    return Z0 .+ e.E0
end

"""
Basic Transformer encoder with learned positional embedding
"""
function TransformerEncoder(MLPHead = nothing; nY = 48, nθ = 0, nZ = 0, pout = 16, psize = 16, chunksize = 16, head = 8, hdim = 256, nhidden = 4)
    topology = if nθ == 0 && nZ == 0
        Transformers.@nntopo(
            Y : # Input (nY × b)
            Y => E : # SignalEmbedder (psize × _ × b)
            E => H : # Transformer encoder (psize × _ × b)
            H => H : # Extract "class" token embedding (psize × b)
            H => C # MLP head mapping "class" embedding to output (pout x b)
        )
    elseif nθ == 0 || nZ == 0
        Transformers.@nntopo(
            (Y,X) : # Inputs (nY × b), (nX x b)
            (Y,X) => E : # SignalEmbedder (psize × _ × b)
            E => H : # Transformer encoder (psize × _ × b)
            H => H : # Extract "class" token embedding (psize × b)
            H => C # MLP head mapping "class" embedding to output (pout x b)
        )
    else
        Transformers.@nntopo(
            (Y,θ,Z) : # Inputs (nY × b), (nθ x b), (nZ x b)
            (θ,Z) => X : # Concatenate θ,Z inputs (nY × b), (nX x b)
            (Y,X) => E : # SignalEmbedder (psize × _ × b)
            E => H : # Transformer encoder (psize × _ × b)
            H => H : # Extract "class" token embedding (psize × b)
            H => C # MLP head mapping "class" embedding to output (pout x b)
        )
    end

    xf = Transformers.Stack(
        topology,
        (nθ != 0 && nZ != 0 ? [vcat] : [])...,
        SignalEmbedder(; p = psize, cY = chunksize, nX = nθ + nZ, nY = nY),
        Flux.Chain(map(1:nhidden) do _
            Transformers.Basic.Transformer(psize, head, hdim; future = true, act = Flux.relu, pdrop = 0.0)
        end...),
        H -> H[:, 1, :],
        (isnothing(MLPHead) ? MMDLearning.MLP(psize => pout, 0, hdim, Flux.relu, identity) : MLPHead),
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
function shard_array(Y::AbstractMatrix, chunksize::Int)
    chunk = 1:chunksize
    offset = 0:size(Y,1)-chunksize
    I = CartesianIndex.(chunk .+ offset')
    return Y[I,:]
end
