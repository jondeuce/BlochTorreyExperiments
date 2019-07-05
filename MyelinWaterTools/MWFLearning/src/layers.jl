"""
Print model
"""
function model_summary(io::IO, model, filename = nothing)
    @info "Model summary..."
    (filename != nothing) && open(filename, "w") do file
        _model_summary(file, model)
    end
    _model_summary(io, model)
end
model_summary(model, filename = nothing) = model_summary(stdout, model, filename)

"""
Recursively print model
"""
_model_summary(io::IO, model, depth::Int = 0) = println(io, model) # fallback
getindent(depth) = reduce(*, "    " for _ in 1:depth; init = "")

function _model_summary(io::IO, model::Flux.Chain, depth::Int = 0)
    indent = getindent(depth)
    for layer in model
        if layer isa Flux.Chain
            println(io, indent * "Chain(")
            _model_summary(io, layer, depth + 1)
            println(io, indent * ")")
        elseif layer != identity
            print(io, indent)
            _model_summary(io, layer, depth)
        end
    end
    if depth == 0
        println(io, "\nParameters: $(sum(length, Flux.params(model)))")
    end
    nothing
end

function _model_summary(io::IO, model::Flux.SkipConnection, depth::Int = 0)
    println(io, getindent(depth) * "SkipConnection(")
    print(io, getindent(depth + 1) * "λ = ")
    print(io, model.connection)
    println(",")
    _model_summary(io, model.layers, depth + 1)
    println(io, getindent(depth) * ")")
end

"""
PrintSize()

Non-learnable layer which simply prints the current size.
"""
printsize(x) = (@show size(x); x)
PrintSize() = @λ(x -> printsize(x))

"""
DenseResize()

Non-learnable layer which resizes input arguments `x` to be a matrix with batchsize(x) columns.
"""
struct DenseResize end
Flux.@treelike DenseResize
(l::DenseResize)(x::AbstractArray) = reshape(x, :, batchsize(x))
Base.show(io::IO, l::DenseResize) = print(io, "DenseResize()")

"""
ChannelResize(c::Int)

Non-learnable layer which resizes input arguments `x` to be an array with `c` channels.
The data height is divided by `c`.
"""
struct ChannelResize
    c::Int
end
Flux.@treelike ChannelResize
(l::ChannelResize)(x::AbstractArray) = reshape(x, heightsize(x) ÷ l.c, l.c, :)
Base.show(io::IO, l::ChannelResize) = print(io, "ChannelResize(", l.c, ")")

"""
Scale(s::AbstractArray)

Non-learnable layer which scales input `x` by array `s`
"""
struct Scale{V}
    s::V
end
Scale(s::Flux.TrackedArray) = Scale(Flux.data(s)) # Layer is not learnable
Flux.@treelike Scale
(l::Scale)(x::AbstractArray) = x .* l.s
Base.show(io::IO, l::Scale) = print(io, "Scale(", length(l.s), ")")

"""
ChannelwiseDense
"""
ChannelwiseDense(H::Int, ch::Pair, σ = identity) = Flux.Chain(DenseResize(), Flux.Dense(H*ch[1], H*ch[2]), ChannelResize(ch[2]))

"""
IdentitySkip

`ResNet`-type skip-connection with identity shortcut.
Wraps `SkipConnection` from the Flux library.
"""
IdentitySkip(layer) = Flux.SkipConnection(layer, @λ (a,b) -> a + b)

"""
CatSkip

`DenseNet`-type skip-connection with concatenation shortcut along dimension `dim`.
Wraps `SkipConnection` from the Flux library.
"""
CatSkip(layer, dims = 1) = Flux.SkipConnection(layer, @λ (a,b) -> cat(a, b; dims = dims))

"""
ResidualBlock

Input has size H x C x B, for some batch size B.
"""
function ResidualBlock(H::Int, C::Int, mode::Symbol = :post, bn::Bool = false, σ = Flux.relu)
    ApplyDense(σ = identity) = ChannelwiseDense(H, C=>C, σ).layers
    layer = if mode == :pre
        BN = bn ? Flux.BatchNorm(C, σ) : identity
        Flux.Chain(
            BN,                   # Flux.BatchNorm(ch[2], σ),
            ApplyDense()...,      # Flux.Conv(k, ch[2]=>ch[2], pad=(k.-1).÷2),
            BN,                   # Flux.BatchNorm(ch[2], σ),
            ApplyDense()...,      # Flux.Conv(k, ch[2]=>ch[2], pad=(k.-1).÷2),
        )
    elseif mode == :post
        BN = bn ? Flux.BatchNorm(C, σ) : identity
        Flux.Chain(
            ApplyDense()...,      # Flux.Conv(k, ch[2]=>ch[2], pad=(k.-1).÷2),
            BN,                   # Flux.BatchNorm(ch[2], σ),
            ApplyDense()...,      # Flux.Conv(k, ch[2]=>ch[2], pad=(k.-1).÷2),
            BN,                   # Flux.BatchNorm(ch[2], σ),
        )
    elseif mode == :hybrid
        BN = bn ? Flux.BatchNorm(C) : identity
        Flux.Chain(
            BN,                   # Flux.BatchNorm(ch[2]),
            ApplyDense(σ)...,     # Flux.Conv(k, ch[2]=>ch[2], pad=(k.-1).÷2, σ),
            BN,                   # Flux.BatchNorm(ch[2]),
            ApplyDense()...,      # Flux.Conv(k, ch[2]=>ch[2], pad=(k.-1).÷2),
        )
    else
        error("Unknown ResidualBlock mode $mode")
    end
    IdentitySkip(layer)
end