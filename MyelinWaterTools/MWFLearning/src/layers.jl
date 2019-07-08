"""
PrintSize()

Non-learnable layer which simply prints the current size.
"""
printsize(x) = (@show size(x); x)
PrintSize() = @λ (x -> printsize(x))

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
    Sumout(over)
`Sumout` is a neural network layer, which has a number of internal layers,
which all have the same input, and returns the elementwise sum of the
internal layers' outputs.
"""
struct Sumout{FS<:Tuple}
    over::FS
end
Sumout(args...) = Sumout(args)
Base.show(io::IO, so::Sumout) = (print(io, "Sumout("); join(io, so.over, ", "); print(io, ")"))

Flux.@treelike Sumout

function (mo::Sumout)(input::AbstractArray)
    mapreduce(f -> f(input), (acc, out) -> acc + out, mo.over)
end

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
CatSkip(dims, layer) = Flux.SkipConnection(layer, @λ (a,b) -> cat(a, b; dims = dims))

"""
DenseCatSkip
"""
function DenseCatSkip(Factory, k::Tuple, ch::Pair, σ = Flux.relu; dims::Int = 2, depth::Int = 1)
    Downsample(ch) = Flux.Conv(k, ch, σ; pad = (k .- 1) .÷ 2)
    DenseBlock(ch) = CatSkip(dims, Flux.Chain(Downsample(ch), Factory()))
    return Flux.Chain(
        [DenseBlock(i * ch[1] => ch[1]) for i in 1:depth]...,
        Downsample((depth + 1) * ch[1] => ch[2]),
    )
end
DenseCatSkip(Factory, k::Int, args...; kwargs...) = DenseCatSkip(Factory, (k,), args...; kwargs...)
DenseCatSkip(Factory, k, C::Int, args...; kwargs...) = DenseCatSkip(Factory, k, C=>C, args...; kwargs...)

"""
DenseResConnection

Input has size H1 x ... x HN x C x B, for some data size (H1,...,HN), channels C,
and batch size B.
"""
function DenseResConnection(
        H::Int, C::Int, σ = Flux.relu;
        mode::Symbol = :hybrid,
        groupnorm::Bool = false,
        batchnorm::Bool = false,
    )
    CD(σ = identity) = ChannelwiseDense(H, C=>C, σ).layers
    BN(σ = identity) = batchnorm ? Flux.BatchNorm(C, σ) : groupnorm ? Flux.GroupNorm(C, C÷2, σ) : identity
    AF() = @λ x -> σ.(x)
    if mode == :pre
        Flux.Chain(BN(), AF(), CD()..., BN(), AF(), CD()...)
    elseif mode == :post
        Flux.Chain(CD()..., BN(), AF(), CD()..., BN(), AF())
    elseif mode == :hybrid
        Flux.Chain(BN(), CD(σ)..., BN(), CD()...)
    else
        error("Unknown DenseResConnection mode $mode")
    end
end
DenseResConnection(H::Tuple, args...; kwargs...) = DenseResConnection(prod(H), args...; kwargs...)

"""
ConvResConnection

Input has size H1 x ... x HN x C x B, for some data size (H1,...,HN), channels C,
and batch size B.
"""
function ConvResConnection(
        k::Tuple, ch::Pair, σ = Flux.relu;
        mode::Symbol = :hybrid,
        numlayers::Int = 3,
        groupnorm::Bool = false,
        batchnorm::Bool = false,
    )
    @assert numlayers >= 2
    CV(ch, σ = identity) = Flux.Conv(k, ch, σ; pad = (k.-1).÷2)
    BN(C,  σ = identity) = batchnorm ? Flux.BatchNorm(C, σ) : groupnorm ? Flux.GroupNorm(C, C÷2, σ) : identity
    AF() = @λ x -> σ.(x)
    if mode == :pre
        Flux.Chain(BN(ch[1]), AF(), CV(ch[1]=>ch[2]), vcat(([BN(ch[2]), AF(), CV(ch[2]=>ch[2])] for _ in 1:numlayers-2)...)..., BN(ch[2]), AF(), CV(ch[2]=>ch[1]))
    elseif mode == :post
        Flux.Chain(CV(ch[1]=>ch[2]), BN(ch[2]), AF(), vcat(([CV(ch[2]=>ch[2]), BN(ch[2]), AF()] for _ in 1:numlayers-2)...)..., CV(ch[2]=>ch[1]), BN(ch[1]), AF())
    elseif mode == :hybrid
        Flux.Chain(BN(ch[1]), CV(ch[1]=>ch[2], σ),    vcat(([BN(ch[2]), CV(ch[2]=>ch[2], σ)]    for _ in 1:numlayers-2)...)..., BN(ch[2]), CV(ch[2]=>ch[1]))
    else
        error("Unknown DenseResConnection mode $mode")
    end
end
ConvResConnection(k::Int, args...; kwargs...) = ConvResConnection((k,), args...; kwargs...)
ConvResConnection(k, C::Int, args...; kwargs...) = ConvResConnection(k, C=>C, args...; kwargs...)

"""
Print model/layer
"""
function model_summary(io::IO, model, filename = nothing; kwargs...)
    @info "Model summary..."
    (filename != nothing) && open(filename, "w") do file
        _model_summary(file, model; kwargs...)
        _model_parameters(file, model)
    end
    _model_summary(io, model; kwargs...)
    _model_parameters(io, model)
end
model_summary(model, filename = nothing; kwargs...) = model_summary(stdout, model, filename; kwargs...)

"""
Print model parameters following `_model_summary`
"""
function _model_parameters(io::IO, model, depth::Int = 0)
    if depth == 0
        nparams = reduce(+, length.(Flux.params(model)); init = 0)
        println(io, "\nParameters: $nparams")
    end
end

"""
Recursively print model/layer

Note: All models implementing `_model_summary` should not end the printing on a new line.
      This is so that, during the recursive printing, parent callers may add commas, etc.,
      following printing. A final newline will be added in the `_model_parameters` function,
      called inside the `model_summary` parent function.
"""
function _model_summary(io::IO, model::Flux.Chain, depth::Int = 0; skipidentity = false)
    println(io, getindent(depth) * "Chain(")
    for (i,layer) in enumerate(model)
        if !(skipidentity && layer == identity)
            _model_summary(io, layer, depth+1; skipidentity = skipidentity)
            (i < length(model)) ? println(io, ",") : println(io, "")
        end
    end
    print(io, getindent(depth) * ")")
    nothing
end

# SkipConnection
function _model_summary(io::IO, model::Flux.SkipConnection, depth::Int = 0; kwargs...)
    println(io, getindent(depth) * "SkipConnection(")
    _model_summary(io, model.layers, depth+1; kwargs...)
    println(io, ",")
    _model_summary(io, model.connection, depth+1; kwargs...)
    println(io, "")
    print(io, getindent(depth) * ")")
end

# Sumout
function _model_summary(io::IO, model::Sumout, depth::Int = 0; kwargs...)
    println(io, getindent(depth) * "Sumout(")
    for (i,layer) in enumerate(model.over)
        _model_summary(io, layer, depth+1; kwargs...)
        (i < length(model.over)) ? println(io, ",") : println(io, "")
    end
    print(io, getindent(depth) * ")")
end

# Fallback method
function _model_summary(io::IO, model::Function, depth::Int = 0; kwargs...)
    print(io, getindent(depth) * "@λ ")
    print(io, model)
end

# Fallback method
function _model_summary(io::IO, model, depth::Int = 0; kwargs...)
    print(io, getindent(depth))
    print(io, model)
end

"""
Indenting for layer `depth`
"""
getindent(depth) = reduce(*, "    " for _ in 1:depth; init = "")
