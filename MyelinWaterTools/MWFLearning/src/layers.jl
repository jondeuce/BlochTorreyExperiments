"""
    dropout(x, p, dims = :)

Fix definition of dropout kernel to apply dropout on both forward and
reverse passes. Test/train mode is handled by [`Dropout`](@ref).

See: https://github.com/FluxML/Flux.jl/issues/1084
"""
function Flux.dropout(x, p; dims = :)
    q = 1 - p
    y = rand!(similar(x, Flux._dropout_shape(x, dims)))
    y .= Flux._dropout_kernel.(y, p, q)
    x .* y
end

"""
    wrapprint(io::IO, layer)
"""    
wrapprint(io::IO, layer) = Flux.Chain(
    @λ(x -> (  print(io, "      layer: "); MWFLearning._model_summary(io, layer); print(io, "\n"); x)),
    @λ(x -> (println(io, " input size: $(size(x))"); x)),
    layer,
    @λ(x -> (println(io, "output size: $(size(x))"); x)),
)
wrapprint(layer) = wrapprint(stdout, layer)
wrapprint(layer::Flux.Chain) = Flux.Chain(wrapprint.(layer.layers)...)

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
Flux.@functor DenseResize
(l::DenseResize)(x::AbstractArray) = reshape(x, :, batchsize(x))
Base.show(io::IO, l::DenseResize) = print(io, "DenseResize()")

"""
ChannelResize(c::Int)

Non-learnable layer which resizes input arguments `x` to be 4D-array with size
`d` x 1 x `c` x `b`, where `c` is the desired channels, `b` is the batch size,
and `d` is `length(x) ÷ (c x b)`.
"""
struct ChannelResize
    c::Int
end
Flux.@functor ChannelResize
(l::ChannelResize)(x::AbstractArray) = reshape(x, :, 1, l.c, batchsize(x))
Base.show(io::IO, l::ChannelResize) = print(io, "ChannelResize(", l.c, ")")

"""
Scale(s::AbstractArray)

Non-learnable layer which scales input `x` by array `s`
"""
struct Scale{V}
    s::V
end
trainable(l::Scale) = () # Layer is not learnable
Flux.@functor Scale
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

Flux.@functor Sumout

function (mo::Sumout)(input::AbstractArray)
    mapreduce(f -> f(input), (acc, out) -> acc + out, mo.over)
end

"""
    MultiInput(layers...)
Applies `layers` to each element of a tuple input.
    See: https://github.com/FluxML/Flux.jl/pull/776
"""
struct MultiInput{T<:Tuple}
    layers::T
    MultiInput(xs...) = new{typeof(xs)}(xs)
end

Flux.@forward MultiInput.layers Base.getindex, Base.length, Base.first, Base.last, Base.iterate, Base.lastindex

Flux.functor(m::MultiInput) = m.layers, ls -> MultiInput(ls...)

(m::MultiInput)(xs) = map((layer, x) -> layer(x), m.layers, xs)

Base.getindex(m::MultiInput, i::AbstractArray) = MultiInput(m.layers[i]...)

function Base.show(io::IO, m::MultiInput)
    print(io, "MultiInput(")
    join(io, m.layers, ", ")
    print(io, ")")
end

"""
    Fanout(N::Int)
Repeat input `x`, outputing an N-tuple.
    See: https://github.com/FluxML/Flux.jl/pull/776
"""
struct Fanout{N} end
Fanout(N::Int) = Fanout{N}()
(f::Fanout{N})(x) where {N} = ntuple(_ -> x, N)
Base.show(io::IO, m::Fanout{N}) where {N} = print(io, "Fanout($N)")

"""
ChannelwiseDense
"""
ChannelwiseDense(H::Int, ch::Pair, σ = identity) = Flux.Chain(DenseResize(), Flux.Dense(H*ch[1], H*ch[2], σ), ChannelResize(ch[2]))

"""
HeightwiseDense
"""
HeightwiseDense(H::Int, C::Int, σ = identity) = Flux.Chain(@λ(x -> reshape(x, H, :)), Flux.Dense(H, H, σ), @λ(x -> reshape(x, H, 1, C, :)))

"""
IdentitySkip

`ResNet`-type skip-connection with identity shortcut.
Wraps `SkipConnection` from the Flux library.
"""
IdentitySkip(layer::Flux.Chain) = Flux.SkipConnection(layer, @λ (a,b) -> a + b)
IdentitySkip(layer) = IdentitySkip(Flux.Chain(layer))

"""
CatSkip

`DenseNet`-type skip-connection with concatenation shortcut along dimension `dim`.
Wraps `SkipConnection` from the Flux library.
"""
CatSkip(dims, layer::Flux.Chain) = Flux.SkipConnection(layer, @λ (a,b) -> cat(a, b; dims = dims))
CatSkip(dims, layer) = CatSkip(dims, Flux.Chain(layer))

"""
BatchDenseConnection

Input has size H1 x ... x HN x C x B, for some data size (H1,...,HN), channels C,
and batch size B.
"""
function BatchDenseConnection(
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
        error("Unknown BatchDenseConnection mode $mode")
    end
end
BatchDenseConnection(H::Tuple, args...; kwargs...) = BatchDenseConnection(prod(H), args...; kwargs...)

"""
BatchConvConnection

Input size:     H1 x ... x HN x ch[1] x B
Output size:    H1 x ... x HN x ch[2] x B
"""
function BatchConvConnection(
        k::Tuple, ch::Pair, σ = Flux.relu;
        mode::Symbol = :pre,
        numlayers::Int = 3,
        groupnorm::Bool = false,
        batchnorm::Bool = false,
    )
    @assert numlayers >= 2 && !(groupnorm && batchnorm)
    CV(ch, σ = identity) = Flux.Conv(k, ch, σ; init = xavier_uniform, pad = (k.-1).÷2)
    BN(C,  σ = identity) = batchnorm ? Flux.BatchNorm(C, σ) : groupnorm ? Flux.GroupNorm(C, C÷2, σ) : identity
    AF() = @λ x -> σ.(x)
    if mode == :pre
        Flux.Chain(BN(ch[1]), AF(), CV(ch[1]=>ch[2]), vcat(([BN(ch[2]), AF(), CV(ch[2]=>ch[2])] for _ in 1:numlayers-2)...)..., BN(ch[2]), AF(), CV(ch[2]=>ch[2]))
    elseif mode == :post
        Flux.Chain(CV(ch[1]=>ch[1]), BN(ch[1]), AF(), vcat(([CV(ch[1]=>ch[1]), BN(ch[1]), AF()] for _ in 1:numlayers-2)...)..., CV(ch[1]=>ch[2]), BN(ch[2]), AF())
    elseif mode == :hybrid
        Flux.Chain(BN(ch[1]), CV(ch[1]=>ch[1], σ),    vcat(([BN(ch[1]), CV(ch[1]=>ch[1], σ)]    for _ in 1:numlayers-2)...)..., BN(ch[1]), CV(ch[1]=>ch[2]))
    else
        error("Unknown BatchDenseConnection mode $mode")
    end
end
BatchConvConnection(k::Int, args...; kwargs...) = BatchConvConnection((k,), args...; kwargs...)
BatchConvConnection(k, C::Int, args...; kwargs...) = BatchConvConnection(k, C=>C, args...; kwargs...)

"""
DenseConnection (Figure 3: https://arxiv.org/abs/1802.08797)

Densely connected layers, concatenating along the feature dimension, followed by
feature fusion via 1x1 convolution:
    `Factory`   Function which takes in a pair `ch` and outputs a layer which maps
                input data with `ch[1]` channels to output data with `ch[2]` channels
    `G0`        Number of channels of input and output data
    `G`         Channel growth rate
    `C`         Number of densely connected `Factory`s
    `dims`      Concatenation dimension
"""
function DenseConnection(Factory, G0::Int, G::Int, C::Int; dims::Int = 3)
    Flux.Chain(
        [CatSkip(dims, Factory(G0 + (c - 1) * G => G)) for c in 1:C]...,
        Flux.Conv((1,1), G0 + C * G => G0, identity; init = xavier_uniform, pad = (0,0)),
    )
end
DenseConnection(G0::Int, G::Int, C::Int; dims::Int = 3, k::Tuple = (3,1), σ = Flux.relu) =
    DenseConnection(
        ch -> Flux.Conv(k, ch, σ; init = xavier_uniform, pad = (k.-1).÷2), # Default factory for RDB's
        G0, G, C; dims = dims)

"""
ResidualDenseBlock (Figure 3: https://arxiv.org/abs/1802.08797)

Residual-type connection on a `DenseConnection` block
"""
ResidualDenseBlock(args...; kwargs...) = IdentitySkip(DenseConnection(args...; kwargs...))

"""
GlobalFeatureFusion (Figure 2: https://arxiv.org/abs/1802.08797)

Performs global feature fusion over the successive output of the `layers`,
concatenating the individual outputs over the feature dimension `dims`:

    F0 -> layers[1] -> F1 -> layers[2] -> ... -> layers[D] -> FD

where F0 is the input. The output is then simply `cat(F1, ..., FD; dims = dims)`.
"""
struct GlobalFeatureFusion{FS<:Tuple}
    dims::Int
    layers::FS
end
GlobalFeatureFusion(dims::Int, args...) = GlobalFeatureFusion(dims, args)
Base.show(io::IO, GFF::GlobalFeatureFusion) = (print(io, "GlobalFeatureFusion($(GFF.dims), "); join(io, GFF.layers, ", "); print(io, ")"))

Flux.@functor GlobalFeatureFusion

function (GFF::GlobalFeatureFusion)(x::AbstractArray)
    y = GFF.layers[1](x)
    out = y
    for d in 2:length(GFF.layers)
        y = GFF.layers[d](y)
        out = cat(out, y; dims = GFF.dims)
    end
    out
end

"""
DenseFeatureFusion (Figure 2: https://arxiv.org/abs/1802.08797)

From the referenced paper:

    'After extracting hierarchical features with a set of RDBs, we further conduct
    dense feature fusion (DFF), which includes global feature fusion (GFF) and
    global residual learning learning (GRL). DFF makes full use of features from
    all the preceding layers...'

The structure is

    F_{-1} -> 3x3 Conv -> F_{0} -> GlobalFeatureFusion -> 1x1 Conv -> 3x3 Conv -> F_{GF}

where the output - the densely fused features - is then given by

    F_{DF} = F_{-1} + F_{GF}
"""
function DenseFeatureFusion(Factory, G0::Int, G::Int, C::Int, D::Int, k::Tuple = (3,1), σ = Flux.relu; dims::Int = 3)
    IdentitySkip(
        Flux.Chain(
            # Flux.Conv(k, G0 => G0, σ; init = xavier_uniform, pad = (k.-1).÷2),
            GlobalFeatureFusion(
                dims,
                [ResidualDenseBlock(Factory, G0, G, C; dims = dims) for d in 1:D]...,
            ),
            # Flux.BatchNorm(D * G0, σ),
            Flux.GroupNorm(D * G0, (D * G0) ÷ 2, σ),
            Flux.Conv((1,1), D * G0 => G0, identity; init = xavier_uniform, pad = (0,0)),
            # Flux.Conv(k, G0 => G0, σ; init = xavier_uniform, pad = (k.-1).÷2),
        )
    )
end
DenseFeatureFusion(G0::Int, G::Int, C::Int, D::Int, k::Tuple = (3,1), σ = Flux.relu; kwargs...) =
    DenseFeatureFusion(
        ch -> Flux.Conv(k, ch, σ; init = xavier_uniform, pad = (k.-1).÷2), # Default factory for RDB's
        G0, G, C, D, k, σ; kwargs...)

"""
Print model/layer
"""
function model_summary(io::IO, models::AbstractArray; kwargs...)
    for (i,m) in enumerate(models)
        _model_summary(io, m; kwargs...)
        _model_parameters(io, m)
        (i < length(models)) && println(io, "")
    end
end
function model_summary(models::AbstractArray, filename = nothing; kwargs...)
    @info "Model summary..."
    (filename != nothing) && open(filename, "w") do file
        model_summary(file, models; kwargs...)
    end
    model_summary(stdout, models; kwargs...)
end
model_summary(model, filename = nothing; kwargs...) =
    model_summary([model], filename; kwargs...)

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

"""
Indenting for layer `depth`
"""
getindent(depth) = reduce(*, ["    " for _ in 1:depth]; init = "")

# SkipConnection
function _model_summary(io::IO, model::Flux.SkipConnection, depth::Int = 0; kwargs...)
    println(io, getindent(depth) * "SkipConnection(")
    _model_summary(io, model.layers, depth+1; kwargs...)
    println(io, ",")
    _model_summary(io, model.connection, depth+1; kwargs...)
    println(io, "")
    print(io, getindent(depth) * ")")
end

# GlobalFeatureFusion
function _model_summary(io::IO, model::GlobalFeatureFusion, depth::Int = 0; kwargs...)
    println(io, getindent(depth) * "GlobalFeatureFusion(")
    _model_summary(io, model.dims, depth+1; kwargs...)
    println(io, ",")
    for (d,layer) in enumerate(model.layers)
        _model_summary(io, layer, depth + 1; kwargs...)
        (d < length(model.layers)) ? println(io, ",") : println(io, "")
    end
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

# MultiInput
function _model_summary(io::IO, model::MultiInput, depth::Int = 0; kwargs...)
    println(io, getindent(depth) * "MultiInput(")
    for (i,layer) in enumerate(model.layers)
        _model_summary(io, layer, depth+1; kwargs...)
        (i < length(model.layers)) ? println(io, ",") : println(io, "")
    end
    print(io, getindent(depth) * ")")
end

# Functions
function _model_summary(io::IO, model::Function, depth::Int = 0; kwargs...)
    print(io, getindent(depth) * "@λ ")
    buf = IOBuffer()
    show(buf, model)
    print(io, String(take!(buf)))
end

# Fallback method
function _model_summary(io::IO, model, depth::Int = 0; kwargs...)
    print(io, getindent(depth))
    buf = IOBuffer()
    show(buf, model)
    print(io, String(take!(buf)))
end
