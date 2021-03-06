"""
batchsize(x::AbstractArray)

Returns the length of the last dimension of the data `x`.
"""
batchsize(x::AbstractArray{T,N}) where {T,N} = size(x, N)

"""
channelsize(x::AbstractArray)

Returns the length of the second-last dimension of the data `x`.
"""
channelsize(x::AbstractArray{T,N}) where {T,N} = N <= 1 ? 1 : size(x, N-1)

"""
heightsize(x::AbstractArray)

Returns the length of the first dimension of the data `x`.
"""
heightsize(x::AbstractArray) = size(x, 1)

"""
Kaiming uniform initialization.
"""
function kaiming_uniform(T::Type, dims; gain = 1)
   fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
   bound = sqrt(3) * gain / sqrt(fan_in)
   return rand(Uniform(-bound, bound), dims) |> Array{T}
end
kaiming_uniform(T::Type, dims...; kwargs...) = kaiming_uniform(T::Type, dims; kwargs...)
kaiming_uniform(args...; kwargs...) = kaiming_uniform(Float64, args...; kwargs...)

"""
Kaiming normal initialization.
"""
function kaiming_normal(T::Type, dims; gain = 1)
   fan_in = length(dims) <= 2 ? dims[end] : prod(dims) ÷ dims[end]
   std = gain / sqrt(fan_in)
   return rand(Normal(0, std), dims) |> Array{T}
end
kaiming_normal(T::Type, dims...; kwargs...) = kaiming_normal(T::Type, dims; kwargs...)
kaiming_normal(args...; kwargs...) = kaiming_normal(Float64, args...; kwargs...)

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
xavier_uniform(args...; kwargs...) = xavier_uniform(Float64, args...; kwargs...)

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
xavier_normal(args...; kwargs...) = xavier_normal(Float64, args...; kwargs...)

"""
wrapchain(layer)

Wraps `layer` in a `Flux.Chain`. No-op if `layer` is already a `Flux.Chain`.
"""
wrapchain(layer::Flux.Chain) = layer
wrapchain(layer) = Flux.Chain(layer)

"""
catchain(chain)

Concatenation of `c1` and `c2` into a `Flux.Chain`. If either or both of
`c1` and `c2` are chain, they will be splatted.
"""
catchain(c1::Flux.Chain, c2::Flux.Chain) = Flux.Chain(c1..., c2...)
catchain(c1::Flux.Chain, c2) = Flux.Chain(c1..., c2)
catchain(c1, c2::Flux.Chain) = Flux.Chain(c1, c2...)
catchain(c1, c2) = Flux.Chain(c1, c2)

"""
flattenchain(chain)

Recursively flattens `chain`, removing redundant `Chain` wrappers.
"""
flattenchain(chain::Flux.Chain) = length(chain) == 1 ? flattenchain(chain[1]) : Flux.Chain(reduce(catchain, flattenchain.(chain))...)
flattenchain(chain) = chain

"""
NotTrainable(layer)

Wraps the callable `layer` such that any parameters internal to `layer`
are ignored by Flux during gradient calls.
"""
struct NotTrainable{F}
    layer::F
end
Flux.@functor NotTrainable # need functor for e.g. `fmap`
Flux.trainable(l::NotTrainable) = () # no trainable parameters
(l::NotTrainable)(x...) = l.layer(x...)
Base.show(io::IO, l::NotTrainable) = (print(io, "NotTrainable("); print(io, l.layer); print(io, ")"))

# Helper function for gradient operators
ConstantFilter(args...; kwargs...) = NotTrainable(Flux.Chain(ChannelResize(1), Flux.Conv(args...; kwargs...), DenseResize()))

"""
CentralDifference()

Non-trainable central-difference layer which convolves the stencil [-1, 0, 1]
along the first dimension of `d x b` inputs, producing `(d-2) x b` outputs.
"""
CentralDifference() = ConstantFilter(reshape(Float32[-1.0, 0.0, 1.0], 3, 1, 1, 1), Float32[0.0], identity; stride = 1, pad = 0)
ForwardDifferemce() = ConstantFilter(reshape(Float32[1.0, -1.0], 2, 1, 1, 1), Float32[0.0], identity; stride = 1, pad = 0)
BackwardDifferemce() = ConstantFilter(reshape(Float32[-1.0, 1.0], 2, 1, 1, 1), Float32[0.0], identity; stride = 1, pad = 0)

"""
Laplacian()

Non-trainable central-difference layer which convolves the stencil [1, -2, 1]
along the first dimension of `d x b` inputs, producing `(d-2) x b` outputs.
"""
Laplacian() = ConstantFilter(reshape(Float32[1.0, -2.0, 1.0], 3, 1, 1, 1), Float32[0.0], identity; stride = 1, pad = 0)

"""
Scale(α = 1, β = zero(α))

Non-trainable wrapper of `Diagonal` layer. Output is `α .* x .+ β`.
"""
Scale(α = 1, β = zero(α)) = NotTrainable(Flux.Diagonal(α, β))

"""
CatScale(bd::Vector{<:NTuple{2}}, n::Vector{Int})

Create a `Scale` instance acting on inputs `x` of height `sum(n)`.
An affine transformation is constructed which represents `length(n)` separate
transformations of chunks of height `n[i]` of the input `x`.
Each transformation is such that values in `(-1,1)` are mapped to `(bd[i][1], bd[i][2])`.
Input/output bounds are not enforced.
"""
CatScale(bd::Vector{<:NTuple{2}}, n::Vector{Int}) = Scale(catscale_slope_and_bias(bd, n)...)
catscale_slope_and_bias(bd::Vector{<:NTuple{2}}, n::Vector{Int}) = catscale_slope(bd, n), catscale_bias(bd, n)
catscale_slope(bd::Vector{<:NTuple{2}}, n::Vector{Int}) = mapreduce(((bd, n)) -> fill((bd[2] - bd[1])/2, n), vcat, bd, n)
catscale_bias(bd::Vector{<:NTuple{2}}, n::Vector{Int}) = mapreduce(((bd, n)) -> fill((bd[1] + bd[2])/2, n), vcat, bd, n)

"""
wrapprint(io::IO, layer)
"""
wrapprint(io::IO, layer) = Flux.Chain(
    @λ(x -> (  print(io, "      layer: "); _model_summary(io, layer); print(io, "\n"); x)),
    @λ(x -> (println(io, " input size: $(size(x))"); x)),
    layer,
    @λ(x -> (println(io, "output size: $(size(x))"); x)),
)
wrapprint(layer) = wrapprint(stdout, layer)
wrapprint(layer::Flux.Chain) = Flux.Chain(wrapprint.(layer.layers)...)

"""
PrintSize()

Non-trainable layer which simply prints the current size.
"""
printsize(x) = (@show size(x); x)
PrintSize() = @λ (x -> printsize(x))

"""
DenseResize()

Non-trainable layer which reshapes input arguments `x` with dimensions `(d1, ..., dN)`
into matrices with `dN` columns.
"""
DenseResize() = Flux.flatten

"""
ChannelResize(c::Int)

Non-trainable layer which resizes input arguments `x` to be 4D-array with size
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
MLP

Multi-layer perceptron mapping inputs with height `sz[1]` to outputs with height `sz[2]`.
`Nhid+2` total dense layers are used with `Dhid` hidden nodes. The first `Nhid+1` layers
use activation `σhid` and the last layer uses activation `σout`. 
"""
function MLP(sz::Pair{Int,Int}, Nhid::Int, Dhid::Int, σhid = Flux.relu, σout = identity; skip = false, dropout = false, layernorm = false, initW = Flux.glorot_uniform, initb = Flux.zeros, initW_last = initW, initb_last = initb)
    maybedropout(l) = dropout > 0 ? Flux.Chain(l, Flux.Dropout(dropout)) : l
    maybelayernorm(l) = layernorm ? Flux.Chain(l, Flux.LayerNorm(Dhid)) : l
    MaybeResidualDense() = skip ?
        Flux.Chain(Flux.SkipConnection(Flux.Dense(Dhid, Dhid, identity; initW, initb), +), x -> σhid.(x)) : # x -> σhid.(x .+ W*x .+ b)
        Flux.Dense(Dhid, Dhid, σhid; initW, initb) # x -> σhid.(W*x .+ b)
    Flux.Chain(
        Flux.Dense(sz[1], Dhid, σhid; initW, initb) |> maybedropout |> maybelayernorm,
        [MaybeResidualDense() |> maybedropout |> maybelayernorm for _ in 1:Nhid]...,
        Flux.Dense(Dhid, sz[2], σout; initW = initW_last, initb = initb_last),
    ) |> flattenchain
end

"""
`Conv` wrapper which initializes using `xavier_uniform`
"""
XavierConv(k::NTuple{2,Int}, ch::Pair{Int,Int}, σ = identity; kwargs...) = XavierConv(Float64, k, ch, σ; kwargs...)
XavierConv(T, k::NTuple{2,Int}, ch::Pair{Int,Int}, σ = identity; kwargs...) = Flux.Conv(xavier_uniform(T, k..., ch...), zeros(T, ch[2]), σ; kwargs...)

"""
`ConvTranspose` wrapper which initializes using `xavier_uniform`
"""
XavierConvTrans(k::NTuple{2,Int}, ch::Pair{Int,Int}, σ = identity; kwargs...) = XavierConvTrans(Float64, k, ch, σ; kwargs...)
XavierConvTrans(T, k::NTuple{2,Int}, ch::Pair{Int,Int}, σ = identity; kwargs...) = Flux.ConvTranspose(xavier_uniform(T, k..., reverse(ch)...), zeros(T, ch[2]), σ; kwargs...)

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
    if mode === :pre
        Flux.Chain(BN(), AF(), CD()..., BN(), AF(), CD()...)
    elseif mode === :post
        Flux.Chain(CD()..., BN(), AF(), CD()..., BN(), AF())
    elseif mode === :hybrid
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
    CV(ch, σ = identity) = XavierConv(k, ch, σ; pad = (k.-1).÷2)
    BN(C,  σ = identity) = batchnorm ? Flux.BatchNorm(C, σ) : groupnorm ? Flux.GroupNorm(C, C÷2, σ) : identity
    AF() = @λ x -> σ.(x)
    if mode === :pre
        Flux.Chain(BN(ch[1]), AF(), CV(ch[1]=>ch[2]), vcat(([BN(ch[2]), AF(), CV(ch[2]=>ch[2])] for _ in 1:numlayers-2)...)..., BN(ch[2]), AF(), CV(ch[2]=>ch[2]))
    elseif mode === :post
        Flux.Chain(CV(ch[1]=>ch[1]), BN(ch[1]), AF(), vcat(([CV(ch[1]=>ch[1]), BN(ch[1]), AF()] for _ in 1:numlayers-2)...)..., CV(ch[1]=>ch[2]), BN(ch[2]), AF())
    elseif mode === :hybrid
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
        XavierConv((1,1), G0 + C * G => G0; pad = (0,0)),
    )
end
DenseConnection(G0::Int, G::Int, C::Int; dims::Int = 3, k::Tuple = (3,1), σ = Flux.relu) =
    DenseConnection(
        ch -> XavierConv(k, ch, σ; pad = (k.-1).÷2), # Default factory for RDB's
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
            # XavierConv(k, G0 => G0, σ; pad = (k.-1).÷2),
            GlobalFeatureFusion(
                dims,
                [ResidualDenseBlock(Factory, G0, G, C; dims = dims) for d in 1:D]...,
            ),
            # Flux.BatchNorm(D * G0, σ),
            Flux.GroupNorm(D * G0, (D * G0) ÷ 2, σ),
            XavierConv((1,1), D * G0 => G0; pad = (0,0)),
            # XavierConv(k, G0 => G0, σ; pad = (k.-1).÷2),
        )
    )
end
DenseFeatureFusion(G0::Int, G::Int, C::Int, D::Int, k::Tuple = (3,1), σ = Flux.relu; kwargs...) =
    DenseFeatureFusion(
        ch -> XavierConv(k, ch, σ; pad = (k.-1).÷2), # Default factory for RDB's
        G0, G, C, D, k, σ; kwargs...)

"""
Basic CNN with optional skip connection
"""
function RESCNN(sz::Pair{Int,Int}, Nhid::Int, Dhid::Int, σhid = Flux.relu, σout = identity; skip = false)
    Flux.Chain(
        x::AbstractMatrix -> reshape(x, sz[1], 1, 1, size(x,2)),
        Flux.Conv((3,1), 1=>Dhid, identity; pad = Flux.SamePad()),
        mapreduce(vcat, 1:Nhid÷2) do _
            convlayers = [Flux.Conv((3,1), Dhid=>Dhid, σhid; pad = Flux.SamePad()) for _ in 1:2]
            skip ? [Flux.SkipConnection(Flux.Chain(convlayers...), +)] : convlayers
        end...,
        Flux.Conv((1,1), Dhid=>1, identity; pad = Flux.SamePad()),
        x::AbstractArray{<:Any,4} -> reshape(x, sz[1], size(x,4)),
        Flux.Dense(sz[1], sz[2], σout),
    )
end

"""
Print all output activations for a `Transformers.Stack` for debugging purposes
"""
function stack_activations(m::Transformers.Stack, xs...; verbose = false)
    _m = Transformers.Stack(
        m.topo,
        map(enumerate(m.models)) do (i,f)
            function (args...,)
                @info "layer $i", "NaN params: $(any([any(isnan, p) for p in Flux.params(f)]))", "Inf params: $(any([any(isinf, p) for p in Flux.params(f)]))", typeof(f)
                for (j,arg) in enumerate(args)
                    @info "layer $i", "input $j", "NaN: $(any(isnan, arg))", "Inf: $(any(isinf, arg))", typeof(arg), size(arg)
                    verbose && display(arg)
                end
                y = f(args...)
                @info "layer $i", "output", "NaN: $(any(isnan, y))", "Inf: $(any(isinf, y))", typeof(y), size(y)
                verbose && display(y)
                println("")
                return y
            end
        end...
    )
    _m(xs...)
end

"""
Print model/layer
"""
function model_summary(io::IO, models::AbstractDict)
    for (i,(k,m)) in enumerate(models)
        (k != "") && println(io, string(k) * ":")
        _model_summary(io, m)
        _model_parameters(io, m)
        (i < length(models)) && println(io, "")
    end
end
function model_summary(models::AbstractDict, filename = nothing)
    @info "Model summary..."
    (filename !== nothing) && open(filename, "w") do file
        model_summary(file, models)
    end
    model_summary(stdout, models)
end
model_summary(model, filename = nothing) = model_summary(Dict("" => model), filename)

"""
Print model parameters following `_model_summary`
"""
function _model_parameters(io::IO, model; depth::Int = 0)
    if depth == 0
        nparams = mapreduce(length, +, Flux.params(model); init = 0)
        println(io, "\nParameters: $nparams")
    end
end

"""
Indenting for layer `depth`
"""
_getprefix(depth::Int, pre = "", suf = "") = pre * "    " ^ depth * suf

"""
Recursively print model/layer

Note: All models implementing `_model_summary` should not end the printing on a new line.
      This is so that, during the recursive printing, parent callers may add commas, etc.,
      following printing. A final newline will be added in the `_model_parameters` function,
      called inside the `model_summary` parent function.
"""
function _model_summary(io::IO, model; depth::Int = 0, pre = "", suf = "")
    println(io, _getprefix(depth, pre, suf * "$(typeof(model).name)("))
    fs = fieldnames(typeof(model))
    for (i,f) in enumerate(fs)
        _model_summary(io, getfield(model, f); depth = depth + 1, suf = "$f: ")
        println(io, i < length(fs) ? "," : "")
    end
    print(io, _getprefix(depth, pre, ")"))
end

# Flux.Chain
function _model_summary(io::IO, model::Flux.Chain; depth::Int = 0, pre = "", suf = "")
    println(io, _getprefix(depth, pre, suf * "Chain("))
    for (i,layer) in enumerate(model)
        _model_summary(io, layer; depth = depth + 1, suf = "$i: ")
        (i < length(model)) ? println(io, ",") : println(io, "")
    end
    print(io, _getprefix(depth, pre, ")"))
    nothing
end

# Transformers.NNTopo
function _model_summary(io::IO, model::Transformers.Stacks.NNTopo; depth::Int = 0, pre = "", suf = "")
    # Workaround (https://github.com/chengchingwen/Transformers.jl/pull/32)
    topo_print = let original_stdout = stdout
        read_pipe, write_pipe = redirect_stdout()
        try
            show(model)
        finally
            close(write_pipe)
            redirect_stdout(original_stdout)
        end
        read(read_pipe, String)
    end
    topo_print = _getprefix(depth, pre, suf) * topo_print
    topo_print = replace(topo_print, "\t" => _getprefix(1, pre, suf))
    topo_print = replace(topo_print, "end\n" => "end")
    topo_print = replace(topo_print, "\n" => "\n" * _getprefix(depth, pre, suf))
    print(io, topo_print)
end

# Transformers.Stack
function _model_summary(io::IO, model::Transformers.Stack; depth::Int = 0, pre = "", suf = "")
    println(io, _getprefix(depth, pre, suf * "Stack("))
    _model_summary(io, model.topo; depth = depth + 1)
    println(io, ",")
    for (i,layer) in enumerate(model.models)
        _model_summary(io, layer; depth = depth + 1, suf = "$i: ")
        (i < length(model.models)) ? println(io, ",") : println(io, "")
    end
    print(io, _getprefix(depth, pre, ")"))
end

# Arrays
function _model_summary(io::IO, model::AbstractArray; depth::Int = 0, pre = "", suf = "")
    print(io, _getprefix(depth, pre, suf * "size " * string(size(model)) * " " * string(typeof(model))))
end

# Numbers, Symbols, ...
function _model_summary(io::IO, model::Union{<:Number, Nothing, Symbol, String}; depth::Int = 0, pre = "", suf = "")
    print(io, _getprefix(depth, pre, suf * "$model :: $(typeof(model))"))
end

# Functions
function _model_summary(io::IO, model::Function; depth::Int = 0, pre = "", suf = "")
    print(io, _getprefix(depth, pre, suf * "@λ "))
    show(io, model)
end
