"""
batchsize(x::AbstractArray)

Returns the size of the last dimension if ndims(x) >= 2, else returns 1.
"""
batchsize(x::AbstractArray{T,N}) where {T,N} = N <= 1 ? 1 : size(x, N)

"""
channelsize(x::AbstractArray)

Returns the size of the second-last dimension if ndims(x) >= 3, else returns 1.
"""
channelsize(x::AbstractArray{T,N}) where {T,N} = N <= 2 ? 1 : size(x, N-1)

"""
heightsize(x::AbstractArray)

Returns the size of the first dimension.
"""
heightsize(x::AbstractArray) = size(x, 1)

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

Wraps the callable `layer` such any that parameters internal to `layer`
are treated as constant by Flux during training.
"""
struct NotTrainable{F}
    layer::F
end
Flux.@functor NotTrainable # need functor for e.g. `fmap`
Flux.trainable(::NotTrainable) = () # no trainable parameters
Base.show(io::IO, l::NotTrainable) = (print(io, "NotTrainable("); print(io, l.layer); print(io, ")"))

(l::NotTrainable)(xs...) = whitebox_apply(l.layer, xs...)

whitebox_apply(f, xs...) = f(xs...)

Zygote.@adjoint function whitebox_apply(f, xs...)
    y, J = Zygote.pullback(f, xs...)
    y, Δ -> (nothing, J(Δ)...)
end

"""
Scale(α = 1, β = zero(α))

Non-trainable wrapper of `Flux.Diagonal` layer. Output is `α .* x .+ β`.
"""
Scale(α = 1, β = zero(α)) = NotTrainable(Flux.Diagonal(α, β))

"""
CatScale(intervals::Vector{<:PairOfTuples{2}}, n::Vector{Int})

Create a `Scale` instance acting on inputs `x` of height `sum(n)`.
An affine transformation is constructed which represents `length(intervals) == length(n)`
separate affine transformations to chunks of height `n[i]` of the input `x`.
The `i`th pair of tuples `intervals[i]` defines an affine transformation such that
the interval defined by the tuple `intervals[i][1]` is mapped to `intervals[i][2]`.

For example,

    CatScale([(-1,1) => (0,1), (0,1) => (1,-1)], [2, 4])

applies the function `y = (x+1)/2` to the first 2 rows and `y = -2x+1`
to the next 4 rows of the inputs with first dimension of height 6.
"""
CatScale(intervals::AbstractVector, n::AbstractVector{Int}) = Scale(catscale_slope_and_bias(intervals, n)...)
catscale_slope_and_bias(intervals::AbstractVector, n::AbstractVector{Int}) = catscale_slope(intervals, n), catscale_bias(intervals, n)
catscale_slope(intervals::AbstractVector, n::AbstractVector{Int}) = mapreduce(((t, n)) -> fill(linear_xform_slope(t), n), vcat, intervals, n)
catscale_bias(intervals::AbstractVector, n::AbstractVector{Int}) = mapreduce(((t, n)) -> fill(linear_xform_bias(t), n), vcat, intervals, n)

@inline linear_xform_slope_and_bias(t) = linear_xform_slope_and_bias(_unpack_xform(t)...)
@inline linear_xform_slope(t) = linear_xform_slope(_unpack_xform(t)...)
@inline linear_xform_bias(t) = linear_xform_bias(_unpack_xform(t)...)
@inline linear_xform_slope_and_bias(x1,x2,y1,y2) = linear_xform_slope(x1,x2,y1,y2), linear_xform_bias(x1,x2,y1,y2)
@inline linear_xform_slope(x1,x2,y1,y2) = (y2 - y1) / (x2 - x1)
@inline linear_xform_bias(x1,x2,y1,y2) = (y1*x2 - y2*x1) / (x2 - x1)
@inline _unpack_xform(((x1,x2), (y1,y2))) = float.(promote(x1,x2,y1,y2))

"""
Resize(dims...)

Non-trainable layer which reshapes input argument `x` as `reshape(x, dims...)`.
"""
struct Resize{dims} end
Resize(dims...) = Resize{dims}()
Base.show(io::IO, ::Resize{dims}) where {dims} = print(io, "Resize(" * join(dims, ", ") * ")")
(r::Resize{dims})(x) where {dims} = reshape(x, dims...)

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
IdentitySkip

`ResNet`-type skip-connection with identity shortcut.
Wraps `SkipConnection` from the Flux library.
"""
IdentitySkip(layer) = Flux.SkipConnection(layer, +)

"""
CatSkip

`DenseNet`-type skip-connection with concatenation shortcut along dimensions `dims`.
Wraps `SkipConnection` from the Flux library.
"""
CatSkip(dims::Int, layer) = Flux.SkipConnection(layer, dims == 1 ? vcat : dims == 2 ? hcat : (a,b) -> cat(a, b; dims = dims))

"""
MLP

Multi-layer perceptron mapping inputs with height `sz[1]` to outputs with height `sz[2]`.
`Nhid+2` total dense layers are used with `Dhid` hidden nodes. The first `Nhid+1` layers
use activation `σhid` and the last layer uses activation `σout`. 
"""
function MLP(sz::Pair{Int,Int}, Nhid::Int, Dhid::Int, σhid = Flux.relu, σout = identity; skip = false, dropout = false, layernorm = false, initW = Flux.glorot_uniform, initb = Flux.zeros, initW_last = initW, initb_last = initb)
    σfactory(f) =
        f === identity ? () -> () : # Identity layers can be skipped
        f isa Base.BroadcastFunction ? () -> (f,) : # Thunk just returns BroadcastFunction
        f isa Tuple ? () -> (f[1](f[2:end]...),) : # Assume f[1] is a factory returning structs which take array inputs, and f[2:end] are factory arguments; thunk returns new instance with each call
        () -> (Base.BroadcastFunction(f),) # Assume f is scalar function capable of broadcasting; thunk returns BroadcastFunction wrapping f
    maybedropout(l) =
        dropout > 0 ? Flux.Chain(l, Flux.Dropout(dropout)) : l
    maybelayernorm(l) =
        layernorm ? Flux.Chain(l, Flux.LayerNorm(Dhid)) : l
    Dense(in, out, f; kwargs...) =
        Flux.Chain(Flux.Dense(in, out, identity; kwargs...), σfactory(f)()...)
    MaybeResidualDense(in, out, f; kwargs...) =
        skip ?
            Flux.Chain(Flux.SkipConnection(Dense(in, out, identity; kwargs...), +), σfactory(f)()...) : # x -> f.(x .+ W*x .+ b)
            Dense(in, out, f; kwargs...) # x -> f.(W*x .+ b)
    Flux.Chain(
        Dense(sz[1], Dhid, σhid; initW, initb) |> maybedropout |> maybelayernorm,
        [MaybeResidualDense(Dhid, Dhid, σhid; initW, initb) |> maybedropout |> maybelayernorm for _ in 1:Nhid]...,
        Dense(Dhid, sz[2], σout; initW = initW_last, initb = initb_last),
    ) |> flattenchain
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
function model_summary(models::AbstractDict; filename = nothing, verbose = true)
    verbose && @info "Model summary..."
    (filename !== nothing) && open(filename, "w") do file
        model_summary(file, models)
    end
    verbose && model_summary(stdout, models)
end
model_summary(model; kwargs...) = model_summary(Dict("" => model); kwargs...)

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
    fs = fieldnames(typeof(model))
    if length(fs) == 0
        print(io, _getprefix(depth, pre, suf * "$(nameof(typeof(model)))()"))
    else
        println(io, _getprefix(depth, pre, suf * "$(nameof(typeof(model)))("))
        for (i,f) in enumerate(fs)
            _model_summary(io, getfield(model, f); depth = depth + 1, suf = "$f: ")
            println(io, i < length(fs) ? "," : "")
        end
        print(io, _getprefix(depth, pre, ")"))
    end
end

# Flux.Chain
function _model_summary(io::IO, model::Flux.Chain; depth::Int = 0, pre = "", suf = "")
    println(io, _getprefix(depth, pre, suf * "Chain("))
    for (i,layer) in enumerate(model.layers)
        _model_summary(io, layer; depth = depth + 1, suf = "$i: ")
        (i < length(model)) ? println(io, ",") : println(io, "")
    end
    print(io, _getprefix(depth, pre, ")"))
    nothing
end

# Transformers.NNTopo
function _model_summary(io::IO, model::NNTopo; depth::Int = 0, pre = "", suf = "")
    topo_print = sprint(Stacks.print_topo, model)
    topo_print = _getprefix(depth, pre, suf * "@λ ") * topo_print
    topo_print = replace(topo_print, "\t" => _getprefix(1, pre))
    topo_print = replace(topo_print, "end\n" => "end")
    topo_print = replace(topo_print, "\n" => "\n" * _getprefix(depth, pre))
    print(io, topo_print)
end

# Transformers.Stack
function _model_summary(io::IO, model::Stack; depth::Int = 0, pre = "", suf = "")
    println(io, _getprefix(depth, pre, suf * "Stack("))
    _model_summary(io, model.topo; depth = depth + 1, suf = "topo: ")
    println(io, ",")
    _model_summary(io, model.models; depth = depth + 1, suf = "models: ")
    println(io, "")
    print(io, _getprefix(depth, pre, ")"))
end

# Arrays
function _model_summary(io::IO, model::AbstractArray; depth::Int = 0, pre = "", suf = "")
    print(io, _getprefix(depth, pre, suf * "size " * string(size(model)) * " " * string(typeof(model))))
end

# Resize
function _model_summary(io::IO, model::Resize; depth::Int = 0, pre = "", suf = "")
    print(io, _getprefix(depth, pre, suf * repr(model)))
end

# Print as scalars (Numbers, Symbols, ...)
const ScalarPrint = Union{<:Number, Nothing, Symbol, String}
const TupleOfScalarPrint = Tuple{Vararg{<:ScalarPrint}}
const NamedTupleOfScalarPrint = NamedTuple{Keys, <:TupleOfScalarPrint} where {Keys}

function _model_summary(io::IO, model::Union{<:ScalarPrint, <:TupleOfScalarPrint, <:NamedTupleOfScalarPrint}; depth::Int = 0, pre = "", suf = "")
    print(io, _getprefix(depth, pre, suf * "$model :: $(typeof(model))"))
end

# Functions
function _model_summary(io::IO, model::Function; depth::Int = 0, pre = "", suf = "")
    print(io, _getprefix(depth, pre, suf * "@λ "))
    show(io, model)
end
