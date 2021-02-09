"""
DummyArray(x::AbstractArray{T,N}) <: AbstractArray{T,N}

Data-less dummy array wrapping a tuple representating the array size.
`DummyArray`s implement the bare minimum `AbstractArray` interface, and are
additionally mutable, thence providing a unique `objectid`, which e.g.
`IdDict`, `Flux.fmap`, and `===` rely on.
"""
mutable struct DummyArray{T,N,A} <: AbstractArray{T,N}
    dims::NTuple{N,Int}
end
DummyArray(x::AbstractArray{T,N}) where {T,N} = DummyArray{T,N,typeof(x)}(size(x))
Base.size(x::DummyArray) = x.dims
Base.length(x::DummyArray) = prod(size(x))
Base.getindex(::DummyArray, I...) = error("Base.getindex not defined for DummyArray")
Base.setindex!(::DummyArray, v, I...) = error("Base.setindex! not defined for DummyArray")
Base.show(io::IO, x::DummyArray{T,N,A}) where {T,N,A} = print(io, "DummyArray{$T,$N,$A}(dims = $(size(x)))")
Base.show(io::IO, ::MIME"text/plain", x::DummyArray) = show(io, x) # avoid falling back to AbstractArray default

struct HyperNet{F,M}
    hyper::F
    frame::M
end
Flux.@functor HyperNet
Flux.trainable(h::HyperNet) = (h.hyper,)

(h::HyperNet)(x) = restructure(h.frame, h.hyper(x))
(h::HyperNet{F,Nothing})(x) where {F} = h.hyper

hypernet_from_template(hyper, template) = HyperNet(hyper, fmap_trainables(DummyArray, hyperify(template)))
hypernet_wrapper(model) = HyperNet(model, nothing)

ishyperleaf(m) = false # fallback
ishyperleaf(m::Flux.Dense{<:Any, <:AbstractMatrix, <:AbstractVector}) = true
ishyperleaf(m::Flux.Diagonal{<:AbstractVector}) = true

hyperify1(m) = m # fallback
hyperify1(m::Flux.Dense{<:Any, <:AbstractMatrix, <:AbstractVector}) = BatchedDense(copy(reshape(m.W, size(m.W)..., 1)), copy(reshape(m.b, size(m.b, 1), 1, 1)), m.σ)
hyperify1(m::Flux.Diagonal{<:AbstractVector}) = BatchedDiagonal(copy(reshape(m.α, size(m.α, 1), 1)), copy(reshape(m.β, size(m.β, 1), 1)))
hyperify(m) = fmap_(hyperify1, m, ishyperleaf)

"""
Same as `Flux.Dense`, except `W` and `b` act along batch dimensions:

    y[:,k] = σ.(W[:,:,k] * x[:,k] .+ b[:,:,k])
"""
struct BatchedDense{F,S<:AbstractTensor3D,T<:AbstractTensor3D}
    W::S
    b::T
    σ::F
end
Flux.@functor BatchedDense
BatchedDense(W, b) = BatchedDense(W, b, identity)
Base.show(io::IO, l::BatchedDense) = print(io, "BatchedDense(", size(l.W, 2), ", ", size(l.W, 1), (l.σ === identity ? () : (", ", l.σ))..., ")")

function (a::BatchedDense)(x::AbstractVecOrMat)
    W, b, σ = a.W, a.b, a.σ
    sz = size(x)
    x = reshape(x, sz[1], 1, :) # treat dims > 1 as batch dimensions
    x = σ.(Flux.batched_mul(W, x) .+ b)
    x = reshape(x, :, sz[2:end]...)
    return x
end

"""
Same as `Flux.Diagonal`, except `W` and `b` act along batch dimensions:

    y[:,k] = σ.(W[:,:,k] * x[:,k] .+ b[:,:,k])
"""
struct BatchedDiagonal{T}
    α::T
    β::T
end
Flux.@functor BatchedDiagonal
Base.show(io::IO, l::BatchedDiagonal) = print(io, "BatchedDiagonal(", size(l.α, 1), ")")

function (a::BatchedDiagonal)(x::AbstractVecOrMat)
    α, β = a.α, a.β
    sz = size(x)
    x = reshape(x, sz[1], :) # treat dims > 1 as batch dimensions
    x = α .* x .+ β
    x = reshape(x, :, sz[2:end]...)
end

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
movingaverage!(mean_model, model, window)

Accumulate moving average of parameters from `model` into `mean_model` with time constant `window` (units of update steps, need not be integral).
The relative contribution of parameters from `k` updates ago to the average in `mean_model` is proportional to `exp(-k/window)`.
"""
function movingaverage!(mean_model, model, window)
    μs = Flux.params(mean_model)
    xs = Flux.params(model)
    @assert length(μs) == length(xs)
    for (μ, x) in zip(μs, xs)
        movingaverage!(μ, x, window)
    end
    return mean_model
end
function movingaverage!(μ::AbstractArray, x::AbstractArray, τ)
    α = ofeltype(μ, 1-exp(-1/τ))
    @. μ = α * x + (1-α) * μ
end

"""
CollectAsNamedTuple(keys...)
"""
struct CollectAsNamedTuple{N,Ks} end
CollectAsNamedTuple(ks::Symbol...) = CollectAsNamedTuple{length(ks), ks}()
(c::CollectAsNamedTuple{N,Ks})(Xs::Vararg{<:Any,N}) where {N,Ks} = NamedTuple{Ks}(Xs)

"""
ZipNamedTuples()(nts...)

Convert tuple of named tuples (with matching keys) to named tuple of tuples, e.g.

    ZipNamedTuples()(((a=1,b=2), (a=3,b=4))) == (a=(1,3), b=(2,4))
"""
struct ZipNamedTuples end
(::ZipNamedTuples)(nts::NTupleOfNamedTuples{Ks,M,N}) where {Ks,M,N} = zipnamedtuples(nts)
zipnamedtuples(nts::NTupleOfNamedTuples{Ks,M,N}) where {Ks,M,N} = NamedTuple{Ks}(ntuple(j -> ntuple(i -> nts[i][j], N), M))
Zygote.@adjoint zipnamedtuples(nts::NTupleOfNamedTuples{Ks,M,N}) where {Ks,M,N} = zipnamedtuples(nts), Δ -> (unzipnamedtuple(Δ),)

"""
UnzipNamedTuple()(nt)

Convert named tuple of tuples to tuple of named tuples, e.g.

    UnzipNamedTuple()((a=(1,3), b=(2,4))) == ((a=1,b=2), (a=3,b=4))
"""
struct UnzipNamedTuple end
(::UnzipNamedTuple)(nt::NamedTupleOfNTuples{Ks,M,N}) where {Ks,M,N} = unzipnamedtuple(nt)
unzipnamedtuple(nt::NamedTupleOfNTuples{Ks,M,N}) where {Ks,M,N} = ntuple(j -> NamedTuple{Ks}(ntuple(i -> nt[i][j], M)), N)
Zygote.@adjoint unzipnamedtuple(nt::NamedTupleOfNTuples{Ks,M,N}) where {Ks,M,N} = unzipnamedtuple(nt), Δ -> (zipnamedtuples(Δ),)

function _zip_nt_test()
    tup = ((a=[1.0],b=[2.0]), (a=[3.0],b=[4.0]))
    nt = (a=([1.0],[3.0]), b=([2.0],[4.0]))
    @assert zipnamedtuples(tup) == nt
    @assert unzipnamedtuple(nt) == tup
    @assert unzipnamedtuple(zipnamedtuples(tup)) == tup
    @assert zipnamedtuples(unzipnamedtuple(nt)) == nt
    @assert modelgradcheck(tup; extrapolate = true, verbose = true) do
        nt = zipnamedtuples(tup)
        # sum(abs2, map(+, map(+, nt...)...)) # should work... make MWE and file issue?
        sum(abs2, nt.a[1] + nt.a[2]) + sum(abs2, nt.b[1] + nt.b[2])
    end
    @assert modelgradcheck(nt; extrapolate = true, verbose = true) do
        tup = unzipnamedtuple(nt)
        # sum(abs2, map(+, map(+, tup...)...)) # should work... make MWE and file issue?
        sum(abs2, tup[1].a + tup[2].a) + sum(abs2, tup[1].b + tup[2].b)
    end
    # tup_out, J = Zygote.pullback(unzipnamedtuple, nt); @show J(tup_out)
    # nt_out, J = Zygote.pullback(zipnamedtuples, tup); @show J(nt_out)
end

"""
NotTrainable(layer)

Wraps the callable `layer` such that any parameters internal to `layer`
are ignored by Flux during gradient calls.
"""
struct NotTrainable{F}
    layer::F
end
Flux.@functor NotTrainable # need functor for e.g. `fmap`
Flux.trainable(::NotTrainable) = () # no trainable parameters
(l::NotTrainable)(xs...) = l.layer(xs...)
Base.show(io::IO, l::NotTrainable) = (print(io, "NotTrainable("); print(io, l.layer); print(io, ")"))

```
ApplyOverDims
```
struct ApplyOverDims{dims,F}
    f::F
end
Flux.@functor ApplyOverDims
ApplyOverDims(f; dims) = ApplyOverDims{dims,typeof(f)}(f)
(a::ApplyOverDims{dims})(xs...) where {dims} = a.f(xs...; dims = dims)
Base.show(io::IO, a::ApplyOverDims{dims}) where {dims} = (print(io, "ApplyOverDims("); print(io, a.f); print(io, ", dims = $dims)"))

"""
CentralDifference()

Non-trainable central-difference layer which convolves the stencil [-1, 0, 1]
along the first dimension of `d x b` inputs, producing `(d-2) x b` outputs.
"""
CentralDifference() = ConstantFilter(reshape(Float32[-1.0, 0.0, 1.0], 3, 1, 1, 1), Float32[0.0], identity; stride = 1, pad = 0)
ForwardDifference() = ConstantFilter(reshape(Float32[1.0, -1.0], 2, 1, 1, 1), Float32[0.0], identity; stride = 1, pad = 0)
BackwardDifference() = ConstantFilter(reshape(Float32[-1.0, 1.0], 2, 1, 1, 1), Float32[0.0], identity; stride = 1, pad = 0)

# Helper functions for gradient operators
ConstantFilter(args...; kwargs...) = NotTrainable(Flux.Chain(ChannelResize(1), Flux.Conv(args...; kwargs...), DenseResize()))

Id_matrix(n::Int) = convert(Matrix{Float32}, LinearAlgebra.I(n))
FD_matrix(n::Int) = LinearAlgebra.diagm(n-1, n, 0 => -ones(Float32, n-1), 1 => ones(Float32, n-1))

function DenseFiniteDiff(n::Int, order::Int)
    FD = FD_matrix(n)
    A = foldl(1:order; init = Id_matrix(n)) do acc, i
        @views FD[1:end-i+1, 1:end-i+1] * acc
    end
    NotTrainable(Flux.Dense(A, Float32[0.0]))
end

function CatDenseFiniteDiff(n::Int, order::Int)
    FD = FD_matrix(n)
    A = foldl(1:order; init = Id_matrix(n)) do acc, i
        vcat(acc, @views FD[1:end-i+1, 1:end-i+1] * acc[end-n+i:end, :])
    end
    NotTrainable(Flux.Dense(A, Float32[0.0]))
end

"""
Laplacian()

Non-trainable central-difference layer which convolves the stencil [1, -2, 1]
along the first dimension of `d x b` inputs, producing `(d-2) x b` outputs.
"""
Laplacian() = ConstantFilter(reshape(Float32[1.0, -2.0, 1.0], 3, 1, 1, 1), Float32[0.0], identity; stride = 1, pad = 0)

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
wrapprint(io::IO, layer)
"""
wrapprint(io::IO, layer) = Flux.Chain(
    x -> (  print(io, "      layer: "); _model_summary(io, layer); print(io, "\n"); x),
    x -> (println(io, " input size: $(size(x))"); x),
    layer,
    x -> (println(io, "output size: $(size(x))"); x),
)
wrapprint(layer) = wrapprint(stdout, layer)
wrapprint(layer::Flux.Chain) = Flux.Chain(wrapprint.(layer.layers)...)

"""
PrintSize()

Non-trainable layer which simply prints the current size.
"""
printsize(x) = (@show size(x); x)
PrintSize() = x -> printsize(x)

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
end
MultiInput(xs...) = MultiInput{typeof(xs)}(xs)

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
DistributionUnion(d1, d2, p)

Draw samples from distributions `d1` and `d2` with probabilities `p` and `1-p`, respectively.
"""
struct DistributionUnion{D1,D2,T}
    d1::D1
    d2::D2
    p::T
end
Flux.@functor DistributionUnion
DistributionUnion(d1, d2; p) = p <= 0 ? d2 : p >= 1 ? d1 : DistributionUnion(d1, d2, p)

(u::DistributionUnion)(x) = sample_union(u.d1, u.d2, eltype(x)(u.p), x)

"""
ChannelwiseDense
"""
ChannelwiseDense(H::Int, ch::Pair, σ = identity) = Flux.Chain(DenseResize(), Flux.Dense(H*ch[1], H*ch[2], σ), ChannelResize(ch[2]))

"""
HeightwiseDense
"""
HeightwiseDense(H::Int, C::Int, σ = identity) = Flux.Chain(x -> reshape(x, H, :), Flux.Dense(H, H, σ), x -> reshape(x, H, 1, C, :))

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
ApplyAsMatrix(f)

Apply `f` columnwise to input `X::AbstractArray`.
If `X` is a matrix, return `Y = f(X)`. Else, flatten X first and reshape the intermediate output,
i.e. `Y = f(reshape(X, size(X,1), :))` followed by `reshape(Y, size(Y,1), size(X)[2:end]...)`.
"""
struct ApplyAsMatrix{F}
    f::F
end
Flux.@functor ApplyAsMatrix

(a::ApplyAsMatrix)(X::AbstractMatrix) = a.f(X)

function (a::ApplyAsMatrix)(X::AbstractArray)
    Y = a.f(reshape(X, size(X,1), :))
    return reshape(Y, size(Y,1), Base.tail(size(X))...)
end

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
        Flux.Chain(Flux.SkipConnection(Flux.Dense(Dhid, Dhid, identity; initW, initb), +), Base.BroadcastFunction(σhid)) : # x -> σhid.(x .+ W*x .+ b)
        Flux.Dense(Dhid, Dhid, σhid; initW, initb) # x -> σhid.(W*x .+ b)
    Flux.Chain(
        Flux.Dense(sz[1], Dhid, σhid; initW, initb) |> maybedropout |> maybelayernorm,
        [MaybeResidualDense() |> maybedropout |> maybelayernorm for _ in 1:Nhid]...,
        Flux.Dense(Dhid, sz[2], σout; initW = initW_last, initb = initb_last),
    ) |> flattenchain
end

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
    AF(σ) = Base.BroadcastFunction(σ)
    if mode === :pre
        Flux.Chain(BN(), AF(σ), CD()..., BN(), AF(σ), CD()...)
    elseif mode === :post
        Flux.Chain(CD()..., BN(), AF(σ), CD()..., BN(), AF(σ))
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
    CV(ch, σ = identity) = Flux.Conv(k, ch, σ; pad = (k.-1).÷2)
    BN(C,  σ = identity) = batchnorm ? Flux.BatchNorm(C, σ) : groupnorm ? Flux.GroupNorm(C, C÷2, σ) : identity
    AF(σ) = Base.BroadcastFunction(σ)
    if mode === :pre
        Flux.Chain(BN(ch[1]), AF(σ), CV(ch[1]=>ch[2]), vcat(([BN(ch[2]), AF(σ), CV(ch[2]=>ch[2])] for _ in 1:numlayers-2)...)..., BN(ch[2]), AF(σ), CV(ch[2]=>ch[2]))
    elseif mode === :post
        Flux.Chain(CV(ch[1]=>ch[1]), BN(ch[1]), AF(σ), vcat(([CV(ch[1]=>ch[1]), BN(ch[1]), AF(σ)] for _ in 1:numlayers-2)...)..., CV(ch[1]=>ch[2]), BN(ch[2]), AF(σ))
    elseif mode === :hybrid
        Flux.Chain(BN(ch[1]), CV(ch[1]=>ch[1], σ),     vcat(([BN(ch[1]), CV(ch[1]=>ch[1], σ)]     for _ in 1:numlayers-2)...)..., BN(ch[1]), CV(ch[1]=>ch[2]))
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
        Flux.Conv((1,1), G0 + C * G => G0; pad = (0,0)),
    )
end
DenseConnection(G0::Int, G::Int, C::Int; dims::Int = 3, k::Tuple = (3,1), σ = Flux.relu) =
    DenseConnection(
        ch -> Flux.Conv(k, ch, σ; pad = (k.-1).÷2), # Default factory for RDB's
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
            # Flux.Conv(k, G0 => G0, σ; pad = (k.-1).÷2),
            GlobalFeatureFusion(
                dims,
                [ResidualDenseBlock(Factory, G0, G, C; dims = dims) for d in 1:D]...,
            ),
            # Flux.BatchNorm(D * G0, σ),
            Flux.GroupNorm(D * G0, (D * G0) ÷ 2, σ),
            Flux.Conv((1,1), D * G0 => G0; pad = (0,0)),
            # Flux.Conv(k, G0 => G0, σ; pad = (k.-1).÷2),
        )
    )
end
DenseFeatureFusion(G0::Int, G::Int, C::Int, D::Int, k::Tuple = (3,1), σ = Flux.relu; kwargs...) =
    DenseFeatureFusion(
        ch -> Flux.Conv(k, ch, σ; pad = (k.-1).÷2), # Default factory for RDB's
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
function stack_activations(m::Stack, xs...; verbose = false)
    acts = Any[]
    _m = Stack(
        m.topo,
        map(enumerate(m.models)) do (i,f)
            function (args...,)
                if verbose
                    @info "layer $i", "NaN params: $(any([any(isnan, p) for p in Flux.params(f)]))", "Inf params: $(any([any(isinf, p) for p in Flux.params(f)]))", typeof(f)
                    for (j,arg) in enumerate(args)
                        @info "layer $i", "input $j", "NaN: $(any(isnan, arg))", "Inf: $(any(isinf, arg))", typeof(arg), size(arg)
                        display(arg)
                    end
                end
                y = f(args...)
                push!(acts, y)
                if verbose
                    @info "layer $i", "output", "NaN: $(any(isnan, y))", "Inf: $(any(isinf, y))", typeof(y), size(y)
                    display(y)
                    println("")
                end
                return y
            end
        end...
    )
    out = _m(xs...)
    return acts
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
