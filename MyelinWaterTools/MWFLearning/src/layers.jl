"""
DenseResize()

Non-learnable layer which resizes input arguments `x` to be a matrix with batchsize(x) columns.
"""
struct DenseResize end
Flux.@treelike DenseResize
(l::DenseResize)(x::AbstractArray) = reshape(x, :, batchsize(x))
Base.show(io::IO, l::DenseResize) = print(io, "DenseResize()")

"""
ChannelResize(n::Int)

Non-learnable layer which resizes input arguments `x` to be an array with `n` columns,
preserving the height of the data.
"""
struct ChannelResize
    n::Int
end
ChannelResize(s::Flux.TrackedArray) = ChannelResize(Flux.data(s)) # Layer is not learnable
Flux.@treelike ChannelResize
(l::ChannelResize)(x::AbstractArray) = reshape(x, heightsize(x), l.n, :)
Base.show(io::IO, l::ChannelResize) = print(io, "ChannelResize(", length(l.n), ")")

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
