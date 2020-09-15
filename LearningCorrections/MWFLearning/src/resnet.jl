module ResNet

import Flux
using Flux: onehotbatch, onecold, crossentropy, throttle, @functor
using Base.Iterators: partition
using LegibleLambdas: @λ

export ResidualBlock, BasicBlock, Bottleneck, StandardResnet
export resnet18, resnet34, resnet50, resnet101, resnet152

const DIM = Ref(1)
_rep(n::Int) = DIM[] == 1 ? (n,1) : (n,n)
_pad(p::Int) = DIM[] == 1 ? (p,p,0,0) : (p,p,p,p)

struct ResidualBlock
    conv_layers
    norm_layers
    shortcut
end

@functor ResidualBlock

# ResidualBlock Function allows us to define a Residual Block having any number of Convolution and Batch Normalization Layers
function ResidualBlock(filters, kernels::Array{NTuple{2,Int}}, pads::Array{NTuple{4,Int}}, strides::Array{NTuple{2,Int}}, shortcut = identity)
    local conv_layers = []
    local norm_layers = []
    for i in 2:length(filters)
        push!(conv_layers, Flux.Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]))
        push!(norm_layers, Flux.BatchNorm(filters[i]))
    end
    ResidualBlock(Tuple(conv_layers),Tuple(norm_layers),shortcut)
end

# Function converts the Array of scalar kernel, pad and stride values to tuples
function ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity)
    ResidualBlock(filters, [_rep(i) for i in kernels], [_pad(i) for i in pads], [_rep(i) for i in strides], shortcut)
end

function (block::ResidualBlock)(input)
    local value = input
    for i in 1:length(block.conv_layers)-1
        value = Flux.relu.(block.norm_layers[i](block.conv_layers[i](value)))
    end
    Flux.relu.(block.norm_layers[end](block.conv_layers[end](value)) + block.shortcut(input))
end

# Function to generate the residual blocks for ResNet18 and ResNet34
function BasicBlock(filters::Int, downsample::Bool = false, res_top::Bool = false)
    if !downsample || res_top
        return ResidualBlock([filters for i in 1:3], [3,3], [1,1], [1,1])
    end
    shortcut = Flux.Chain(Flux.Conv(_rep(3), filters÷2=>filters, pad = _pad(1), stride = _rep(2)), Flux.BatchNorm(filters))
    ResidualBlock([filters÷2, filters, filters], [3,3], [1,1], [1,2], shortcut)
end

# Function to generate the residual blocks used for ResNet50, ResNet101 and ResNet152
function Bottleneck(filters::Int, downsample::Bool = false, res_top::Bool = false)
    if !downsample && !res_top
        return ResidualBlock([4 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1])
    elseif downsample && res_top
        return ResidualBlock([filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1], Flux.Chain(Flux.Conv(_rep(1), filters=>4 * filters, pad = _pad(0), stride = _rep(1)), Flux.BatchNorm(4 * filters)))
    else
        shortcut = Flux.Chain(Flux.Conv(_rep(1), 2 * filters=>4 * filters, pad = _pad(0), stride = _rep(2)), Flux.BatchNorm(4 * filters))
        return ResidualBlock([2 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,2], shortcut)
    end
end

# Function to build Standard Resnet models as described in the paper "Deep Residual Learning for Image Recognition"
function StandardResnet(Block, layers, top = nothing, bottom = nothing; initial_filters::Int = 64, nclasses::Int = 1000)
    
    if isnothing(top)
        top = []
        push!(top, Flux.Conv(_rep(7), 3=>initial_filters, pad = _pad(3), stride = _rep(2)))
        push!(top, Flux.MaxPool(_rep(3), pad = _pad(1), stride = _rep(2)))
    end

    filters = initial_filters
    residual = []
    for i in 1:length(layers)
        push!(residual, Block(filters, true, i==1))
        for j in 2:layers[i]
            push!(residual, Block(filters))
        end
        filters *= 2
    end

    if isnothing(bottom)
        bottom = []
        push!(bottom, Flux.MeanPool(_rep(7)))
        push!(bottom, @λ(x -> reshape(x, :, size(x,4))))
        if Block == Bottleneck
            push!(bottom, Flux.Dense(initial_filters * 2^(length(layers) + 1), nclasses))
        else
            push!(bottom, Flux.Dense(initial_filters * 2^(length(layers) - 1), nclasses))
        end
        push!(bottom, Flux.softmax)
    end

    Flux.Chain(top..., residual..., bottom...)
end

# Standard ResNet models for Imagenet as described in the Paper "Deep Residual Learning for Image Recognition"
# Uncomment the model that you want to use
resnet18( args...; kwargs...) = StandardResnet(BasicBlock, [2, 2,  2, 2], args...; kwargs...)
resnet34( args...; kwargs...) = StandardResnet(BasicBlock, [3, 4,  6, 3], args...; kwargs...)
resnet50( args...; kwargs...) = StandardResnet(Bottleneck, [3, 4,  6, 3], args...; kwargs...)
resnet101(args...; kwargs...) = StandardResnet(Bottleneck, [3, 4, 23, 3], args...; kwargs...)
resnet152(args...; kwargs...) = StandardResnet(Bottleneck, [3, 8, 36, 3], args...; kwargs...)

end # module
