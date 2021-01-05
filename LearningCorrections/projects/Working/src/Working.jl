module Working

using Reexport
@reexport using LinearAlgebra, Statistics, Random, Dates, Printf
@reexport using StatsBase, Distributions, DataFrames, SpecialFunctions, FFTW, TimerOutputs, BenchmarkTools
@reexport using Parameters, BangBang, EllipsisNotation, LegibleLambdas, LaTeXStrings
@reexport using StatsPlots
@reexport using PyCall

@reexport using LoopVectorization, Tullio
import UnsafeArrays
using UnsafeArrays: uview, uviews, @uviews

import PyPlot, Conda, TOML, BSON, Glob, PrettyTables, DrWatson, Flux, CUDA, NNlib, Zygote, ChainRules, Transformers, BlackBoxOptim, Optim, NLopt, FiniteDifferences, ForwardDiff, SparseDiffTools, DECAES, HypothesisTests
export PyPlot, Conda, TOML, BSON, Glob, PrettyTables, DrWatson, Flux, CUDA, NNlib, Zygote, ChainRules, Transformers, BlackBoxOptim, Optim, NLopt, FiniteDifferences, ForwardDiff, SparseDiffTools, DECAES, HypothesisTests
using DrWatson: @dict, @ntuple, @pack!, @unpack
export @dict, @ntuple, @pack!, @unpack
using Distributions: log2π
export log2π

const AbstractTensor3D{T} = AbstractArray{T,3}
const AbstractTensor4D{T} = AbstractArray{T,4}
const CuTensor3D{T} = CUDA.CuArray{T,3}
const CuTensor4D{T} = CUDA.CuArray{T,4}

####
#### Init
####

export todevice, to32, to64

const JL_WANDB_LOGGER = Ref(false)
const JL_CHECKPOINT_FOLDER = Ref("")

function __init__()
    # Use CUDA
    Flux.use_cuda[] = CUDA.functional() && get(ENV, "JL_DISABLE_GPU", "0") != "1"
    cuda_device = parse(Int, get(ENV, "JL_CUDA_DEVICE", "0"))
    if Flux.use_cuda[]
        CUDA.allowscalar(false)
        CUDA.device!(cuda_device)
    end

    # Use WandB logger
    JL_WANDB_LOGGER[] = get(ENV, "JL_WANDB_LOGGER", "0") == "1"

    # Load model from checkpoint folder
    JL_CHECKPOINT_FOLDER[] = get(ENV, "JL_CHECKPOINT_FOLDER", "")

    # Treat subnormals as zero
    zero_subnormals = get(ENV, "JL_ZERO_SUBNORMALS", "1") != "0"
    if zero_subnormals
        Threads.@threads for i in 1:Threads.nthreads()
            set_zero_subnormals(true)
        end
    end
end

todevice(x) = Flux.use_cuda[] ? Flux.gpu(x) : Flux.cpu(x)
todevice(d::AbstractDict) = Dict(k => todevice(v) for (k,v) in d)
to32(x) = Flux.fmap(xi -> xi isa AbstractArray ? convert(AbstractArray{Float32}, xi) : xi, todevice(x))
to64(x) = Flux.fmap(xi -> xi isa AbstractArray ? convert(AbstractArray{Float64}, xi) : xi, todevice(x))

####
#### Includes
####

include("PyTools.jl")
@reexport using .PyTools

include("Ignite.jl")
@reexport using .Ignite

include("math/rician.jl")
include("math/batched_math.jl")
include("math/math_utils.jl")
include("math/mmd.jl")

include("utils/utils.jl")
include("utils/plot.jl")

include("models/layers.jl")
include("models/physics.jl")
include("models/cvae.jl")
include("models/xformer.jl")
include("models/eval.jl")
include("models/setup.jl")

end # module
