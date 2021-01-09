module Working

include("Reexport.jl") # workaround until Reexport v1.0 is tagged
using .Reexport: @reexport, @importexport

#### Import/export dependency names

@importexport import ArgParse, BSON, BangBang, BenchmarkTools, BlackBoxOptim, CUDA, ChainRules, Conda, DECAES, DataFrames, Dates, Distributions, DrWatson, EllipsisNotation, FFTW, FiniteDifferences, Flux, ForwardDiff, Functors, Glob, HypothesisTests, LaTeXStrings, LinearAlgebra, LoopVectorization, NLopt, NNlib, Optim, Parameters, Pkg, PrettyTables, PyCall, PyPlot, Random, SparseDiffTools, SpecialFunctions, StatsBase, StatsPlots, Suppressor, TOML, TimerOutputs, Transformers, Tullio, UnicodePlots, UnsafeArrays, Zygote

#### Dependencies' symbols

@reexport using BangBang: push!!, setindex!!
@reexport using BenchmarkTools: @btime
@reexport using CUDA: CuArray, CuVector, CuMatrix
@reexport using DataFrames: DataFrame, dropmissing
@reexport using Distributions: Normal, Uniform, cdf, logpdf, pdf, log2π
@reexport using DrWatson: @dict, @ntuple, projectdir
@reexport using EllipsisNotation: (..)
@reexport using FFTW: fft, ifft, rfft
@reexport using LaTeXStrings: @L_str, latexstring
@reexport using LinearAlgebra: BLAS, diag, diagm, dot, mul!, norm, normalize, tr, ×, ⋅
@reexport using LoopVectorization: @avx
@reexport using Parameters: @unpack, @with_kw, @with_kw_noshow
@reexport using Random: MersenneTwister, randperm, randperm!
@reexport using SpecialFunctions: besseli, besselix, erf, erfinv
@reexport using Statistics: mean, median, quantile, std, var
@reexport using StatsBase: Histogram, UnitWeights, mean_and_std, sample
@reexport using Suppressor: @suppress
@reexport using TimerOutputs: @timeit
@reexport using Tullio: @tullio
@reexport using UnsafeArrays: @uviews, uview, uviews

#### TODO: give plotting functions their own namespace

@reexport using StatsPlots

#### Exports

export AbstractTensor3D, AbstractTensor4D, CuTensor3D, CuTensor4D, todevice, to32, to64

#### Internal modules

include("PyTools.jl")
@importexport import .PyTools
@reexport using .PyTools: torch, wandb, ignite, logging, plt, rcParams

#### Includes

include("utils/common.jl")
include("utils/ignite.jl")
include("utils/flux.jl")
include("utils/plot.jl")

include("math/rician.jl")
include("math/batched_math.jl")
include("math/math_utils.jl")
include("math/mmd.jl")

include("models/layers.jl")
include("models/physics.jl")
include("models/cvae.jl")
include("models/xformer.jl")
include("models/eval.jl")
include("models/setup.jl")

#### Init

const JL_WANDB_LOGGER = Ref(false)
const JL_CHECKPOINT_FOLDER = Ref("")

function __init__()
    # Use CUDA
    Flux.use_cuda[] = CUDA.functional() && get(ENV, "JL_DISABLE_GPU", "0") != "1"
    if Flux.use_cuda[]
        cuda_device = parse(Int, get(ENV, "JL_CUDA_DEVICE", "0"))
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

end # module
