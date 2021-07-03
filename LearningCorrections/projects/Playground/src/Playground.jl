module Playground

include("fix/Reexport.jl") # workaround until Reexport v1.0 is tagged
using .Reexport: @reexport

include("fix/Stacks/Stacks.jl") # workaround until Transformers is updated for julia v1.6
@reexport using .Stacks: @nntopo_str, @nntopo, NNTopo, Stack, show_stackfunc, stack

#### Dependencies

@reexport using ArgParse: ArgParse
@reexport using BSON: BSON
@reexport using BangBang: BangBang, push!!, setindex!!, append!!
@reexport using BenchmarkTools: BenchmarkTools, @btime
@reexport using CUDA: CUDA, @cufunc, CuArray, CuVector, CuMatrix
@reexport using ChainRules: ChainRules
@reexport using Conda: Conda
@reexport using CSV: CSV
@reexport using DECAES: DECAES
@reexport using DataFrames: DataFrames, DataFrame, dropmissing
@reexport using DataStructures: DataStructures, OrderedDict
@reexport using Dates: Dates
@reexport using Distributions: Distributions, Gaussian, Uniform, cdf, logpdf, pdf, normcdf, norminvcdf, log2π, logtwo, sqrt2, sqrt2π, sqrtπ, sqrthalfπ, invsqrt2, invsqrt2π
@reexport using DrWatson: DrWatson, @dict, @ntuple, datadir, projectdir, scriptsdir, srcdir
@reexport using EllipsisNotation: EllipsisNotation, (..)
@reexport using FFTW: FFTW, fft, ifft, rfft
@reexport using FileIO: FileIO
@reexport using FiniteDifferences: FiniteDifferences
@reexport using Flux: Flux
@reexport using ForwardDiff: ForwardDiff
@reexport using Functors: Functors
@reexport using Glob: Glob
@reexport using HypothesisTests: HypothesisTests
@reexport using JLD2: JLD2
@reexport using LaTeXStrings: LaTeXStrings, @L_str, latexstring
@reexport using LinearAlgebra: LinearAlgebra, BLAS, diag, diagm, dot, mul!, norm, normalize, tr, ×, ⋅
@reexport using LoopVectorization: LoopVectorization, @avx, @avxt
@reexport using NLopt: NLopt
@reexport using NNlib: NNlib
@reexport using OMEinsum: OMEinsum, @ein, @ein_str
@reexport using Optim: Optim
@reexport using Parameters: Parameters, @unpack, @with_kw, @with_kw_noshow
@reexport using Pkg: Pkg
@reexport using Plots: Plots, @layout, cgrad, contourf, density!, heatmap, histogram, plot, plot!, pyplot, savefig, scatter, stephist, stephist!, sticks, surface, title!, vline!, xlabel!
@reexport using PrettyTables: PrettyTables
@reexport using ProgressMeter: ProgressMeter
@reexport using PyCall: PyCall, @py_str, PyCall, PyDict, PyNULL, PyObject, pycall, pyimport
@reexport using PyPlot: PyPlot
@reexport using Random: Random, MersenneTwister, randperm, randperm!
@reexport using ReadableRegex: ReadableRegex
@reexport using SpecialFunctions: SpecialFunctions, besseli, besselix, erf, erfc, erfcx, erfinv
@reexport using Statistics: Statistics, mean, median, quantile, std, var
@reexport using StatsBase: StatsBase, Histogram, UnitWeights, mean_and_std, sample
@reexport using StatsPlots: StatsPlots, qqplot
@reexport using Suppressor: Suppressor, @suppress
@reexport using TOML: TOML
@reexport using ThreadPools: ThreadPools
@reexport using TimerOutputs: TimerOutputs, @timeit
@reexport using Tullio: Tullio, @tullio
@reexport using Turing: Turing
@reexport using UUIDs: UUIDs
@reexport using UnicodePlots: UnicodePlots
@reexport using Zygote: Zygote

#### Exports

const lib = @__MODULE__
export lib, plt, rcParams, torch, wandb, ignite, logging, scipy, numpy
export @j2p, cpu, gpu, cpu32, gpu32, cpu64, gpu64
export AbstractTensor3D, AbstractTensor4D, CuTensor3D, CuTensor4D

#### Includes

include("utils/common.jl")
include("utils/pytools.jl")
include("utils/ignite.jl")
include("utils/flux.jl")
include("utils/plot.jl")

include("math/math_utils.jl")
include("math/rician.jl")
include("math/kumaraswamy.jl")
include("math/truncatedgaussian.jl")
include("math/batched_math.jl")
include("math/mmd.jl")

include("models/physics.jl")
include("models/layers.jl")
include("models/transforms.jl")
include("models/cvae.jl")
include("models/xformer.jl")
include("models/losses.jl")
include("models/eval.jl")
include("models/setup.jl")

include("fix/transformers_mha.jl")

#### Init

function __init__()
    # Python utilities
    try
        __pyinit__()
    catch e
        if e isa PyCall.PyError
            @info "Installing python utilities..."
            __pyinstall__()
            __pyinit__()
        else
            rethrow(e)
        end
    end
end

function initenv()
    # Plotting defaults
    pyplot(size=(1200,900))

    # Environment variable defaults
    get!(ENV, "JL_TRAIN_DEBUG", "0")
    get!(ENV, "JL_DISABLE_GPU", "0")
    get!(ENV, "JL_CUDA_DEVICE", "0")
    get!(ENV, "JL_WANDB_LOGGER", "0")
    get!(ENV, "JL_ZERO_SUBNORMALS", "1")
    get!(ENV, "JL_CHECKPOINT_FOLDER", "")

    # Use CUDA
    Flux.use_cuda[] = CUDA.functional() && ENV["JL_DISABLE_GPU"] != "1"
    if Flux.use_cuda[]
        CUDA.allowscalar(false)
        CUDA.device!(parse(Int, ENV["JL_CUDA_DEVICE"]))
    end

    # Use WandB logger
    use_wandb_logger[] = ENV["JL_WANDB_LOGGER"] == "1"

    # Debug mode
    train_debug[] = ENV["JL_TRAIN_DEBUG"] == "1"

    # Load model from checkpoint folder
    if !isempty(ENV["JL_CHECKPOINT_FOLDER"])
        set_checkpointdir!(ENV["JL_CHECKPOINT_FOLDER"])
    else
        clear_checkpointdir!()
    end

    # Treat subnormals as zero
    if ENV["JL_ZERO_SUBNORMALS"] != "0"
        Threads.@threads for i in 1:Threads.nthreads()
            set_zero_subnormals(true)
        end
    end
end

end # module Playground
