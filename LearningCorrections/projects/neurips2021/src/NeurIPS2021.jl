module NeurIPS2021

include("fix/Reexport.jl") # workaround until Reexport v1.0 is tagged
using .Reexport: @reexport, @importexport

include("fix/Stacks/Stacks.jl") # workaround until Transformers is updated for julia v1.6
@reexport using .Stacks: @nntopo_str, @nntopo, NNTopo, Stack, show_stackfunc, stack

#### Import/export dependency names

@importexport import ArgParse, BSON, BangBang, BenchmarkTools, CUDA, ChainRules, Conda, CSV, DECAES, DataFrames, DataStructures, Dates, Distributions, DrWatson, EllipsisNotation, FFTW, FileIO, FiniteDifferences, Flux, ForwardDiff, Functors, Glob, HypothesisTests, JLD2, LaTeXStrings, LinearAlgebra, LoopVectorization, NLopt, NNlib, OMEinsum, Optim, Parameters, Pkg, PrettyTables, ProgressMeter, PyCall, PyPlot, Random, ReadableRegex, SpecialFunctions, Statistics, StatsBase, StatsPlots, Suppressor, TOML, ThreadPools, TimerOutputs, Tullio, Turing, UUIDs, UnicodePlots, Zygote

#### Dependencies' symbols

@reexport using BangBang: push!!, setindex!!, append!!
@reexport using BenchmarkTools: @btime
@reexport using CUDA: @cufunc, CuArray, CuVector, CuMatrix
@reexport using DataFrames: DataFrame, dropmissing
@reexport using DataStructures: OrderedDict
@reexport using Distributions: Gaussian, Uniform, cdf, logpdf, pdf, normcdf, norminvcdf, log2π, logtwo, sqrt2, sqrt2π, sqrtπ, sqrthalfπ, invsqrt2, invsqrt2π
@reexport using DrWatson: @dict, @ntuple, datadir, projectdir, scriptsdir, srcdir
@reexport using EllipsisNotation: (..)
@reexport using FFTW: fft, ifft, rfft
@reexport using LaTeXStrings: @L_str, latexstring
@reexport using LinearAlgebra: BLAS, diag, diagm, dot, mul!, norm, normalize, tr, ×, ⋅
@reexport using LoopVectorization: @avx, @avxt
@reexport using OMEinsum: @ein, @ein_str
@reexport using Parameters: @unpack, @with_kw, @with_kw_noshow
@reexport using PyCall: @py_str, PyCall, PyDict, PyNULL, PyObject, pycall, pyimport
@reexport using Random: MersenneTwister, randperm, randperm!
@reexport using SpecialFunctions: besseli, besselix, erf, erfc, erfcx, erfinv
@reexport using Statistics: mean, median, quantile, std, var
@reexport using StatsBase: Histogram, UnitWeights, mean_and_std, sample
@reexport using Suppressor: @suppress
@reexport using TimerOutputs: @timeit
@reexport using Tullio: @tullio

#### TODO: give plotting functions their own namespace

@reexport using StatsPlots

#### Exports

const lib = @__MODULE__
const Playground = @__MODULE__
export Playground, lib, plt, rcParams, torch, ignite, logging, scipy, numpy
export @j2p, cpu, gpu, cpu32, gpu32, cpu64, gpu64
export AbstractTensor3D, AbstractTensor4D, CuTensor3D, CuTensor4D

#### Includes

include("utils/common.jl")
include("utils/pytools.jl")
include("utils/ignite.jl")
include("utils/flux.jl")

include("math/math_utils.jl")
include("math/rician.jl")
include("math/truncatedgaussian.jl")

include("models/physics.jl")
include("models/layers.jl")
include("models/cvae.jl")
include("models/losses.jl")
include("models/eval.jl")
include("models/setup.jl")

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
    Plots.gr()
    Plots.GR.inline("png") # workaround for "too many open files" issue, c.f. https://github.com/JuliaPlots/Plots.jl/issues/1723#issuecomment-471363504
    # Plots.pyplot()
    Plots.default(;
        fontfamily = "serif",
        size = (1600, 900),
        titlefont = 12,
        guidefont = 12,
        tickfont = 9,
        legendfontsize = 9,
        grid = true,
        yminorgrid = true,
        xminorgrid = true,
    )

    # Environment variable defaults
    get!(ENV, "JL_TRAIN_DEBUG", "0")
    get!(ENV, "JL_DISABLE_GPU", "0")
    get!(ENV, "JL_CUDA_DEVICE", "0")
    get!(ENV, "JL_ZERO_SUBNORMALS", "1")
    get!(ENV, "JL_CHECKPOINT_FOLDER", "")

    # Use CUDA
    Flux.use_cuda[] = CUDA.functional() && ENV["JL_DISABLE_GPU"] != "1"
    if Flux.use_cuda[]
        CUDA.allowscalar(false)
        CUDA.device!(parse(Int, ENV["JL_CUDA_DEVICE"]))
    end

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

end # module NeurIPS2021
