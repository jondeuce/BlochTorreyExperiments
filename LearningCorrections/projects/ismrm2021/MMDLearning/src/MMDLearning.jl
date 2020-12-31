module MMDLearning

using Reexport
@reexport using LinearAlgebra, Statistics, Random, Dates, Printf
@reexport using StatsBase, Distributions, DataFrames, SpecialFunctions, FFTW, TimerOutputs, BenchmarkTools
@reexport using Parameters, BangBang, EllipsisNotation, LegibleLambdas, LaTeXStrings
@reexport using StatsPlots

@reexport using LoopVectorization, Tullio
import UnsafeArrays
using UnsafeArrays: uview, uviews, @uviews

import Pkg.TOML, BSON, Glob, PrettyTables, DrWatson, Flux, Flux.CUDA, NNlib, Zygote, ChainRules, Transformers, BlackBoxOptim, Optim, NLopt, FiniteDifferences, ForwardDiff, SparseDiffTools, DECAES, HypothesisTests
export     TOML, BSON, Glob, PrettyTables, DrWatson, Flux,      CUDA, NNlib, Zygote, ChainRules, Transformers, BlackBoxOptim, Optim, NLopt, FiniteDifferences, ForwardDiff, SparseDiffTools, DECAES, HypothesisTests
using DrWatson: @dict, @ntuple, @pack!, @unpack
export @dict, @ntuple, @pack!, @unpack
using Distributions: log2π
export log2π

export MMDKernel, FunctionKernel, DeepExponentialKernel
export mmd, mmdvar, mmd_and_mmdvar, tstat_flux, kernel_loss, train_kernel!
export mmd_perm_test_power, mmd_perm_test_power_plot
export fast_hist_1D, signal_histograms, pyheatmap
export map_dict, sum_dict, apply_dim1, clamp_dim1
export std_thresh, split_mean_std, split_mean_exp_std, split_mean_softplus_std, sample_mv_normal, pow2
export arr_similar, arr32, arr64, zeros_similar, ones_similar, randn_similar, rand_similar, fill_similar
export handleinterrupt, saveprogress, saveplots

####
#### Global aliases
####
const AbstractTensor3D{T} = AbstractArray{T,3}
const AbstractTensor4D{T} = AbstractArray{T,4}
const CuTensor3D{T} = CUDA.CuArray{T,3}
const CuTensor4D{T} = CUDA.CuArray{T,4}

export AbstractTensor3D, AbstractTensor4D, CuTensor3D, CuTensor4D

####
#### Includes
####

include("PyTools.jl")
@reexport using .PyTools

include("Ignite.jl")
@reexport using .Ignite

include("rician.jl")
include("batched_math.jl")
include("math_utils.jl")
include("mmd.jl")
include("utils.jl")
include("layers.jl")

end # module MMDLearning
