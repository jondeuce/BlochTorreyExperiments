module MMDLearning

using Reexport
@reexport using LinearAlgebra, Statistics, Random, Dates, Printf
@reexport using StatsBase, Distributions, DataFrames, SpecialFunctions, FFTW, TimerOutputs, BenchmarkTools
@reexport using Parameters, BangBang, EllipsisNotation, LegibleLambdas, LaTeXStrings
@reexport using StatsPlots

@reexport using LoopVectorization, Tullio
import UnsafeArrays
using UnsafeArrays: uview, uviews, @uviews

import Pkg.TOML, BSON, Glob, DrWatson, Flux, Flux.CUDA, NNlib, Zygote, ChainRules, Transformers, BlackBoxOptim, Optim, NLopt, FiniteDiff, ForwardDiff, SparseDiffTools, DECAES, HypothesisTests
export     TOML, BSON, Glob, DrWatson, Flux,      CUDA, NNlib, Zygote, ChainRules, Transformers, BlackBoxOptim, Optim, NLopt, FiniteDiff, ForwardDiff, SparseDiffTools, DECAES, HypothesisTests
using DrWatson: @dict, @ntuple, @pack!, @unpack
export @dict, @ntuple, @pack!, @unpack

export load_settings
export handleinterrupt, saveprogress, saveplots
# export make_toy_samplers, make_mle_data_samplers
# export signal_loglikelihood_inference
# export signal_theta_error, theta_bounds, signal_model, signal_model!, signal_model_work
# export toy_theta_error, toy_theta_bounds, toy_signal_model, toy_theta_sampler
export mmd, mmdvar, mmd_and_mmdvar, tstat_flux, kernel_loss, train_kernel!
export mmd_perm_test_power, mmd_perm_test_power_plot, fast_hist_1D

export NotTrainable, DenseResize, Scale
export Rician, RicianCorrector, NormalizedRicianCorrector, VectorRicianCorrector, FixedNoiseVectorRicianCorrector, LatentVectorRicianCorrector, LatentVectorRicianNoiseCorrector, LatentScalarRicianNoiseCorrector
export correction, noiselevel, correction_and_noiselevel, corrected_signal_instance, add_correction, add_noise_instance, rician_params

export PhysicsModel, ClosedForm, ToyModel
export arr_similar, arr32, arr64, zeros_similar, ones_similar, randn_similar, rand_similar
export physicsmodel, hasclosedform, initialize!, nsignal, nlatent, ninput, noutput, ntheta, signal_model
export θbounds, θlower, θupper, θlabels, θasciilabels, θerror
export sampleθprior, sampleZprior, sampleWprior
export sampleθ, sampleX, sampleY
export initialize_callback, update_callback!

export CVAE, KLDivUnitNormal, KLDivergence, EvidenceLowerBound, KL_and_ELBO
export θZposterior_sampler, sampleθZposterior, θZ_sampler
export sampleθZ, sampleXθZ, sampleX̂θZ, sampleX̂

####
#### Global aliases
####
const AbstractTensor3D{T} = AbstractArray{T,3}
const AbstractTensor4D{T} = AbstractArray{T,4}
const CuTensor3D{T} = CUDA.CuArray{T,3}
const CuTensor4D{T} = CUDA.CuArray{T,4}

export AbstractTensor3D, AbstractTensor4D, CuTensor3D, CuTensor4D

####
#### PyPlot
####

import PyPlot
const plt = PyPlot
const rcParams = plt.PyDict(plt.matplotlib."rcParams")

rcParams["font.size"] = 10
rcParams["text.usetex"] = false

export PyPlot, plt, rcParams

####
#### Includes
####

include("rician.jl")
include("batched_math.jl")
include("math_utils.jl")
include("mmd.jl")
include("physics.jl")
include("utils.jl")
include("layers.jl")
include("mmdcvae.jl")
include("models.jl")

include("Ignite.jl")
@reexport using .Ignite

end # module