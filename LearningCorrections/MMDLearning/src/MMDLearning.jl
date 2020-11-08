module MMDLearning

using Reexport
@reexport using LinearAlgebra, Statistics, Random, Dates, Printf
@reexport using StatsBase, Distributions, DataFrames, SpecialFunctions, FFTW, TimerOutputs, BenchmarkTools
@reexport using Parameters, BangBang, EllipsisNotation, LegibleLambdas, LaTeXStrings
@reexport using StatsPlots

@reexport using LoopVectorization, Tullio
import UnsafeArrays
using UnsafeArrays: uview, uviews, @uviews

import Pkg.TOML, BSON, DrWatson, Flux, Flux.CUDA, NNlib, Zygote, ChainRules, Transformers, BlackBoxOptim, Optim, ForwardDiff, DECAES, HypothesisTests
export     TOML, BSON, DrWatson, Flux,      CUDA, NNlib, Zygote, ChainRules, Transformers, BlackBoxOptim, Optim, ForwardDiff, DECAES, HypothesisTests
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
export Rician, RicianCorrector, NormalizedRicianCorrector, VectorRicianCorrector, FixedNoiseVectorRicianCorrector, LatentVectorRicianCorrector, LatentVectorRicianNoiseCorrector
export correction, noiselevel, correction_and_noiselevel, corrected_signal_instance, add_correction, add_noise_instance, rician_params

export PhysicsModel, ClosedForm, ToyModel
export randn_similar, rand_similar
export physicsmodel, hasclosedform, initialize!, nsignal, nlatent, ninput, noutput, ntheta, signal_model
export θbounds, θlower, θupper, θlabels, θasciilabels, θerror
export sampleθprior, sampleWprior
export sampleθ, sampleX, sampleY
export initialize_callback, update_callback!

####
#### Global aliases
####
const AbstractTensor3D{T} = AbstractArray{T,3}
const CuTensor3D{T} = CUDA.CuArray{T,3}

export AbstractTensor3D, CuTensor3D

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
include("models.jl")

include("Ignite.jl")
@reexport using .Ignite

end # module
