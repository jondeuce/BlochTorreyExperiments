module MMDLearning

using Reexport
@reexport using LinearAlgebra, Statistics, Random, Dates, Printf
@reexport using StatsBase, SpecialFunctions, Distributions, DataFrames, TimerOutputs, BenchmarkTools
@reexport using Parameters, EllipsisNotation, LegibleLambdas, LaTeXStrings
@reexport using StatsPlots

import UnsafeArrays, LoopVectorization
using UnsafeArrays: uview, uviews, @uviews
using LoopVectorization: @avx

import TOML, BSON, DrWatson, Flux, NNlib, Zygote, ChainRules, BlackBoxOptim, Optim, ForwardDiff, DECAES
export TOML, BSON, DrWatson, Flux, NNlib, Zygote, ChainRules, BlackBoxOptim, Optim, ForwardDiff, DECAES
using DrWatson: @dict, @ntuple, @pack!, @unpack
export @dict, @ntuple, @pack!, @unpack

export load_settings
export handleinterrupt, saveprogress, saveplots
# export make_toy_samplers, make_mle_data_samplers
# export signal_loglikelihood_inference
# export signal_theta_error, theta_bounds, signal_model, signal_model!, signal_model_work
# export toy_theta_error, toy_theta_bounds, toy_signal_model, toy_theta_sampler
export mmd_flux, mmd_and_mmdvar_flux, tstat_flux, kernel_bandwidth_loss_flux, train_kernel_bandwidth_flux!
export mmd_perm_test_power, mmd_perm_test_power_plot

export NotTrainable, DenseResize, Scale
export Rician, RicianCorrector, VectorRicianCorrector, FixedNoiseVectorRicianCorrector
export correction, noiselevel, correction_and_noiselevel, noise_instance, corrected_signal_instance, rician_params

export PhysicsModel, ClosedForm, ToyModel
export physicsmodel, hasclosedform, initialize!, ntheta, nsignal, signal_model, epsilon
export θbounds, θlower, θupper, θlabels, θerror
export sampleθ, sampleX, sampleY
export initialize_callback, update_callback!

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

end # module
