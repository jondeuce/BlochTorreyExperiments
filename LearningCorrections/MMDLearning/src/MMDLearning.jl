module MMDLearning

using Reexport
@reexport using LinearAlgebra, Distributions, Statistics, Random, SpecialFunctions, StatsBase, Dates, Printf, DataFrames
@reexport using DrWatson, Parameters, EllipsisNotation, LegibleLambdas, BenchmarkTools, TimerOutputs, LaTeXStrings
@reexport using StatsPlots

import UnsafeArrays
using UnsafeArrays: uview, uviews, @uviews

import TOML, BSON, Flux, NNlib, Zygote, ChainRules, BlackBoxOptim, Optim, ForwardDiff, Yeppp, DECAES
export TOML, BSON, Flux, NNlib, Zygote, ChainRules, BlackBoxOptim, Optim, ForwardDiff, Yeppp, DECAES

export load_settings
export make_toy_samplers, make_mle_data_samplers
export signal_loglikelihood_inference
export signal_theta_error, theta_bounds, signal_model, signal_model!, signal_model_work
export toy_theta_error, toy_theta_bounds, toy_signal_model, toy_theta_sampler
export mmd_flux, mmd_and_mmdvar_flux, mmd_flux_bandwidth_optfun
export mmd_perm_test_power, mmd_perm_test_power_plot
export Rician, NotTrainable, DenseResize, Scale

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

end # module
