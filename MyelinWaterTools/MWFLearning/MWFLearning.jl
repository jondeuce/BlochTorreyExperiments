# ============================================================================ #
# MWFLearning
# ============================================================================ #

module MWFLearning

using Reexport
@reexport using LinearAlgebra
using Statistics: mean, median, cov, std, var
using StatsBase: quantile

import Flux, Flux.NNlib, Flux.Tracker, Flux.Optimise, DrWatson, BSON, TOML, Dates, MultivariateStats, Wavelets
export Flux,      NNlib,      Tracker,      Optimise, DrWatson, BSON, TOML, Dates, MultivariateStats, Wavelets

using DrWatson: @dict, @ntuple
using Parameters: @unpack
using LegibleLambdas: @λ
export @dict, @ntuple, @unpack, @λ

export verify_settings, model_summary, get_model, get_activation
export heightsize, batchsize, channelsize, log10range
export prepare_data, label_fun, init_data, init_labels, init_signal
export ilaplace, ilaplace!

include("src/transforms.jl")
include("src/utils.jl")
include("src/models.jl")

end # module MWFLearning