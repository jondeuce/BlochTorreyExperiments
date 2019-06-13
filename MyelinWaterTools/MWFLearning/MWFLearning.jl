# ============================================================================ #
# MWFLearning
# ============================================================================ #

module MWFLearning

using Reexport
@reexport using LinearAlgebra
using Statistics: mean, median
#using StatsBase: quantile

import Flux, Flux.NNlib, Flux.Tracker, Flux.Optimise, DrWatson, BSON, TOML, Dates, MultivariateStats
export Flux,      NNlib,      Tracker,      Optimise, DrWatson, BSON, TOML, Dates, MultivariateStats

using DrWatson: @dict, @ntuple
using Parameters: @unpack
export @dict, @ntuple, @unpack

export model_summary, get_model, get_activation
export heightsize, batchsize, channelsize, log10range
export prepare_data, label_fun, init_data, init_labels, init_signal
export project_onto_exp, project_onto_exp!

include("src/utils.jl")
include("src/models.jl")

end # module MWFLearning