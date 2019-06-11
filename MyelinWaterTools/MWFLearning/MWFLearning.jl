# ============================================================================ #
# MWFLearning
# ============================================================================ #

module MWFLearning

using Reexport
@reexport using LinearAlgebra
using Statistics: mean, median
#using StatsBase: quantile

import Flux, Flux.NNlib, Flux.Tracker, Flux.Optimise, DrWatson, BSON, TOML, Dates
export Flux,      NNlib,      Tracker,      Optimise, DrWatson, BSON, TOML, Dates

using DrWatson: @dict, @ntuple
using Parameters: @unpack
export @dict, @ntuple, @unpack

export get_model, get_activation
export log10range, batchsize, channelsize
export prepare_data, label_fun, init_data, init_labels, init_signal
export project_onto_exp, project_onto_exp!

include("src/utils.jl")
include("src/models.jl")

end # module MWFLearning