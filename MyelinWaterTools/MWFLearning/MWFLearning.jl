# ============================================================================ #
# MWFLearning
# ============================================================================ #

module MWFLearning

import Flux, Flux.NNlib, Flux.Tracker, Flux.Optimise, DrWatson, BSON, TOML, Dates
export Flux,      NNlib,      Tracker,      Optimise, DrWatson, BSON, TOML, Dates

using Reexport
@reexport using LinearAlgebra
using Statistics: mean, median, cov, std, var
using StatsBase: quantile
import MultivariateStats

@reexport using RecipesBase, LaTeXStrings
using RecipesBase: plot, plot!
export plot, plot!, mean, median, cov, std, var, quantile

@reexport using EllipsisNotation
@reexport using Wavelets
using DrWatson: @dict, @ntuple
using Parameters: @unpack
using LegibleLambdas: @λ
export @dict, @ntuple, @unpack, @λ

export verify_settings, model_summary, get_model, get_activation
export heightsize, batchsize, channelsize, log10range
export prepare_data, label_fun, init_data, init_labels, init_signal

# Layers
export PrintSize, DenseResize, ChannelResize, Scale, Sumout
export IdentitySkip, CatSkip, DenseCatSkip
export DenseResConnection, ConvResConnection
export ResidualDenseBlock

include("src/transforms.jl")
include("src/utils.jl")
include("src/layers.jl")
include("src/models.jl")

end # module MWFLearning