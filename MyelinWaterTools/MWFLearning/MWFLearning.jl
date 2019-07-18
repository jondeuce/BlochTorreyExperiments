# ============================================================================ #
# MWFLearning
# ============================================================================ #

module MWFLearning

import Flux, Flux.NNlib, Flux.Tracker, Flux.Optimise, DrWatson, BSON, TOML, Dates
export Flux,      NNlib,      Tracker,      Optimise, DrWatson, BSON, TOML, Dates

using Statistics: mean, median, cov, std, var
using StatsBase: quantile
import MultivariateStats

using Reexport

@reexport using RecipesBase, LaTeXStrings
using RecipesBase: plot, plot!
export plot, plot!, mean, median, cov, std, var, quantile

@reexport using Random
@reexport using Distributions
@reexport using Base.Iterators
@reexport using GeometryUtils
@reexport using EllipsisNotation
@reexport using Wavelets
@reexport using NNLS

using DrWatson: @dict, @ntuple
using Parameters: @unpack
using LegibleLambdas: @λ
export @dict, @ntuple, @unpack, @λ

export verify_settings, model_summary, get_model, get_activation
export heightsize, batchsize, channelsize, log10range
export prepare_data, label_fun, init_data, init_labels, init_signal

# Layers
export AdaBound
export PrintSize, DenseResize, ChannelResize, Scale, Sumout
export IdentitySkip, CatSkip, ChannelwiseDense, HeightwiseDense
export BatchDenseConnection, BatchConvConnection
export DenseConnection, ResidualDenseBlock
export GlobalFeatureFusion, DenseFeatureFusion

include("src/transforms.jl")
include("src/utils.jl")
include("src/loading.jl")
include("src/layers.jl")
include("src/optimizers.jl")
include("src/models.jl")

end # module MWFLearning