# ============================================================================ #
# MWFLearning
# ============================================================================ #

module MWFLearning

import Flux, Flux.NNlib, Flux.Tracker, Flux.Optimise, DrWatson, BSON, TOML, Dates
export Flux,      NNlib,      Tracker,      Optimise, DrWatson, BSON, TOML, Dates

using Statistics: mean, median, cov, std, var
using StatsBase: quantile
using StaticArrays
using Tensors
import MultivariateStats
import Interpolations
import FFTW

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
@reexport using Printf
@reexport using StatsPlots

using DrWatson: @dict, @ntuple
using Parameters: @unpack
using LegibleLambdas: @λ
export @dict, @ntuple, @unpack, @λ

export getnow, savebson, epochthrottle
export verify_settings, model_summary, get_model, get_activation
export make_minibatch, training_batches, testing_batches, param_summary, makelosses, lr, lr!, features, labels
export heightsize, batchsize, channelsize, log10range, linspace, logspace, unitsum
export prepare_data, label_fun, init_data, init_labels, init_signal

# Layers
export MomentumW, AdaBound
export printsize, wrapprint
export PrintSize, DenseResize, ChannelResize, Scale, Sumout
export IdentitySkip, CatSkip, ChannelwiseDense, HeightwiseDense
export BatchDenseConnection, BatchConvConnection
export DenseConnection, ResidualDenseBlock
export GlobalFeatureFusion, DenseFeatureFusion

# ResNet
include("resnet.jl")
@reexport using .ResNet

# Source
include("transforms.jl")
include("utils.jl")
include("loading.jl")
include("layers.jl")
include("optimizers.jl")
include("models.jl")

end # module MWFLearning