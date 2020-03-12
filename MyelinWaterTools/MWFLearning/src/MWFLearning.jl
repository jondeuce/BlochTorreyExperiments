# ============================================================================ #
# MWFLearning
# ============================================================================ #

module MWFLearning

import Flux, Flux.NNlib, Flux.Optimise, DrWatson, BSON, TOML, Dates
export Flux,      NNlib,      Optimise, DrWatson, BSON, TOML, Dates

using Statistics: mean, median, cov, std, var
using StatsBase: quantile
using StaticArrays
using Tensors
import MultivariateStats
import Interpolations
import FFTW
import NonNegLeastSquares
using NonNegLeastSquares: NNLS

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
@reexport using Printf
@reexport using StatsPlots
@reexport using DrWatson
@reexport using Parameters
@reexport using LegibleLambdas
@reexport using DataFrames
@reexport using TimerOutputs

export getnow, savebson, epochthrottle
export make_model, make_activation, model_summary, model_string
export make_minibatch, training_batches, testing_batches, param_summary, make_losses, lr, lr!, features, labels
export heightsize, batchsize, channelsize, log10range, linspace, logspace, unitsum, unitsum!
export prepare_data, label_fun, init_data, init_labels

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