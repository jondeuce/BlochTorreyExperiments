####
#### Load packages
####

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra, Statistics, Random, SpecialFunctions, StatsBase
using DrWatson, Parameters, BenchmarkTools, Dates, TimerOutputs, ThreadPools, LegibleLambdas
using TOML, BSON, DataFrames
using Distributions #Turing, MCMCChains
using BlackBoxOptim, Optim, ForwardDiff
using UnsafeArrays, TensorCast, Yeppp
using Flux, Zygote
using DECAES
using StatsPlots

# pyplot(size = (500,400))
pyplot(size = (800,600))
# Turing.turnprogress(false)
# empty!(Revise.queue_errors);

####
#### Includes
####

using Revise
Revise.includet(joinpath(@__DIR__, "rician.jl"))
Revise.includet(joinpath(@__DIR__, "batchedmath.jl"))
Revise.includet(joinpath(@__DIR__, "mmd_math.jl"))
Revise.includet(joinpath(@__DIR__, "mmd_flux.jl"))
Revise.includet(joinpath(@__DIR__, "mmd_utils.jl"))

nothing