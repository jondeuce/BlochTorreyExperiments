####
#### Load packages
####

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra, Statistics, Random, SpecialFunctions
using DrWatson, Parameters, BenchmarkTools, Dates, TimerOutputs, ThreadPools, LegibleLambdas
using TOML, BSON, DataFrames
using Turing, MCMCChains, Distributions
using BlackBoxOptim, Optim, ForwardDiff, TensorCast, Yeppp
using Flux
using DECAES
using StatsPlots

# pyplot(size = (500,400))
pyplot(size = (800,600))
Turing.turnprogress(false)
# empty!(Revise.queue_errors);

####
#### Includes
####

include(joinpath(@__DIR__, "rician.jl")) #Revise.includet
include(joinpath(@__DIR__, "batchedmath.jl")) #Revise.includet
include(joinpath(@__DIR__, "mmd_math.jl")) #Revise.includet
include(joinpath(@__DIR__, "mmd_flux.jl")) #Revise.includet
include(joinpath(@__DIR__, "mmd_utils.jl")) #Revise.includet

####
#### Load settings file
####

settings = let
    # Load default settings + merge in custom settings, if given
    settings = TOML.parsefile(joinpath(@__DIR__, "default_settings.toml"))
    mergereducer!(x, y) = deepcopy(y) # fallback
    mergereducer!(x::Dict, y::Dict) = merge!(mergereducer!, x, y)
    haskey(ENV, "SETTINGSFILE") && merge!(mergereducer!, settings, TOML.parsefile(ENV["SETTINGSFILE"]))

    # Save + print resulting settings
    outpath = settings["data"]["out"]
    !isdir(outpath) && mkpath(outpath)
    open(joinpath(outpath, "settings.toml"); write = true) do io
        TOML.print(io, settings)
    end
    TOML.print(stdout, settings)
    settings
end

####
#### Load image data
####

#=
const image   = DECAES.load_image("/project/st-arausch-1/jcd1994/MMD-Learning/data/masked-image-240x240x48x48.mat")
const t2dist  = DECAES.load_image("/project/st-arausch-1/jcd1994/MMD-Learning/data/masked-image-240x240x48x48.t2dist.mat")
const t2parts = DECAES.MAT.matread("/project/st-arausch-1/jcd1994/MMD-Learning/data/masked-image-240x240x48x48.t2parts.mat")
const opts    = T2mapOptions(
    MatrixSize = size(image)[1:3],
    nTE        = size(image)[4],
    TE         = 8e-3,
    T2Range    = (15e-3, 2.0),
    nT2        = 40,
);
=#

nothing