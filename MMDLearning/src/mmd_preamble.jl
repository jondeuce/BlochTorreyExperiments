####
#### Load packages
####

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DECAES
using LinearAlgebra, Statistics, Distributions, Random, GaussianMixtures
using DrWatson, Parameters, BenchmarkTools, Dates, TimerOutputs
using Optim, BlackBoxOptim, Flux, ForwardDiff
using TOML, BSON, DataFrames
using StatsPlots
# pyplot(size = (500,400))
pyplot(size = (800,600))
# empty!(Revise.queue_errors);

####
#### Load settings file
####

# Load default settings
settings = TOML.parsefile(joinpath(@__DIR__, "default_settings.toml"))

# Load custom settings + merge into default settings
mergereducer!(x, y) = deepcopy(y) # fallback
mergereducer!(x::Dict, y::Dict) = merge!(mergereducer!, x, y)
if haskey(ENV, "SETTINGSFILE")
    merge!(mergereducer!, settings, TOML.parsefile(ENV["SETTINGSFILE"]))
end

# Save + print resulting settings
let outpath = settings["data"]["out"]
    !isdir(outpath) && mkpath(outpath)
    open(joinpath(outpath, "settings.toml"); write = true) do io
        TOML.print(io, settings)
    end
end
TOML.print(stdout, settings)

####
#### Load image data
####

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

nothing