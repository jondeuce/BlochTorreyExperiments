# ENV["MATLAB_HOME"] = "C:\\Users\\Jonathan\\Downloads\\Mathworks Matlab R2016a\\R2016a"
# HOME = "C:\\Users\\Jonathan\\Documents\\MATLAB\\"
# HOME = "/home/jon/Documents/UBCMRI/"
HOME = "/home/coopar7/Documents/code/"
BTMASTERPATH = HOME * "BlochTorreyExperiments-master/"
MWOPATH = BTMASTERPATH * "Experiments/MyelinWaterOrientation/"
MWOFOLDERS = ["BlochTorrey/", "CirclePacking/", "DistMesh/", "Expmv/", "Geometry/", "Utils/"]
push!(LOAD_PATH, MWOPATH .* ("", MWOFOLDERS...)...)

# ============================================================================ #
# Revise and Rebugger
# ============================================================================ #
try
    @eval using Revise
    # Turn on Revise's automatic-evaluation behavior
    Revise.async_steal_repl_backend()
catch err
    @warn "Could not load Revise."
end

try
    @eval using Rebugger
    # Activate Rebugger's key bindings
    Rebugger.keybindings[:stepin] = "\e[17~"      # Add the keybinding F6 to step into a function.
    Rebugger.keybindings[:stacktrace] = "\e[18~"  # Add the keybinding F7 to capture a stacktrace.
    atreplinit(Rebugger.repl_init)
catch
    @warn "Could not load Rebugger."
end

@static if VERSION >= v"0.7.0"
    # Packages moved out of base for v0.7.0+
    using Statistics
    using StatsBase
    using Printf
    using SparseArrays
    using LinearAlgebra
    using Arpack
    using Profile
    using Random
    using Distributed
end

# # Packages currently not working on v0.7.0:
# @static if VERSION < v"0.7.0"
#     using Traceur
#     using ReverseDiff
# end

# My files and modules to load
using Expmv
using GeometryUtils
using CirclePackingUtils
using MeshUtils
using BlochTorreyUtils
using BlochTorreySolvers

import EnergyCirclePacking
import GreedyCirclePacking

# Packages to load
using JuAFEM
# using JuAFEM: vertices, faces, edges
using MATLAB
# using DifferentialEquations
using OrdinaryDiffEq, DiffEqOperators, Sundials
# using Interpolations
# using LsqFit
# using Expokit
using BenchmarkTools
# using LinearMaps
# using Parameters
# using StaticArrays
# using Optim
# using Roots
# using Distributions
# using ForwardDiff
# using PolynomialRoots
# using ApproxFun
# using Flatten
# using Plots

# ============================================================================ #
# Moved to ~/.julia/config/startup.jl
# ============================================================================ #
#
# # ENV["MATLAB_HOME"] = "C:\\Users\\Jonathan\\Downloads\\Mathworks Matlab R2016a\\R2016a"
# # HOME = "C:\\Users\\Jonathan\\Documents\\MATLAB\\"
# # HOME = "/home/jon/Documents/UBCMRI/"
# HOME = "/home/coopar7/Documents/code/"
# BTMASTERPATH = HOME * "BlochTorreyExperiments-master/"
# MWOPATH = BTMASTERPATH * "Experiments/MyelinWaterOrientation/"
# MWOFOLDERS = ["BlochTorrey/", "CirclePacking/", "DistMesh/", "Expmv/", "Geometry/", "Utils/"]
# push!(LOAD_PATH, MWOPATH .* ("", MWOFOLDERS...)...)
