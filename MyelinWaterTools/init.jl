# ============================================================================ #
# Module path loading (do this before using Revise to ensure Revise sees them)
# ============================================================================ #
using Revise
include(joinpath(@__DIR__, "initpaths.jl"))

#####
##### Standard library packages
#####
using Statistics
using StatsBase
using Printf
using SparseArrays
using SuiteSparse
using LinearAlgebra
using Random
using BenchmarkTools
using Profile
# using Arpack
# using Distributed

#####
##### Debugging
#####
# using Debugger

#####
##### Plotting
#####
# NOTE: Must use pyplot() backend before using MATLAB (on some versions of Matlab);
#       MATLAB is loaded within MWFUTils below, so it is best to invoke pyplot() now
using StatsPlots; pyplot()

#####
##### My files and modules to load
#####
using GlobalUtils
using MWFUtils

#####
##### Misc. useful packages
#####
# using Parameters
# using JuAFEM, Tensors
# using JuAFEM: vertices, faces, edges
# using StaticArrays
# using DiffEqBase, OrdinaryDiffEq
# using DifferentialEquations
# using DiffEqOperators
# using Sundials

#####
##### Other packages
#####
# using Interpolations
# using LsqFit
# using Expokit
# using LinearMaps
# using Parameters
# using Optim
# using Roots
# using Distributions
# using ForwardDiff
# using PolynomialRoots
# using ApproxFun
# using Flatten
# using Plots