# ============================================================================ #
# Module path loading (do this before using Revise to ensure Revise sees them)
# ============================================================================ #
using Revise
include(joinpath(@__DIR__, "initpaths.jl"))

#####
##### Packages moved out of base for v0.7.0+
#####
@static if VERSION >= v"0.7.0"
    using Statistics
    using StatsBase
    using Printf
    using SparseArrays
    using SuiteSparse
    using LinearAlgebra
    using Profile
    using Random
    # using Arpack
    # using Distributed
end

#####
##### My files and modules to load
#####
using GlobalUtils
using GeometryUtils
using CirclePackingUtils
using MeshUtils
using DistMesh
using BlochTorreyUtils
using BlochTorreySolvers
using MATLABPlots
using MWFUtils

import EnergyCirclePacking
import GreedyCirclePacking

#####
##### Debugging packages
#####
using BenchmarkTools
using Debugger

#####
##### Misc. useful packages
#####
using Parameters
using JuAFEM, Tensors
# using JuAFEM: vertices, faces, edges
using StaticArrays
using DiffEqBase, OrdinaryDiffEq
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