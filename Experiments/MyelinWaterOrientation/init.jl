# ============================================================================ #
# Module path loading (do this before using Revise to ensure Revise sees them)
# ============================================================================ #
using Revise
include(joinpath(@__DIR__, "initpaths.jl"))

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

# My files and modules to load
using GeometryUtils
using CirclePackingUtils
using MeshUtils
using BlochTorreyUtils
using BlochTorreySolvers
using MATLABPlots
using MWFUtils
# using TestBlochTorrey2D

import EnergyCirclePacking
import GreedyCirclePacking

# Debugging packages
using BenchmarkTools
using Debugger
# using Traceur

# Useful packages to have loaded
using JuAFEM, Tensors
# using JuAFEM: vertices, faces, edges
using StaticArrays
# using DifferentialEquations
# using DiffEqOperators
# using OrdinaryDiffEq, Sundials

# # Other packages
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
