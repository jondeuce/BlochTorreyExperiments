# ENV["MATLAB_HOME"] = "C:\\Users\\Jonathan\\Downloads\\Mathworks Matlab R2016a\\R2016a"
# HOME = "C:\\Users\\Jonathan\\Documents\\MATLAB\\"
# HOME = "/home/jon/Documents/UBCMRI/"
HOME = "/home/coopar7/Documents/code/"
BTMASTERPATH = HOME * "BlochTorreyExperiments-master/"
MWOPATH = BTMASTERPATH * "Experiments/MyelinWaterOrientation/"

using Revise
push!(LOAD_PATH, MWOPATH .* ("", "Geometry/", "Utils/", "Expmv/")...)

@static if VERSION >= v"0.7.0"
    # Packages moved out of base for v0.7.0
    using Statistics
    using StatsBase
    using Printf
    using SparseArrays
    using LinearAlgebra
    using Arpack
    using Profile
end

# My files and modules to load
using Expmv
using GeometryUtils
using CirclePackingUtils
using MeshUtils
using BlochTorreyUtils
using BlochTorreySolvers

# Packages to load
using BenchmarkTools
# using IterTools
# using Parameters
# using StaticArrays
using JuAFEM
# using JuAFEM: vertices, faces, edges
using MATLAB
# using LinearMaps
using DifferentialEquations
using Expokit
# using Optim
# using Roots
# using Distributions
# using ForwardDiff
# using PolynomialRoots
# using ApproxFun
using Interpolations
using LsqFit
# using Flatten
# using Plots

# Packages currently not working on v0.7.0:
@static if VERSION < v"0.7.0"
    using Traceur
    using ReverseDiff
end
