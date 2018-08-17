# My files and modules

# ENV["MATLAB_HOME"] = "C:\\Users\\Jonathan\\Downloads\\Mathworks Matlab R2016a\\R2016a"
# HOME = "C:\\Users\\Jonathan\\Documents\\MATLAB\\"
# HOME = "/home/jon/Documents/UBCMRI/"
HOME = "/home/coopar7/Documents/code/"
BTMASTER = HOME * "BlochTorreyExperiments-master/"
MWOPATH = BTMASTER * "Experiments/MyelinWaterOrientation/"

cd(BTMASTER)

@static if VERSION >= v"0.7.0"
    # Packages moved out of base for v0.7.0
    using Statistics
    using StatsBase
    using Printf
    using SparseArrays
    using LinearAlgebra
    using Profile

    # Flatten caused Pkg errors on v0.6.4 for some reason...?
    # using Flatten
end

# Packages
using Revise
using BenchmarkTools
using IterTools
using Parameters
using StaticArrays
using JuAFEM
using JuAFEM: vertices, faces, edges
using MATLAB
using LinearMaps
using DifferentialEquations
using Expokit
using Optim
using Roots
using Distributions
using ForwardDiff
using PolynomialRoots
using ApproxFun
using Interpolations
using LsqFit
# using Plots

# Packages currently not working on v0.7.0:
@static if VERSION < v"0.7.0"
    using Traceur
    using ReverseDiff
end

include(MWOPATH * "Utils/normest1.jl")
include(MWOPATH * "Expmv/src/Expmv.jl")
Revise.track(MWOPATH * "Utils/normest1.jl")
Revise.track(MWOPATH * "Expmv/src/Expmv.jl")

@static if VERSION < v"0.7.0"
    using Normest1
    using Expmv
else
    # using Main.Normest1
    # using Main.Expmv
end

include(MWOPATH * "Geometry/geometry_utils.jl")
include(MWOPATH * "Geometry/circle_packing.jl")
include(MWOPATH * "Utils/mesh_utils.jl")
include(MWOPATH * "Utils/blochtorrey_utils.jl")
Revise.track(MWOPATH * "Geometry/geometry_utils.jl")
Revise.track(MWOPATH * "Geometry/circle_packing.jl")
Revise.track(MWOPATH * "Utils/mesh_utils.jl")
Revise.track(MWOPATH * "Utils/blochtorrey_utils.jl")

# add Arpack Atom BenchmarkTools BlockArrays DifferentialEquations Distributions Expokit Flatten ForwardDiff GR IJulia Interpolations IterTools JuAFEM Juno LinearMaps MATLAB Optim Parameters Plots PolynomialRoots Revise Roots StaticArrays StatsBase Tensors TimerOutputs
# add https://github.com/KristofferC/JuAFEM.jl
