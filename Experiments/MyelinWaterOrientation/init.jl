# My files and modules
ENV["MATLAB_HOME"] = "C:\\Users\\Jonathan\\Downloads\\Mathworks Matlab R2016a\\R2016a"

HOME = "C:\\Users\\Jonathan\\Documents\\MATLAB\\"
# HOME = "/home/jon/Documents/UBCMRI/"
# HOME = "/home/coopar7/Documents/code/"
BTMASTER = HOME * "BlochTorreyExperiments-master/"
MWOPATH = BTMASTER * "Experiments/MyelinWaterOrientation/"

cd(BTMASTER)

# Packages
using Traceur
using BenchmarkTools
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
using ApproxFun
#using Plots
using ForwardDiff
using ReverseDiff
using IterTools

include(MWOPATH * "Geometry/geometry_utils.jl")
include(MWOPATH * "Geometry/circle_packing.jl")
include(MWOPATH * "Utils/mesh_utils.jl")
include(MWOPATH * "Utils/normest1.jl")
include(MWOPATH * "Utils/blochtorrey_utils.jl")
Revise.track(MWOPATH * "Geometry/geometry_utils.jl")
Revise.track(MWOPATH * "Geometry/circle_packing.jl")
Revise.track(MWOPATH * "Utils/mesh_utils.jl")
Revise.track(MWOPATH * "Utils/normest1.jl")
Revise.track(MWOPATH * "Utils/blochtorrey_utils.jl")

using Normest1
