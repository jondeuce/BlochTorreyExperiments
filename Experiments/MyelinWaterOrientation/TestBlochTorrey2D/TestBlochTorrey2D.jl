module TestBlochTorrey2D

# Packages to load
using LinearAlgebra
using BenchmarkTools
using MATLAB
using JuAFEM
using LinearMaps
using OrdinaryDiffEq, DiffEqOperators, Sundials
using Expokit

# My files and modules to load
using GeometryUtils
using CirclePackingUtils
using MeshUtils
using BlochTorreyUtils
using BlochTorreySolvers
using ExpmvHigham
using MWFUtils

export lap, testbtfindiff2D, testbtfinelem2D, testblochtorrey2D

include("src/testneumannfindiff.jl")
include("src/testbtfindiff2D.jl")
include("src/testbtfinelem2D.jl")
include("src/testblochtorrey2D.jl")

end # TestBlochTorrey2D