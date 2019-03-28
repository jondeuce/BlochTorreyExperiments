module MWFUtils

using LinearAlgebra, Statistics
using GeometryUtils
using CirclePackingUtils
using MeshUtils
using DistMesh
using BlochTorreyUtils
using BlochTorreySolvers

using JuAFEM
using OrdinaryDiffEq, DiffEqOperators, Sundials
using BenchmarkTools
using Parameters: @with_kw, @unpack
using IterableTables, DataFrames, BSON, CSV, Dates

# Plotting
using StatsPlots, MATLABPlots

export packcircles, creategrids, createdomains
export calcomegas, calcomega
export calcsignals, calcsignal
export solveblochtorrey, default_algfun, get_algfun
export plotmagnitude, plotphase, plotSEcorr, plotbiexp
export compareMWFmethods
export mxsavefig, getnow

export MWFResults

include("src/utils.jl")
include("src/plotutils.jl")
include("src/mwfresults.jl")

end # module MWFUtils

nothing