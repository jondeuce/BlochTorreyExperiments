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

# MWF Calculation
import ForwardDiff
import LsqFit
import BlackBoxOptim

# Plotting
using StatsPlots
using MATLABPlots

export packcircles
export creategeometry, loadgeometry
export createdomains
export calcomegas, calcomega
export calcsignals, calcsignal
export solveblochtorrey, default_algfun, get_algfun
export plotmagnitude, plotphase, plotSEcorr, plotbiexp
export mxsavefig, getnow

export AbstractMWIFittingModel, NNLSRegression, TwoPoolMagnToMagn, ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx
export getmwf, fitmwfmodel, mwimodel, initialparams, compareMWFmethods

export MWFResults

include("src/mwftypes.jl")
include("src/mwfutils.jl")
include("src/mwfmodels.jl")
include("src/mwfplotutils.jl")

end # module MWFUtils

nothing