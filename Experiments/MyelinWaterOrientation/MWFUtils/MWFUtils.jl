module MWFUtils

# using LinearAlgebra, Statistics
# using GeometryUtils
# using MeshUtils
# using DistMesh
# using BlochTorreyUtils
# using BlochTorreySolvers

# using JuAFEM
# using OrdinaryDiffEq, DiffEqOperators, Sundials
# using Parameters: @with_kw, @unpack

using Reexport
@reexport using BlochTorreyUtils
@reexport using CirclePackingUtils

const AVOID_MAT_PLOTS = true # avoid external matlab calls, if possible
using MATLABPlots

using StatsPlots
using IterableTables, DataFrames
import BSON, CSV, Dates

# For curve fitting/optimization in calculating MWF
import ForwardDiff
import LsqFit
import BlackBoxOptim

export packcircles
export creategeometry, loadgeometry
export createdomains
export calcomegas, calcomega
export calcsignals, calcsignal
export solveblochtorrey, default_algorithm
export plotomega, plotmagnitude, plotphase, plotSEcorr, plotbiexp, plotMWF
export mxsavefig, getnow

export AbstractMWIFittingModel, NNLSRegression, TwoPoolMagnToMagn, ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx
export getmwf, fitmwfmodel, mwimodel, initialparams, compareMWFmethods
export blank_results_dict, load_results_dict

include("src/mwftypes.jl")
include("src/mwfutils.jl")
include("src/mwfmodels.jl")
include("src/mwfplotutils.jl")

# TODO: deprecate MWFResults
# export MWFResults
# include("src/mwfresults.jl")

end # module MWFUtils