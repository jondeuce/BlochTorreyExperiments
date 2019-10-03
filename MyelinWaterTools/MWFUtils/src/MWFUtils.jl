module MWFUtils

const AVOID_MAT_PLOTS = true # avoid external matlab calls, if possible

using Reexport
@reexport using BlochTorreyUtils
@reexport using CirclePackingUtils
@reexport using MATLABPlots
@reexport using StatsPlots
@reexport using LaTeXStrings
@reexport using BenchmarkTools

import BSON, Dates, DrWatson
export BSON, Dates, DrWatson

# For curve fitting/optimization in calculating MWF
import ForwardDiff
import LsqFit
import BlackBoxOptim

export packcircles
export creategeometry, createdomains, loadgeometry, geometrytuple
export calcomegas, calcomega
export calcsignals, calcsignal
export solveblochtorrey, saveblochtorrey, default_algorithm
export plotcircles, plotgrids, plotSEcorr, plotmultiexp, plotsignal, plotMWFvsAngle, plotMWFvsMethod
export mxplotomega, mxplotmagnitude, mxplotphase, mxplotlongitudinal, mxgifmagnitude, mxgifphase, mxgiflongitudinal
export mxsavefig, getnow

export AbstractMWIFittingModel, NNLSRegression, TwoPoolMagnToMagn, ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx
export AbstractGeometry, AbstractMyelinatedFibresGeometry, AbstractPackedFibresGeometry, PeriodicPackedFibres, SingleFibre
export getmwf, fitmwfmodel, mwimodel, initialparams, compareMWFmethods
export blank_results_dict, load_results_dict
export wrap_string, partitionby

include("types.jl")
include("utils.jl")
include("signalmodels.jl")
include("plotutils.jl")

end # module MWFUtils