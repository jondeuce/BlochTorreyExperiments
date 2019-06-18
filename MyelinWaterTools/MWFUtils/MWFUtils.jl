module MWFUtils

const AVOID_MAT_PLOTS = true # avoid external matlab calls, if possible

using Reexport
@reexport using BlochTorreyUtils
@reexport using CirclePackingUtils
@reexport using MATLABPlots
@reexport using StatsPlots
@reexport using LaTeXStrings

import BSON, Dates, DrWatson
export BSON, Dates, DrWatson

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
export plotcircles, plotgrids, plotSEcorr, plotbiexp, plotsignal, plotMWFvsAngle, plotMWFvsMethod
export mxplotomega, mxplotmagnitude, mxplotphase, mxplotlongitudinal
export mxsavefig, getnow

export AbstractMWIFittingModel, NNLSRegression, TwoPoolMagnToMagn, ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx
export getmwf, fitmwfmodel, mwimodel, initialparams, compareMWFmethods
export blank_results_dict, load_results_dict
export wrap_string, partitionby

include("src/mwftypes.jl")
include("src/mwfutils.jl")
include("src/mwfmodels.jl")
include("src/mwfplotutils.jl")

end # module MWFUtils