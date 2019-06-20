# ---------------------------------------------------------------------------- #
# BlochTorreyUtils
# ---------------------------------------------------------------------------- #

module BlochTorreyUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using Reexport
@reexport using GeometryUtils
@reexport using DiffEqBase, OrdinaryDiffEq, DiffEqCallbacks, DiffEqOperators#, Sundials
@reexport using LinearMaps
@reexport using Rotations
@reexport using BlockArrays

using DrWatson: @dict, @ntuple
using Parameters: @with_kw, @unpack
export @with_kw, @unpack, @dict, @ntuple

import SuiteSparse # for defining ldiv! on SuiteSparse.CHOLMOD.Factor's
import ExpmV, Expokit
import Distributions
import Random
import Lazy

include("src/types.jl")
include("src/btparams.jl")
include("src/domains.jl")
include("src/linearmaps.jl")
include("src/frequencyfields.jl")
include("src/algorithms.jl")
include("src/callbacks.jl")

# ---------------------------------------------------------------------------- #
# Exported Methods
# ---------------------------------------------------------------------------- #
export normest1_norm, radiidistribution
export doassemble!, factorize!, interpolate, interpolate!, integrate
export fieldvectype, fieldfloattype, getgrid, getdomain, numfibres, createmyelindomains, omegamap
export getmass, getmassfact, getstiffness
       getdofhandler, getcellvalues, getfacevalues,
       getregion, getoutercircles, getinnercircles, getoutercircle, getinnercircle, getouterradius, getinnerradius
export shift_longitudinal, shift_longitudinal!, pi_flip, pi_pulse!, apply_pulse!, init_fliptimes, init_savetimes
export testproblem

# ---------------------------------------------------------------------------- #
# Exported Types
# ---------------------------------------------------------------------------- #

export BlochTorreyParameters
export FieldType, DofType, MassType, MassFactType, StiffnessType, MyelinBoundary
export AbstractParabolicProblem, MyelinProblem, BlochTorreyProblem
export AbstractDomain, ParabolicDomain, MyelinDomain, TriangularMyelinDomain, TriangularGrid
export AbstractRegion, AbstractRegionUnion, AxonRegion, MyelinRegion, TissueRegion, PermeableInterfaceRegion
export ParabolicLinearMap, LinearOperatorWrapper
export MultiSpinEchoCallback
export ExpokitExpmv, HighamExpmv

end # module BlochTorreyUtils