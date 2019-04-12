# ---------------------------------------------------------------------------- #
# BlochTorreyUtils
# ---------------------------------------------------------------------------- #

module BlochTorreyUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using GeometryUtils
using MeshUtils
using DistMesh
using JuAFEM
using LinearAlgebra
using SparseArrays
using SuiteSparse # need to define ldiv! for SuiteSparse.CHOLMOD.Factor
using StatsBase
using LinearMaps
# using Normest1
import ExpmV

using Parameters: @with_kw, @unpack

import Distributions
import Lazy

# ---------------------------------------------------------------------------- #
# Exported Methods
# ---------------------------------------------------------------------------- #
export normest1_norm, radiidistribution
export doassemble!, factorize!, interpolate, interpolate!, integrate #, addquadweights
export getgrid, getdomain, numfibres, createmyelindomains, omegamap
export getdofhandler, getcellvalues, getfacevalues,
       getmass, getmassfact, getstiffness, # getquadweights
       getregion, getoutercircles, getinnercircles, getoutercircle, getinnercircle, getouterradius, getinnerradius
export testproblem

# ---------------------------------------------------------------------------- #
# Exported Types
# ---------------------------------------------------------------------------- #

export BlochTorreyParameters
export FieldType, DofType, MassType, MassFactType, StiffnessType, MyelinBoundary
export AbstractParabolicProblem, MyelinProblem, BlochTorreyProblem
export AbstractDomain, ParabolicDomain, MyelinDomain, TriangularMyelinDomain, TriangularGrid
export AbstractRegion, AbstractRegionUnion, AxonRegion, MyelinRegion, TissueRegion, PermeableInterfaceRegion
export ParabolicLinearMap, DiffEqParabolicLinearMapWrapper

include("src/types.jl")
include("src/btparams.jl")
include("src/domains.jl")
include("src/linearmaps.jl")
include("src/frequencyfields.jl")

end # module BlochTorreyUtils