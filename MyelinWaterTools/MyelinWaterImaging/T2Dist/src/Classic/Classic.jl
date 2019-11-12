# ============================================================================ #
# Classic
# ============================================================================ #

module Classic

using LinearAlgebra, Statistics, SparseArrays
using NNLS, Dierckx
using DrWatson: @ntuple
using Parameters, Printf

include("utils.jl")
include("lsqnonneg_reg.jl")
include("EPGdecaycurve.jl")
include("T2map_SEcorr.jl")
include("T2part_SEcorr.jl")

end # module Classic