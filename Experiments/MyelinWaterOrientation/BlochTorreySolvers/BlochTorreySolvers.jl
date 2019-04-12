# ---------------------------------------------------------------------------- #
# BlochTorreySolvers
# ---------------------------------------------------------------------------- #

module BlochTorreySolvers

using BlochTorreyUtils
import ExpmV
import Expokit

using LinearAlgebra, SparseArrays, JuAFEM, Tensors
using Parameters: @with_kw, @pack, @unpack
using DiffEqBase, OrdinaryDiffEq, DiffEqCallbacks, DiffEqOperators#, Sundials
using LinearMaps

export MultiSpinEchoCallback
export ExpokitExpmv, HighamExpmv

include("src/algorithms.jl")
include("src/callbacks.jl")

end # module BlochTorreySolvers