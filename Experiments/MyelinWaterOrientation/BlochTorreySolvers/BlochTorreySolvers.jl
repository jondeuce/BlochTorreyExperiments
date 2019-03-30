# ---------------------------------------------------------------------------- #
# BlochTorreySolvers
# ---------------------------------------------------------------------------- #

module BlochTorreySolvers

using BlochTorreyUtils, GeometryUtils # MeshUtils
import ExpmV
import Expokit

using LinearAlgebra, SparseArrays, JuAFEM, Tensors
using Parameters: @with_kw, @pack, @unpack
using DiffEqBase, OrdinaryDiffEq, DiffEqCallbacks # DiffEqOperators, Sundials
# using MATLAB
using TimerOutputs
# using LinearMaps

export MultiSpinEchoCallback
export ExpokitExpmv, HighamExpmv

include("src/diffeq.jl")

end # module BlochTorreySolvers

nothing
