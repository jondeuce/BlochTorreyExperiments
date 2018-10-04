module Expmv

using LinearAlgebra

# package code goes here
include("normAm.jl")

include("select_taylor_degree.jl")

include("expmv_fun.jl")

export expmv, expmv!

end # module

nothing
