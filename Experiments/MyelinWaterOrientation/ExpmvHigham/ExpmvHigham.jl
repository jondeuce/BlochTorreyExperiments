module ExpmvHigham

using LinearAlgebra

include("src/normAm.jl")
include("src/select_taylor_degree.jl")
include("src/expmv_fun.jl")

export expmv, expmv!, select_taylor_degree

end # module

nothing
