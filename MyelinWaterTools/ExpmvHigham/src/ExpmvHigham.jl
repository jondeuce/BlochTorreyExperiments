module ExpmvHigham

using LinearAlgebra

include("normAm.jl")
include("select_taylor_degree.jl")
include("expmv_fun.jl")

export expmv, expmv!, select_taylor_degree

end # module