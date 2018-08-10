module Expmv

  # package code goes here
  include("normAm.jl")

  include("select_taylor_degree.jl")

  include("expmv_fun.jl")

  export expmv, expmv!, normAm

end # module
