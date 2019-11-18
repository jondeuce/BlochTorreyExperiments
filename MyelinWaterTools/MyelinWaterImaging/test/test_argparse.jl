import Pkg
Pkg.activate(joinpath(@__DIR__, "../../."))
include(joinpath(@__DIR__, "../../initpaths.jl"))
using T2Dist
T2Dist.Rewrite.main()