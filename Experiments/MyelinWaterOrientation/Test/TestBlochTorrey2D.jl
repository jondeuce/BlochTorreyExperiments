module TestBlochTorrey2D

using LinearAlgebra
using LinearMaps: FunctionMap
using Expokit

using BlochTorreyUtils

export lap, testblochtorrey2D

include(joinpath(@__DIR__, "testneumannfindiff.jl"))
include(joinpath(@__DIR__, "testblochtorrey2D.jl"))

end # TestBlochTorrey2D
