using Normest1
using LinearMaps
using Test
using Profile
using BenchmarkTools

function profile_normest1(;Asize::Int = 5000, Nloops::Int = 100, prnt::Bool = false)
    A = randn(Asize,Asize)
    normest1(A) # dry run to precompile

    Profile.clear()
    Profile.init(;n=10_000_000)
    @profile for i = 1:Nloops
        normest1(A)[1]
    end
    prnt && Profile.print()

    return nothing
end
 
function benchmark_normest1(;Asize::Int = 5000, Nloops::Int = 1, prnt::Bool = false)
    A = randn(Asize,Asize)
    normest1(A) # dry run to precompile
    @benchmark (for i = 1:$Nloops; normest1($A)[1]; end)
end