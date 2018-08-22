module Normest1Test

using Normest1
using LinearMaps
using Test
using Profile
using BenchmarkTools

# test `normest1` using a matrix wrapped in a `LinearMap`
get_wrapped_map(A) = LinearMaps.WrappedMap(A)

function test()
   RealAndCplx(n) = (randn(n,n), Complex.(randn(n,n), randn(n,n)))

   #TODO: scalar case for completeness (necessary?)
   # A, Ac = RealAndCplx(1)
   # A, Ac = A[1], Ac[1]
   # @test normest1(A)[1] ≈ norm(A,1)
   # @test normest1(Ac)[1] ≈ norm(Ac,1)

   # Small matrices: should be exact
   for n = 1:4
      A, Ac = RealAndCplx(n)
      @test normest1(A)[1] ≈ norm(A,1)
      @test normest1(Ac)[1] ≈ norm(Ac,1)
      @test normest1(get_wrapped_map(A))[1] ≈ norm(A,1)
      @test normest1(get_wrapped_map(Ac))[1] ≈ norm(Ac,1)
   end

   # Larger matrices: should be exact when t = size(A,1)
   for n = [100]
      A, Ac = RealAndCplx(n)
      @test normest1(A, n)[1] ≈ norm(A,1)
      @test normest1(Ac, n)[1] ≈ norm(Ac,1)
      @test normest1(get_wrapped_map(A), n)[1] ≈ norm(A,1)
      @test normest1(get_wrapped_map(Ac), n)[1] ≈ norm(Ac,1)
   end

   # Larger matrices: should almost always be exact when t = size(A,1)/2
   for n = [100]
      A, Ac = RealAndCplx(n)
      @test normest1(A, div(n,2))[1] ≈ norm(A,1)
      @test normest1(Ac, div(n,2))[1] ≈ norm(Ac,1)
      @test normest1(get_wrapped_map(A), div(n,2))[1] ≈ norm(A,1)
      @test normest1(get_wrapped_map(Ac), div(n,2))[1] ≈ norm(Ac,1)
   end

end

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

end # module Normest1Test
