module CirclePackingUtilsTest

using GeometryUtils
using CirclePackingUtils

using Test
using Tensors: Vec

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

function runtests()
    @testset "Greedy Circle Packing" begin
        for dim = 1:3
            T = Float64
            N = 5

            c0 = rand(Circle{dim,T}, N)
            x0, r0 = zeros(T, dim*N), zeros(T, N)

            x, r = tovectors(c0)
            c = tocircles(x, r, Val(dim))
            @test c0 == c
            @test (x, r) == tovectors!(x0, r0, c0)
            @test c0 == tocircles!(c, x, r)
        end
    end
    nothing
end

end # module CirclePackingUtilsTest

nothing
