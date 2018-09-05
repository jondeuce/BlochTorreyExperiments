module ManifoldCirclePackingTest

using GeometryUtils
using ManifoldCirclePacking

using Test
using LinearAlgebra
using Tensors: Vec

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

function runtests()
    @testset "Manifold Circle Packing" begin
        dim = 2
        T = Float64
        d = signed_edge_distance

        # Degenerate case and simple cases
        r = T[]
        @test pack(r) == Vector{Circle{dim,T}}[]
        r = rand(T, 1)
        @test pack(r) == [Circle{dim,T}(Vec{dim,T}((zero(T), zero(T))), r[1])]
        r = rand(T, 2)
        @test pack(r) == [Circle{dim,T}(Vec{dim,T}((zero(T), zero(T))), r[1]),
                          Circle{dim,T}(Vec{dim,T}((r[1]+r[2], zero(T))), r[2])]

        # 3 circles
        r = rand(T, 3)
        c = pack(r)
        for i in 1:2, j in i+1:3
            @test d(c[i], c[j]) ≈ zero(T) atol = 10*eps(T)
        end

        # Tangent circles testing
        c = pack(rand(T, 2)) # random tangent circles
        c = translate_shape.(c, [rand(Vec{dim,T})]) # random translation
        ts = ManifoldCirclePacking.tangent_circles(c[1], c[2], rand())
        for t in ts, i in 1:2
            @test d(c[i], t) ≈ zero(T) atol = 10*eps(T)
        end
    end
    nothing
end

end # module ManifoldCirclePackingTest

nothing
