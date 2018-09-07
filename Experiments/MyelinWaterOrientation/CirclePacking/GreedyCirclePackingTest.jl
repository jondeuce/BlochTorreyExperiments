module GreedyCirclePackingTest

using GeometryUtils
using GreedyCirclePacking

using Test
using LinearAlgebra
using Tensors: Vec

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

function runtests()
    @testset "Greedy Circle Packing" begin
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
        for j in 2:3, i in 1:j-1
            @test d(c[i], c[j]) ≈ zero(T) atol = 10*eps(T)
        end

        # Tangent circles where c[1] and c[2] are tangent
        c = pack(rand(T, 2)) # random tangent circles
        c = translate_shape.(c, [rand(Vec{dim,T})]) # random translation
        r = rand()
        ts = GreedyCirclePacking.tangent_circles(c[1], c[2], r)
        for t in ts, i in 1:2
            @test d(c[i], t) ≈ zero(T) atol = 10*eps(T)
        end

        # Tangent circles where c[1] and c[2] are close, but not tangent
        dx = origin(c[2]) - origin(c[1]) # vector from c1 to c2
        dx /= norm(dx) # unit vector
        dx *= 2r * rand() # random length <= 2r
        c[2] = translate_shape(c[2], dx)
        ts = GreedyCirclePacking.tangent_circles(c[1], c[2], r)
        for t in ts, i in 1:2
            @test d(c[i], t) ≈ zero(T) atol = 10*eps(T)
        end

        # Tangent circles where c[1] and c[2] are far, but not tangent
        dx *= 2r/norm(dx) # length == 2r to add to previous displacement
        c[2] = translate_shape(c[2], dx)
        ts = GreedyCirclePacking.tangent_circles(c[1], c[2], r)
        for (i,j) in zip(1:2, 2:-1:1)
            @test  isapprox(d(c[i], ts[i]), zero(T); atol = 10*eps(T))
            @test !isapprox(d(c[i], ts[j]), zero(T); atol = 10*eps(T))
        end
    end
    nothing
end

end # module GreedyCirclePackingTest

nothing
