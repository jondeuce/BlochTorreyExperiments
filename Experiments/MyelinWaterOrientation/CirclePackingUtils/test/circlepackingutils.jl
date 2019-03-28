module CirclePackingUtilsTest

using GeometryUtils
using CirclePackingUtils
using CirclePackingUtils: d, ∇d, ∇²d, d², ∇d², ∇²d², d²_overlap, ∇d²_overlap, ∇²d²_overlap

using Test
using BenchmarkTools

import Tensors
import ForwardDiff
using ForwardDiff: GradientConfig, HessianConfig, Chunk

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

function runtests()
    @testset "Circle Packing Utils" begin
        # -------------------------------------------------------------------- #
        # tocircles! and tovectors! testing
        # -------------------------------------------------------------------- #
        for dim = 1:3, T in [Float64], N in [5]
            c1 = rand(Circle{dim,T}, N)
            x1, r1 = zeros(T, dim*N), zeros(T, N)

            x, r = tovectors(c1)
            c = tocircles(x, r, Val(dim))
            @test c1 == c
            @test (x, r) == tovectors!(x1, r1, c1)
            @test c1 == tocircles!(c, x, r)
        end

        # -------------------------------------------------------------------- #
        # Test gradient and Hessian of distance function `d`
        # -------------------------------------------------------------------- #
        T = Float64
        V = Tensors.Vec{2,T}
        x1, x2, r1, r2 = -rand(V), rand(V), rand() + one(T), rand() + one(T)

        @test ∇d(x1, x2, r[1], r[2]) ≈ Tensors.gradient(x -> d(x, x2, r[1], r[2]), x1)
        @test ∇²d(x1, x2, r[1], r[2]) ≈ Tensors.hessian(x -> d(x, x2, r[1], r[2]), x1)

        # -------------------------------------------------------------------- #
        # Test gradient and Hessian of squared distance function `d²`
        # -------------------------------------------------------------------- #
        T = Float64
        V = Tensors.Vec{2,T}
        x1, x2, r1, r2 = -rand(V), rand(V), rand() + one(T), rand() + one(T)

        @test ∇d²(x1, x2, r[1], r[2]) ≈ Tensors.gradient(x -> d²(x, x2, r[1], r[2]), x1)
        @test ∇²d²(x1, x2, r[1], r[2]) ≈ Tensors.hessian(x -> d²(x, x2, r[1], r[2]), x1)

        # -------------------------------------------------------------------- #
        # `pairwise_sum` Gradient and Hessian testing
        # -------------------------------------------------------------------- #
        dim = 2
        T = Float64
        N = 5 # num circles

        x1, x2, r = randn(2N), randn(2N), rand(N) .+ one(T)
        g, H = zeros(2N), zeros(2N, 2N)

        for (f,∇f,∇²f) in [(d,∇d,∇²d), (d²,∇d²,∇²d²), (d²_overlap,∇d²_overlap,∇²d²_overlap)]
            F = x -> pairwise_sum(f, x, r)

            # Test `pairwise_sum` gradient
            gfwd = ForwardDiff.gradient(F, x1)
            gpair = copy(pairwise_grad!(g, ∇f, x1, r))
            @test gfwd ≈ gpair

            # Test `pairwise_sum` Hessian
            Hfwd = ForwardDiff.hessian(F, x1)
            Hpair = copy(pairwise_hess!(H, ∇²f, x1, r))
            @test Hfwd ≈ Hpair
        end

        # # Benchmarks
        # F = x -> pairwise_sum(d, x, r)
        #
        # println("\nFunction Call (N = $N circles):\n");
        # display(@benchmark $F($x1))
        #
        # println("\nForwardDiff Gradient (N = $N circles):\n");
        # cfg = GradientConfig(F, x1, Chunk{min(20,N)}())
        # display(@benchmark ForwardDiff.gradient!($g, $F, $x1, $cfg))
        #
        # println("\nManual Gradient (N = $N circles):\n");
        # display(@benchmark pairwise_grad!($g, $∇d, $x1, $r))
        #
        # println("\nForwardDiff Hessian (N = $N circles):\n");
        # cfg = HessianConfig(F, x1, Chunk{min(20,N)}())
        # display(@benchmark ForwardDiff.hessian!($H, $F, $x1, $cfg))
        #
        # println("\nManual Hessian (N = $N circles):\n");
        # display(@benchmark pairwise_hess!($H, $∇²d, $x1, $r))
    end
    nothing
end

end # module CirclePackingUtilsTest

nothing
