using CirclePackingUtils
using CirclePackingUtils: d, ∇d, ∇²d, d², ∇d², ∇²d², d²_overlap, ∇d²_overlap, ∇²d²_overlap
using CirclePackingUtils: barrier, ∇barrier, ∇²barrier, softplusbarrier, ∇softplusbarrier

import ForwardDiff
using Test
using BenchmarkTools

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

        @test ∇d(x1, x2, r1, r2) ≈ Tensors.gradient(x -> d(x, x2, r1, r2), x1)
        @test ∇²d(x1, x2, r1, r2) ≈ Tensors.hessian(x -> d(x, x2, r1, r2), x1)

        # -------------------------------------------------------------------- #
        # Test gradient and Hessian of squared distance function `d²`
        # -------------------------------------------------------------------- #
        T = Float64
        V = Tensors.Vec{2,T}
        x1, x2, r1, r2 = -rand(V), rand(V), rand() + one(T), rand() + one(T)

        @test ∇d²(x1, x2, r1, r2) ≈ Tensors.gradient(x -> d²(x, x2, r1, r2), x1)
        @test ∇²d²(x1, x2, r1, r2) ≈ Tensors.hessian(x -> d²(x, x2, r1, r2), x1)

        # -------------------------------------------------------------------- #
        # `pairwise_sum` Gradient and Hessian testing
        # -------------------------------------------------------------------- #
        dim = 2
        T = Float64
        
        N = 3 # num circles
        z = cis.(range(0,2π,length=N+1))[1:end-1] # evenly spaced points around the unit circle
        R1 = abs(z[2]-z[1])/2 # half-distance between each pair of closest points
        x1 = reinterpret(Float64, z) |> copy
        ϵ = 0.1 * R1 # overlap thresh
        g, H = zeros(2N), zeros(2N, 2N)
        
        wrap(f, ϵ) = (x...) -> f(x..., ϵ)
        D, ∇D, ∇²D = wrap(d, ϵ), wrap(∇d, ϵ), wrap(∇²d, ϵ)
        b, ∇b, ∇²b = wrap(barrier, ϵ), wrap(∇barrier, ϵ), wrap(∇²barrier, ϵ)
        s, ∇s = wrap(softplusbarrier, ϵ), wrap(∇softplusbarrier, ϵ)
        
        # Gradient tests
        for (f,∇f) in [(b,∇b), (s,∇s), (d,∇d), (D,∇D), (d²,∇d²), (d²_overlap,∇d²_overlap)]
            for r in [fill(R1-ϵ, N), fill(R1+ϵ, N)]
                # Test `pairwise_sum` gradient
                F = x -> pairwise_sum(f, x, r)
                gfwd = ForwardDiff.gradient(F, x1)
                gpair = copy(pairwise_grad!(g, ∇f, x1, r))
                @test gfwd ≈ gpair
            end
        end

        # Hessian tests
        for (f,∇²f) in [(b,∇²b), (d,∇²d), (D,∇²D), (d²,∇²d²), (d²_overlap,∇²d²_overlap)]
            for r in [fill(R1-ϵ, N), fill(R1+ϵ, N)]
                # Test `pairwise_sum` Hessian
                F = x -> pairwise_sum(f, x, r)
                Hfwd = ForwardDiff.hessian(F, x1)
                Hpair = copy(pairwise_hess!(H, ∇²f, x1, r))
                @test Hfwd ≈ Hpair
            end
        end

        # # Benchmarks
        # F = x -> pairwise_sum(d, x, r)
        #
        # println("\nFunction Call (N = $N circles):\n");
        # display(@benchmark $F($x1))
        #
        # println("\nForwardDiff Gradient (N = $N circles):\n");
        # cfg = ForwardDiff.GradientConfig(F, x1, ForwardDiff.Chunk{min(20,N)}())
        # display(@benchmark ForwardDiff.gradient!($g, $F, $x1, $cfg))
        #
        # println("\nManual Gradient (N = $N circles):\n");
        # display(@benchmark pairwise_grad!($g, $∇d, $x1, $r))
        #
        # println("\nForwardDiff Hessian (N = $N circles):\n");
        # cfg = ForwardDiff.HessianConfig(F, x1, ForwardDiff.Chunk{min(20,N)}())
        # display(@benchmark ForwardDiff.hessian!($H, $F, $x1, $cfg))
        #
        # println("\nManual Hessian (N = $N circles):\n");
        # display(@benchmark pairwise_hess!($H, $∇²d, $x1, $r))
    end
    nothing
end

nothing