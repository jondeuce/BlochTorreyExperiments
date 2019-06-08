using CirclePackingUtils
using CirclePackingUtils: d, ∇d, ∇²d, d², ∇d², ∇²d²
using CirclePackingUtils: d²_overlap, ∇d²_overlap, ∇²d²_overlap
using CirclePackingUtils: expbarrier, ∇expbarrier, ∇²expbarrier
using CirclePackingUtils: softplusbarrier, ∇softplusbarrier
using CirclePackingUtils: genericbarrier, ∇genericbarrier, ∇²genericbarrier

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
        G, H = zeros(2N), zeros(2N, 2N)
        
        # Form gradient triples for testing
        wrap(f, ϵ) = f == nothing ? nothing : (x...) -> f(x..., ϵ)
        test_triples = []
        
        # Handwritten gradients with default ϵ=0 methods
        for triples in [(d, ∇d, ∇²d), (d², ∇d², ∇²d²),
                        (d²_overlap, ∇d²_overlap, ∇²d²_overlap)]
            push!(test_triples, triples)
            push!(test_triples, map(f -> wrap(f, ϵ), triples))
        end

        # Handwritten gradients which must be wrapped
        for triples in [(expbarrier, ∇expbarrier, ∇²expbarrier),
                        (softplusbarrier, ∇softplusbarrier, nothing)]
            push!(test_triples, map(f -> wrap(f, ϵ), triples))
        end

        # Generic gradients
        for triples in [(sin, cos, x -> -sin(x)),
                        (x -> x^3, x -> 3x^2, x -> 6x)]
            b, ∂b, ∂²b = triples
            f = (o1, o2, r1, r2) -> genericbarrier(b, o1, o2, r1, r2, ϵ)
            ∇f = (o1, o2, r1, r2) -> ∇genericbarrier(∂b, o1, o2, r1, r2, ϵ)
            ∇²f = (o1, o2, r1, r2) -> ∇²genericbarrier(∂b, ∂²b, o1, o2, r1, r2, ϵ)
            push!(test_triples, (f, ∇f, ∇²f))
        end
        
        for r in [fill(R1-ϵ, N), fill(R1+ϵ, N)]
            for triples in test_triples
                f, ∇f, ∇²f = triples
                F = x -> pairwise_sum(f, x, r)
                
                if !(∇f == nothing)
                    # Test `pairwise_sum` gradient
                    gfwd = ForwardDiff.gradient(F, x1)
                    gpair = copy(pairwise_grad!(G, ∇f, x1, r))
                    @test gfwd ≈ gpair
                end

                if !(∇²f == nothing)
                    # Test `pairwise_sum` Hessian
                    Hfwd = ForwardDiff.hessian(F, x1)
                    Hpair = copy(pairwise_hess!(H, ∇²f, x1, r))
                    @test Hfwd ≈ Hpair
                end
            end
        end
    end
    nothing
end

function runbenchmarks()
    # Benchmarks
    F = x -> pairwise_sum(d, x, r)
    
    println("\nFunction Call (N = $N circles):\n");
    display(@benchmark $F($x1))
    
    println("\nForwardDiff Gradient (N = $N circles):\n");
    cfg = ForwardDiff.GradientConfig(F, x1, ForwardDiff.Chunk{min(20,N)}())
    display(@benchmark ForwardDiff.gradient!($G, $F, $x1, $cfg))
    
    println("\nManual Gradient (N = $N circles):\n");
    display(@benchmark pairwise_grad!($G, $∇d, $x1, $r))
    
    println("\nForwardDiff Hessian (N = $N circles):\n");
    cfg = ForwardDiff.HessianConfig(F, x1, ForwardDiff.Chunk{min(20,N)}())
    display(@benchmark ForwardDiff.hessian!($H, $F, $x1, $cfg))
    
    println("\nManual Hessian (N = $N circles):\n");
    display(@benchmark pairwise_hess!($H, $∇²d, $x1, $r))
    nothing
end

nothing