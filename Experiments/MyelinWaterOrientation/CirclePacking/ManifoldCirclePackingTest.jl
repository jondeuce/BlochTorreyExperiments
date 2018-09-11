module ManifoldCirclePackingTest

using Test
using BenchmarkTools

using GeometryUtils
using CirclePackingUtils
using GreedyCirclePacking
using EnergyCirclePacking
using ManifoldCirclePacking

using Distributions
using LinearAlgebra
using Tensors: Vec
using Optim

using CirclePackingUtils: d, ∇d, ∇²d
const greedypack = GreedyCirclePacking.pack
const manifoldpack = ManifoldCirclePacking.pack
const energypack = EnergyCirclePacking.pack

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

function runtests()
    @testset "Manifold Circle Packing" begin
        dim = 2
        T = Float64
        N = 10 # num circles

        init(N) = (zeros(2N), randn(2N), rand(N) .+ one(T))
        x0, r = randn(2N), rand(N) .+ one(T)

        # Test `retract!`
        x = copy(x0)
        ϵ = T(0.001 * (2*rand() - 1.0)) # should work for negative ϵ as well
        m = ManifoldCirclePacking.MinimallyTangent(ϵ, r)
        x = Optim.retract!(m, x)
        os = reinterpret(Vec{2,T}, x)
        cs = [Circle(os[i],r[i]) for i in 1:length(os)]
        @test minimum_signed_edge_distance(cs) ≈ ϵ

        # Use greedy result as initialization for manifold packing
        N = 20
        r = rand(Distributions.Gamma(5.7, 0.46/5.7), N)
        c0 = greedypack(r; iters = 5)
        @show estimate_density(c0)
        c1 = manifoldpack(c0)
        @show estimate_density(c1)
        c2 = energypack(c1; goaldensity = 0.8)
        @show estimate_density(c2)
    end

    nothing
end

end # module ManifoldCirclePackingTest

nothing
