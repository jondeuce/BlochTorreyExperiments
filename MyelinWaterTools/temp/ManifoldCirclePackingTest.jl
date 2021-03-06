module ManifoldCirclePackingTest

using Test
using BenchmarkTools

using GeometryUtils
using CirclePackingUtils

using Distributions
using LinearAlgebra
using Tensors: Vec
using Optim

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
        N = 5 # num circles

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
    end

    nothing
end

function testpacking(;N = 5, eta = 0.6, it = 5)
    # Use greedy result as initialization for manifold/energy/etc. packing
    r = rand(Distributions.Gamma(5.7, 0.46/5.7), N)

    println("\nGreedy circle packing")
    @time c0 = greedypack(r; iters = it)
    @show estimate_density(c0)

    println("\nManifold circle packing")
    @time c1 = manifoldpack(c0)
    @show estimate_density(c1)

    println("\nFirst order (greedy init)")
    @time c2 = energypack(c0; goaldensity = eta,
        autodiff = false, setcallback = true, secondorder = false)
    @show estimate_density(c2)

    println("\nFirst order (radii init)")
    @time c3 = energypack(r; goaldensity = eta,
        autodiff = false, setcallback = true, secondorder = false)
    @show estimate_density(c3)

    println("\nSecond order (greedy init)")
    @time c4 = energypack(c0; goaldensity = eta,
        autodiff = false, setcallback = true, secondorder = true)
    @show estimate_density(c4)

    println("\nSecond order (radii init)")
    @time c5 = energypack(r; goaldensity = eta,
        autodiff = false, setcallback = true, secondorder = true)
    @show estimate_density(c5)

    return c3
end

end # module ManifoldCirclePackingTest

nothing
