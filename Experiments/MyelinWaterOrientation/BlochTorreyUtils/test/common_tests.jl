include("init.jl")
using Test
using TimerOutputs

# ---------------------------------------------------------------------------- #
# ParabolicDomain integration testing
# ---------------------------------------------------------------------------- #

@testset "ParabolicDomain Integration" begin
    dim, T = 2, Float64
    Zero, One = zeros(Vec{dim,T}), ones(Vec{dim,T})
    e1, e2 = basevec(Vec{dim,T}, 1), basevec(Vec{dim,T}, 2)

    function getintegral(f; cellshape = Triangle, refshape = RefTetrahedron, qorder = 1, forder = 1, gorder = 1, N = 20)
        grid = generate_grid(cellshape, (N, N))
        domain = ParabolicDomain(grid; refshape = refshape, quadorder = qorder, funcinterporder = forder, geominterporder = gorder)
        # addquadweights!(domain)
        u = BlochTorreyUtils.interpolate(f, domain)
        I = BlochTorreyUtils.integrate(u, domain)
        return I
    end

    function getL2norm(f; cellshape = Triangle, refshape = RefTetrahedron, qorder = 1, forder = 1, gorder = 1, N = 20)
        grid = generate_grid(cellshape, (N, N))
        domain = ParabolicDomain(grid; refshape = refshape, quadorder = qorder, funcinterporder = forder, geominterporder = gorder)
        doassemble!(domain)
        u = BlochTorreyUtils.interpolate(f, domain)
        L² = norm(u, domain)
        return L²
    end

    # ---- All integration methods should return exact area ---- #
    Agrid = 4.0 # [-1,1]² grid
    for (q,f) in Iterators.product(1:2, 1:2)
        @test getintegral(x->One; qorder = q, forder = f, gorder = 1) ≈ Agrid*One
        @test_broken getintegral(x->One; qorder = q, forder = f, gorder = 2) ≈ Agrid*One
    end

    # ---- All integration methods should integrate linear functions exactly ---- #
    func = x -> Vec{dim,T}((2x[1]+x[2]+1, 3x[2]-x[1]+1))
    Iexact = Agrid * One # linear terms cancel over [-1,1]^2
    for (q,f) in Iterators.product(1:2, 1:2)
        @test getintegral(func; qorder = q, forder = f, gorder = 1) ≈ Iexact
        @test_broken getintegral(func; qorder = q, forder = f, gorder = 2) ≈ Iexact
    end

    # ---- Quadratic integration of quadratic interpolated function integrates quadratics exactly ---- #
    func = x -> Vec{dim,T}((x[1]^2-x[1]*x[2]+1, 2x[1]*x[2] + x[2]^2))
    Iexact = Vec{dim,T}((16/3, 4/3))
    for (q,f) in Iterators.product(2:2, 2:2)
        @test getintegral(func; qorder = q, forder = f, gorder = 1) ≈ Iexact
        @test_broken getintegral(func; qorder = q, forder = f, gorder = 2) ≈ Iexact
    end

    # ---- L²-norm of linear function should be exact for quadratic quad rules ---- #
    func = x -> Vec{dim,T}((2x[1]+x[2]+1, 3x[2]-x[1]+1))
    L²exact = √28.0
    for (q,f) in Iterators.product(2:2, 1:2)
        @test getL2norm(func; qorder = q, forder = f, gorder = 1) ≈ L²exact
        @test_broken getL2norm(func; qorder = q, forder = f, gorder = 2) ≈ L²exact
    end

    # ---- Integrate more complicated smooth functions, expecting errors ---- #
    func = x -> Vec{dim,T}((cos(2x[1]+x[2]), cos(x[1]+x[2])*exp(x[1]^2)))
    Iexact = Vec{dim,T}((2*sin(1)*sin(2), 3.9236263410199394))
    for f in 1:2, q in 1:f, N in (25,50)
        h, p, C = √2*(2/N), min(q,f)^2+2, 5.0 # heuristic error ~ C*h^p
        @test getintegral(func; N = N, qorder = q, forder = f, gorder = 1) ≈ Iexact rtol = C*h^p
        @test_broken getintegral(func; N = N, qorder = q, forder = f, gorder = 2) ≈ Iexact rtol = C*h^p
    end
end

# ---------------------------------------------------------------------------- #
# Expmv methods testing for consistency
# ---------------------------------------------------------------------------- #

function expmv_tests(;N = 10, h = √2*(2/N), grid = generate_grid(Triangle, (N, N)))
    # Assemble mass and (Neumann BC) stiffness matrix. `quadorder` must be at
    # least one larger than `funcinterporder` to create exact (and
    # non-singular!) mass matrix
    domain = ParabolicDomain(grid; quadorder = 2, funcinterporder = 1)
    doassemble!(domain)
    factorize!(domain)
    M, Mfact, K = getmass(domain), getmassfact(domain), getstiffness(domain)

    # Smallest 2 eigenvalues should be zero (one for each dimension of u),
    # and the rest negative
    d, v = eigs(-K; nev=3, which=:SR) # smallest real
    dmin = max(abs(d[1]), abs(d[2])) # largest null eigenvalue
    ispossemidef_K = isapprox(dmin, 0.0; atol=1e-12) && (d[3] > 1e-2*h^2) # minimum non-zero eigenvalue should be positive and least O(h^2)

    @test isposdef(M) && ispossemidef_K

    A = ParabolicLinearMap(M, Mfact, K)
    Af = Matrix(A)
    n = size(A,2)

    t = 1e-3
    x0 = randn(n); #x0 = zeros(n); @views x0[2:2:n] .= 1.0;
    Ef = exp(t*Af) # dense exponential matrix

    μ = tr(A)/n
    Ashift = A - μ*I
    M = ExpmvHigham.select_taylor_degree(A, x0; opnorm = normest1_norm)[1]
    Mshift = ExpmvHigham.select_taylor_degree(Ashift, x0; opnorm = normest1_norm)[1]

    Ys = [zeros(n) for i in 1:5]
    @btime $(Ys[1]) .= $Ef * $x0  evals = 1
    @btime Expokit.expmv!($(Ys[2]), $t, $A, $x0; tol=1e-14, anorm = normest1_norm($A,Inf), m=30)  evals = 1
    @btime ExpmvHigham.expmv!($(Ys[3]), $t, $A, $x0; opnorm = normest1_norm)  evals = 1
    @btime ExpmvHigham.expmv!($(Ys[4]), $t, $A, $x0; M = $M, opnorm = normest1_norm)  evals = 1
    @btime $(Ys[5]) .= exp($μ*$t) .* ExpmvHigham.expmv!($(Ys[5]), $t, $Ashift, $x0; M = $M, shift = false, opnorm = normest1_norm)  evals = 1

    @test norm(Ys[1] - Ys[2], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[2] - Ys[3], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[3] - Ys[4], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[4] - Ys[5], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
end

@testset "Expmv Methods" begin
    expmv_tests()
end

function testcomparemethods(prob::MyelinProblem,
                            domains::MyelinDomain;
                            tspan = (0.0, 1e-3),
                            rtol = 1e-6,
                            atol = 1e-6)
    doassemble!(prob, domains)
    factorize!(domains)

    u0 = Vec{2}((0.0, 1.0)) # Initial π/2-pulse
    U0 = BlochTorreyUtils.interpolate(x->u0, domains) # vector of vectors of degrees of freedom with `u0` at each node
    U, Uexpmv, Uexpokit, Udiffeq = deepcopy(U0), deepcopy(U0), deepcopy(U0), deepcopy(U0)

    for (i, subdomain) in enumerate(getsubdomains(domains))
        println("i = $i/$(numsubdomains(domains)):")

        A = ParabolicLinearMap(subdomain)
        copyto!(U[i], U0[i])

        # Method 1: expmv! from Higham's `Expmv`
        solver! = expmv_solver(subdomain)
        solver!(U[i], A, tspan, U0[i])
        copyto!(Uexpmv[i], U[i])

        # Method 2: expmv! from Expokit.jl
        solver! = expokit_solver(subdomain)
        solver!(U[i], A, tspan, U0[i])
        copyto!(Uexpokit[i], U[i])

        # Method 3: direct ODE solution using DifferentialEquations.jl
        solver! = diffeq_solver(subdomain)
        solver!(U[i], A, tspan, U0[i])
        copyto!(Udiffeq[i], U[i])

        # Compare simulation results
        @test norm(Uexpmv - Uexpokit, Inf) ≈ 0.0 rtol = rtol atol = atol
        @test norm(Udiffeq - Uexpokit, Inf) ≈ 0.0 rtol = rtol atol = atol

        flush(stdout)
    end

    return Uexpmv, Uexpokit, Udiffeq
end

btparams = BlochTorreyParameters{Float64}()
prob, domains = testproblem(SingleAxonSetup(), btparams)
@testset "Myelin Problem Solutions" begin
    testcomparemethods(prob, domains)
end

# ---------------------------------------------------------------------------- #
# Benchmark different expmv methods on a single axon geometry
# ---------------------------------------------------------------------------- #

function benchmark_method(prob, domains;
                          solvertype = :diffeq,
                          tspan = (0.0,1e-3))
    doassemble!(prob, domains)
    factorize!(domains)

    u0 = Vec{2}((0.0, 1.0)) # Initial π/2-pulse
    U0 = BlochTorreyUtils.interpolate(x->u0, domains) # vector of vectors of degrees of freedom with `u0` at each node
    U = deepcopy(U0)

    for (i, subdomain) in enumerate(getsubdomains(domains))
        solver! = if solvertype == :diffeq
            diffeq_solver(subdomain)
        elseif solvertype == :expokit
            expokit_solver(subdomain)
        elseif solvertype == :expmv
            expmv_solver(subdomain)
        end

        print("subdomain $i/$(numsubdomains(domains)) ($solvertype): ")

        A = ParabolicLinearMap(subdomain)
        @btime $solver!($(U[i]), $A, $tspan, $(U0[i])) evals = 2

        flush(stdout)
    end
end

function run_benchmarks(prob, domains; tspan = (0.0, 1e-4))
    benchmark_method(prob, domains; tspan = tspan, solvertype = :diffeq)
    benchmark_method(prob, domains; tspan = tspan, solvertype = :expokit)
    benchmark_method(prob, domains; tspan = tspan, solvertype = :expmv)
end

btparams = BlochTorreyParameters{Float64}()
prob, domains = testproblem(SingleAxonSetup(), btparams)
doassemble!(prob, domains)
factorize!(domains)
run_benchmarks(prob, domains; tspan = (0.0, 1e-6))

# ---------------------------------------------------------------------------- #
# Benchmark different expmv methods on a single axon geometry
# ---------------------------------------------------------------------------- #

function benchmark_views(;n = 1000)
    m = sprandn(n, n, 3/n) # approx 3 elements per row
    m = m + m' # approx 6 elements per row
    m = m + 2*opnorm(m,1)*I # approx 7 elements per row

    M = spzeros(2n, 2n)
    M[1:2:end, 1:2:end] .= m
    M[2:2:end, 2:2:end] .= m

    mfact = cholesky(m)
    Mfact = cholesky(M)

    benchmark_views(Mfact, mfact)

    return nothing
end

function benchmark_views(domain::ParabolicDomain)
    m = Symmetric(copy(getmass(domain)[1:2:end,1:2:end]))
    mfact = cholesky(m)
    Mfact = cholesky(getmass(domain))
    benchmark_views(Mfact, mfact)
end

function benchmark_views(Mfact::Factorization, mfact::Factorization)
    U = randn(size(Mfact,2))
    ux, uy = copy(U[1:2:end]), copy(U[2:2:end])
    uxview, uyview = view(U, 1:2:lastindex(U)), view(U, 2:2:lastindex(U))

    function reassemble!(U,ux,uy)
        @assert length(ux) == length(uy)
        @assert length(U) == 2*length(ux)
        @inbounds for (i,iU) in enumerate(1:2:2*length(ux))
            U[iU  ] = ux[i]
            U[iU+1] = uy[i]
        end
        return U
    end

    tmp = similar(U)
    @assert Mfact\U ≈ reassemble!(tmp, mfact\ux, mfact\uy)

    @btime $Mfact \ $U
    @btime reassemble!($tmp, $mfact\$ux, $mfact\$uy)
    @btime reassemble!($tmp, $mfact\$uxview, $mfact\$uyview)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Solving using DifferentialEquations.jl
# ---------------------------------------------------------------------------- #

btparams = BlochTorreyParameters{Float64}()
prob, domains = testproblem(SingleAxonSetup(), btparams)
doassemble!(prob, domains)
factorize!(domains)
sols, signals = solve(prob, domains, (0.0,40e-3); abstol = 1e-6; reltol = 1e-4)

function plotsignal(signals::Vector{SignalIntegrator};
                    softmaxpts = 200)
    N = length(signals)
    if N > 2*softmaxpts
        idx = 1:div(N,softmaxpts):N
        (idx[end] < N) && (idx = vcat(idx, N)) # ensure endpoint is included
    else
        idx = 1:N
    end

    Plots.plot()
    for (i,signal) in enumerate(signals)
        t, S = gettime(signal)[idx], relativesignalnorm(signal)[idx]
        Plots.plot!(t, S, title = "Signal vs. Time", label = "S vs. t: subdomain $i")
    end
end

using Plots
gr()
plotsignal(signals)
