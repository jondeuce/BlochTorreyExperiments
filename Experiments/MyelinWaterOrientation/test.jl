include("init.jl")
using Test
using TimerOutputs
using Arpack

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

@testset "Geometry Utils" begin
    dim = 2
    T = Float64

    # ---- Ellipse misc. ---- #
    e = rand(Ellipse{dim,T})
    @test geta(e)^2 ≈ getb(e)^2 + getc(e)^2

    # ---- Ellipse is exactly a circle ---- #
    c = rand(Circle{dim,T})
    e = Ellipse(origin(c), origin(c), radius(c))
    X = rand(Vec{dim,T})

    # Still may only be approximate, as origin(e) = 0.5*(F1+f2)
    @test area(c) ≈ area(e)
    @test signed_edge_distance(X, c) ≈ signed_edge_distance(X, e)

    # ---- Ellipse signed_edge_distance testing ---- #
    e = rand(Ellipse{dim,T})
    O, F1, F2, a, b, c = origin(e), getF1(e), getF2(e), geta(e), getb(e), getc(e)

    Ua = (F2-F1)/norm(F2-F1) # unit vector for major axis
    Ub = Vec{dim,T}((-Ua[2], Ua[1])) # unit vector for minor
    A, B = O + a*Ua, O + b*Ub # point at major/minor axis boundary

    R = getrotmat(e)
    ϕ = 2π*rand()
    P = O + R ⋅ Vec{dim,T}((a*cos(ϕ), b*sin(ϕ))) # rand point on ellipse
    N = R ⋅ Vec{dim,T}((cos(ϕ)/a, sin(ϕ)/b)) # normal vector at P
    N = N / norm(N)
    # dmax = min(b^2/a*abs(cos(ϕ)), a^2/b*abs(sin(ϕ))) # distance to x- or y-intercept
    dmax = b^2/a*abs(cos(ϕ)) # x-axis dist is always less than y-axis dist
    d = rand(T) * dmax # rand distance less distance to x- or y-axis
    Xout = (1+rand(T)) * (P-O) + O # random point outside ellipse
    Xin = rand(T) * (P-O) + O # random point inside ellipse

    @test signed_edge_distance(Xout, e) > zero(T)
    @test signed_edge_distance(Xin, e) < zero(T)
    @test signed_edge_distance(A, e) ≈ zero(T) atol = 10eps(T)
    @test signed_edge_distance(B, e) ≈ zero(T) atol = 10eps(T)
    @test signed_edge_distance(P, e) ≈ zero(T) atol = 10eps(T)
    @test signed_edge_distance(O, e) ≈ -b atol = 10eps(T)
    @test signed_edge_distance(F1, e) ≈ c - a atol = 10eps(T)
    @test signed_edge_distance(P + d*N, e) ≈ d atol = 10eps(T)
    @test signed_edge_distance(P - d*N, e) ≈ -d atol = 10eps(T)

    X0 = O + R ⋅ Vec{dim,T}((c^2/a, zero(T))) # center of curvature at vertice (a,0)
    X1 = O + R ⋅ Vec{dim,T}((zero(T), c^2/b)) # center of curvature at co-vertice (0,b)
    ρ0, ρ1 = b^2/a, a^2/b # ρ0 < b and ρ1 > a
    c0, c1 = Circle(X0, 0.9ρ0), Circle(X1, ρ1)

    # c0 should be inside e
    @test is_inside(Circle(O,rand(T)*b), e) # should be true, as ρ < b
    @test is_inside(c0, e) # should be true, as ρ0 < b
    @test !is_inside(c1, e) # should be false, as ρ1 > a > b
end

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
        addquadweights!(domain)
        u = interpolate(f, domain)
        I = integrate(u, domain)
        return I
    end

    function getL2norm(f; cellshape = Triangle, refshape = RefTetrahedron, qorder = 1, forder = 1, gorder = 1, N = 20)
        grid = generate_grid(cellshape, (N, N))
        domain = ParabolicDomain(grid; refshape = refshape, quadorder = qorder, funcinterporder = forder, geominterporder = gorder)
        doassemble!(domain)
        u = interpolate(f, domain)
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

    t = 1.0
    x0 = randn(n); #x0 = zeros(n); @views x0[2:2:n] .= 1.0;
    Ef = exp(t*Af) # dense exponential matrix

    μ = tr(A)/n
    Ashift = A - μ*I
    M = Expmv.select_taylor_degree(A, x0; opnorm = expmv_norm)[1]
    Mshift = Expmv.select_taylor_degree(Ashift, x0; opnorm = expmv_norm)[1]

    Ys = [zeros(n) for i in 1:5]
    @btime $(Ys[1]) .= $Ef * $x0  evals = 1
    @btime Expokit.expmv!($(Ys[2]), $t, $A, $x0; tol=1e-14, anorm = expmv_norm($A,Inf), m=30)  evals = 1
    @btime Expmv.expmv!($(Ys[3]), $t, $A, $x0; opnorm = expmv_norm)  evals = 1
    @btime Expmv.expmv!($(Ys[4]), $t, $A, $x0; M = $M, opnorm = expmv_norm)  evals = 1
    @btime $(Ys[5]) .= exp($μ*$t) .* Expmv.expmv!($(Ys[5]), $t, $Ashift, $x0; M = $M, shift = false, opnorm = expmv_norm)  evals = 1

    @test norm(Ys[1] - Ys[2], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[2] - Ys[3], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[3] - Ys[4], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[4] - Ys[5], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
end

@testset "Expmv Methods" begin
    expmv_tests()
end

# ---------------------------------------------------------------------------- #
# Single axon geometry testing
# ---------------------------------------------------------------------------- #

function setup(;params = BlochTorreyParameters{Float64}())
    rs = [params.R_mu] # one radius of average size
    os = zeros(Vec{2}, 1) # one origin at the origin
    outer_circles = Circle.(os, rs)
    inner_circles = scale_shape.(outer_circles, params.g_ratio)
    bcircle = scale_shape(outer_circles[1], 1.5)

    h0 = 0.2 * params.R_mu * (1.0 - params.g_ratio) # fraction of size of average torus width
    eta = 5.0 # approx ratio between largest/smallest edges

    mxcall(:figure,0); mxcall(:hold,0,"on")
    @time grid = circle_mesh_with_tori(bcircle, inner_circles, outer_circles, h0, eta)
    @time exteriorgrid, torigrids, interiorgrids = form_tori_subgrids(grid, bcircle, inner_circles, outer_circles)

    all_tori = form_subgrid(grid, getcellset(grid, "tori"), getnodeset(grid, "tori"), getfaceset(grid, "boundary"))
    all_int = form_subgrid(grid, getcellset(grid, "interior"), getnodeset(grid, "interior"), getfaceset(grid, "boundary"))
    mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(exteriorgrid); sleep(0.5)
    mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(all_tori); sleep(0.5)
    mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(all_int)

    prob = MyelinProblem(params)
    domains = MyelinDomain(grid, outer_circles, inner_circles, bcircle,
        exteriorgrid, torigrids, interiorgrids;
        quadorder = 3, funcinterporder = 1)

    doassemble!(prob, domains)
    factorize!(domains)

    return prob, domains
end

function testcomparemethods(prob::MyelinProblem,
                            domains::MyelinDomain;
                            tspan = (0.0, 1e-3),
                            rtol = 1e-6,
                            atol = 1e-6)
    doassemble!(prob, domains)
    factorize!(domains)

    u0 = Vec{2}((0.0, 1.0)) # Initial π/2-pulse
    U0 = interpolate(x->u0, domains) # vector of vectors of degrees of freedom with `u0` at each node
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

params = BlochTorreyParameters{Float64}()
prob, domains = setup(;params = params)
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
    U0 = interpolate(x->u0, domains) # vector of vectors of degrees of freedom with `u0` at each node
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

params = BlochTorreyParameters{Float64}()
prob, domains = setup(;params = params)
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
    @btime $reassemble!($tmp, $mfact\$ux, $mfact\$uy)
    @btime $reassemble!($tmp, $mfact\$uxview, $mfact\$uyview)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Solving using DifferentialEquations.jl
# ---------------------------------------------------------------------------- #

function DiffEqBase.solve(prob, domains, tspan = (0.0,1e-3);
                          abstol = 1e-8,
                          reltol = 1e-8,
                          linear_solver = :GMRES)
    to = TimerOutput()
    u0 = Vec{2}((0.0, 1.0))
    sols = ODESolution[]
    signals = SignalIntegrator[]

    @timeit to "Assembly" doassemble!(prob, domains)
    @timeit to "Factorization" factorize!(domains)
    @timeit to "Interpolation" U0 = interpolate(u0, domains) # π/2-pulse at each node

    @timeit to "Solving on subdomains" for (i, subdomain) in enumerate(getsubdomains(domains))

        A = ParabolicLinearMap(subdomain)
        signal, callbackfun = IntegrationCallback(U0[i], tspan[1], subdomain)
        prob = ODEProblem((du,u,p,t)->mul!(du,p[1],u), U0[i], tspan, (A,));

        print("Subdomain $i/$(numsubdomains(domains)): ")
        @time begin # time twice so that can see each iteration
            @timeit to "Subdomain $i/$(numsubdomains(domains))" begin
                sol = solve(prob, CVODE_BDF(linear_solver = linear_solver);
                            abstol = abstol,
                            reltol = reltol,
                            saveat = tspan,
                            alg_hints = :stiff,
                            callback = callbackfun)
            end
        end
        push!(sols, sol)
        push!(signals, signal)

        flush(stdout)
    end

    print_timer(to)
    return sols, signals
end

params = BlochTorreyParameters{Float64}()
prob, domains = setup(;params = params)
doassemble!(prob, domains)
factorize!(domains)
sols, signals = solve(prob, domains, (0.0,40e-3))

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
