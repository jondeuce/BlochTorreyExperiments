include("init.jl")
using Test
using Arpack

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

@testset "Parabolic Integration" begin
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
    for (q,f) in IterTools.product(1:2, 1:2)
        @test getintegral(x->One; qorder = q, forder = f, gorder = 1) ≈ Agrid*One
        @test_broken getintegral(x->One; qorder = q, forder = f, gorder = 2) ≈ Agrid*One
    end

    # ---- All integration methods should integrate linear functions exactly ---- #
    func = x -> Vec{dim,T}((2x[1]+x[2]+1, 3x[2]-x[1]+1))
    Iexact = Agrid * One # linear terms cancel over [-1,1]^2
    for (q,f) in IterTools.product(1:2, 1:2)
        @test getintegral(func; qorder = q, forder = f, gorder = 1) ≈ Iexact
        @test_broken getintegral(func; qorder = q, forder = f, gorder = 2) ≈ Iexact
    end

    # ---- Quadratic integration of quadratic interpolated function integrates quadratics exactly ---- #
    func = x -> Vec{dim,T}((x[1]^2-x[1]*x[2]+1, 2x[1]*x[2] + x[2]^2))
    Iexact = Vec{dim,T}((16/3, 4/3))
    for (q,f) in IterTools.product(2:2, 2:2)
        @test getintegral(func; qorder = q, forder = f, gorder = 1) ≈ Iexact
        @test_broken getintegral(func; qorder = q, forder = f, gorder = 2) ≈ Iexact
    end

    # ---- L²-norm of linear function should be exact for quadratic quad rules ---- #
    func = x -> Vec{dim,T}((2x[1]+x[2]+1, 3x[2]-x[1]+1))
    L²exact = √28.0
    for (q,f) in IterTools.product(2:2, 1:2)
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
    M = Expmv.select_taylor_degree(A, x0; norm = expmv_norm)[1]
    Mshift = Expmv.select_taylor_degree(Ashift, x0; norm = expmv_norm)[1]

    Ys = [zeros(n) for i in 1:5]
    @btime $(Ys[1]) .= $Ef * $x0 evals = 1
    @btime Expokit.expmv!($(Ys[2]), $t, $A, $x0; tol=1e-14, norm = expmv_norm, m=30) evals = 1
    @btime Expmv.expmv!($(Ys[3]), $t, $A, $x0; norm = expmv_norm) evals = 1
    @btime Expmv.expmv!($(Ys[4]), $t, $A, $x0; M = $M, norm = expmv_norm) evals = 1
    @btime $(Ys[5]) .= exp($μ*$t) .* Expmv.expmv!($(Ys[5]), $t, $Ashift, $x0; M = $M, shift = false, norm = expmv_norm) evals = 1

    @test norm(Ys[1] - Ys[2], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[2] - Ys[3], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[3] - Ys[4], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
    @test norm(Ys[4] - Ys[5], Inf) ≈ 0.0 rtol = 1e-12 atol = 1e-12
end

@testset "Expmv Methods" begin
    expmv_tests()
end

# ---- Single axon geometry testing ---- #
function setup()
    params = BlochTorreyParameters{Float64}()
    rs = [params.R_mu] # one radius of average size
    os = zeros(Vec{2}, 1) # one origin at the origin
    outer_circles = Circle.(os, rs)
    inner_circles = scale_shape.(outer_circles, params.g_ratio)
    bcircle = scale_shape(outer_circles[1], 1.5)

    h0 = 0.3 * params.R_mu * (1.0 - params.g_ratio) # fraction of size of average torus width
    eta = 5.0 # approx ratio between largest/smallest edges

    mxcall(:figure,0); mxcall(:hold,0,"on")
    @time grid = circle_mesh_with_tori(bcircle, inner_circles, outer_circles, h0, eta)
    @time exteriorgrid, torigrids, interiorgrids = form_tori_subgrids(grid, bcircle, inner_circles, outer_circles)

    all_tori = form_subgrid(grid, getcellset(grid, "tori"), getnodeset(grid, "tori"), getfaceset(grid, "boundary"))
    all_int = form_subgrid(grid, getcellset(grid, "interior"), getnodeset(grid, "interior"), getfaceset(grid, "boundary"))
    mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(exteriorgrid)
    mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(all_tori)
    mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(all_int)

    prob = MyelinProblem(params)
    domain = MyelinDomain(grid, outer_circles, inner_circles, bcircle,
        exteriorgrid, torigrids, interiorgrids;
        quadorder = 3, funcinterporder = 1)

    return prob, domain
end

function DiffEqBase.solve(prob::MyelinProblem,
                          domain::MyelinDomain;
                          tspan = (0.0, 40e-3))
    doassemble!(prob, domain)
    factorize!(domain)

    u0 = Vec{2}((0.0, 1.0)) # Initial π/2-pulse
    U0 = interpolate(x->u0, domain) # vector of vectors of degrees of freedom with `u0` at each node
    U = deepcopy(U0)

    for i in 1:numsubdomains(domain)
        print("i = $i: ")

        subdomain = getsubdomain(domain, i)
        A = ParabolicLinearMap(subdomain)
        U[i] = copy(U0[i])

        # Method 1: expmv! from Higham's `Expmv`
        M = Expmv.select_taylor_degree(A, U0[i]; norm = expmv_norm)[1]
        @time Expmv.expmv!(U[i], tspan[end], A, U0[i]; prec = "single", M = M, norm = expmv_norm)
        Uexpmv = copy(U[i])

        # Method 2: expmv! from Expokit.jl
        @time Expokit.expmv!(U[i], tspan[end], A, U0[i]; tol=1e-6, norm=expmv_norm, m=30)
        Uexpokit = copy(U[i])

        # Method 3: direct ODE solution using DifferentialEquations.jl
        prob = ODEProblem((du,u,p,t)->mul!(du,p[1],u), U0[i], tspan, (A,));
        @time sol = solve(prob, CVODE_BDF(linear_solver=:GMRES); saveat=tspan, reltol=1e-6, alg_hints=:stiff)
        U[i] = sol.u[end]
        Udiffeq = copy(U[i])

        # Compare simulation results
        @test norm(Uexpmv - Uexpokit, Inf) ≈ 0.0 rtol = 1e-4 atol = 1e-4
        @test norm(Udiffeq - Uexpokit, Inf) ≈ 0.0 rtol = 1e-4 atol = 1e-4

        flush(stdout)
    end

    return U
end

@testset "Myelin Problem Solutions" begin
    prob, domain = setup()
    U = solve(prob, domain)
end

prob, domain = setup()
U = solve(prob, domain)
