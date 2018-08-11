include("init.jl")
using Base.Test

@testset "Geometry Utils" begin
    const dim = 2
    const T = Float64

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

function expmv_tests(;N = 4, h = √2*(2/N), grid = generate_grid(Triangle, (N, N)))
    # Assemble mass and Neumann stiffness matrix
    domain = ParabolicDomain(grid; quadorder = 1, funcinterporder = 1)
    doassemble!(domain)
    factorize!(domain)
    M, Mfact, K = getmass(domain), getmassfact(domain), getstiffness(domain)

    # Smallest 2 eigenvalues should be zero (one for each dimension of u), and the rest negative
    λ, ϕ = eigs(-K; nev=3, which=:SR) # smallest real
    λmin = max(abs(λ[1]), abs(λ[2])) # largest null eigenvalue
    ispossemidef_K = isapprox(λmin, 0.0; atol=1e-12) && (λ[3] > h^2)

    # @test isposdef(M) && ispossemidef_K

    A = ParabolicLinearMap(M, Mfact, K)
    Af = full(A)

    # @show eigs(M; nev=5, which=:SM)[1]

    t = 0.1
    x0 = randn(size(A,2))
    Ef = expm(t*Af) # dense exponential matrix

    y1 = Ef * x0
    y2 = Expmv.expmv(t, A, x0)
    @test norm(y1 - y2, Inf) ≈ 0.0 rtol = 1e-6

    # x = Expmv.expmv(1e-6, A, x0; prnt = true)
end

@testset "Expmv Methods" begin
    expmv_tests()
end

# ---- Single axon geometry testing ---- #
const p = BlochTorreyParameters()
rs = [p[:R_mu]] # one radius of average size
os = zeros(Vec{2}, 1) # one origin at the origin
outer_circles = Circle.(os, rs)
inner_circles = scale_shape.(outer_circles, p[:g_ratio])
bcircle = scale_shape(outer_circles[1], 1.5)

h0 = 0.3 * p[:R_mu] * (1 - p[:g_ratio]) # fraction of size of average torus width
eta = 5.0 # approx ratio between largest/smallest edges
@time grid = circle_mesh_with_tori(bcircle, inner_circles, outer_circles, h0, eta)
@time exteriorgrid, torigrids, interiorgrids = form_tori_subgrids(grid, bcircle, inner_circles, outer_circles)

all_tori = form_subgrid(grid, getcellset(grid, "tori"), getnodeset(grid, "tori"), getfaceset(grid, "boundary"))
all_int = form_subgrid(grid, getcellset(grid, "interior"), getnodeset(grid, "interior"), getfaceset(grid, "boundary"))
mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(exteriorgrid)
mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(all_tori)
mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(all_int)

domain = MyelinDomain(grid, outer_circles, inner_circles, bcircle,
    exteriorgrid, torigrids, interiorgrids;
    quadorder = 1, funcinterporder = 1)

prob = MyelinProblem(p)
doassemble!(prob, domain)
factorize!(domain)

tspan = (0.0, 1e-3) # 40ms simulation
u0 = Vec{2}((0.0, 1.0)) # Initial π/2-pulse
U0 = interpolate(x->u0, domain) # vector of vectors of degrees of freedom with `u0` at each node
U = deepcopy(U0)

for i in 1:numsubdomains(domain)
    print("i = $i: ")

    subdomain = getsubdomain(domain, i)
    A = paraboliclinearmap(subdomain)
    U[i] = copy(U0[i])

    # Method 1: expmv! from Expokit.jl
    @time Expokit.expmv!(U[i], tspan[end], A, U0[i]; tol=1e-4, norm=expmv_norm, m=30);

    # Method 2: direct ODE solution using DifferentialEquations.jl
    # prob = ODEProblem((du,u,p,t)->A_mul_B!(du,p[1],u), U0[i], tspan, (A,));
    # @time sol = solve(prob, CVODE_BDF(linear_solver=:GMRES); saveat=tspan, reltol=1e-4, alg_hints=:stiff)
    # U[i] = sol.u[end]

    flush(STDOUT)
end

subdomain = getsubdomain(domain, 2)
A = ParabolicLinearMap(getmass(subdomain), getmassfact(subdomain), getstiffness(subdomain))
Af = full(A)

x0 = randn(size(A,2))
x = Expmv.expmv(1e-6, A, x0; prnt = true)
