module GeometryUtilsTest

using GeometryUtils

using Test
using LinearAlgebra
using Tensors: Vec

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

function runtests()
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
        @test signed_edge_distance(P + d*N, e) ≈ d atol = 50eps(T)
        @test signed_edge_distance(P - d*N, e) ≈ -d atol = 50eps(T)

        X0 = O + R ⋅ Vec{dim,T}((c^2/a, zero(T))) # center of curvature at vertice (a,0)
        X1 = O + R ⋅ Vec{dim,T}((zero(T), c^2/b)) # center of curvature at co-vertice (0,b)
        ρ0, ρ1 = b^2/a, a^2/b # ρ0 < b and ρ1 > a
        c0, c1 = Circle(X0, 0.9ρ0), Circle(X1, ρ1)

        # c0 should be inside e
        @test is_inside(Circle(O,rand(T)*b), e) # should be true, as ρ < b
        @test is_inside(c0, e) # should be true, as ρ0 < b
        @test !is_inside(c1, e) # should be false, as ρ1 > a > b
    end
    nothing
end

end # module GeometryUtilsTest

nothing
