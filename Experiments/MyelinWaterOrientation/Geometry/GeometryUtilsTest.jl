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
        for T in (Float32, Float64, BigFloat)
            V = Vec{dim,T}
            C = Circle{dim,T}
            R = Rectangle{dim,T}
            E = Ellipse{dim,T}

            # ---- Ellipse misc. ---- #
            e = rand(Ellipse{dim,T})
            @test geta(e)^2 ≈ getb(e)^2 + getc(e)^2

            # ---- Ellipse is exactly a circle ---- #
            circ = rand(Circle{dim,T})
            e = Ellipse(origin(circ), origin(circ), radius(circ))
            X = rand(Vec{dim,T})

            # Still may only be approximate, as origin(e) = 0.5*(F1+f2)
            @test area(circ) ≈ area(e)
            @test signed_edge_distance(X, circ) ≈ signed_edge_distance(X, e)

            # ---- Ellipse signed_edge_distance testing ---- #
            e = rand(Ellipse{dim,T})
            O, F1, F2, a, b, c = origin(e), getF1(e), getF2(e), geta(e), getb(e), getc(e)

            Ua = (F2-F1)/norm(F2-F1) # unit vector for major axis
            Ub = Vec{dim,T}((-Ua[2], Ua[1])) # unit vector for minor
            A, B = O + a*Ua, O + b*Ub # point at major/minor axis boundary

            Rot = rotmat(e)
            ϕ = 2*T(π)*rand(T)
            P = O + Rot ⋅ Vec{dim,T}((a*cos(ϕ), b*sin(ϕ))) # rand point on ellipse
            N = Rot ⋅ Vec{dim,T}((cos(ϕ)/a, sin(ϕ)/b)) # normal vector at P
            N = N / norm(N)
            # dmax = min(b^2/a*abs(cos(ϕ)), a^2/b*abs(sin(ϕ))) # distance to x- or y-intercept
            dmax = b^2/a*abs(cos(ϕ)) # x-axis dist is always less than y-axis dist
            d = rand(T) * dmax # rand distance less distance to x- or y-axis
            Xout = (1+rand(T)) * (P-O) + O # random point outside ellipse
            Xin = rand(T) * (P-O) + O # random point inside ellipse

            @test signed_edge_distance(Xout, e) > zero(T)
            @test signed_edge_distance(Xin, e) < zero(T)
            @test signed_edge_distance(A, e) ≈ zero(T) atol = 10*eps(T)
            @test signed_edge_distance(B, e) ≈ zero(T) atol = 10*eps(T)
            @test signed_edge_distance(P, e) ≈ zero(T) atol = 10*eps(T)
            @test signed_edge_distance(O, e) ≈ -b atol = 10*eps(T)
            @test signed_edge_distance(F1, e) ≈ c - a atol = 10*eps(T)
            @test signed_edge_distance(P + d*N, e) ≈ d atol = 50*eps(T)
            @test signed_edge_distance(P - d*N, e) ≈ -d atol = 50*eps(T)

            X0 = O + Rot ⋅ Vec{dim,T}((c^2/a, zero(T))) # center of curvature at vertice (a,0)
            X1 = O + Rot ⋅ Vec{dim,T}((zero(T), c^2/b)) # center of curvature at co-vertice (0,b)
            ρ0, ρ1 = b^2/a, a^2/b # ρ0 < b and ρ1 > a
            c0, c1 = Circle(X0, T(0.9)*ρ0), Circle(X1, ρ1)

            # c0 should be inside e
            @test is_inside(Circle(O,rand(T)*b), e) # should be true, as ρ < b
            @test is_inside(c0, e) # should be true, as ρ0 < b
            @test !is_inside(c1, e) # should be false, as ρ1 > a > b

            # ---- Intersect area testing between circle/rectangle ---- #
            circ = rand(Circle{2,T})
            test_intersect_area(circ)

            # ---- Intersect points testing between circle/rectangle ---- #
            # sort by x first, unless x-coords are approx equal; then sort by y
            approxlt(x,y) = x[1] ≈ y[1] ? x[2] < y[2] : x[1] < y[1]
            sortpts(p) = sort(p; lt = approxlt)
            sorted_int_pts(c, r) = intersection_points(c, r) |> sortpts

            circ = C(zero(V), one(T)) # unit circle
            rect = R(-ones(V), ones(V)) # unit square
            @test sorted_int_pts(circ, rect) ≈ V.([(-1,0), (0,-1), (0,1), (1,0)])

            # circle and it's bounding box
            circ = rand(C)
            rect = bounding_box(circ)
            @test length(intersection_points(circ, rect)) == 4 # should only be the 4 tangent points

            # circle and rectangle which splits it in half vertically
            circ = rand(C)
            o, r = origin(circ), radius(circ)
            rect = R(V((xmin(circ)-r, ymin(circ)-r)), V((o[1], ymax(circ) + r)))
            @test sorted_int_pts(circ, rect) ≈ V.([(o[1], o[2]-r), (o[1], o[2]+r)])

            # circle and rectangle which splits it in half vertically
            circ = rand(C)
            o, r = origin(circ), radius(circ)
            rect = scale_shape(bounding_box(circ), inv(sqrt(T(2))))
            pts = sorted_int_pts(circ, rect)
            @test length(pts) == 4
            @test pts ≈ [o] .+ (r/sqrt(T(2))) .* V.([(-1,-1), (-1,1), (1,-1), (1,1)])

            # circle and rectangle which overlap such that there are 8 equally
            # spaced intersection points around the circle
            circ = rand(C)
            o, r = origin(circ), radius(circ)
            rect = scale_shape(bounding_box(circ), sqrt(1/T(2)+sqrt(1/T(8))) )
            pts = sorted_int_pts(circ, rect)

            ϕ = T(π)/8 .* collect(1:2:15) # 8 equally spaced angles
            u = V[V((cos(ϕᵢ), sin(ϕᵢ))) for ϕᵢ in ϕ] |> sortpts # equally spaced unit vectors
            pts_exact = [o] .+ r .* u

            @test length(pts) == 8
            @test pts ≈ pts_exact
        end
    end
    nothing
end

function test_intersect_area(c::Circle{2,T}) where {T}
    # allow for points to be inside or slightly outside of circle
    rect = scale_shape(bounding_box(c), T(1.5))

    x0 = origin(rect)[1] - (xmax(rect)-xmin(rect))/2 * rand(T)
    x1 = origin(rect)[1] + (xmax(rect)-xmin(rect))/2 * rand(T)
    y0 = origin(rect)[2] - (ymax(rect)-ymin(rect))/2 * rand(T)
    y1 = origin(rect)[2] + (ymax(rect)-ymin(rect))/2 * rand(T)
    @assert (x0 <= x1 && y0 <= y1)

    R = radius(c)
    h = max(R - (min(y1, ymax(c)) - origin(c)[2]), zero(T))

    A_top_test = (
        intersect_area(c, -T(Inf),   y1,     x0, T(Inf)) + # 1: top left
        intersect_area(c,      x0,   y1,     x1, T(Inf)) + # 2: top middle
        intersect_area(c,      x1,   y1, T(Inf), T(Inf))   # 3: top right
    )

    A_top = R^2 * acos(1-h/R) - (R-h)*√(2R*h-h^2)
    @test isapprox(A_top, A_top_test; atol = 100*eps(T))

    A_test = (intersect_area(c, -T(Inf),      y1,     x0, T(Inf)) + # 1: top left
              intersect_area(c,      x0,      y1,     x1, T(Inf)) + # 2: top middle
              intersect_area(c,      x1,      y1, T(Inf), T(Inf)) + # 3: top right
              intersect_area(c, -T(Inf),      y0,     x0,     y1) + # 4: middle left
              intersect_area(c,      x0,      y0,     x1,     y1) + # 5: middle middle
              intersect_area(c,      x1,      y0, T(Inf),     y1) + # 6: middle right
              intersect_area(c, -T(Inf), -T(Inf),     x0,     y0) + # 7: bottom left
              intersect_area(c,      x0, -T(Inf),     x1,     y0) + # 8: bottom middle
              intersect_area(c,      x1, -T(Inf), T(Inf),     y0))  # 9: bottom right

    A_exact = pi*radius(c)^2
    @test A_test ≈ A_exact

    return nothing
end

end # module GeometryUtilsTest

nothing
