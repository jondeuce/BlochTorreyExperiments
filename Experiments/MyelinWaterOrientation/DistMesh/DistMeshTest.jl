module DistMeshTest

using DistMesh
using Tensors
using MATLAB

using Test
using BenchmarkTools

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

function runtests()
    @testset "DistMesh" begin
        N = 1000
        T = Float64
        V = Vec{2,T}
        d1, d2 = randn(N), randn(N)
        p = V[randn(V) for i in 1:N]
        P = reshape(reinterpret(T, p), (2,N)) |> transpose |> copy

        @test ddiff.(d1, d2) ≈ mxcall(:ddiff, 1, d1, d2)
        @test dintersect.(d1, d2) ≈ mxcall(:dintersect, 1, d1, d2)
        @test dunion.(d1, d2) ≈ mxcall(:dunion, 1, d1, d2)

        x1, x2, y1, y2 = -rand(), rand(), -rand(), rand()
        @test drectangle.(p,x1,x2,y1,y2) ≈ mxcall(:drectangle,1,P,x1,x2,y1,y2)
        @test drectangle0.(p,x1,x2,y1,y2) ≈ mxcall(:drectangle0,1,P,x1,x2,y1,y2)
    end
    nothing
end

function runbenchmarks()
    fd = x -> x⋅x - one(eltype(x))
    fh = x -> one(eltype(x))
    h0 = 0.2
    bbox = [-1.0 -1.0; 1.0 1.0]

    p, t = distmesh2d(fd, fh, h0, bbox; PLOT = false)
    b = @benchmark distmesh2d($fd, $fh, $h0, $bbox; DETERMINISTIC = true)
    display(b)

    nothing
end

function runexamples()
    to_vec(P) = reinterpret(Vec{2,eltype(P)}, transpose(P)) |> vec |> copy

    # Example: (Uniform Mesh on Unit Circle)
    fd = p -> p⋅p - one(eltype(p))
    fh = huniform
    h0 = 0.2
    bbox = [-1.0 -1.0; 1.0 1.0]
    p, t = distmesh2d(fd, fh, h0, bbox; PLOTLAST = true)

    # Example: (Rectangle with circular hole, refined at circle boundary)
    fd = p -> ddiff(drectangle0(p,-1.0,1.0,-1.0,1.0), dcircle(p,0.0,0.0,0.5))
    fh = p -> 0.05 + 0.3 * dcircle(p,0.0,0.0,0.5)
    h0 = 0.05
    bbox = [-1.0 -1.0; 1.0 1.0]
    pfix = to_vec([-1.0 -1.0; -1.0 1.0; .01 -1.0; 1.0 1.0])
    p, t = distmesh2d(fd, fh, h0, bbox, pfix; PLOTLAST = true);

    nothing
end

end # module DistMeshTest

nothing
