module DistMeshTest

using DistMesh
using Tensors
using MATLAB

using Test
using BenchmarkTools

# ---------------------------------------------------------------------------- #
# Utils Testing
# ---------------------------------------------------------------------------- #
function utilstests(;N = 1000)
    @testset "Distances" begin
            
    end
end

# ---------------------------------------------------------------------------- #
# Geometry Testing
# ---------------------------------------------------------------------------- #

function geomtests(;N = 1000)
    @testset "Distances" begin
        d1, d2 = randn(N), randn(N)
        @test ddiff.(d1, d2) ≈ mxcall(:ddiff, 1, d1, d2)
        @test dintersect.(d1, d2) ≈ mxcall(:dintersect, 1, d1, d2)
        @test dunion.(d1, d2) ≈ mxcall(:dunion, 1, d1, d2)

        T = Float64
        dim = 2
        V = Vec{dim,T}
        p = V[randn(V) for i in 1:N]
        P = DistMesh.to_mat(p)

        x1, x2, y1, y2 = -rand(T), rand(T), -rand(T), rand(T)
        p1, p2 = Vec{dim,T}((x1,y1)), Vec{dim,T}((x2,y2))
        r = one(T) + rand(T)

        @test drectangle.(p,x1,x2,y1,y2) ≈ mxcall(:drectangle,1,P,x1,x2,y1,y2)
        @test drectangle.(p,[p1],[p2]) ≈ mxcall(:drectangle,1,P,x1,x2,y1,y2)
        @test drectangle0.(p,x1,x2,y1,y2) ≈ mxcall(:drectangle0,1,P,x1,x2,y1,y2)
        @test drectangle0.(p,[p1],[p2]) ≈ mxcall(:drectangle0,1,P,x1,x2,y1,y2)
        @test dcircle.(p,[p1],[r]) ≈ mxcall(:dcircle,1,P,x1,y1,r)
        @test dcircle.(p,[x1],[y1],[r]) ≈ mxcall(:dcircle,1,P,x1,y1,r)

        T = Float64
        dim = 3
        V = Vec{dim,T}
        p = V[randn(V) for i in 1:N]
        P = DistMesh.to_mat(p)

        x1, x2, y1, y2, z1, z2 = -rand(T), rand(T), -rand(T), rand(T), -rand(T), rand(T)
        p1, p2 = Vec{dim,T}((x1,y1,z1)), Vec{dim,T}((x2,y2,z2))
        r = one(T) + rand(T)

        @test dblock.(p,x1,x2,y1,y2,z1,z2) ≈ mxcall(:dblock,1,P,x1,x2,y1,y2,z1,z2)
        @test dblock.(p,[p1],[p2]) ≈ mxcall(:dblock,1,P,x1,x2,y1,y2,z1,z2)
        @test dsphere.(p,[p1],[r]) ≈ mxcall(:dsphere,1,P,x1,y1,z1,r)
        @test dsphere.(p,[x1],[y1],[z1],[r]) ≈ mxcall(:dsphere,1,P,x1,y1,z1,r)
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
