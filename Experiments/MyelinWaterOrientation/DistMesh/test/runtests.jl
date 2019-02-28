
using DistMesh
using Tensors
using MATLAB

using Test
using BenchmarkTools

# ---------------------------------------------------------------------------- #
# Distances Testing
# ---------------------------------------------------------------------------- #

function runtests(;N = 1000)
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

runtests()
nothing