using Test, BenchmarkTools
using SparseArrays, LinearAlgebra, LinearMaps
# import ExpmV, Expokit

####
#### InterleavedMatrix
####

@testset "Sparse: $sparsemat" for sparsemat in [false,true]
    @testset "Columns: $cols" for cols in 1:2
        @testset "Skip: $skip" for skip in 1:3
            d = 24
            dA = div(d,skip)
            
            A = sparsemat ?
                sprandn(ComplexF64, dA, dA, 6/dA) :
                randn(ComplexF64, dA, dA)
            x = cols == 1 ? randn(ComplexF64, d) : randn(ComplexF64, d, cols)
            y = similar(x)
            S = InterleavedMatrix(A,skip)

            (skip == 1) && @test A*x ≈ S*x
            (skip == 1) && @test mul!(y,A,x) ≈ mul!(y,S,x)

            for i in 1:skip
                y[i:skip:end, :] .= A * x[i:skip:end, :]
            end
            @test y ≈ S*x

            for i in 1:skip
                y[i:skip:end, :] .= A \ x[i:skip:end, :]
            end
            @test y ≈ S\x

            FA = lu(A)
            FS = InterleavedMatrix(FA,skip)
            for i in 1:skip
                y[i:skip:end, :] .= FA \ x[i:skip:end, :]
            end
            @test y ≈ FS\x

            # Force positive definite and test cholesky
            Apd = (A+A')/2
            Apd = Apd + d*I
            @assert isposdef(Apd)
            
            FA = cholesky(Apd)
            FS = InterleavedMatrix(FA,skip)
            for i in 1:skip
                y[i:skip:end, :] .= FA \ x[i:skip:end, :]
            end
            @test y ≈ FS\x

            if skip > 1
                # Pad x, check that dimension mismatch is thrown
                xpad = vcat(x, ones(1,size(x,2)))
                @test_throws DimensionMismatch S*xpad
                @test_throws DimensionMismatch S\xpad
            end
        end
    end
end

####
#### BlockDiagonalMatrix
####

@testset "Sparse: $sparsemat" for sparsemat in [false, true]
    @testset "Posdef: $posdef" for posdef in [false, true]
        @testset "Columns: $cols" for cols in 1:2
            ds = [10,9,13,11]
            d = sum(ds)
            
            to_posdef = A -> posdef ? (A = (A+A')/2; A = A + d*I; @assert isposdef(A); A) : A
            As = sparsemat ?
                [rand(ComplexF64, di, di) |> to_posdef for di in ds] :
                [sprand(ComplexF64, di, di, 6/di) |> to_posdef for di in ds]
            
            S = BlockDiagonalMatrix(As)
            A = Matrix(S)
            x = cols == 1 ? randn(ComplexF64, d) : randn(ComplexF64, d, cols)
            y = similar(x)

            @test mul!(y,A,x) ≈ mul!(y,S,x)
            @test A*x ≈ S*x
            @test A\x ≈ S\x

            FA = posdef ? cholesky(A) : lu(A)
            FS = posdef ? cholesky(S) : lu(S)
            @test FA\x ≈ FS\x
        end
    end
end
