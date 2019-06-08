using Test, BenchmarkTools
using SparseArrays, LinearAlgebra, Statistics
# import ExpmV, Expokit

#BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

SUITE = BenchmarkGroup([],
    "Matrix" => BenchmarkGroup(["Matrix"]),
    "BlockDiag" => BenchmarkGroup(["BlockDiag"]))

# posdefrange = [true]
# sparsematrange = [true]
# dimensions = [1024]
# partitions = [16]
posdefrange = [false, true]
sparsematrange = [true]
dimensions = [d for d in 2 .^ (10:12)]
partitions = [p for p in 2 .^ (1:4)]

for sp in sparsematrange
    SUITE["Matrix"][sp] = BenchmarkGroup(["Sparse", sp])
    SUITE["BlockDiag"][sp] = BenchmarkGroup(["Sparse", sp])
    for pd in posdefrange
        SUITE["Matrix"][sp][pd] = BenchmarkGroup(["Posdef", pd])
        SUITE["BlockDiag"][sp][pd] = BenchmarkGroup(["Posdef", pd])
        for d in dimensions
            SUITE["Matrix"][sp][pd][d] = BenchmarkGroup(["Dimension", d])
            SUITE["BlockDiag"][sp][pd][d] = BenchmarkGroup(["Dimension", d])
            for p in partitions

                ds = [div(d,p) for _ in 1:p]
                As = sp ? [sprand(ComplexF64, di, di, min(6/di,0.5)) for di in ds] :
                          [rand(ComplexF64, di, di) for di in ds]
                As = if pd
                    map(As) do A
                        A = (A+A')/2 + size(A,1)*I; @assert isposdef(A); return A
                    end
                else
                    map(A -> A + size(A,1)*I, As) # ensure non-singular A
                end

                B = BlockDiagonalMatrix(As)
                FB = pd ? cholesky(B) : lu(B)
                A = sp ? sparse(B) : Matrix(B)
                FA = pd ? cholesky(A) : lu(A)
                x = randn(eltype(A), size(A,2))

                SUITE["Matrix"][sp][pd][d][p] = @benchmarkable $FA\$x
                SUITE["BlockDiag"][sp][pd][d][p] = @benchmarkable $FB\$x

                # @testset "isposdef=$pd, issparse=$sp, dim=$d, partition=$p" begin
                #     @test FB\x ≈ FA\x
                # end
            end
        end
    end
end

results = run(SUITE, verbose = true, seconds = 5)

for sp in sparsematrange
    for pd in posdefrange
        for d in dimensions
            mat_results = results["Matrix"][sp][pd][d]
            blk_results = results["BlockDiag"][sp][pd][d]
            names = sort(collect(keys(mat_results)))
            len = maximum(length, names)

            println("---- d=$d, isposdef=$pd, issparse=$sp ----")
            for name in names
                print("p = " * rpad(string(name) * ": ", len+6))
                ratio(median(blk_results[name]),median(mat_results[name])) |> judge |> show
                println("")
            end
            println("")

            # println("Matrix:");
            # for name in names
            #     print(rpad(string(name) * ": ", len+2)); show(mat_results[name]); println("")
            # end
            # println("")

            # println("BlockDiagonalMatrix:");
            # for name in names
            #     print(rpad(string(name) * ": ", len+2)); show(blk_results[name]); println("")
            # end
            # println("")
        end
    end
end
