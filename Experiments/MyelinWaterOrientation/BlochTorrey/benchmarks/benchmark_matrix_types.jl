using Test, BenchmarkTools
using SparseArrays, LinearAlgebra, LinearMaps
# import ExpmV, Expokit

#BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

SUITE = BenchmarkGroup([],
    "General" => BenchmarkGroup([],
        "Sparse" => BenchmarkGroup(),
        "Full" => BenchmarkGroup()),
    "Posdef" => BenchmarkGroup([],
        "Sparse" => BenchmarkGroup(),
        "Full" => BenchmarkGroup()));

dimensions = [d for d in 2 .^ (6:2:10)]
partitions = [p for p in 2 .^ (1:4)]

for posdef in [false, true]
    for sparsemat in [false, true]
        
        posdefstr = posdef ? "Posdef" : "General"
        sparsestr = sparsemat ? "Sparse" : "Full"
        
        for d in dimensions
            for p in partitions
                ds = [div(d,p) for _ in 1:p]
                As = sparsemat ?
                    [sprand(ComplexF64, di, di, 6/di) for di in ds] :
                    [rand(ComplexF64, di, di) for di in ds]
                if posdef
                    As = map(As) do A
                        A = (A+A')/2; A = A + size(A,1)*I; @assert isposdef(A)
                        return A
                    end
                end

                B = BlockDiagonalMatrix(As)
                A = Matrix(B)
                FB = posdef ? cholesky(B) : lu(B)
                FA = posdef ? cholesky(A) : lu(A)
                x = similar(A, size(A,2))
                
                SUITE[posdefstr][sparsestr][d][p]  = @benchmarkable $FA\$x
                SUITE[posdefstr][sparsestr][d][p] = @benchmarkable $FB\$x
            end
        end
    end
end

results = run(SUITE, verbose = true, seconds = 5)

# for d in dimensions
#     for p in partitions
#         results_d = results[@tagged d]
#         names = collect(keys(results_d))
#         len = maximum(length, names)
#         sort!(names; by = k -> minimum(results[k][d][p]).time)
        
#         @show d
#         for name in names
#             print(rpad(name * ": ", len+2))
#             show(results[name][d][p])
#             println("")
#         end
#         println("")
#     end
# end
