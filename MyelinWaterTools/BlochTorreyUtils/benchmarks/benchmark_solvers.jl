include(realpath(joinpath(@__DIR__,"../../initpaths.jl")))
using Test, BenchmarkTools
# using SparseArrays, LinearAlgebra, Statistics
# using BlochTorreyUtils, BlochTorreySolvers, MWFUtils
# using GeometryUtils, JuAFEM # needed for loading grids
# using OrdinaryDiffEq, Sundials
using MWFUtils
import BSON
import ExpmV, Expokit

function setup()
    default_btparams = BlochTorreyParameters{Float64}(
        theta = π/2,
        AxonPDensity = 0.7,
        g_ratio = 0.75,
        K_perm = 0.5, #0.0 # [μm/s]
        D_Tissue = 100.0, #0.5, # [μm²/s]
        D_Sheath = 100.0, #0.5, # [μm²/s]
        D_Axon   = 100.0, #0.5, # [μm²/s]
    )
    
    grids = BSON.load(joinpath(BTHOME, "BlochTorreyResults/Experiments/MyelinWaterOrientation/geom_sweep_2/2019-02-15-T-13-45-04-394__N-10_g-0.7500_p-0.7000__grids.bson"))

    G, C = typeof(grids[:exteriorgrids][1]), typeof(grids[:outercircles][1])
    myelinprob, myelinsubdomains, myelindomains = createdomains(
        default_btparams,
        Vector{G}(grids[:exteriorgrids][:]), Vector{G}(grids[:torigrids][:]), Vector{G}(grids[:interiorgrids][:]),
        Vector{C}(grids[:outercircles][:]), Vector{C}(grids[:innercircles][:])
    )

    return myelinprob, myelinsubdomains, myelindomains
end

function run_benchmarks()
    myelinprob, myelinsubdomains, myelindomains = setup()
    M, K = getmass(myelindomains[1]), getstiffness(myelindomains[1])
    A = ParabolicLinearMap(getdomain(myelindomains[1]))
    
    algs = Dict(
        :Tsit5 => Tsit5(),
        :BS5 => BS5(),
        :BS3 => BS3(),
        # :CVODE_BDF => CVODE_BDF(;method = :Functional), # this method is broken!
        :HighamExpmv => HighamExpmv(A,1; precision = :half),
        :ExpokitExpmv => ExpokitExpmv(A)
    )
    
    sols = Dict()
    for (algname,alg) in algs
        println("---- Alg = $(algname) ----")
        sols[algname] = solveblochtorrey(myelinprob, myelindomains, prob->alg;
            tspan = (0.0, 40e-3),
            reltol=1e-4,
            abstol=1e-6
        )
        println("")
    end

    return sols
end

relerr(x,y) = (m = max(abs(x),abs(y),5*eps(typeof(x)),5*eps(typeof(y))); return (x-y)/m)

sols = run_benchmarks()

nothing

# #BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120
# 
# SUITE = BenchmarkGroup([],
#     "Matrix" => BenchmarkGroup(["Matrix"]),
#     "BlockDiag" => BenchmarkGroup(["BlockDiag"]))
# 
# # posdefrange = [true]
# # sparsematrange = [true]
# # dimensions = [1024]
# # partitions = [16]
# posdefrange = [false, true]
# sparsematrange = [true]
# dimensions = [d for d in 2 .^ (10:12)]
# partitions = [p for p in 2 .^ (1:4)]
# 
# for sp in sparsematrange
#     SUITE["Matrix"][sp] = BenchmarkGroup(["Sparse", sp])
#     SUITE["BlockDiag"][sp] = BenchmarkGroup(["Sparse", sp])
#     for pd in posdefrange
#         SUITE["Matrix"][sp][pd] = BenchmarkGroup(["Posdef", pd])
#         SUITE["BlockDiag"][sp][pd] = BenchmarkGroup(["Posdef", pd])
#         for d in dimensions
#             SUITE["Matrix"][sp][pd][d] = BenchmarkGroup(["Dimension", d])
#             SUITE["BlockDiag"][sp][pd][d] = BenchmarkGroup(["Dimension", d])
#             for p in partitions
# 
#                 ds = [div(d,p) for _ in 1:p]
#                 As = sp ? [sprand(ComplexF64, di, di, min(6/di,0.5)) for di in ds] :
#                           [rand(ComplexF64, di, di) for di in ds]
#                 As = if pd
#                     map(As) do A
#                         A = (A+A')/2 + size(A,1)*I; @assert isposdef(A); return A
#                     end
#                 else
#                     map(A -> A + size(A,1)*I, As) # ensure non-singular A
#                 end
# 
#                 B = BlockDiagonalMatrix(As)
#                 FB = pd ? cholesky(B) : lu(B)
#                 A = sp ? sparse(B) : Matrix(B)
#                 FA = pd ? cholesky(A) : lu(A)
#                 x = randn(eltype(A), size(A,2))
# 
#                 SUITE["Matrix"][sp][pd][d][p] = @benchmarkable $FA\$x
#                 SUITE["BlockDiag"][sp][pd][d][p] = @benchmarkable $FB\$x
# 
#                 # @testset "isposdef=$pd, issparse=$sp, dim=$d, partition=$p" begin
#                 #     @test FB\x ≈ FA\x
#                 # end
#             end
#         end
#     end
# end
# 
# results = run(SUITE, verbose = true, seconds = 5)
# 
# for sp in sparsematrange
#     for pd in posdefrange
#         for d in dimensions
#             mat_results = results["Matrix"][sp][pd][d]
#             blk_results = results["BlockDiag"][sp][pd][d]
#             names = sort(collect(keys(mat_results)))
#             len = maximum(length, names)
# 
#             println("---- d=$d, isposdef=$pd, issparse=$sp ----")
#             for name in names
#                 print("p = " * rpad(string(name) * ": ", len+6))
#                 ratio(median(blk_results[name]),median(mat_results[name])) |> judge |> show
#                 println("")
#             end
#             println("")
# 
#             # println("Matrix:");
#             # for name in names
#             #     print(rpad(string(name) * ": ", len+2)); show(mat_results[name]); println("")
#             # end
#             # println("")
# 
#             # println("BlockDiagonalMatrix:");
#             # for name in names
#             #     print(rpad(string(name) * ": ", len+2)); show(blk_results[name]); println("")
#             # end
#             # println("")
#         end
#     end
# end
# 