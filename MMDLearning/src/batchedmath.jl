####
#### Batched multiplication of 3D arrays "slicewise"
####

# Borrowed from Transformers.jl, which borrowed it from BatchedRoutines.jl
#   https://github.com/chengchingwen/Transformers.jl
#   https://github.com/Roger-luo/BatchedRoutines.jl

#batched cpu gemm by BatchedRoutines.jl
for (gemm, elty) in ((:dgemm_,:Float64), (:sgemm_,:Float32),)
    @eval begin
        function batched_gemm!(
                transA::AbstractChar,
                transB::AbstractChar,
                alpha::($elty),
                A::AbstractArray{$elty, 3},
                B::AbstractArray{$elty, 3},
                beta::($elty),
                C::AbstractArray{$elty, 3},
            )
            @assert !Base.has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end
            LinearAlgebra.BLAS.chkstride1(A)
            LinearAlgebra.BLAS.chkstride1(B)
            LinearAlgebra.BLAS.chkstride1(C)

            ptrA = Base.unsafe_convert(Ptr{$elty}, A)
            ptrB = Base.unsafe_convert(Ptr{$elty}, B)
            ptrC = Base.unsafe_convert(Ptr{$elty}, C)

            for k in 1:size(A, 3)
                ccall((LinearAlgebra.BLAS.@blasfunc($gemm), LinearAlgebra.BLAS.libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt},
                     Ref{LinearAlgebra.BLAS.BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{LinearAlgebra.BLAS.BlasInt},
                     Ptr{$elty}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{$elty}, Ptr{$elty},
                     Ref{LinearAlgebra.BLAS.BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,stride(A,2)),
                     ptrB, max(1,stride(B,2)), beta, ptrC,
                     max(1,stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * sizeof($elty)
                ptrB += size(B, 1) * size(B, 2) * sizeof($elty)
                ptrC += size(C, 1) * size(C, 2) * sizeof($elty)
            end

            C
        end
    end
end

function batchedmul(
        A::AbstractArray{T,3},
        B::AbstractArray{T,3};
        transA::Bool = false,
        transB::Bool = false,
    ) where {T}
    (bs = size(A, 3)) == size(B, 3) || error("batch size mismatch")
    C = similar(A, size(A, transA ? 2 : 1), size(B, transB ? 1 : 2), bs)
    batchedmul!(C, A, B; transA=transA, transB=transB)
    return C
end

function batchedmul!(
        C::AbstractArray{T,3},
        A::AbstractArray{T,3},
        B::AbstractArray{T,3};
        transA::Bool = false,
        transB::Bool = false,
    ) where {T}
    At = transA ? 'T' : 'N'
    Bt = transB ? 'T' : 'N'
    batched_gemm!(At, Bt, one(T), A, B, zero(T), C)
    C
end

Flux.Zygote.@adjoint function batchedmul(
        A::AbstractArray{<:Real,3},
        B::AbstractArray{<:Real,3};
        transA::Bool = false,
        transB::Bool = false,
    )
    batchedmul(A, B; transA=transA, transB=transB),
    if transA
        if transB
            Δ -> (batchedmul(B, Δ; transA=true, transB=true), batchedmul(Δ, A; transA=true, transB=true))
        else
            Δ -> (batchedmul(B, Δ; transB=true), batchedmul(A, Δ))
        end
    else
        if transB
            Δ -> (batchedmul(Δ, B), batchedmul(Δ, A; transA=true))
        else
            Δ -> (batchedmul(Δ, B; transB=true), batchedmul(A, Δ; transA=true))
        end
    end
end

####
#### Batched diagonal extraction of 3D arrays
####

function batcheddiag(x::AbstractArray{T,3}) where {T}
    nbatch = size(x,3)
    ndiag = min(size(x,1), size(x,2))
    y = similar(x, ndiag, 1, nbatch)
    # Threads.@threads
    for k in 1:nbatch
        @simd for i in 1:ndiag
            @inbounds y[i,1,k] = x[i,i,k]
        end
    end
    return y
end

Flux.Zygote.@adjoint function batcheddiag(x::AbstractArray{T,3}) where {T}
    return batcheddiag(x), function(Δ)
        nbatch = size(x,3)
        ndiag = min(size(x,1), size(x,2))
        y = zero(x)
        # Threads.@threads
        for k in 1:nbatch
            @simd for i in 1:ndiag
                @inbounds y[i,i,k] = Δ[i,1,k]
            end
        end
        return (y,)
    end
end
Flux.Zygote.refresh()

function batcheddiag_brute(x::AbstractArray{T,3}) where {T}
    nbatch = size(x,3)
    ndiag = min(size(x,1), size(x,2))

    idx = CartesianIndex.(1:ndiag, 1:ndiag)
    y = reshape(x[idx,:], ndiag, 1, nbatch)

    return y
end

#=
let
    x = randn(256,256,4)

    @assert batcheddiag(x) == batcheddiag_brute(x)
    @btime batcheddiag($x)
    @btime batcheddiag_brute($x)

    f = x -> sum(batcheddiag(x))
    f_brute = x -> sum(batcheddiag_brute(x))

    g = Flux.Zygote.gradient(f, x)
    g_brute = Flux.Zygote.gradient(f_brute, x)
    @assert all(isapprox.(g, g_brute))

    @btime Flux.Zygote.gradient($f, $x)
    @btime Flux.Zygote.gradient($f_brute, $x)
end;
=#

nothing