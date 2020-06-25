####
#### Fast exponential function using Yeppp
####

function fast_exp!(B::AbstractMatrix, A::AbstractMatrix)
    @assert size(A) == size(B)
    Threads.@threads for j in 1:size(A,2)
        Yeppp.exp!(view(B,:,j), view(A,:,j))
    end
    return B
end

function fast_exp!(A::AbstractMatrix)
    Threads.@threads for j in 1:size(A,2)
        Yeppp.exp!(view(A,:,j))
    end
    return A
end

function fast_exp!(B::AbstractArray{<:Any,3}, A::AbstractArray{<:Any,3})
    @assert size(A) == size(B)
    for k in 1:size(A,3)
        Threads.@threads for j in 1:size(A,2)
            Yeppp.exp!(view(B,:,j,k), view(A,:,j,k))
        end
    end
    return B
end

function fast_exp!(A::AbstractArray{<:Any,3})
    for k in 1:size(A,3)
        Threads.@threads for j in 1:size(A,2)
            Yeppp.exp!(view(A,:,j,k))
        end
    end
    return A
end

####
#### In-place transpose-related helpers
####

function add_transpose!(A::AbstractMatrix)
    for j in 1:size(A, 2)
        @inbounds @simd for i in 1:j
            A[i,j] += A[j,i]
            A[j,i]  = A[i,j]
        end
    end
    return A
end

function add_transpose!(A::AbstractArray{<:Any,3})
    for k in 1:size(A, 3)
        for j in 1:size(A, 2)
            @inbounds @simd for i in 1:j
                A[i,j,k] += A[j,i,k]
                A[j,i,k]  = A[i,j,k]
            end
        end
    end
    return A
end

function self_transpose!(A::AbstractMatrix)
    for j in 1:size(A, 2)
        @inbounds @simd for i in 1:j
            A[i,j], A[j,i] = A[j,i], A[i,j]
        end
    end
    return A
end

function self_transpose!(A::AbstractArray{<:Any,3})
    for k in 1:size(A, 3)
        for j in 1:size(A, 2)
            @inbounds @simd for i in 1:j
                A[i,j,k], A[j,i,k] = A[j,i,k], A[i,j,k]
            end
        end
    end
    return A
end

####
#### Batched diagonal extraction of 3D arrays
####

batcheddiag(x::AbstractMatrix) = LinearAlgebra.diag(x)

Zygote.@adjoint function batcheddiag(x::AbstractMatrix)
    return LinearAlgebra.diag(x), function(Δ)
        # Why is Δ sometimes an nx1 matrix? Related to adjoint... e.g. loss = sum(diag(x)')
        # (LinearAlgebra.Diagonal(Δ),) # Should be this...
        (LinearAlgebra.Diagonal(reshape(Δ,:)),) # ...but need to reshape nx1 matrix Δ to n-vector
    end
end
# Zygote.refresh() #TODO

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

Zygote.@adjoint function batcheddiag(x::AbstractArray{T,3}) where {T}
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
# Zygote.refresh() #TODO

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

    g = Zygote.gradient(f, x)
    g_brute = Zygote.gradient(f_brute, x)
    @assert all(isapprox.(g, g_brute))

    @btime Zygote.gradient($f, $x)
    @btime Zygote.gradient($f_brute, $x)
end;
=#

nothing