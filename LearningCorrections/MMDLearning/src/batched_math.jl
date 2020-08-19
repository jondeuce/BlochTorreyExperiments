####
#### Accelerated batched functions
####

function avx_map!(f, dst::AbstractArray, src::AbstractArray)
    if length(dst) < Threads.nthreads()
        @avx for i in eachindex(src,dst)
            dst[i] = f(src[i])
        end
    else
        Threads.@sync for I in Iterators.partition(eachindex(src,dst), length(dst) ÷ Threads.nthreads())
            Threads.@spawn begin
                @avx for i in I
                    dst[i] = f(src[i])
                end
            end
        end
    end
    return dst
end
avx_map!(f, A::AbstractArray) = avx_map!(f, A, A)
fast_exp!(A...) = avx_map!(exp, A...)

# TODO: Some necessary piracy here: broadcasted `+.(::Diagonal{Fill}, ::CuMatrix)` falls back to scalar indexing
Zygote.accum(x::Diagonal{<:Any, <:Zygote.FillArrays.Fill}, y::CUDA.CuMatrix) = Zygote.accum.(Diagonal(CUDA.CuVector(diag(x))), y)
Zygote.accum(x::CUDA.CuMatrix, y::Diagonal{<:Any, <:Zygote.FillArrays.Fill}) = Zygote.accum.(x, Diagonal(CUDA.CuVector(diag(y))))

####
#### In-place transpose-related helpers
####

# Generic versions of `NNlib.batched_transpose` for use outside of `NNlib.batched_mul`
batched_transpose(A::AbstractMatrix) = transpose(A)
batched_transpose(A::AbstractTensor3D) = permutedims(A, (2,1,3))
batched_transpose(A::CUDA.CuMatrix) = transpose(A)
batched_transpose(A::CuTensor3D) = permutedims(A, (2,1,3))

add_transpose!(A::CUDA.CuMatrix) = A .= A .+ A'
add_transpose!(A::CuTensor3D) = A .= A .+ batched_transpose(A)

self_transpose!(A::CUDA.CuMatrix) = A .= A'
self_transpose!(A::CuTensor3D) = A .= batched_transpose(A)

function add_transpose!(A::AbstractMatrix)
    for j in 1:size(A, 2)
        @inbounds @simd for i in 1:j
            A[i,j] += A[j,i]
            A[j,i]  = A[i,j]
        end
    end
    return A
end

function add_transpose!(A::AbstractTensor3D)
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

function self_transpose!(A::AbstractTensor3D)
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

# batched_diag(x::CuTensor3D) = batched_diag_brute(x)
batched_diag(x::CuTensor3D) = batched_diag!(similar(x, min(size(x,1), size(x,2)), 1, size(x,3)), x)

function batched_diag!(out::CuTensor3D, in::CuTensor3D)
    w, h, d = size(in)
    w_out = min(w,h)
    @assert size(out) == (w_out, 1, d)

    function batched_diag_kernel!(out, in)
        i = CUDA.threadIdx().x + (CUDA.blockIdx().x-1) * CUDA.blockDim().x
        j = CUDA.threadIdx().y + (CUDA.blockIdx().y-1) * CUDA.blockDim().y
        if i <= w_out && j <= d
            @inbounds out[i,1,j] = in[i,i,j]
        end
        return
    end

    function configurator(kernel)
        # See: https://github.com/JuliaGPU/CUDA.jl/blob/463a41295bfede5125c584e6be9c51a4b9074e12/examples/pairwise.jl#L88
        function get_threads(threads)
            if w_out * d <= threads
                return (w_out, d)
            else
                threads_x = min(2 ^ floor(Int, log2(max(w_out, d))), threads)
                threads_y = threads ÷ threads_x
                return w_out >= d ? (threads_x, threads_y) : (threads_y, threads_x)
            end
        end
        config = CUDA.launch_configuration(kernel.fun)
        threads = get_threads(config.threads)
        blocks = ceil.(Int, (w_out, d) ./ threads)
        return (threads=threads, blocks=blocks)
    end

    CUDA.@cuda name="batched_diag!" config=configurator batched_diag_kernel!(out, in)

    return out
end

batched_diag(x::AbstractMatrix) = LinearAlgebra.diag(x)

function batched_diag_brute(x::AbstractTensor3D)
    ndiag = min(size(x,1), size(x,2))
    return reshape(x[CartesianIndex.(1:ndiag, 1:ndiag), :], ndiag, 1, size(x,3))
end

Zygote.@adjoint function batched_diag(x::AbstractMatrix)
    return LinearAlgebra.diag(x), function(Δ)
        # Why is Δ sometimes an nx1 matrix? Related to adjoint... e.g. loss = sum(diag(x)')
        # (LinearAlgebra.Diagonal(Δ),) # Should be this...
        (LinearAlgebra.Diagonal(reshape(Δ,:)),) # ...but need to reshape nx1 matrix Δ to n-vector
    end
end

function batched_diag(x::AbstractTensor3D)
    ndiag = min(size(x,1), size(x,2))
    nbatch = size(x,3)
    y = similar(x, ndiag, 1, nbatch)
    # Threads.@threads
    @avx for k in 1:nbatch, i in 1:ndiag
        y[i,1,k] = x[i,i,k]
    end
    return y
end

Zygote.@adjoint function batched_diag(x::AbstractTensor3D)
    return batched_diag(x), function(Δ)
        ndiag = min(size(x,1), size(x,2))
        nbatch = size(x,3)
        y = zero(x)
        # Threads.@threads
        @avx for k in 1:nbatch, i in 1:ndiag
            y[i,i,k] = Δ[i,1,k]
        end
        return (y,)
    end
end

#=
let
    x = randn(256,256,4)

    @assert batched_diag(x) == batched_diag_brute(x)
    @btime batched_diag($x)
    @btime batched_diag_brute($x)

    f = x -> sum(batched_diag(x))
    f_brute = x -> sum(batched_diag_brute(x))

    g = Zygote.gradient(f, x)
    g_brute = Zygote.gradient(f_brute, x)
    @assert all(isapprox.(g, g_brute))

    @btime Zygote.gradient($f, $x)
    @btime Zygote.gradient($f_brute, $x)
end;
=#

nothing