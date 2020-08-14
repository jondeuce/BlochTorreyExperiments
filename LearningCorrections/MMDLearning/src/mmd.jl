####
#### Abstract MMD kernel interface
####

abstract type MMDKernel{T} end

struct KernelFunction{T,F} <: MMDKernel{T}
    f::F
    KernelFunction{T}(f) where {T} = new{T,typeof(f)}(f)
end
(k::KernelFunction)(Δ) = k.f(Δ)

struct MMDResults{T}
    m::T
    e_Kxx_e::T
    e_Kyy_e::T
    e_Kxy_e::T
end

struct MMDVarResults{T}
    m::T
    Kxx_F2::T
    Kyy_F2::T
    Kxy_F2::T
    e_Kxx_e::T
    e_Kyy_e::T
    e_Kxy_e::T
    Kxx_e_F2::T
    Kyy_e_F2::T
    Kxy_e_F2::T
    Kyx_e_F2::T
    e_Kxx_Kxy_e::T
    e_Kyy_Kyx_e::T
end

function mmd(res::Union{<:MMDResults, <:MMDVarResults})
    @unpack m, e_Kxx_e, e_Kyy_e, e_Kxy_e = res

    # See: http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    # MMDsq = e_Kxx_e/(m*(m-1)) + e_Kyy_e/(m*(m-1)) - 2*e_Kxy_e/(m^2)

    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    MMDsq = (e_Kxx_e + e_Kyy_e - 2e_Kxy_e) / (m * (m-1))

    return MMDsq
end

function mmd_and_mmdvar(res::MMDVarResults)
    #   NOTE: Here we assume a symmetric kernel k(x,y) == k(y,x),
    #         and therefore that Kxx == Kxx', Kyy = Kyy', Kxy == Kxy'
    # See:
    #   [1] https://arxiv.org/pdf/1906.02104.pdf
    #   [2] http://www.gatsby.ucl.ac.uk/~dougals/slides/dali/#/50
    @unpack m, Kxx_F2, Kyy_F2, Kxy_F2, e_Kxx_e, e_Kyy_e, e_Kxy_e, Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2, e_Kxx_Kxy_e, e_Kyy_Kyx_e = res
    m_2 = m * (m-1)
    m_3 = m_2 * (m-2)
    m_4 = m_3 * (m-3)

    # Var[MMD²_U]: Variance estimator
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    # MMDvar =
    #     ((          4) / (m_4    )) * (Kxx_e_F2 + Kyy_e_F2) +
    #     ((4*(m^2-m-1)) / (m*m_2^2)) * (Kxy_e_F2 + Kyx_e_F2) -
    #     ((          8) / (m*m_3  )) * (e_Kxx_Kxy_e + e_Kyy_Kyx_e) +
    #     ((          8) / (m^2*m_3)) * ((e_Kxx_e + e_Kyy_e) * e_Kxy_e) -
    #     ((   2*(2m-3)) / (m_2*m_4)) * (e_Kxx_e^2 + e_Kyy_e^2) -
    #     ((   4*(2m-3)) / (m_2^3  )) * (e_Kxy_e^2) -
    #     ((          2) / (m_4    )) * (Kxx_F2 + Kyy_F2) +
    #     ((   4m*(m-2)) / (m_2^3  )) * (Kxy_F2)

    t1_4 = ((   4) / (m_4    )) * (Kxx_e_F2 + Kyy_e_F2)
    t2_4 = ((4m^2) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t2_5 = ((  4m) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t2_6 = ((   4) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t3_4 = ((   8) / (m*m_3  )) * (e_Kxx_Kxy_e + e_Kyy_Kyx_e)
    t4_5 = ((   8) / (m^2*m_3)) * ((e_Kxx_e + e_Kyy_e) * e_Kxy_e)
    t5_4 = ((  4m) / (m_2*m_4)) * (e_Kxx_e^2 + e_Kyy_e^2)
    t5_5 = ((   6) / (m_2*m_4)) * (e_Kxx_e^2 + e_Kyy_e^2)
    t6_4 = ((  8m) / (m_2^3  )) * (e_Kxy_e^2)
    t6_5 = ((  12) / (m_2^3  )) * (e_Kxy_e^2)
    t7_4 = ((   2) / (m_4    )) * (Kxx_F2 + Kyy_F2)
    t8_4 = ((4m^2) / (m_2^3  )) * (Kxy_F2)
    t8_5 = ((  8m) / (m_2^3  )) * (Kxy_F2)
    MMDvar = (((t1_4 + t2_4) - (t3_4 + t5_4 + t6_4 + t7_4 + t8_4)) + ((t4_5 + t5_5 + t6_5 + t8_5) - t2_5)) - t2_6 # NOTE: typo in original paper: +t8 --> -t8

    #=
    MMDvar =
        ((      2) / (m^2 * (m-1)^2)) * (2*Kxx_e_F2 - Kxx_F2 + 2*Kyy_e_F2 - Kyy_F2) - # Units: (m*(m*[K])^2)/m^4 + (m^2*[K]^2)/m^4 = [K]^2/m + [K]^2/m^2 ~ [K]^2/m ~ O(1/m)
        ((   4m-6) / (m^3 * (m-1)^3)) * (e_Kxx_e^2 + e_Kyy_e^2) +                     # Units: (m^2*[K])^2*m/m^6 = [K]^2/m ~ O(1/m)
        ((4*(m-2)) / (m^3 * (m-1)^2)) * (2*Kxy_e_F2) -                                # Units: (m^2*[K])^2*m/m^6 = [K]/m ~ O(1/m). Note: used Kxy symmetry
        ((4*(m-3)) / (m^3 * (m-1)^2)) * (Kxy_F2) -                                    # Units: (m^2*[K]^2)*m/m^5 = [K]/m^2 ~ O(1/m^2)
        ((  8m-12) / (m^5 * (m-1)  )) * (e_Kxy_e^2) +                                 # Units: (m^2*[K])^2*m/m^6 = [K]/m ~ O(1/m)
        ((      8) / (m^4 * (m-1)  )) * ((e_Kxx_e + e_Kyy_e) * e_Kxy_e) -             # Units: (m^2*[K])^2/m^5   = [K]/m ~ O(1/m)
        ((      8) / (m^3 * (m-1)  )) * (e_Kxx_Kxy_e + Kyy_e'Kxy_e)                   # Units: (m*(m*[K])^2)/m^4 = [K]/m ~ O(1/m). Note: used Kxx, Kyy, Kxy symmetry
    =#

    MMDsq = mmd(res)

    return MMDsq, MMDvar
end

mmdvar(res::MMDVarResults) = mmd_and_mmdvar(res)[2]

####
#### Generic MMD using buffer matrices
####

mmd!(k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmd!(mmd_work(X,Y), k, X, Y)
mmd!(work, k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmd(kernel_mmd_stats!(work, k, X, Y))
mmdvar!(k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmdvar!(mmd_work(X,Y), k, X, Y)
mmdvar!(work, k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmdvar(kernel_mmdvar_stats!(work, k, X, Y))
mmd_and_mmdvar!(k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmd_and_mmdvar!(mmd_work(X,Y), k, X, Y)
mmd_and_mmdvar!(work, k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmd_and_mmdvar(kernel_mmdvar_stats!(work, k, X, Y))

function mmd_work(T::Type, sz::NTuple{2,Int})
    Kxx, Kyy, Kxy = ntuple(_ -> zeros(T, sz[2], sz[2]), 3)
    Kxx_e, Kyy_e, Kxy_e, Kyx_e = ntuple(_ -> zeros(T, sz[2]), 4)
    return @ntuple(Kxx, Kyy, Kxy, Kxx_e, Kyy_e, Kxy_e, Kyx_e)
end
mmd_work(sz::NTuple{2,Int}) = mmd_work(Float64, sz)
mmd_work(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = (@assert size(X) == size(Y); return mmd_work(T, size(X)))

function kernel_pairwise!(Kxy::AbstractMatrix{T}, k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, ::Val{skipdiag} = Val(false)) where {T,skipdiag}
    @assert size(X) == size(Y) && size(Kxy) == (size(X, 2), size(Y, 2))
    m = size(X, 2)
    if X === Y
        @inbounds for j = 1:m
            for i in (j + 1):m
                Kxy[i, j] = k(column_mse(X, Y, i, j))
            end
            Kxy[j, j] = skipdiag ? zero(T) : k(column_mse(X, Y, j, j))
            @simd for i in 1:j-1
                Kxy[i, j] = Kxy[j, i] # k(x, y) = k(y, x) for all x, y
            end
        end
    else
        @inbounds for j = 1:m
            for i in 1:j-1
                Kxy[i, j] = k(column_mse(X, Y, i, j))
            end
            Kxy[j, j] = skipdiag ? zero(T) : k(column_mse(X, Y, j, j))
            for i in j+1:m
                Kxy[i, j] = k(column_mse(X, Y, i, j))
            end
        end
    end
    return Kxy
end

function kernel_mmd_stats!(work, k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy = work
    m = size(X, 2)
    e_Kxx_e = sum(kernel_pairwise!(Kxx, k, X, X, Val(true)))
    e_Kyy_e = sum(kernel_pairwise!(Kyy, k, Y, Y, Val(true)))
    e_Kxy_e = sum(kernel_pairwise!(Kxy, k, X, Y, Val(true)))
    return MMDResults(T(m), e_Kxx_e, e_Kyy_e, e_Kxy_e)
end

function kernel_mmdvar_stats!(work, k::KernelFunction{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy, Kxx_e, Kyy_e, Kxy_e, Kyx_e = work

    kernel_pairwise!(Kxx, k, X, X, Val(true))
    kernel_pairwise!(Kyy, k, Y, Y, Val(true))
    kernel_pairwise!(Kxy, k, X, Y, Val(false))

    sum_columns!(Kxx_e, Kxx)
    sum_columns!(Kyy_e, Kyy)
    sum_columns!(Kxy_e, Kxy)
    sum_rows!(Kyx_e, Kxy)

    Kxx_F2, Kyy_F2, Kxy_F2 = frob_norm2(Kxx), frob_norm2(Kyy), frob_norm2(Kxy)
    e_Kxx_e, e_Kyy_e, e_Kxy_e = sum(Kxx), sum(Kyy), sum(Kxy)
    e_Kxx_Kxy_e, e_Kyy_Kyx_e = Kxx_e'Kxy_e, Kyy_e'Kyx_e
    Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2 = frob_norm2(Kxx_e), frob_norm2(Kyy_e), frob_norm2(Kxy_e), frob_norm2(Kyx_e)

    return MMDVarResults(T(size(X,2)), Kxx_F2, Kyy_F2, Kxy_F2, e_Kxx_e, e_Kyy_e, e_Kxy_e, Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2, e_Kxx_Kxy_e, e_Kyy_Kyx_e)
end

#=
for a in [3,5,8], m in 30:10:100
    k = Δ -> exp(-Δ)
    # sampleX = () -> randn(2,m)
    # sampleY = () -> ((1/√2) * [1 -1; 1 1] * [√a 0; 0 1/√a]) * randn(2,m)
    sampleX = () -> rand(2,m)
    sampleY = () -> [2-1/a; 1/a] .* rand(2,m)
    work = mmd_work((2, m))
    mmds = [mmd!(work, k, sampleX(), sampleY()) for _ in 1:50000]
    mmdvars = [mmdvar!(work, k, sampleX(), sampleY()) for _ in 1:50000]
    V, Vm, dV, rdV = m*var(mmds), m*mean(mmdvars), m^3*abs(mean(mmdvars) - var(mmds)), abs(mean(mmdvars) - var(mmds))/var(mmds)
    @show m, a, V, Vm, dV, rdV
end
=#

####
#### Generic MMD without buffers
####

mmd(k::KernelFunction, X::AbstractMatrix, Y::AbstractMatrix) = mmd(kernel_mmd_stats(k, X, Y))
mmdvar(k::KernelFunction, X::AbstractMatrix, Y::AbstractMatrix) = mmdvar(kernel_mmdvar_stats(k, X, Y))
mmd_and_mmdvar(k::KernelFunction, X::AbstractMatrix, Y::AbstractMatrix) = mmd_and_mmdvar(kernel_mmdvar_stats(k, X, Y))

function kernel_pairwise_sum(k::KernelFunction, X::AbstractMatrix, Y::AbstractMatrix, ::Val{skipdiag} = Val(false)) where {skipdiag}
    @assert size(X) == size(Y)
    m  = size(X, 2)
    Tk = typeof(k(zero(promote_type(eltype(X), eltype(Y)))))
    Σ  = zero(Tk)
    if X === Y
        @inbounds for j = 1:m
            for i in (j + 1):m
                Σ += 2*k(column_mse(X, Y, i, j)) # k(x, y) = k(y, x) for all x, y
            end
            !skipdiag && (Σ += k(column_mse(X, Y, j, j)))
        end
    else
        @inbounds for j = 1:m
            for i in 1:j-1
                Σ += k(column_mse(X, Y, i, j))
            end
            !skipdiag && (Σ += k(column_mse(X, Y, j, j)))
            for i in j+1:m
                Σ += k(column_mse(X, Y, i, j))
            end
        end
    end
    return Σ
end

function kernel_mmd_stats(k::KernelFunction, X::AbstractMatrix, Y::AbstractMatrix)
    # Ref: http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    @assert size(X) == size(Y)
    e_Kxx_e = kernel_pairwise_sum(k, X, X, Val(true))
    e_Kyy_e = kernel_pairwise_sum(k, Y, Y, Val(true))
    e_Kxy_e = kernel_pairwise_sum(k, X, Y, Val(true))
    return MMDResults(typeof(e_Kxy_e)(size(X,2)), e_Kxx_e, e_Kyy_e, e_Kxy_e)
end

function kernel_mmdvar_stats(k::KernelFunction, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)
    m  = size(X, 2)
    Tk = typeof(k(zero(promote_type(eltype(X), eltype(Y)))))
    Kxx_F2 = Kyy_F2 = Kxy_F2 = e_Kxx_e = e_Kyy_e = e_Kxy_e = Kxx_e_F2 = Kyy_e_F2 = Kxy_e_F2 = Kyx_e_F2 = e_Kxx_Kxy_e = e_Kyy_Kyx_e = zero(Tk)

    @inbounds for j = 1:m
        Kxx_e_j = Kyy_e_j = Kxy_e_j = Kyx_e_j = zero(Tk)
        @inbounds @simd for i in 1:j-1
            kxx_ij = k(column_mse(X, X, i, j))
            kyy_ij = k(column_mse(Y, Y, i, j))
            kxy_ij = k(column_mse(X, Y, i, j))
            kyx_ij = k(column_mse(Y, X, i, j))
            Kxx_F2 += 2 * kxx_ij * kxx_ij
            Kyy_F2 += 2 * kyy_ij * kyy_ij
            Kxy_F2 += kxy_ij * kxy_ij
            e_Kxx_e += 2 * kxx_ij
            e_Kyy_e += 2 * kyy_ij
            e_Kxy_e += kxy_ij
            Kxx_e_j += kxx_ij
            Kyy_e_j += kyy_ij
            Kxy_e_j += kyx_ij
            Kyx_e_j += kxy_ij
        end
        kxy_jj = k(column_mse(X, Y, j, j))
        Kxy_F2  += kxy_jj * kxy_jj
        e_Kxy_e += kxy_jj
        Kxy_e_j += kxy_jj
        Kyx_e_j += kxy_jj
        @inbounds @simd for i in j+1:m
            kxx_ij = k(column_mse(X, X, i, j))
            kyy_ij = k(column_mse(Y, Y, i, j))
            kxy_ij = k(column_mse(X, Y, i, j))
            kyx_ij = k(column_mse(Y, X, i, j))
            # Kxx_F2 += kxx_ij * kxx_ij
            # Kyy_F2 += kyy_ij * kyy_ij
            Kxy_F2 += kxy_ij * kxy_ij
            # e_Kxx_e += kxx_ij
            # e_Kyy_e += kyy_ij
            e_Kxy_e += kxy_ij
            Kxx_e_j += kxx_ij
            Kyy_e_j += kyy_ij
            Kxy_e_j += kyx_ij
            Kyx_e_j += kxy_ij
        end
        Kxx_e_F2 += Kxx_e_j * Kxx_e_j
        Kyy_e_F2 += Kyy_e_j * Kyy_e_j
        Kxy_e_F2 += Kxy_e_j * Kxy_e_j
        Kyx_e_F2 += Kyx_e_j * Kyx_e_j
        e_Kxx_Kxy_e += Kxx_e_j * Kxy_e_j
        e_Kyy_Kyx_e += Kyy_e_j * Kyx_e_j
    end

    return MMDVarResults(Tk(m), Kxx_F2, Kyy_F2, Kxy_F2, e_Kxx_e, e_Kyy_e, e_Kxy_e, Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2, e_Kxx_Kxy_e, e_Kyy_Kyx_e)
end

####
#### Flux differentiable MMD kernel matrices
####

function _mmd_flux_kernel_matrices(X::AbstractMatrix, Y::AbstractMatrix)
    Kxx, Kyy, Kxy = X'X, Y'Y, X'Y
    xx, yy = batcheddiag(Kxx), batcheddiag(Kyy)
    Threads.@threads for j in 1:size(Kxx,2)
        @avx for i in 1:size(Kxx,1)
            Kxx[i,j] = 2 * Kxx[i,j] - xx[i] - xx[j]
            Kyy[i,j] = 2 * Kyy[i,j] - yy[i] - yy[j]
            Kxy[i,j] = 2 * Kxy[i,j] - xx[i] - yy[j]
        end
    end
    fast_exp!(Kxx)
    fast_exp!(Kyy)
    fast_exp!(Kxy)

    @ntuple(Kxx, Kyy, Kxy)
end

#=
function _mmd_flux_kernel_matrices(X::AbstractMatrix, Y::AbstractMatrix)
    Kxx, Kyy, Kxy = X'X, Y'Y, X'Y
    xx, yy = batcheddiag(Kxx), batcheddiag(Kyy)
    Kxx .= 2 .* Kxx .- xx .- xx'
    Kyy .= 2 .* Kyy .- yy .- yy'
    Kxy .= 2 .* Kxy .- xx .- yy'
    fast_exp!(Kxx)
    fast_exp!(Kyy)
    fast_exp!(Kxy)

    @ntuple(Kxx, Kyy, Kxy)
end
=#

Zygote.@adjoint function _mmd_flux_kernel_matrices(X::AbstractMatrix, Y::AbstractMatrix)
    # Store kernel matrices for reverse pass
    @unpack Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices(X, Y)

    return @ntuple(Kxx, Kyy, Kxy), function(Δ) # much faster + much less memory usage
        # dK_dX
        Δ_buf = Δ.Kxx .* Kxx
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        add_transpose!(Δ_buf)
        mul_buf = X * Δ_buf
        dK_dX = 2 .* (mul_buf .- X .* (Δ_buf_colsum' .+ Δ_buf_rowsum))

        # dK_dX/dK_dY cross-terms
        Δ_buf .= Δ.Kxy .* Kxy
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        mul!(mul_buf, X, Δ_buf)
        dK_dY = 2 .* (mul_buf .- Y .* Δ_buf_rowsum)

        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        self_transpose!(Δ_buf)
        mul!(mul_buf, Y, Δ_buf)
        dK_dX .+= 2 .* (mul_buf .- X .* Δ_buf_colsum')

        # dK_dY
        Δ_buf .= Δ.Kyy .* Kyy
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        add_transpose!(Δ_buf)
        mul!(mul_buf, Y, Δ_buf)
        dK_dY .+= 2 .* (mul_buf .- Y .* (Δ_buf_colsum' .+ Δ_buf_rowsum))

        return dK_dX, dK_dY
    end
    #=
    return @ntuple(Kxx, Kyy, Kxy), function(Δ) # moderately faster + less memory usage
        # dK_dX
        Δ_buf1 = Δ.Kxx .* Kxx
        Δ_buf2 = Δ.Kxy .* Kxy
        mul_buf1 = X * (Δ_buf1 .+ Δ_buf1')
        mul_buf2 = Y * Δ_buf2'
        dK_dX = 2 .* (mul_buf1 .+ mul_buf2 .- X .* (sum(Δ_buf1; dims = 2)' .+ sum(Δ_buf1; dims = 1) .+ sum(Δ_buf2; dims = 2)'))

        # dK_dY
        Δ_buf1 .= Δ.Kyy .* Kyy # Δ_buf2 same as above
        mul!(mul_buf1, Y, Δ_buf1 .+ Δ_buf1')
        mul!(mul_buf2, X, Δ_buf2 )
        dK_dY = 2 .* (mul_buf1 .+ mul_buf2 .- Y .* (sum(Δ_buf1; dims = 2)' .+ sum(Δ_buf1; dims = 1) .+ sum(Δ_buf2; dims = 1)))

        return dK_dX, dK_dY
    end
    =#
    #=
    return @ntuple(Kxx, Kyy, Kxy), function(Δ) # moderately slower + more memory usage
        Δ_Kxx = Δ.Kxx .* Kxx
        Δ_Kyy = Δ.Kyy .* Kyy
        Δ_Kxy = Δ.Kxy .* Kxy        
        dK_dX = 2 .* ((X * (Δ_Kxx .+ Δ_Kxx')) .+ (Y * Δ_Kxy') .- X .* (sum(Δ_Kxx; dims = 2)' .+ sum(Δ_Kxx; dims = 1) .+ sum(Δ_Kxy; dims = 2)'))
        dK_dY = 2 .* ((Y * (Δ_Kyy .+ Δ_Kyy')) .+ (X * Δ_Kxy)  .- Y .* (sum(Δ_Kyy; dims = 2)' .+ sum(Δ_Kyy; dims = 1) .+ sum(Δ_Kxy; dims = 1)))
        return dK_dX, dK_dY
    end
    =#
end

function _mmd_flux_kernel_matrices_inner(X::AbstractArray{<:Any,3}, Y::AbstractArray{<:Any,3})
    Kxx = NNlib.batched_mul(NNlib.batched_adjoint(X), X)
    Kyy = NNlib.batched_mul(NNlib.batched_adjoint(Y), Y)
    Kxy = NNlib.batched_mul(NNlib.batched_adjoint(X), Y)
    xx  = batcheddiag(Kxx)
    yy  = batcheddiag(Kyy)
    Threads.@sync for k in 1:size(Kxx,3) #TODO can be improved; k is usually small
        Threads.@spawn begin
            @avx for j in 1:size(Kxx,2), i in 1:size(Kxx,1)
                Kxx[i,j,k] = 2 * Kxx[i,j,k] - xx[i,1,k] - xx[j,1,k]
                Kyy[i,j,k] = 2 * Kyy[i,j,k] - yy[i,1,k] - yy[j,1,k]
                Kxy[i,j,k] = 2 * Kxy[i,j,k] - xx[i,1,k] - yy[j,1,k]
            end
        end
    end
    fast_exp!(Kxx)
    fast_exp!(Kyy)
    fast_exp!(Kxy)

    @ntuple(Kxx, Kyy, Kxy)
end

#=
function _mmd_flux_kernel_matrices_inner(X::AbstractArray{<:Any,3}, Y::AbstractArray{<:Any,3})
    Kxx = NNlib.batched_mul(NNlib.batched_adjoint(X), X)
    Kyy = NNlib.batched_mul(NNlib.batched_adjoint(Y), Y)
    Kxy = NNlib.batched_mul(NNlib.batched_adjoint(X), Y)
    xx  = batcheddiag(Kxx)
    yy  = batcheddiag(Kyy)
    xxT = NNlib.batched_adjoint(xx)
    yyT = NNlib.batched_adjoint(yy)
    Kxx .= 2 .* Kxx .- xx .- xxT
    Kyy .= 2 .* Kyy .- yy .- yyT
    Kxy .= 2 .* Kxy .- xx .- yyT
    fast_exp!(Kxx)
    fast_exp!(Kyy)
    fast_exp!(Kxy)

    @ntuple(Kxx, Kyy, Kxy)
end
=#

function _mmd_flux_kernel_matrices(X::AbstractArray{<:Any,3}, Y::AbstractArray{<:Any,3})
    @unpack Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices_inner(X, Y)
    Kxx = dropdims(mean(Kxx; dims = 3); dims = 3)
    Kyy = dropdims(mean(Kyy; dims = 3); dims = 3)
    Kxy = dropdims(mean(Kxy; dims = 3); dims = 3)
    @ntuple(Kxx, Kyy, Kxy)
end

Zygote.@adjoint function _mmd_flux_kernel_matrices(X::AbstractArray{<:Any,3}, Y::AbstractArray{<:Any,3})
    # Store kernel matrices for reverse pass
    @unpack Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices_inner(X, Y)
    out = (
        Kxx = dropdims(mean(Kxx; dims = 3); dims = 3),
        Kyy = dropdims(mean(Kyy; dims = 3); dims = 3),
        Kxy = dropdims(mean(Kxy; dims = 3); dims = 3),
    )
    return out, function(Δ) # much faster + much less memory usage
        T = x -> NNlib.batched_adjoint(x)
        nbw = size(X,3)

        # dK_dX
        Δ_buf = Δ.Kxx .* Kxx ./ nbw
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        add_transpose!(Δ_buf)
        mul_buf = NNlib.batched_mul(X, Δ_buf)
        dK_dX = 2 .* (mul_buf .- X .* (T(Δ_buf_colsum) .+ Δ_buf_rowsum))

        # dK_dX/dK_dY cross-terms
        Δ_buf .= Δ.Kxy .* Kxy ./ nbw
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        NNlib.batched_mul!(mul_buf, X, Δ_buf)
        dK_dY = 2 .* (mul_buf .- Y .* Δ_buf_rowsum)

        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        NNlib.batched_mul!(mul_buf, Y, T(Δ_buf))
        dK_dX .+= 2 .* (mul_buf .- X .* T(Δ_buf_colsum))

        # dK_dY
        Δ_buf .= Δ.Kyy .* Kyy ./ nbw
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        add_transpose!(Δ_buf)
        NNlib.batched_mul!(mul_buf, Y, Δ_buf)
        dK_dY .+= 2 .* (mul_buf .- Y .* (T(Δ_buf_colsum) .+ Δ_buf_rowsum))

        return dK_dX, dK_dY
        # return dropdims(mean(dK_dX; dims = 3); dims = 3), dropdims(mean(dK_dY; dims = 3); dims = 3)
    end
    #=
    return out, function(Δ) # moderately faster + less memory usage
        T = x -> permutedims(x, (2,1,3))
        nbw = size(X,3)

        # dK_dX
        Δ_buf1 = Δ.Kxx .* Kxx ./ nbw
        Δ_buf2 = Δ.Kxy .* Kxy ./ nbw
        Δ_buf1_rowsum = sum(Δ_buf1; dims = 1)
        Δ_buf1_colsum = sum(Δ_buf1; dims = 2)
        Δ_buf2_colsum = sum(Δ_buf2; dims = 2)
        add_transpose!(Δ_buf1)
        mul_buf1 = NNlib.batched_mul(X, Δ_buf1)
        mul_buf2 = NNlib.batched_mul(Y, T(Δ_buf2))
        dK_dX = 2 .* (mul_buf1 .+ mul_buf2 .- X .* (T(Δ_buf1_colsum) .+ Δ_buf1_rowsum .+ T(Δ_buf2_colsum)))
        # dK_dX = -2 .* muladd.(X, T(Δ_buf1_colsum) .+ Δ_buf1_rowsum .+ T(Δ_buf2_colsum), .-mul_buf1 .- mul_buf2)

        # dK_dY
        Δ_buf1 .= Δ.Kyy .* Kyy ./ nbw # Δ_buf2 same as above
        Δ_buf1_rowsum = sum(Δ_buf1; dims = 1)
        Δ_buf1_colsum = sum(Δ_buf1; dims = 2)
        Δ_buf2_rowsum = sum(Δ_buf2; dims = 1)
        add_transpose!(Δ_buf1)
        NNlib.batched_mul!(mul_buf1, Y, Δ_buf1)
        NNlib.batched_mul!(mul_buf2, X, Δ_buf2)
        dK_dY = 2 .* (mul_buf1 .+ mul_buf2 .- Y .* (T(Δ_buf1_colsum) .+ Δ_buf1_rowsum .+ Δ_buf2_rowsum))
        # dK_dY = -2 .* muladd.(Y, T(Δ_buf1_colsum) .+ Δ_buf1_rowsum .+ Δ_buf2_rowsum, .-mul_buf1 .- mul_buf2)

        return dK_dX, dK_dY
        # return dropdims(mean(dK_dX; dims = 3); dims = 3), dropdims(mean(dK_dY; dims = 3); dims = 3)
    end
    =#
    #=
    return out, function(Δ) # moderately slower + more memory usage
        T = x -> permutedims(x, (2,1,3))
        nbw = size(X,3)
        Δ_Kxx = Δ.Kxx .* Kxx ./ nbw
        Δ_Kyy = Δ.Kyy .* Kyy ./ nbw
        Δ_Kxy = Δ.Kxy .* Kxy ./ nbw
        dK_dX = 2 .* (NNlib.batched_mul(X, Δ_Kxx .+ T(Δ_Kxx)) .+ NNlib.batched_mul(Y, T(Δ_Kxy)) .- X .* (T(sum(Δ_Kxx; dims = 2)) .+ sum(Δ_Kxx; dims = 1) .+ T(sum(Δ_Kxy; dims = 2))))
        dK_dY = 2 .* (NNlib.batched_mul(Y, Δ_Kyy .+ T(Δ_Kyy)) .+ NNlib.batched_mul(X,   Δ_Kxy ) .- Y .* (T(sum(Δ_Kyy; dims = 2)) .+ sum(Δ_Kyy; dims = 1) .+   sum(Δ_Kxy; dims = 1)))
        return dK_dX, dK_dY
    end
    =#
end

#=
# Testing adjoint for _mmd_flux_kernel_matrices
let
    Random.seed!(0)

    # Dummy version for Zygote to auto-diff through
    function _kernel_mats(X::AbstractMatrix, Y::AbstractMatrix)
        XX, YY, XY = X'X, Y'Y, X'Y
        xx, yy = batcheddiag(XX), batcheddiag(YY)
        Kxx = exp.(2 .* XX .- xx .- xx')
        Kyy = exp.(2 .* YY .- yy .- yy')
        Kxy = exp.(2 .* XY .- xx .- yy')
        @ntuple(Kxx, Kyy, Kxy)
    end
    function _kernel_mats(X::AbstractArray{<:Any,3}, Y::AbstractArray{<:Any,3})
        XX = NNlib.batched_mul(NNlib.batched_adjoint(X), X)
        YY = NNlib.batched_mul(NNlib.batched_adjoint(Y), Y)
        XY = NNlib.batched_mul(NNlib.batched_adjoint(X), Y)
        xx = batcheddiag(XX)
        yy = batcheddiag(YY)

        T = x -> permutedims(x, (2,1,3))
        Kxx = exp.(2 .* XX .- xx .- T(xx))
        Kyy = exp.(2 .* YY .- yy .- T(yy))
        Kxy = exp.(2 .* XY .- xx .- T(yy))
        # @ntuple(Kxx, Kyy, Kxy)
        return (
            Kxx = dropdims(mean(Kxx; dims = 3); dims = 3),
            Kyy = dropdims(mean(Kyy; dims = 3); dims = 3),
            Kxy = dropdims(mean(Kxy; dims = 3); dims = 3),
        )
    end

    n, m = 128, 64 #2048
    for nbw in [1,4]
        arrsize = nbw == 1 ? (n,m) : (n,m,nbw)
        Ksize = (m,m) #nbw == 1 ? (m,m) : (m,m,nbw)
        A, B = rand(arrsize...), rand(arrsize...)
        Δ = (Kxx = rand(Ksize...), Kyy = rand(Ksize...), Kxy = rand(Ksize...))

        _y, _back = Zygote.pullback(_kernel_mats, A, B)
        _dyA, _dyB = _back(Δ)

        y, back = Zygote.pullback(_mmd_flux_kernel_matrices, A, B)
        dyA, dyB = back(Δ)

        @assert all(values(_y) .≈ values(y))
        @assert _dyA ≈ dyA
        @assert _dyB ≈ dyB

        #=
        for f in (_mmd_flux_kernel_matrices,) #_kernel_mats
            print("$f call:   "); @btime $f($A, $B)
            print("$f forward:"); _, back = @btime Zygote.pullback($f, $A, $B)
            print("$f reverse:"); @btime $back($Δ)
        end
        =#
    end
end
=#

#=
# mmd_flux and mmdvar_flux speed testing
let
    Random.seed!(0)
    n, m = 128, 3072
    for nbw in [1,4]
        X, Y = rand(n,m), rand(n,m)
        logsigma = rand(nbw, n)
        for f in (mmd_flux, mmdvar_flux) #mmd_and_mmdvar_flux returns two outputs
            print("$f call:   \t"); @btime $f($logsigma, $X, $Y)
            print("$f forward:\t"); y, back = @btime Zygote.pullback(logσ -> $f(logσ, $X, $Y), $logsigma)
            print("$f value:  \t$y\n")
            print("$f reverse:\t"); @btime $back(1.0)
        end
    end
end
=#

#=
# Various useful adjoints
let
    let
        A = rand(3,4)
        Δ = rand(3,4)
        f = (A) -> exp.(A)
        y, back = Zygote.pullback(f, A)
        dyA, = back(Δ)
        @assert y == f(A)
        @assert dyA == Δ .* exp.(A)
    end

    let
        A = rand(3,4)
        B = rand(3,4)
        Δ = rand(4,4)
        f = (A,B) -> exp.(A'B)
        y, back = Zygote.pullback(f, A, B)
        dyA, dyB = back(Δ)
        @assert y == f(A,B)
        @assert dyA == B * (Δ .* exp.(A'B))'
        @assert dyB == A * (Δ .* exp.(A'B))
    end

    let
        A = rand(3,4)
        Δ = rand(4)
        f = (A) -> exp.(diag(A'A))
        y, back = Zygote.pullback(f, A)
        dyA, = back(Δ)
        @assert y == f(A)
        @assert dyA == 2 .* A .* (Δ .* exp.(diag(A'A)))'
    end

    let
        A = rand(3,4)
        B = rand(3,4)
        Δ = rand(4)
        f = (A,B) -> exp.(diag(A'B))
        y, back = Zygote.pullback(f, A, B)
        dyA, dyB = back(Δ)
        @assert y == f(A,B)
        @assert dyA == B .* (Δ .* exp.(diag(A'B)))'
        @assert dyB == A .* (Δ .* exp.(diag(A'B)))'
    end

    let
        Random.seed!(0)
        A = rand(3,4)
        B = rand(3,4)
        Δ = rand(4,4)
        f = (A,B) -> exp.(diag(A'B)) .* ones(1,4)
        y, back = Zygote.pullback(f, A, B)
        dyA, dyB = back(Δ)

        _y = f(A,B)
        _dyA = B .* sum(Δ .* f(A,B); dims=2)'
        @assert y == _y
        @assert dyA == _dyA
    end

    let
        Random.seed!(0)
        A = rand(3,4)
        Δ = rand(4,4)
        f = (A) -> exp.(diag(A'A)) .* ones(1,4)
        y, back = Zygote.pullback(f, A)
        dyA, = back(Δ)

        _y = f(A)
        _dyA = 2 .* A .* sum(Δ .* f(A); dims=2)'
        @assert y == _y
        @assert dyA ≈ _dyA
    end

    let
        Random.seed!(0)
        A = rand(3,4)
        B = rand(3,4)
        Δ = rand(4,4)
        f = (A,B) -> exp.(A'B .+ diag(A'B))
        y, back = Zygote.pullback(f, A, B)
        dyA, dyB = back(Δ)

        _y = f(A,B)
        _dyA = B * (Δ .* _y)' .+ B .* sum(Δ .* _y; dims = 2)'
        _dyB = A * (Δ .* _y)  .+ A .* sum(Δ .* _y; dims = 2)'
        @assert y == _y
        @assert dyA == _dyA
        @assert dyB == _dyB
    end

    let
        A = rand(3,4)
        B = rand(3,4)
        Δ = rand(4)
        f = (A) -> exp.(diag(A'A))
        y, back = Zygote.pullback(f, A)
        dyA, = back(Δ)
        @assert y == f(A)
        @assert dyA == 2 .* A .* (Δ .* exp.(diag(A'A)))'
    end

    let
        Random.seed!(0)
        A = rand(3,4)
        Δ = rand(4,4)
        f = (A) -> exp.(diag(A'A)) .* ones(1,4)
        y, back = Zygote.pullback(f, A)
        dyA, = back(Δ)

        _y = f(A)
        _dyA = 2 .* A .* sum(Δ .* f(A); dims=2)'
        @assert y == _y
        @assert dyA ≈ _dyA
    end

    let # Symm.
        Random.seed!(0)
        A = rand(3,4)
        Δ = rand(4,4)
        f = (A) -> exp.(2 .* (A'A) .- diag(A'A) .- diag(A'A)')
        y, back = Zygote.pullback(f, A)
        dyA, = back(Δ)

        _y = f(A)
        _dy = Δ .* _y
        _dyA = 2 .* ((A * _dy) .+ (A * _dy') .- A .* sum(_dy; dims = 2)' .- A .* sum(_dy; dims = 1))
        @assert y == _y
        @assert dyA ≈ _dyA
    end

    let # Anti-symm.
        Random.seed!(0)
        A = rand(3,4)
        B = rand(3,4)
        Δ = rand(4,4)
        f = (A,B) -> exp.(2 .* (A'B) .- diag(A'A) .- diag(B'B)')
        y, back = Zygote.pullback(f, A, B)
        dyA, dyB = back(Δ)

        _y = f(A,B)
        _dy = Δ .* _y
        _dyA = 2 .* (B * _dy' .- A .* sum(_dy; dims = 2)')
        _dyB = 2 .* (A * _dy  .- B .* sum(_dy; dims = 1))
        @assert y == _y
        @assert dyA ≈ _dyA
        @assert dyB ≈ _dyB
    end
end
=#

####
#### Kernel matrices with generic kernel function
####

function mmd_flux_kernel_matrices(k::KernelFunction, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)

    n = size(X,1)
    XX, XY, YY = X'X, X'Y, Y'Y
    xx, yy = batcheddiag(XX), batcheddiag(YY) # squared norms on diagonal
    Kxx = k.((xx .- 2 .* XX .+ xx')./n) # note: mean is over data length n, not number of data m
    Kyy = k.((yy .- 2 .* YY .+ yy')./n)
    Kxy = k.((xx .- 2 .* XY .+ yy')./n)

    return @ntuple(Kxx, Kyy, Kxy)
end

####
#### Kernel matrices specialized for sums of exponential kernels
####

# Bandwidth array `logsigma` may be:
#   1D `nbandwidth`-length vector
#   2D `nbandwidth x n` matrix, where n == size(X,1) == size(Y,1)
#   3D `n x 1 x nbandwidth` array (not meant for direct use)

function mmd_flux_kernel_matrices(logsigma::AbstractArray{<:Any,3}, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)

    n, m = size(X)
    gamma = @. inv(2n * exp(2 * logsigma)) # gamma = 1/2sigma^2 = 1/2exp(2logsigma)

    T = promote_type(eltype(X), eltype(Y), eltype(gamma))
    Kxx = zeros(T, m, m)
    Kyy = zeros(T, m, m)
    Kxy = zeros(T, m, m)
    ngamma = size(gamma, 3)
    for k in 1:ngamma
        _g = sqrt.(gamma[:,1,k])
        U = _g .* X
        V = _g .* Y
        _Kxx, _Kyy, _Kxy = _mmd_flux_kernel_matrices(U, V)
        Kxx += _Kxx
        Kyy += _Kyy
        Kxy += _Kxy
    end
    Kxx /= ngamma
    Kyy /= ngamma
    Kxy /= ngamma

    return @ntuple(Kxx, Kyy, Kxy)
end
mmd_flux_kernel_matrices(logsigma::AbstractMatrix, args...) = mmd_flux_kernel_matrices(reshape(permutedims(logsigma), size(logsigma,2), 1, :), args...) # reshape for broadcasting
mmd_flux_kernel_matrices(logsigma::AbstractVector, args...) = mmd_flux_kernel_matrices(reshape(logsigma, 1, 1, length(logsigma)), args...) # reshape for broadcasting

function mmd_flux_kernel_matrices_batched(logsigma::AbstractArray{<:Any,3}, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)
    n, m = size(X)
    gamma = @. inv(2n * exp(2 * logsigma)) # gamma = 1/2sigma^2 = 1/2exp(2logsigma)
    _g = sqrt.(gamma)
    U, V = _g .* X, _g .* Y
    return _mmd_flux_kernel_matrices(U, V)
end
mmd_flux_kernel_matrices_batched(logsigma::AbstractMatrix, args...) = mmd_flux_kernel_matrices_batched(reshape(permutedims(logsigma), size(logsigma,2), 1, :), args...) # reshape for broadcasting
mmd_flux_kernel_matrices_batched(logsigma::AbstractVector, args...) = mmd_flux_kernel_matrices_batched(reshape(logsigma, 1, 1, length(logsigma)), args...) # reshape for broadcasting

#=
let # Speed testing of mmd_flux_kernel_matrices
    n, m = 128, 2048 #64
    X = randn(n,m)
    Y = randn(n,m)
    Δ = (Kxx = rand(m,m), Kyy = rand(m,m), Kxy = rand(m,m))
    for nbw in [1,4]#2,4,8,16,32]
        @show n, m, nbw
        logsigma = randn(nbw, n)

        @assert all(values(mmd_flux_kernel_matrices(logsigma,X,Y)) .≈ values(mmd_flux_kernel_matrices_batched(logsigma,X,Y)))

        _y, _back = Zygote.pullback((_X,_Y) -> mmd_flux_kernel_matrices(logsigma,_X,_Y), X, Y)
        _dyA, _dyB = _back(Δ)

        y, back = Zygote.pullback((_X,_Y) -> mmd_flux_kernel_matrices_batched(logsigma,_X,_Y), X, Y)
        dyA, dyB = back(Δ)

        @assert all(values(_y) .≈ values(y))
        @assert _dyA ≈ dyA
        @assert _dyB ≈ dyB

        for f in (mmd_flux_kernel_matrices, mmd_flux_kernel_matrices_batched)
            print("$f call:   "); @btime $f($logsigma, $X, $Y)
            print("$f forward:"); _, back = @btime Zygote.pullback((_X,_Y) -> $f($logsigma,_X,_Y), $X, $Y)
            print("$f reverse:"); @btime $back($Δ)
        end
        #=
        =#
    end
end;
=#

#=
let # Consistency between vectors/matrices of logsigma
    logsigma1 = randn(4)
    logsigma2 = repeat(logsigma1, 1, 10)
    X, Y = randn(10,4), randn(10,4)
    out1 = mmd_flux_kernel_matrices(logsigma1, X, Y)
    out2 = mmd_flux_kernel_matrices(logsigma2, X, Y)
    isapprox.(values(out1), values(out2))
end
=#

####
#### Compute U-statistics using kernel matrices
####

function mmd_flux_u_statistic(Kxx, Kyy, Kxy)
    @assert size(Kxx) == size(Kyy) == size(Kxy)

    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    T = promote_type(eltype(Kxx), eltype(Kyy), eltype(Kxy))
    m = T(size(Kxx,1))
    e_Kxx_e = sum(Kxx) - m # assumes k(0) == 1 --> tr(Kxx) = m
    e_Kyy_e = sum(Kyy) - m # assumes k(0) == 1 --> tr(Kyy) = m
    e_K̃xy_e = sum(Kxy) - tr(Kxy)
    MMDsq = (e_Kxx_e + e_Kyy_e - 2*e_K̃xy_e) / (m*(m-1))

    return MMDsq
end

function mmdvar_flux_u_statistic(Kxx, Kyy, Kxy)
    @assert size(Kxx) == size(Kyy) == size(Kxy)

    # Var[MMD²_U]: Variantes of U-statistic MMD estimator
    #   See: https://arxiv.org/pdf/1906.02104.pdf

    T = promote_type(eltype(Kxx), eltype(Kyy), eltype(Kxy))
    m = T(size(Kxx,1)) # m must be float, else denominator can overflow
    m_2 = m*(m-1)
    m_3 = m*(m-1)*(m-2)
    m_4 = m*(m-1)*(m-2)*(m-3)

    e_Kxx_e = sum(Kxx) - m # assumes k(0) == 1
    e_Kyy_e = sum(Kyy) - m # assumes k(0) == 1
    e_Kxy_e = sum(Kxy)
    Kxx_F2 = sum(abs2, Kxx) - m # assumes k(0) == 1
    Kyy_F2 = sum(abs2, Kyy) - m # assumes k(0) == 1
    Kxy_F2 = sum(abs2, Kxy)
    Kxx_e = dropdims(sum(Kxx; dims = 2); dims = 2) .- 1 # assumes k(0) == 1
    Kyy_e = dropdims(sum(Kyy; dims = 2); dims = 2) .- 1 # assumes k(0) == 1
    Kxy_e = dropdims(sum(Kxy; dims = 2); dims = 2)
    Kyx_e = dropdims(sum(Kxy; dims = 1); dims = 1)
    Kxx_e_F2 = sum(abs2, Kxx_e)
    Kyy_e_F2 = sum(abs2, Kyy_e)
    Kxy_e_F2 = sum(abs2, Kxy_e)
    Kyx_e_F2 = sum(abs2, Kyx_e)
    e_Kxx_Kxy_e = dot(Kxx_e, Kxy_e)
    e_Kyy_Kyx_e = dot(Kyy_e, Kyx_e)

    t1_4 = ((   4) / (m_4    )) * (Kxx_e_F2 + Kyy_e_F2)
    t2_4 = ((4m^2) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t2_5 = ((  4m) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t2_6 = ((   4) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t3_4 = ((   8) / (m*m_3  )) * (e_Kxx_Kxy_e + e_Kyy_Kyx_e)
    t4_5 = ((   8) / (m^2*m_3)) * ((e_Kxx_e + e_Kyy_e) * e_Kxy_e)
    t5_4 = ((  4m) / (m_2*m_4)) * (e_Kxx_e^2 + e_Kyy_e^2)
    t5_5 = ((   6) / (m_2*m_4)) * (e_Kxx_e^2 + e_Kyy_e^2)
    t6_4 = ((  8m) / (m_2^3  )) * (e_Kxy_e^2)
    t6_5 = ((  12) / (m_2^3  )) * (e_Kxy_e^2)
    t7_4 = ((   2) / (m_4    )) * (Kxx_F2 + Kyy_F2)
    t8_4 = ((4m^2) / (m_2^3  )) * (Kxy_F2)
    t8_5 = ((  8m) / (m_2^3  )) * (Kxy_F2)
    MMDvar = (((t1_4 + t2_4) - (t3_4 + t5_4 + t6_4 + t7_4 + t8_4)) + ((t4_5 + t5_5 + t6_5 + t8_5) - t2_5)) - t2_6 # NOTE: typo in original paper: t8_* sign flips

    return MMDvar
end

function mmd_and_mmdvar_flux_u_statistic(Kxx, Kyy, Kxy)
    @assert size(Kxx) == size(Kyy) == size(Kxy)

    T = promote_type(eltype(Kxx), eltype(Kyy), eltype(Kxy))
    m = T(size(Kxx,1))
    m_2 = m*(m-1)
    m_3 = m*(m-1)*(m-2)
    m_4 = m*(m-1)*(m-2)*(m-3)
    
    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    e_Kxx_e = sum(Kxx) - m # assumes k(0) == 1 --> tr(Kxx) = m
    e_Kyy_e = sum(Kyy) - m # assumes k(0) == 1 --> tr(Kyy) = m
    e_Kxy_e = sum(Kxy)
    e_K̃xy_e = e_Kxy_e - tr(Kxy)

    MMDsq = (e_Kxx_e + e_Kyy_e - 2*e_K̃xy_e) / (m*(m-1))

    # Var[MMD²_U]: Variantes of U-statistic MMD estimator
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    Kxx_F2 = sum(abs2, Kxx) - m # assumes k(0) == 1
    Kyy_F2 = sum(abs2, Kyy) - m # assumes k(0) == 1
    Kxy_F2 = sum(abs2, Kxy)
    Kxx_e = dropdims(sum(Kxx; dims = 2); dims = 2) .- 1 # assumes k(0) == 1
    Kyy_e = dropdims(sum(Kyy; dims = 2); dims = 2) .- 1 # assumes k(0) == 1
    Kxy_e = dropdims(sum(Kxy; dims = 2); dims = 2)
    Kyx_e = dropdims(sum(Kxy; dims = 1); dims = 1)
    Kxx_e_F2 = sum(abs2, Kxx_e)
    Kyy_e_F2 = sum(abs2, Kyy_e)
    Kxy_e_F2 = sum(abs2, Kxy_e)
    Kyx_e_F2 = sum(abs2, Kyx_e)
    e_Kxx_Kxy_e = dot(Kxx_e, Kxy_e)
    e_Kyy_Kyx_e = dot(Kyy_e, Kyx_e)

    t1_4 = ((   4) / (m_4    )) * (Kxx_e_F2 + Kyy_e_F2)
    t2_4 = ((4m^2) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t2_5 = ((  4m) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t2_6 = ((   4) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t3_4 = ((   8) / (m*m_3  )) * (e_Kxx_Kxy_e + e_Kyy_Kyx_e)
    t4_5 = ((   8) / (m^2*m_3)) * ((e_Kxx_e + e_Kyy_e) * e_Kxy_e)
    t5_4 = ((  4m) / (m_2*m_4)) * (e_Kxx_e^2 + e_Kyy_e^2)
    t5_5 = ((   6) / (m_2*m_4)) * (e_Kxx_e^2 + e_Kyy_e^2)
    t6_4 = ((  8m) / (m_2^3  )) * (e_Kxy_e^2)
    t6_5 = ((  12) / (m_2^3  )) * (e_Kxy_e^2)
    t7_4 = ((   2) / (m_4    )) * (Kxx_F2 + Kyy_F2)
    t8_4 = ((4m^2) / (m_2^3  )) * (Kxy_F2)
    t8_5 = ((  8m) / (m_2^3  )) * (Kxy_F2)

    MMDvar = (((t1_4 + t2_4) - (t3_4 + t5_4 + t6_4 + t7_4 + t8_4)) + ((t4_5 + t5_5 + t6_5 + t8_5) - t2_5)) - t2_6 # NOTE: typo in original paper: t8_* sign flips

    return MMDsq, MMDvar
end

function mmd_flux(
        kernelargs::Union{<:Function, <:AbstractArray},
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices_batched(kernelargs, X, Y)
    # @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices(kernelargs, X, Y)
    return mmd_flux_u_statistic(Kxx, Kyy, Kxy)
end

function mmdvar_flux(
        kernelargs::Union{<:Function, <:AbstractArray},
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices_batched(kernelargs, X, Y)
    # @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices(kernelargs, X, Y)
    return mmdvar_flux_u_statistic(Kxx, Kyy, Kxy)
end

function mmd_and_mmdvar_flux(
        kernelargs::Union{<:Function, <:AbstractArray},
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices_batched(kernelargs, X, Y)
    # @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices(kernelargs, X, Y)
    return mmd_and_mmdvar_flux_u_statistic(Kxx, Kyy, Kxy)
end

#=
let
    rng = Random.seed!(0)
    a = 5.0
    X = a .* rand(10,10)
    Y = a .* randn(10,10)
    k = d -> exp(-d/2a^2)
    loss1 = (X,Y) -> mmd_flux(k, X, Y)
    loss2 = (X,Y) -> mmd_flux([log(a)], X, Y)
    @show loss1(X,Y) ≈ loss2(X,Y)
    @show gradcheck(loss1, X, Y)
    @show gradcheck(loss2, X, Y)
    Random.seed!(rng)
end;
=#

#=
let
    model = Flux.Dense(10,10)
    X, Y = randn(10,100), randn(10,100)
    k = d -> exp(-d/2)
    loss1 = () -> mmd_flux(k, model(X), Y)
    loss2 = () -> mmd_flux(zeros(8), model(X), Y)
    @btime $loss1()
    @btime $loss2()
    @btime Flux.gradient($loss1, $(Flux.params(model)))
    @btime Flux.gradient($loss2, $(Flux.params(model)))
end
=#

#= mmdvar!, mmdvar, mmdvar_flux speed + consistency testing
for a in [3], m in [30]
    k = Δ -> exp(-Δ/2)
    logsigma = [0.0]

    # sampleX = () -> randn(2,m)
    # sampleY = () -> ((1/√2) * [1 -1; 1 1] * [√a 0; 0 1/√a]) * randn(2,m)
    sampleX = () -> rand(2,m)
    sampleY = () -> [2-1/a; 1/a] .* rand(2,m)
    X, Y = sampleX(), sampleY()
    # @btime kernel_mmdvar_stats($k, $X, $Y)

    v1 = @btime mmdvar!($(mmd_work(X, Y)), $k, $X, $Y)
    v2 = @btime mmdvar($k, $X, $Y)
    v3 = @btime mmdvar_flux($logsigma, $X, $Y)
    @show v1, v2, v3
    @show v1-v2, v2-v3
    @show (v1-v2)/v1, (v2-v3)/v2
end
=#

#= mmd!, mmd, and mmd_flux consistency testing
for m in [50]
    k = Δ -> exp(-Δ/2)
    logsigma = [0.0]
    X, Y = randn(2,m), 2 .* randn(2,m)
    v1 = mmd!(mmd_work(X, Y), k, X, Y)
    v2 = mmd(k, X, Y)
    v3 = mmd_flux(k, X, Y)
    v4 = mmd_flux(logsigma, X, Y)
    @assert v1 ≈ v2 && v2 ≈ v3 && v3≈v4
    @show v1, v2, v3, v4
    @show v1-v2, v2-v3, v3-v4
end
=#

#= mmdvar qqplot testing
for a in [4], m in [30], nsamples in [100]
    s = x->round(x; sigdigits = 4)
    qq(x,y) = qqplot(x, y; title = "a=$a, m=$m, nsamples=$nsamples, mean_x = $(s(mean(x))), mean_y = $(s(mean(y)))") |> display
    
    mean_var_samples = []
    mean_mmdvar_samples = []
    mean_mmdvar!_samples = []
    mean_mmdvar_flux_samples = []

    for _ in 1:1
        k = Δ -> exp(-Δ/2)
        logsigma = [0.0]
        sampleX = () -> rand(2,m)
        sampleY = () -> [2-1/a; 1/a] .* rand(2,m)
        work = mmd_work(sampleX(), sampleY())

        var_samples = [var([mmd(k, sampleX(), sampleY()) for _ in 1:nsamples]) for _ in 1:nsamples]
        mmdvar_samples = [mmdvar(k, sampleX(), sampleY()) for _ in 1:nsamples]
        mmdvar!_samples = [mmdvar!(work, k, sampleX(), sampleY()) for _ in 1:nsamples]
        mmdvar_flux_samples = [mmdvar_flux(logsigma, sampleX(), sampleY()) for _ in 1:nsamples]

        qq(mmdvar_samples, var_samples)
        qq(mmdvar_samples, mmdvar!_samples)
        qq(mmdvar_samples, mmdvar_flux_samples)
        
        push!(mean_var_samples, mean(var_samples))
        push!(mean_mmdvar_samples, mean(mmdvar_samples))
        push!(mean_mmdvar!_samples, mean(mmdvar!_samples))
        push!(mean_mmdvar_flux_samples, mean(mmdvar_flux_samples))
    end

    # qq(mean_var_samples, mean_mmdvar_samples)
    # qq(mean_var_samples, mean_mmdvar!_samples)
    # qq(mean_var_samples, mean_mmdvar_flux_samples)
end
=#

#=
for m in [50], nbw = [1,4], n in [2,10]
    # for m in [1024], nbw = [4], n in [128]
    logσ_vec = rand(nbw)
    logσ_mat = repeat(logσ_vec, )
    X, Y = randn(n,m), 2 .* randn(n,m)
    k = Δ -> mean(@. exp(-Δ/(2*exp(2*logσ_vec))))

    mmds = [
        mmd(k, X, Y),
        mmd_flux(k, X, Y),
        mmd_flux(logσ_vec, X, Y),
        mmd_flux(logσ_mat, X, Y),
        mmd_and_mmdvar_flux(logσ_mat, X, Y)[1],
    ]

    mmdvars = [
        mmdvar(k, X, Y),
        mmdvar_flux(k, X, Y),
        mmdvar_flux(logσ_vec, X, Y),
        mmdvar_flux(logσ_mat, X, Y),
        mmd_and_mmdvar_flux(logσ_mat, X, Y)[2],
    ]

    for x in [mmds, mmdvars]
        # @show x
        # @show maximum(abs, diff(x))
        @assert all(x .≈ x[1])
    end

    # @btime mmd_flux($logσ_mat, $X, $Y)
    # @btime mmdvar_flux($logσ_mat, $X, $Y)
    # @btime mmd_and_mmdvar_flux($logσ_mat, $X, $Y)
end
=#

####
#### Permutation testing
####

combine_kernel_matrices(Kxx, Kyy, Kxy) = [Kxx Kxy; Kxy' Kyy]
split_kernel_matrices(K) = (Kxx = K[1:end÷2,1:end÷2], Kyy = K[end÷2+1:end,end÷2+1:end], Kxy = K[1:end÷2,end÷2+1:end])

function perm_u_statistic(K)
    m = size(K,1)÷2
    @assert size(K) == (2m,2m)
    p = randperm(2m)
    mmd_flux_u_statistic(split_kernel_matrices(K[p,p])...)
end

function perm_u_statistic!(K, ipermvec)
    m = size(K,1)÷2
    @assert size(K) == (2m,2m) && length(ipermvec) == 2m
 
    randperm!(ipermvec)
    kxx = zero(eltype(K))
    kyy = zero(eltype(K))
    kxy = zero(eltype(K))

    @inbounds for j in 1:2m
        jp = ipermvec[j]
        Xblock_j = jp <= m
        @inbounds @simd for i in j+1:2m
            ip = ipermvec[i]
            Xblock_i = ip <= m
            Kij = K[i,j]
            kxx += ifelse( Xblock_i &&  Xblock_j, 2*Kij, zero(eltype(K)))
            kyy += ifelse(!Xblock_i && !Xblock_j, 2*Kij, zero(eltype(K)))
            kxy += ifelse( Xblock_i ⊻   Xblock_j && ip - jp != m && jp - ip != m, Kij, zero(eltype(K)))
        end
    end

    return (kxx + kyy - 2kxy) / (m*(m-1))
end

function mmd_perm_test_brute(kernelargs, X, Y; nperms = size(X,2), alpha = 1//100)
    m = size(X,2)
    c_alpha_perms = [m * mmd_flux(kernelargs, mix_columns(X, Y)...) for _ in 1:nperms]
    c_alpha = quantile(c_alpha_perms, 1-alpha)
    MMDsq, MMDvar = mmd_and_mmdvar_flux(kernelargs, X, Y)
    return @ntuple(MMDsq, MMDvar, c_alpha, c_alpha_perms)
end

function mmd_perm_test(kernelargs, X, Y; nperms = size(X,2), alpha = 1//100)
    @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices_batched(kernelargs, X, Y)
    # @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices(kernelargs, X, Y)
    MMDsq, MMDvar = mmd_and_mmdvar_flux_u_statistic(Kxx, Kyy, Kxy)
    K = combine_kernel_matrices(Kxx, Kyy, Kxy)

    m = size(X,2)
    nt = Threads.nthreads()
    if nt > 1
        # Compute c_α permutations in parallel
        work = [zeros(Int, 2m) for _ in 1:nt]
        c_alpha_perms = zeros(eltype(K), nperms)
        Threads.@sync for i in 1:nperms
            Threads.@spawn begin
                ipermvec = work[Threads.threadid()]
                c_alpha_perms[i] = m * perm_u_statistic!(K, ipermvec)
            end
        end
        c_alpha_perms
    else
        # Compute c_α permutations serially
        ipermvec = zeros(Int, 2m)
        c_alpha_perms = [m * perm_u_statistic!(K, ipermvec) for _ in 1:nperms]
    end
    c_alpha = quantile(c_alpha_perms, 1-alpha)
    
    return @ntuple(MMDsq, MMDvar, c_alpha, c_alpha_perms)
end

#=
let a = 2.0
    for m = [100, 250], nperms = [128, 1024]
        X, Y = randn(2,m), a*randn(2,m)
        k = d -> exp(-d)
        @show m, nperms
        # @btime mmd_perm_test_brute($k, $X, $Y; nperms = $nperms, alpha = 1//10)
        @btime mmd_perm_test($k, $X, $Y; nperms = $nperms, alpha = 1//10)
        qqplot(
            mmd_perm_test_brute(k, X, Y; nperms = nperms, alpha = 1//10).c_alpha_perms,
            mmd_perm_test(k, X, Y; nperms = nperms, alpha = 1//10).c_alpha_perms,
        ) |> display
    end
end
=#

#=
let m = 100
    for a in 1.5:0.5:3
        X, Y = randn(2,m), a*randn(2,m)
        p = plot()

        @time res1 = mmd_perm_test_brute(d->exp(-d), X, Y; nperms = 1000, alpha = 1//10)
        @show a, res1.MMDsq, res1.c_alpha
        density!(p, res1.c_alpha_perms; label = "brute")

        @time res2 = mmd_perm_test(d->exp(-d), X, Y; nperms = 1000, alpha = 1//10)
        @show a, res2.MMDsq, res2.c_alpha
        density!(p, res2.c_alpha_perms; label = "fast")

        display(p)
        qqplot(res1.c_alpha_perms, res2.c_alpha_perms) |> display

        p = plot()
        density!(p, res2.c_alpha_perms; label = "c_α samples", line = (2,))
        vline!(p, [res2.c_alpha]; label = "c_α", line = (2,))
        vline!(p, [m * res2.MMDsq]; label = "MMD", line = (2,))
        display(p)
    end
end
=#

# Perform permutation test:
#   An initial (X,Y) pair is sampled and a permutation test with `nperms` permutations is performed
#   An additional `nsamples-1` (X,Y) pairs are drawn from the samplers and their MMD/variance are also computed
function mmd_perm_test_power(
        kernelargs,
        sampleX,
        sampleY;
        batchsize = 100,
        nperms = batchsize,
        nsamples = 10,
        alpha = 1//100
    )
    @unpack MMDsq, MMDvar, c_alpha, c_alpha_perms =
        mmd_perm_test(kernelargs, sampleX(batchsize), sampleY(batchsize); nperms = nperms, alpha = alpha)

    mmd_samples, mmdvar_samples = [MMDsq], [MMDvar]
    for _ in 1:nsamples-1
        _MMDsq, _MMDvar = mmd_and_mmdvar_flux(kernelargs, sampleX(batchsize), sampleY(batchsize))
        push!(mmd_samples, _MMDsq)
        push!(mmdvar_samples, _MMDvar)
    end

    m = batchsize
    MMDsq = mean(mmd_samples)
    MMDvar = mean(mmdvar_samples) # var(mmd_samples) is less accurate for small nsamples
    MMDσ = √max(MMDvar, eps(typeof(MMDvar))/m^2) # ensure m^2 * MMDvar >= ϵ
    z = MMDsq / MMDσ - c_alpha / (m * MMDσ)
    P_alpha_approx = cdf(Normal(), z) |> typeof(MMDsq)
    P_alpha = count(MMDsq -> m * MMDsq > c_alpha, mmd_samples) / nsamples |> typeof(MMDsq)

    return @ntuple(alpha, m, c_alpha, P_alpha, P_alpha_approx, MMDsq, MMDvar, MMDσ, c_alpha_perms, mmd_samples, mmdvar_samples)
end

# Perform permutation test with a single explicit (X,Y) pair
mmd_perm_test_power(kernelargs, X::AbstractMatrix, Y::AbstractMatrix; kwargs...) = mmd_perm_test_power(kernelargs, m->X, m->Y; kwargs..., batchsize = size(X,2), nsamples = 1)

####
#### MMD plotting
####

function mmd_heatmap(X, Y, σ; skipdiag = true)
    γ = inv(2*σ^2)
    k = Δ -> exp(-γ*Δ)

    # compute mmd
    work = mmd_work(X, Y)
    mmd = mmd!(work, k, X, Y)

    # recompute with diag for plotting
    @unpack Kxx, Kyy, Kxy = work
    if !skipdiag
        kernel_pairwise!(Kxx, k, X, X, Val(false))
        kernel_pairwise!(Kyy, k, Y, Y, Val(false))
        kernel_pairwise!(Kxy, k, X, Y, Val(false))
    end

    s = x -> string(round(x; sigdigits = 3))
    m = size(Kxx, 1)
    K = [Kxx Kxy; Kxy' Kyy]
    Kplot = (x -> x == 0 ? 0.0 : log10(x)).(K[end:-1:1,:])
    p = heatmap(Kplot; title = "m*MMD = $(s(m*mmd)), sigma = $(s(σ))", clims = (max(minimum(Kplot), -10), 0))

    return p
end

function mmd_witness(X, Y, σ; skipdiag = false)
    γ = inv(2*σ^2)
    k = Δ -> exp(-γ*Δ)

    # compute mmd
    work = mmd_work(X, Y)
    mmd = mmd!(work, k, X, Y)

    # recompute with diag for plotting
    @unpack Kxx, Kyy, Kxy = work
    if !skipdiag
        kernel_pairwise!(Kxx, k, X, X, Val(false))
        kernel_pairwise!(Kyy, k, Y, Y, Val(false))
        kernel_pairwise!(Kxy, k, X, Y, Val(false))
    end

    s = x -> string(round(x; sigdigits = 3))
    m = size(Kxx, 1)
    fX = vec(mean(Kxx; dims = 1)) - mean(Kxy; dims = 2)
    fY = vec(mean(Kxy; dims = 1)) - vec(mean(Kyy; dims = 1))
    phist = plot(; title = "mmd = $(m * (mean(fX) - mean(fY)))")
    density!(phist, m .* fY; l = (4, :blue), label = "true data: m*fY")
    density!(phist, m .* fX; l = (4, :red),  label = "simulated: m*fX")

    return phist
end

# Plot permutation test results
function mmd_perm_test_power_plot(perm_test_results; showplot = false)
    @timeit "permutation plot" try
        @unpack alpha, m, c_alpha, P_alpha, P_alpha_approx, MMDsq, MMDvar, MMDσ, c_alpha_perms, mmd_samples, mmdvar_samples = perm_test_results

        s = x -> string(round(x; sigdigits = 4))
        p = plot(; title = "P_α = $(s(P_alpha)) ~ $(s(P_alpha_approx))")
        xl = extrema([extrema(c_alpha_perms)..., extrema(m .* mmd_samples)...])

        # Permutation test plot:
        #   c_alpha_perms: m*MMD^2 for mixed (X,Y) data
        #   c_alpha: the (1-alpha)'th quantile for c_alpha_perms
        density!(p, c_alpha_perms; label = "c_α samples (α = $alpha)", line = (3,:blue))
        vline!(p, [c_alpha]; label = "c_α threshold (α = $alpha)", line = (2,:blue,:dash))

        # m*MMD^2 samples plot:
        #   mmd_samples: m*MMD^2 samples for different (X,Y) batches
        #   MMDsq, MMDσ: estimates for mean, std of mmd_samples
        vline!(p, [m*MMDsq]; line = (2,:red,:dash), label = "m*MMD² mean (μ = $(s(m*MMDsq)))")
        vline!(p, m*MMDsq .+ [-m*MMDσ, m*MMDσ]; line = (2,:red,:dot), label = "±1σ bounds (σ = $(s(m*MMDσ)))")
        if m^2 * MMDvar > eps(typeof(MMDvar))
            plot!(p, Normal(m*MMDsq, m*MMDσ); label = "m*MMD² distbn ~ N(μ,σ)", line = (3,:red))
        end

        if length(mmd_samples) > 1
            density!(p, m .* mmd_samples; label = "m*MMD² samples", line = (2,:green))
        end

        showplot && !isnothing(p) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error during permutation plot")
    end
end

#=
let m = 100, n = 2, nbw = 4, nperms = 128, nsamples = 100, ntrials = 10
    # gamma = inv(2 * 2.0^2)
    # kernelargs = d -> exp(-gamma * d)
    # kernelargs = log.([1.5, 1.0, 0.5])
    kernelargs = randn(nbw, n)
    for a in 1.1:0.1:1.5
        @time all_res = map(1:ntrials) do _
            mmd_perm_test_power(kernelargs, m->randn(n,m), m->a*randn(n,m);
                batchsize = m, nperms = nperms, nsamples = nsamples)
        end
        c_alpha = mean(r->r.c_alpha, all_res); @show c_alpha
        P_alpha = mean(r->r.P_alpha, all_res); @show P_alpha
        P_alpha_approx = mean(r->r.P_alpha_approx, all_res); @show P_alpha_approx
        # @show P_alpha, P_alpha_approx
        mmd_perm_test_power_plot(all_res[1]) |> display
    end
end
=#

####
#### Kernel bandwidth opt
####

function tstat(logσ::Real, X, Y)
    m = size(X,2)
    σ = exp(logσ)
    γ = inv(2σ^2)
    k(Δ) = exp(-γ*Δ)
    MMDsq  = m * mmd(k, X, Y)
    MMDvar = m^2 * mmdvar(k, X, Y)
    ϵ = eps(typeof(MMDvar))
    t = MMDsq / √max(MMDvar, ϵ) # avoid division by zero/negative
    return t
end
∇tstat(logσ::Real, args...; kwargs...) = ForwardDiff.derivative(logσ -> tstat(logσ, args...; kwargs...), logσ)

function tstat_flux(logσ::AbstractArray, X, Y, isillposed = nothing)
    # Avoiding div by zero/negative:
    #   m^2 * MMDvar >= ϵ  -->  m * MMDσ >= √ϵ
    MMDsq, MMDvar = mmd_and_mmdvar_flux(logσ, X, Y)
    m = size(X,2)
    ϵ = eps(typeof(MMDvar))
    t = m*MMDsq / √max(m^2*MMDvar, ϵ)
    if !isnothing(isillposed)
        isillposed[] = (MMDsq < 0) || (m^2*MMDvar < ϵ)
    end
    return t
end

function ∇tstat_flux(logσ::AbstractArray, args...; kwargs...)
    if length(logσ) <= 16
        ForwardDiff.gradient(logσ -> tstat_flux(logσ, args...; kwargs...), logσ)
    else
        Zygote.gradient(logσ -> tstat_flux(logσ, args...; kwargs...), logσ)[1]
    end
end

∇tstat_flux_fwddiff(logσ::AbstractArray, args...; kwargs...) = ForwardDiff.gradient(logσ -> tstat_flux(logσ, args...; kwargs...), logσ)
∇tstat_flux_zygote(logσ::AbstractArray, args...; kwargs...) = Zygote.gradient(logσ -> tstat_flux(logσ, args...; kwargs...), logσ)[1]

function kernel_bandwidth_loss_flux(logσ::AbstractArray, X, Y, ::Val{losstype}, isillposed = nothing) where {losstype}
    if losstype === :mmd
        !isnothing(isillposed) && (isillposed[] = false)
        return mmd_flux(logσ, X, Y)
    elseif losstype === :tstatistic
        return tstat_flux(logσ, X, Y, isillposed)
    else
        error("Loss type must be :mmd or :tstatistic")
    end
end

function train_kernel_bandwidth_flux!(logσ::AbstractArray, X, Y; kernelloss::String, kernellr = 0.01, bwbounds = nothing)
    # Kernel is trained with new optimizer for each X, Y; loss jumps too wildly
    opt = Flux.ADAM(kernellr)
    isillposed = Ref(false)
    losstype = Val(Symbol(kernelloss))
    loss(_logσ) = kernel_bandwidth_loss_flux(_logσ, X, Y, losstype, isillposed)

    # Training should not be performed using t-statistic loss function if MMD^2 < 0 or Var[MMD^2] < 0
    @timeit "forward" ℓ, back = Zygote.pullback(loss, logσ)
    if !isillposed[]
        @timeit "reverse" ∇ℓ = back(one(eltype(logσ)))[1]
        Flux.Optimise.update!(opt, logσ, ∇ℓ)
        if !isnothing(bwbounds)
            clamp!(logσ, bwbounds...)
        end
    end

    # Let caller know if training was applied
    success = !isillposed[]
    return success
end

#= (∇)mmd_(flux_)bandwidth_loss speed + consistency testing
for m in [32,64,128,256,512], nsigma in [16], a in [2.0]
    logsigma = rand(nsigma)
    X, Y = randn(2,m), a.*randn(2,m)
    
    if nsigma == 1
        t1 = tstat(logsigma, X, Y)
        t2 = tstat_flux(logsigma, X, Y)
        # @show t1, t2
        # @show t1-t2
        @assert t1≈t2
    end

    g1 = nsigma == 1 ? ∇tstat(logsigma, X, Y) : nothing
    g2 = ∇tstat_flux_fwddiff(logsigma, X, Y)
    g3 = ∇tstat_flux_zygote(logsigma, X, Y)
    # @show g1, g2, g3
    # @show g1-g2, g2-g3
    @assert (nsigma != 1 || g1≈g2) && g2≈g3

    # @btime tstat($logsigma, $X, $Y)
    # @btime tstat_flux($logsigma, $X, $Y)

    @show m, nsigma
    (nsigma == 1) && @btime ∇tstat($logsigma, $X, $Y)
    @btime ∇tstat_flux_fwddiff($logsigma, $X, $Y)
    @btime ∇tstat_flux_zygote($logsigma, $X, $Y)
end
=#

function tstat_bandwidth_bruteopt(sampleX, sampleY, bounds; nsigma = 100, nevals = 100)
    # work = mmd_work(sampleX(), sampleY())
    function f(logσ)
        σ = exp(logσ)
        γ = inv(2*σ^2)
        k = Δ -> exp(-γ*Δ)
        X, Y = sampleX(), sampleY()
        mmds = [mmd(k, sampleX(), sampleY()) for _ in 1:nevals]
        MMDsq = mean(mmds)
        MMDvar = var(mmds)
        σ = √MMDvar
        MMDsq / σ
    end
    logσ = range(bounds[1], bounds[2]; length = nsigma)
    return logσ, [f(logσ) for logσ in logσ]
end
