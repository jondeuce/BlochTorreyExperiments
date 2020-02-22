####
#### Helper functions
####

# Smoothed version of max(x,e) for fixed e > 0
smoothmax(x,e) = e + e * Flux.softplus((x-e) / e)

function mix_columns(X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)
    m = size(X, 2)
    XY = hcat(X, Y)
    idx = randperm(2m)
    return (XY[:, idx[1:m]], XY[:, idx[m+1:end]])
end

function sample_columns(X::AbstractMatrix, batchsize)
    X[:, sample(1:size(X,2), batchsize; replace = false)]
end

function column_mse(X::AbstractVecOrMat, Y::AbstractVecOrMat, i, j)
    @assert size(X) == size(Y) && 1 <= min(i,j) <= max(i,j) <= size(X,2)
    T = promote_type(eltype(X), eltype(Y))
    Σ = zero(T)
    n, m = size(X,1), size(X,2)
    if !(X === Y && i == j) && n > 0
        @inbounds @simd for k in 1:n
            δxy = X[k,i] - Y[k,j]
            Σ += δxy * δxy
        end
        Σ /= n
    end
    return Σ
end
function column_mse(X::AbstractVector, Y::AbstractVector)
    @assert size(X) == size(Y)
    T = promote_type(eltype(X), eltype(Y))
    Σ = zero(T)
    n = length(X)
    if !(X === Y) && n > 0
        @inbounds @simd for k in 1:n
            δxy = X[k] - Y[k]
            Σ += δxy * δxy
        end
        Σ /= n
    end
    return Σ
end
function sum_columns!(out::AbstractVector{T}, X::AbstractMatrix{T}) where {T}
    @assert length(out) == size(X,1)
    @inbounds @simd for i in 1:size(X,1)
        out[i] = X[i,1]
    end
    @inbounds for j in 2:size(X,2)
        @simd for i in 1:size(X,1)
            out[i] += X[i,j]
        end
    end
    return out
end
function sum_rows!(out::AbstractVector{T}, X::AbstractMatrix{T}) where {T}
    @assert length(out) == size(X,1)
    @inbounds for j in 1:size(X,2)
        Σ = zero(T)
        @simd for i in 1:size(X,1)
            Σ += X[i,j]
        end
        out[j] = Σ
    end
    return out
end
function frob_norm2(X::AbstractArray)
    Σ = zero(eltype(X))
    @inbounds @simd for i in 1:length(X)
        xi = X[i]
        Σ += xi * xi
    end
    return Σ
end

####
#### MMD using buffer matrices
####

function kernel_pairwise!(Kxy::AbstractMatrix{T}, k, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, ::Val{skipdiag} = Val(false)) where {T,skipdiag}
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
function kernel_var_stats!(work, k, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy, Kyx, Kxx_e, Kyy_e, Kxy_e, Kyx_e = work

    kernel_pairwise!(Kxx, k, X, X, Val(true))
    kernel_pairwise!(Kyy, k, Y, Y, Val(true))
    kernel_pairwise!(Kxy, k, X, Y, Val(false))

    sum_columns!(Kxx_e, Kxx)
    sum_columns!(Kyy_e, Kyy)
    sum_columns!(Kxy_e, Kxy)
    sum_rows!(Kyx_e, Kxy)

    Kxx_F2, Kyy_F2, Kxy_F2    = frob_norm2(Kxx), frob_norm2(Kyy), frob_norm2(Kxy)
    e_Kxx_e, e_Kyy_e, e_Kxy_e = sum(Kxx), sum(Kyy), sum(Kxy)
    e_Kxx_Kxy_e, e_Kyy_Kyx_e  = Kxx_e'Kxy_e, Kyy_e'Kyx_e
    Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2 = frob_norm2(Kxx_e), frob_norm2(Kyy_e), frob_norm2(Kxy_e), frob_norm2(Kyx_e)

    return @ntuple(Kxx_F2, Kyy_F2, Kxy_F2, e_Kxx_e, e_Kyy_e, e_Kxy_e, Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2, e_Kxx_Kxy_e, e_Kyy_Kyx_e)
end
function mmd_work(T::Type, sz::NTuple{2,Int})
    n, m = sz
    Kxx, Kyy, Kxy, Kyx = (_ -> zeros(T, m, m)).(1:4)
    Kxx_e, Kyy_e, Kxy_e, Kyx_e = (_ -> zeros(T, m)).(1:4)
    return @ntuple(Kxx, Kyy, Kxy, Kyx, Kxx_e, Kyy_e, Kxy_e, Kyx_e)
end
mmd_work(sz::NTuple{2,Int}) = mmd_work(Float64, sz)
mmd_work(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = (@assert size(X) == size(Y); return mmd_work(T, size(X)))

function mmd!(work, k, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy, Kyx = work
    m = size(X, 2)

    # Ref: http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    # e_Kxx_e = sum(kernel_pairwise!(Kxx, k, X, X))
    # e_Kyy_e = sum(kernel_pairwise!(Kyy, k, Y, Y))
    # e_Kxy_e = sum(kernel_pairwise!(Kxy, k, X, Y))
    # MMDsq = e_Kxx_e/(m*(m-1)) + e_Kyy_e/(m*(m-1)) - 2*e_Kxy_e/(m^2) # (m^2 * [K])/m^2 = [K] ~ O(1)

    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    e_Kxx_e = sum(kernel_pairwise!(Kxx, k, X, X, Val(true)))
    e_Kyy_e = sum(kernel_pairwise!(Kyy, k, Y, Y, Val(true)))
    e_Kxy_e = sum(kernel_pairwise!(Kxy, k, X, Y, Val(true)))
    #e_Kyx_e= sum(kernel_pairwise!(Kyx, k, Y, X, Val(true)))
    #MMDsq= (e_Kxx_e + e_Kyy_e - e_Kxy_e - e_Kxy_e)/(m*(m-1))
    MMDsq = (e_Kxx_e + e_Kyy_e - 2e_Kxy_e)/(m*(m-1))

    return MMDsq
end
function mmdvar!(work, k, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    #   NOTE: Here we assume a symmetric kernel k(x,y) == k(y,x),
    #         and therefore that Kxx == Kxx', Kyy = Kyy', Kxy == Kxy'
    # See:
    #   [1] https://arxiv.org/pdf/1906.02104.pdf
    #   [2] http://www.gatsby.ucl.ac.uk/~dougals/slides/dali/#/50
    @assert size(X) == size(Y)
    @unpack Kxx_F2, Kyy_F2, Kxy_F2, e_Kxx_e, e_Kyy_e, e_Kxy_e, Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2, e_Kxx_Kxy_e, e_Kyy_Kyx_e = kernel_var_stats!(work, k, X, Y)
    m = size(X, 2)
    m_2 = m*(m-1)
    m_3 = m*(m-1)*(m-2)
    m_4 = m*(m-1)*(m-2)*(m-3)

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

    # @show t1_4, t2_4, t2_5, t2_6, t3_4, t4_5, t5_4, t5_5, t6_4, t6_5, t7_4, t8_4, t8_5

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

    return MMDvar
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
#### MMD without buffers
####

function kernel_pairwise_sum(k, X::AbstractMatrix, Y::AbstractMatrix, ::Val{skipdiag} = Val(false)) where {skipdiag}
    # @assert size(X) == size(Y)
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
function kernel_var_stats(k, X::AbstractMatrix, Y::AbstractMatrix)
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
    return @ntuple(Kxx_F2, Kyy_F2, Kxy_F2, e_Kxx_e, e_Kyy_e, e_Kxy_e, Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2, e_Kxx_Kxy_e, e_Kyy_Kyx_e)
end
function mmd(k, X::AbstractMatrix, Y::AbstractMatrix)
    # Ref: http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    # @assert size(X) == size(Y)
    m = size(X, 2)

    # Σxx = kernel_pairwise_sum(k, X)
    # Σxy = kernel_pairwise_sum(k, X, Y)
    # Σyy = kernel_pairwise_sum(k, Y)
    # MMDsq = Σxx/(m*(m-1)) + Σyy/(m*(m-1)) - 2*Σxy/(m^2) # generic unbiased estimator (m^2 -> m*n when m != n)

    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    e_Kxx_e = kernel_pairwise_sum(k, X, X, Val(true))
    e_Kyy_e = kernel_pairwise_sum(k, Y, Y, Val(true))
    e_Kxy_e = kernel_pairwise_sum(k, X, Y, Val(true))
    #e_Kyx_e= kernel_pairwise_sum(k, Y, X, Val(true))
    #MMDsq= (e_Kxx_e + e_Kyy_e - e_Kxy_e - e_Kxy_e)/(m*(m-1))
    MMDsq = (e_Kxx_e + e_Kyy_e - 2e_Kxy_e)/(m*(m-1))

    return MMDsq
end
function mmdvar(k, X::AbstractMatrix, Y::AbstractMatrix)
    #   NOTE: Here we assume a symmetric kernel k(x,y) == k(y,x),
    #         and therefore that Kxx == Kxx', Kyy = Kyy', Kxy == Kxy'
    # See:
    #   [1] https://arxiv.org/pdf/1906.02104.pdf
    #   [2] http://www.gatsby.ucl.ac.uk/~dougals/slides/dali/#/50
    @assert size(X) == size(Y)
    @unpack Kxx_F2, Kyy_F2, Kxy_F2, e_Kxx_e, e_Kyy_e, e_Kxy_e, Kxx_e_F2, Kyy_e_F2, Kxy_e_F2, Kyx_e_F2, e_Kxx_Kxy_e, e_Kyy_Kyx_e = kernel_var_stats(k, X, Y)
    m = size(X, 2)
    m_2 = m*(m-1)
    m_3 = m*(m-1)*(m-2)
    m_4 = m*(m-1)*(m-2)*(m-3)

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

#= mmdvar!, mmdvar, mmdvar_flux speed + consistency testing
for a in [3], m in [30]
    k = Δ -> exp(-Δ/2)
    logsigma = [0.0]

    # sampleX = () -> randn(2,m)
    # sampleY = () -> ((1/√2) * [1 -1; 1 1] * [√a 0; 0 1/√a]) * randn(2,m)
    sampleX = () -> rand(2,m)
    sampleY = () -> [2-1/a; 1/a] .* rand(2,m)
    X, Y = sampleX(), sampleY()
    # @btime kernel_var_stats($k, $X, $Y)

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

####
#### Flux differentiable MMD
####

Flux.Zygote.@adjoint function LinearAlgebra.diag(x::AbstractMatrix)
    return LinearAlgebra.diag(x), function(Δ)
        # @show typeof(Δ), size(Δ) # Why is Δ a nx1 matrix? (something to do with adjoint... doesn't happen unless you have a loss like e.g. sum(diag(A)'))
        # (LinearAlgebra.Diagonal(Δ),) # Should be this
        (LinearAlgebra.Diagonal(reshape(Δ,:)),) # Need to reshape nx1 Δ to vector
    end
end
Flux.Zygote.refresh()

#=
let
    X = rand(2,2)
    loss = (X) -> sum(abs2, LinearAlgebra.diag(X)')
    @show gradient(loss, X)[1]
    @show Diagonal(2X)
end;
=#

function mmd_flux_kernel_matrices(logsigma::AbstractVector, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)

    n, m = size(X)
    gamma = @. inv(-2n * exp(2 * logsigma)) # absorb -1/n factor into gamma = 1/2sigma^2 = 1/2exp(2logsigma)
    gamma = reshape(gamma, 1, 1, :) # reshape for broadcasting

    XX, XY, YY = X'X, X'Y, Y'Y
    xx, yy = LinearAlgebra.diag(XX), LinearAlgebra.diag(YY) # squared norms on diagonal
    Kxx = reshape(mean(@. exp(gamma * (xx - 2 * XX + xx')); dims = 3), m, m) # note: mean is over data length n, not number of data m
    Kyy = reshape(mean(@. exp(gamma * (yy - 2 * YY + yy')); dims = 3), m, m)
    Kxy = reshape(mean(@. exp(gamma * (xx - 2 * XY + yy')); dims = 3), m, m)

    return @ntuple(Kxx, Kyy, Kxy)
end

function mmd_flux_kernel_matrices(k::Function, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)

    n = size(X,1)
    XX, XY, YY = X'X, X'Y, Y'Y
    xx, yy = LinearAlgebra.diag(XX), LinearAlgebra.diag(YY) # squared norms on diagonal
    Kxx = k.((xx .- 2 .* XX .+ xx')./n) # note: mean is over data length n, not number of data m
    Kyy = k.((yy .- 2 .* YY .+ yy')./n)
    Kxy = k.((xx .- 2 .* XY .+ yy')./n)

    return @ntuple(Kxx, Kyy, Kxy)
end

function mmd_flux_u_statistic(Kxx, Kyy, Kxy)
    @assert size(Kxx) == size(Kyy) == size(Kxy)

    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    m = size(Kxx,1)
    e_Kxx_e = sum(Kxx) - m # assumes k(0) == 1 --> tr(Kxx) = m
    e_Kyy_e = sum(Kyy) - m # assumes k(0) == 1 --> tr(Kyy) = m
    e_Kxy_e = sum(Kxy) - tr(Kxy)
    MMDsq = (e_Kxx_e + e_Kyy_e - 2e_Kxy_e) / (m*(m-1))

    return MMDsq
end

function mmdvar_flux_u_statistic(Kxx, Kyy, Kxy)
    @assert size(Kxx) == size(Kyy) == size(Kxy)

    # Var[MMD²_U]: Variantes of U-statistic MMD estimator
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    
    m = size(Kxx,1)
    m_2 = m*(m-1)
    m_3 = m*(m-1)*(m-2)
    m_4 = m*(m-1)*(m-2)*(m-3)

    e_Kxx_e = sum(Kxx) - m # assumes k(0) == 1
    e_Kyy_e = sum(Kyy) - m # assumes k(0) == 1
    e_Kxy_e = sum(Kxy)
    Kxx_F2 = sum(abs2, Kxx) - m # assumes k(0) == 1
    Kyy_F2 = sum(abs2, Kyy) - m # assumes k(0) == 1
    Kxy_F2 = sum(abs2, Kxy)
    Kxx_e = reshape(sum(Kxx; dims = 2), :) .- 1 # assumes k(0) == 1
    Kyy_e = reshape(sum(Kyy; dims = 2), :) .- 1 # assumes k(0) == 1
    Kxy_e = reshape(sum(Kxy; dims = 2), :)
    Kyx_e = reshape(sum(Kxy; dims = 1), :)
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

function mmd_flux(
        kernelargs::Union{<:Function, <:AbstractVector},
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices(kernelargs, X, Y)
    return mmd_flux_u_statistic(Kxx, Kyy, Kxy)
end

function mmdvar_flux(
        kernelargs::Union{<:Function, <:AbstractVector},
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices(kernelargs, X, Y)
    return mmdvar_flux_u_statistic(Kxx, Kyy, Kxy)
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
            kxy += ifelse((Xblock_i ⊻   Xblock_j) && ip - jp != m && jp - ip != m, Kij, zero(eltype(K)))
        end
    end

    # randperm!(ipermvec)
    # kxx_t = zeros(eltype(K), Threads.nthreads())
    # kyy_t = zeros(eltype(K), Threads.nthreads())
    # kxy_t = zeros(eltype(K), Threads.nthreads())

    # Threads.@threads for j in 1:2m
    #     @inbounds begin
    #         tid = Threads.threadid()
    #         kxx = kxx_t[tid]
    #         kyy = kyy_t[tid]
    #         kxy = kxy_t[tid]

    #         jp = ipermvec[j]
    #         Xblock_j = jp <= m
    #         @inbounds @simd for i in 1:2m
    #             ip = ipermvec[i]
    #             Xblock_i = ip <= m
    #             Kij = K[i,j]
    #             kxx += ifelse(( Xblock_i &&  Xblock_j) && ip != jp, Kij, zero(eltype(K)))
    #             kyy += ifelse((!Xblock_i && !Xblock_j) && ip != jp, Kij, zero(eltype(K)))
    #             kxy += ifelse(( Xblock_i  ⊻  Xblock_j) && ip - jp != m && jp - ip != m, Kij, zero(eltype(K)))
    #         end

    #         kxx_t[tid] += kxx
    #         kyy_t[tid] += kyy
    #         kxy_t[tid] += kxy
    #     end
    # end

    # kxx = sum(kxx_t)
    # kyy = sum(kyy_t)
    # kxy = sum(kxy_t)

    return (kxx + kyy - 2kxy) / (m*(m-1))
end

function mmd_perm_test_brute(kernelargs, X, Y; nperms = size(X,2), alpha = 0.01)
    m = size(X,2)
    c_alpha_perms = [m * mmd_flux(kernelargs, mix_columns(X, Y)...) for _ in 1:nperms]
    c_alpha = quantile(c_alpha_perms, 1-alpha)
    MMDsq = mmd_flux(kernelargs, X, Y)
    return @ntuple(MMDsq, c_alpha, c_alpha_perms)
end

function mmd_perm_test(kernelargs, X, Y; nperms = size(X,2), alpha = 0.01)
    @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices(kernelargs, X, Y)
    K = combine_kernel_matrices(Kxx, Kyy, Kxy)

    m = size(X,2)
    nt = Threads.nthreads()
    c_alpha_perms = if nt > 1
        # Compute c_α permutations in parallel
        work = [zeros(Int, 2m) for _ in 1:nt]
        c_alpha_perms = zeros(eltype(K), nperms)
        Threads.@threads for i in 1:nperms
            ipermvec = work[Threads.threadid()]
            c_alpha_perms[i] = m * perm_u_statistic!(K, ipermvec)
        end
        c_alpha_perms
    else
        # Compute c_α permutations serially
        ipermvec = zeros(Int, 2m)
        c_alpha_perms = [m * perm_u_statistic!(K, ipermvec) for _ in 1:nperms]
    end

    c_alpha = quantile(c_alpha_perms, 1-alpha)
    MMDsq = mmd_flux_u_statistic(Kxx, Kyy, Kxy)

    return @ntuple(MMDsq, c_alpha, c_alpha_perms)
end

#=
let a = 2.0
    for m = [100, 250], nperms = [128, 1024]
        X, Y = randn(2,m), a*randn(2,m)
        k = d -> exp(-d)
        @show m, nperms
        # @btime mmd_perm_test_brute($k, $X, $Y; nperms = $nperms, alpha = 0.1)
        @btime mmd_perm_test($k, $X, $Y; nperms = $nperms, alpha = 0.1)
        qqplot(
            mmd_perm_test_brute(k, X, Y; nperms = nperms, alpha = 0.1).c_alpha_perms,
            mmd_perm_test(k, X, Y; nperms = nperms, alpha = 0.1).c_alpha_perms,
        ) |> display
    end
end
=#

#=
let m = 100
    for a in 1.5:0.5:3
        X, Y = randn(2,m), a*randn(2,m)
        p = plot()

        @time res1 = mmd_perm_test_brute(d->exp(-d), X, Y; nperms = 1000, alpha = 0.1)
        @show a, res1.MMDsq, res1.c_alpha
        density!(p, res1.c_alpha_perms; label = "brute")

        @time res2 = mmd_perm_test(d->exp(-d), X, Y; nperms = 1000, alpha = 0.1)
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

function mmd_perm_test_power(
        kernelargs,
        sampleX,
        sampleY;
        batchsize = 100,
        nperms = batchsize,
        nsamples = 10,
        alpha = 0.01
    )
    @unpack MMDsq, c_alpha, c_alpha_perms = mmd_perm_test(kernelargs, sampleX(batchsize), sampleY(batchsize); nperms = nperms, alpha = alpha)
    mmd_samples = vcat(MMDsq, [mmd_flux(kernelargs, sampleX(batchsize), sampleY(batchsize)) for _ in 1:nsamples-1])

    m = batchsize
    P_alpha = count(MMDsq -> m * MMDsq > c_alpha, mmd_samples) / nsamples

    MMDsq = mean(mmd_samples)
    MMDvar = var(mmd_samples)
    MMDσ = √MMDvar
    z = MMDsq / MMDσ - c_alpha / (m * MMDσ)
    P_alpha_approx = cdf(Normal(), z)

    return @ntuple(alpha, m, c_alpha, P_alpha, P_alpha_approx, MMDsq, MMDσ, c_alpha_perms, mmd_samples)
end

function mmd_perm_test_power_plot(perm_test_results)
    @unpack alpha, m, c_alpha, P_alpha, P_alpha_approx, MMDsq, MMDσ, c_alpha_perms, mmd_samples = perm_test_results

    s = x -> string(round(x; sigdigits = 4))
    p = plot(; title = "P_α = $(s(P_alpha)) ~ $(s(P_alpha_approx))")
    density!(p, c_alpha_perms; label = "c_α samples", line = (3,:blue))
    vline!(p, [c_alpha]; label = "c_α (α = $alpha)", line = (3,:blue))
    density!(p, m .* mmd_samples; label = "m * MMD^2 samples", line = (3,:red))
    vline!(p, [m * MMDsq]; label = "m * MMD^2 (σ = $(s(m * MMDσ)))", line = (3,:red))
    vline!(p, m .* [MMDsq-MMDσ, MMDsq+MMDσ]; label = "", line = (2,:red,:dash))
    
    return p
end

#=
let m = 100, nperms = 1024, nsamples = 128, ntrials = 10
    # gamma = inv(2 * 2.0^2)
    # kernelargs = d -> exp(-gamma * d)
    kernelargs = log.([1.5, 1.0, 0.5])
    for a in 1.1:0.1:1.3
        @time all_res = map(1:ntrials) do _
            mmd_perm_test_power(kernelargs, m->randn(2,m), m->a*randn(2,m);
                batchsize = m, nperms = nperms, nsamples = nsamples)
        end
        c_alpha = mean(r->r.c_alpha, all_res); @show c_alpha
        P_alpha = mean(r->r.P_alpha, all_res); @show P_alpha
        P_alpha_approx = mean(r->r.P_alpha_approx, all_res); @show P_alpha_approx
        mmd_perm_test_power_plot(all_res[1]) |> display
    end
end
=#

####
#### Gradient testing
####

function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        tmp = x[i]
        δ = cbrt(eps()) # cbrt seems to be slightly better than sqrt; larger step size helps
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return grads
end

function gradcheck(f, xs...)
    dx0 = Flux.gradient(f, xs...)
    dx1 = ngradient(f, xs...)
    @show maximum.(abs, dx0)
    @show maximum.(abs, dx1)
    @show maximum.(abs, (dx0 .- dx1) ./ dx0)
    all(isapprox.(dx0, dx1, rtol = 1e-4, atol = 0))
end

####
#### Rician Distribution
####

#### Rician distribution: https://en.wikipedia.org/wiki/Rice_distribution
struct Rician{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    ν::T
    σ::T
    Rician{T}(ν::T, σ::T) where {T<:Real} = new{T}(ν, σ)
end

function Rician(ν::T, σ::T; check_args = true) where {T <: Real}
    check_args && Distributions.@check_args(Rician, σ >= zero(σ) && ν >= zero(ν))
    return Rician{T}(ν, σ)
end

#### Outer constructors
Rician(ν::Real, σ::Real) = Rician(promote(ν, σ)...)
Rician(ν::Integer, σ::Integer) = Rician(float(ν), float(σ))
Rician(ν::T) where {T <: Real} = Rician(ν, one(T))
Rician() = Rician(0.0, 1.0, check_args = false)

#### Conversions
Base.convert(::Type{Rician{T}}, ν::S, σ::S) where {T <: Real, S <: Real} = Rician(T(ν), T(σ))
Base.convert(::Type{Rician{T}}, d::Rician{S}) where {T <: Real, S <: Real} = Rician(T(d.ν), T(d.σ), check_args = false)

# Distributions.@distr_support Rician 0 Inf
Base.minimum(d::Union{Rician, Type{Rician}}) = 0
Base.maximum(d::Union{Rician, Type{Rician}}) = Inf

#### Parameters
Distributions.params(d::Rician) = (d.ν, d.σ)
@inline Distributions.partype(d::Rician{T}) where {T<:Real} = T

Distributions.location(d::Rician) = d.ν
Distributions.scale(d::Rician) = d.σ

Base.eltype(::Type{Rician{T}}) where {T} = T

#### Bessel function
_L½_bessel_kernel(x) = exp(x/2) * ((1-x) * besseli(0, -x/2) - x * besseli(1, -x/2))
_L½_series_kernel(x) = sqrt(-x/pi) * (256 - 64/x + 8/x^2 - 6/x^3 + 75/x^4) / 128
laguerre½(x, t = 20) = -x < t ? _L½_bessel_kernel(x) : _L½_series_kernel(x)

_logI0_bessel_kernel(z) = log(besseli(0, z) + eps(eltype(z)))
_logI0_series_kernel(z) = z - log(2*(pi*z) + eps(eltype(z)))/2 + log1p(1/8z + 9/(2*(8z)^2) - 9*25/(6*(8z)^3))
logbesseli0(z, t = 20)  = z < t ? _logI0_bessel_kernel(z) : _logI0_series_kernel(z)

#### Statistics
Distributions.mean(d::Rician) = d.σ * sqrt(pi/2) * laguerre½(-d.ν^2 / 2d.σ^2)
# Distributions.mode(d::Rician) = ?
# Distributions.median(d::Rician) = ?

Distributions.var(d::Rician) = 2 * d.σ^2 + d.ν^2 - pi * d.σ^2 * laguerre½(-d.ν^2 / 2d.σ^2)^2 / 2
Distributions.std(d::Rician) = sqrt(var(d))
# Distributions.skewness(d::Rician{T}) where {T<:Real} = ?
# Distributions.kurtosis(d::Rician{T}) where {T<:Real} = ?
# Distributions.entropy(d::Rician) = ?

#### Evaluation
Distributions.logpdf(d::Rician, x::Real) = log(x / d.σ^2 + eps(eltype(x))) + logbesseli0(x * d.ν / d.σ^2) - (x^2 + d.ν^2) / (2*d.σ^2)
Distributions.logpdf(d::Rician, x::AbstractVector{<:Real}) = logpdf.(d, x)
Distributions.pdf(d::Rician, x::Real) = exp(logpdf(d, x)) # below version errors for large x (besseli throws); otherwise is consistent
# Distributions.pdf(d::Rician, x::Real) = x * besseli(0, x * d.ν / d.σ^2) * exp(-(x^2 + d.ν^2) / (2*d.σ^2)) / d.σ^2

#### Sampling
Distributions.rand(rng::Distributions.AbstractRNG, d::Rician{T}) where {T} = sqrt((d.ν + d.σ * randn(rng, T))^2 + (d.σ * randn(rng, T))^2)

#### Testing
#= laguerre½
let
    f₊ = x -> laguerre½(-x, x + sqrt(eps()))
    f₋ = x -> laguerre½(-x, x - sqrt(eps()))
    df = x -> abs((f₊(x) - f₋(x))/f₊(x))
    # xs = range(1.0, 1000.0; length = 100)
    xs = range(1.0f0, 50.0f0; length = 100)
    p = plot()
    plot!(p, xs, (x -> laguerre½(-x)).(xs); lab = "laguerre½(-x)")
    plot!(p, xs, f₊.(xs); lab = "f_+")
    plot!(p, xs, f₋.(xs); lab = "f_-")
    display(p)
    plot(xs, log10.(df.(xs)); lab = "df") |> display
    log10.(df.(xs))
end
=#

#= logbesseli0
let
    f₊ = z -> logbesseli0(z, z + sqrt(eps()))
    f₋ = z -> logbesseli0(z, z - sqrt(eps()))
    df = z -> abs((f₊(z) - f₋(z))/f₊(z))
    # xs = range(1.0, 500.0; length = 100)
    xs = range(1.0f0, 50.0f0; length = 100)
    # plot(xs, (z -> logbesseli0(-z, 1000)).(xs); lab = "logbesseli0(-z)")
    # plot!(xs, f₊.(xs); lab = "f_+")
    # plot!(xs, f₋.(xs); lab = "f_-")
    plot(xs, log10.(df.(xs)); lab = "df") |> display
    log10.(df.(xs))
end
=#

#= (log)pdf
let
    p = plot()
    σ = 0.23
    xs = range(0.0, 8.0; length = 500)
    for ν in [0.0, 0.5, 1.0, 2.0, 4.0]
        d = Rician(ν, σ)
        plot!(p, xs, pdf.(d, xs); lab = "nu = $ν, sigma = $σ")
        x = 8.0 #rand(Uniform(xs[1], xs[end]))
        @show log(pdf(d, x))
        @show logpdf(d, x)
        @assert log(pdf(d, x)) ≈ logpdf(d, x)
    end
    display(p)
end
=#

#= mean/std/rand
using Plots
for ν in [0, 1, 10], σ in [1e-3, 1.0, 10.0]
    d = Rician(ν, σ)
    vline!(histogram([mean(rand(d,1000)) for _ in 1:1000]; nbins = 50), [mean(d)], line = (:black, :solid, 5), title = "nu = $ν, sigma = $σ") |> display
    vline!(histogram([std(rand(d,1000)) for _ in 1:1000]; nbins = 50), [std(d)], line = (:black, :solid, 5), title = "nu = $ν, sigma = $σ") |> display
end
=#
