# Helper functions
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

# MMD using buffer matrices
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

# Flux differentiable MMD
DiagmOp(x::AbstractVector) = diagm(x)
Flux.Zygote.@adjoint function DiagmOp(x::AbstractVector)
    return diagm(x)::AbstractMatrix, Δ::AbstractMatrix -> (diag(Δ),)
end
#=
let
    l = x -> sum(DiagmOp(x))
    x = randn(3)
    @show ngradient(l, x)[1]
    @show Flux.gradient(l, x)
    gradcheck(l, x)
end
=#

DiagOp(x::AbstractMatrix) = diag(x)
Flux.Zygote.@adjoint function DiagOp(x::AbstractMatrix)
    return diag(x)::AbstractVector, Δ -> (diagm(Δ[:]),) # why is Δ a matrix?
end
#=
let
    l = x -> sum(DiagOp(x))
    x = randn(3,3)
    @show ngradient(l, x)[1]
    @show Flux.gradient(l, x)
    gradcheck(l, x)
end
=#

function mmd_flux(k, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)
    n, m = size(X)

    # XX, XY, YY = X'X, X'Y, Y'Y
    # xx, yy = LinearAlgebra.diag(XX), LinearAlgebra.diag(YY) # squared norms on diagonal
    # Kxx = k.((xx .- 2 .* XX .+ xx')./n) # note: mean is over data length n, not number of data m
    # Kyy = k.((yy .- 2 .* YY .+ yy')./n)
    # Kxy = k.((xx .- 2 .* XY .+ yy')./n)
    # Kxy = Kxy - LinearAlgebra.Diagonal(Kxy)

    XX, XY, YY = X'X, X'Y, Y'Y
    xx, yy = DiagOp(XX), DiagOp(YY) # squared norms on diagonal
    Kxx = k.((xx .- 2 .* XX .+ xx')./n) # note: mean is over data length n, not number of data m
    Kyy = k.((yy .- 2 .* YY .+ yy')./n)
    Kxy = k.((xx .- 2 .* XY .+ yy')./n)
    Kxy = Kxy - DiagmOp(DiagOp(Kxy))

    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    e_Kxx_e = sum(Kxx) - m # assumes k(0) == 1
    e_Kyy_e = sum(Kyy) - m # assumes k(0) == 1
    e_Kxy_e = sum(Kxy)
    MMDsq = (e_Kxx_e + e_Kyy_e - 2e_Kxy_e) / (m*(m-1))

    return MMDsq
end
#=
let
    model = Flux.Dense(10,10)
    X, Y = randn(10,100), randn(10,100)
    k = d -> exp(-d)
    loss = () -> mmd_flux(k, model(X), Y)
    @btime $loss()
    @btime Flux.gradient($loss, Flux.params($model))
end
=#

# MMD without buffers
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
#=
for a in [3], m in [30]
    k = Δ -> exp(-Δ)
    # sampleX = () -> randn(2,m)
    # sampleY = () -> ((1/√2) * [1 -1; 1 1] * [√a 0; 0 1/√a]) * randn(2,m)
    sampleX = () -> rand(2,m)
    sampleY = () -> [2-1/a; 1/a] .* rand(2,m)
    X, Y = sampleX(), sampleY()
    v1 = @btime mmdvar!($(mmd_work(X, Y)), $k, $X, $Y)
    # @btime kernel_var_stats($k, $X, $Y)
    v2 = @btime mmdvar($k, $X, $Y)
    @show v1, v2, v1-v2, (v1 - v2)/v1
end
=#

#=
for m in [50]
    k = Δ -> exp(-Δ)
    X, Y = randn(2,m), 2 .* randn(2,m)
    v1 = mmd!(mmd_work(X, Y), k, X, Y)
    v2 = mmd(k, X, Y)
    v3 = mmd_flux(k, X, Y)
    @assert v1 ≈ v2 && v2 ≈ v3
    # @show v1, v2, v3, v1-v2, v2-v3
end
=#

# Permutation testing
function mmd_permutation_test(k, sampleX, sampleY; niters = 1000, alpha = 0.01)
    n, m = size(sampleX())
    c_α_samples = [m * mmd(k, mix_columns(sampleX(), sampleY())...) for _ in 1:niters]
    c_α = quantile(c_α_samples, 1-alpha)
    mmd_samples = [mmd(k, sampleX(), sampleY()) for _ in 1:niters]
    P_α = count(MMD -> m * MMD > c_α, mmd_samples) / niters

    MMDsq = mean(mmd_samples)
    MMDvar = var(mmd_samples)
    MMDσ = √(m * MMDvar)
    z = √m * MMDsq / MMDσ - c_α / (√m * MMDσ)
    P_α_approx = cdf(Normal(), z)

    return @ntuple(c_α, P_α, P_α_approx, MMDsq, MMDσ, c_α_samples, mmd_samples)
end

# Gradient testing
#=
function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return grads
end

gradcheck(f, xs...) = all(isapprox.(ngradient(f, xs...), gradient(f, xs...), rtol = 1e-3, atol = 1e-16))

gs = Flux.gradient(loss, X, Y)
fs = ngradient(loss, X, Y)
gradcheck(loss, X, Y)

∇loss = (X, Y) -> Flux.gradient(() -> loss(X, Y), Flux.params(model))
gs = collect(values(∇loss(X,Y).grads))
fs = ngradient((ps...) -> loss(X, Y), Flux.params(model)...)
=#
