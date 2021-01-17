####
#### Abstract MMD kernel interface
####

abstract type MMDKernel{T} end

struct FunctionKernel{T,F} <: MMDKernel{T}
    f::F
    FunctionKernel{T}(f) where {T} = new{T,typeof(f)}(f)
end
Flux.functor(::Type{<:FunctionKernel{T}}, k) where {T} = (f = k.f,), k -> FunctionKernel{T}(k)
(k::FunctionKernel)(Δ) = k.f(Δ)

struct DeepExponentialKernel{T,N,A<:AbstractArray{T,N},F} <: MMDKernel{T}
    logσ::A
    phi::F
    DeepExponentialKernel(logσ::AbstractArray, phi = identity) = new{eltype(logσ),ndims(logσ),typeof(logσ),typeof(phi)}(logσ, phi)
end
Flux.@functor DeepExponentialKernel
logbandwidths(k::DeepExponentialKernel) = k.logσ
featuremap(k::DeepExponentialKernel) = k.phi

struct MMDResults{T}
    m::T
    e_K̃xx_e::T
    e_K̃yy_e::T
    e_K̃xy_e::T
end

struct MMDVarResults{T}
    m::T
    K̃xx_F2::T
    K̃yy_F2::T
    Kxy_F2::T
    e_K̃xx_e::T
    e_K̃yy_e::T
    e_Kxy_e::T
    e_K̃xy_e::T
    K̃xx_e_F2::T
    K̃yy_e_F2::T
    Kxy_e_F2::T
    Kyx_e_F2::T
    e_K̃xx_Kxy_e::T
    e_K̃yy_Kyx_e::T
end

function mmd(res::Union{<:MMDResults, <:MMDVarResults})
    @unpack m, e_K̃xx_e, e_K̃yy_e, e_K̃xy_e = res

    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    MMDsq = (e_K̃xx_e + e_K̃yy_e - 2e_K̃xy_e) / (m * (m-1))

    return MMDsq
end

function mmd_and_mmdvar(res::MMDVarResults)
    #   NOTE: Here we assume a symmetric kernel k(x,y) == k(y,x),
    #         and therefore that Kxx == Kxx', Kyy = Kyy', Kxy == Kxy'
    # See:
    #   [1] https://arxiv.org/pdf/1906.02104.pdf
    #   [2] http://www.gatsby.ucl.ac.uk/~dougals/slides/dali/#/50
    @unpack m, K̃xx_F2, K̃yy_F2, Kxy_F2, e_K̃xx_e, e_K̃yy_e, e_Kxy_e, e_K̃xy_e, K̃xx_e_F2, K̃yy_e_F2, Kxy_e_F2, Kyx_e_F2, e_K̃xx_Kxy_e, e_K̃yy_Kyx_e = res
    m_2 = m * (m-1)
    m_3 = m_2 * (m-2)
    m_4 = m_3 * (m-3)

    # MMD²_U: MMD estimator which is a U-statistic
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    MMDsq = mmd(res)

    # Var[MMD²_U]: Variance estimator
    #   See: https://arxiv.org/pdf/1906.02104.pdf
    t1_4 = ((   4) / (m_4    )) * (K̃xx_e_F2 + K̃yy_e_F2)
    t2_4 = ((4m^2) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t2_5 = ((  4m) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t2_6 = ((   4) / (m_2^3  )) * (Kxy_e_F2 + Kyx_e_F2) # NOTE: typo in original paper: m^3*(m-1)^2 --> m^3*(m-1)^3 = m_2^3
    t3_4 = ((   8) / (m*m_3  )) * (e_K̃xx_Kxy_e + e_K̃yy_Kyx_e)
    t4_5 = ((   8) / (m^2*m_3)) * ((e_K̃xx_e + e_K̃yy_e) * e_Kxy_e)
    t5_4 = ((  4m) / (m_2*m_4)) * (e_K̃xx_e^2 + e_K̃yy_e^2)
    t5_5 = ((   6) / (m_2*m_4)) * (e_K̃xx_e^2 + e_K̃yy_e^2)
    t6_4 = ((  8m) / (m_2^3  )) * (e_Kxy_e^2)
    t6_5 = ((  12) / (m_2^3  )) * (e_Kxy_e^2)
    t7_4 = ((   2) / (m_4    )) * (K̃xx_F2 + K̃yy_F2)
    t8_4 = ((4m^2) / (m_2^3  )) * (Kxy_F2)
    t8_5 = ((  8m) / (m_2^3  )) * (Kxy_F2)
    MMDvar = (((t1_4 + t2_4) - (t3_4 + t5_4 + t6_4 + t7_4 + t8_4)) + ((t4_5 + t5_5 + t6_5 + t8_5) - t2_5)) - t2_6 # NOTE: typo in original paper: +t8 --> -t8

    return MMDsq, MMDvar
end

mmdvar(res::MMDVarResults) = mmd_and_mmdvar(res)[2]

function tstat(k::MMDKernel, X, Y, isillposed = nothing)
    # Avoiding div by zero/negative:
    #   m^2 * MMDvar >= ϵ  -->  m * MMDσ >= √ϵ
    MMDsq, MMDvar = mmd_and_mmdvar(k, X, Y)
    m = size(X,2)
    ϵ = eps(typeof(MMDvar))
    t = m*MMDsq / √max(m^2*MMDvar, ϵ)
    (isillposed !== nothing) && Zygote.@ignore(isillposed[] = m^2*MMDvar < ϵ) #TODO: MMDsq < 0 --> ill-posed?
    return t
end

####
#### Generic MMD using buffer matrices
####

mmd!(k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmd!(mmd_work(X,Y), k, X, Y)
mmd!(work, k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmd(kernel_mmd_stats!(work, k, X, Y))
mmdvar!(k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmdvar!(mmd_work(X,Y), k, X, Y)
mmdvar!(work, k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmdvar(kernel_mmdvar_stats!(work, k, X, Y))
mmd_and_mmdvar!(k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmd_and_mmdvar!(mmd_work(X,Y), k, X, Y)
mmd_and_mmdvar!(work, k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = mmd_and_mmdvar(kernel_mmdvar_stats!(work, k, X, Y))

function mmd_work(T::Type, sz::NTuple{2,Int})
    Kxx, Kyy, Kxy = ntuple(_ -> zeros(T, sz[2], sz[2]), 3)
    K̃xx_e, K̃yy_e, Kxy_e, Kyx_e = ntuple(_ -> zeros(T, sz[2]), 4)
    return @ntuple(Kxx, Kyy, Kxy, K̃xx_e, K̃yy_e, Kxy_e, Kyx_e)
end
mmd_work(sz::NTuple{2,Int}) = mmd_work(Float64, sz)
mmd_work(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T} = (@assert size(X) == size(Y); return mmd_work(T, size(X)))

function kernel_pairwise!(Kxy::AbstractMatrix{T}, k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, ::Val{skipdiag} = Val(false)) where {T,skipdiag}
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

function kernel_mmd_stats!(work, k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy = work
    m = size(X, 2)
    e_K̃xx_e = sum(kernel_pairwise!(Kxx, k, X, X, Val(true)))
    e_K̃yy_e = sum(kernel_pairwise!(Kyy, k, Y, Y, Val(true)))
    e_Kxy_e = sum(kernel_pairwise!(Kxy, k, X, Y, Val(true)))
    return MMDResults{T}(m, e_K̃xx_e, e_K̃yy_e, e_Kxy_e)
end

function kernel_mmdvar_stats!(work, k::FunctionKernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    @assert size(X) == size(Y)
    @unpack Kxx, Kyy, Kxy, K̃xx_e, K̃yy_e, Kxy_e, Kyx_e = work

    kernel_pairwise!(Kxx, k, X, X, Val(true))
    kernel_pairwise!(Kyy, k, Y, Y, Val(true))
    kernel_pairwise!(Kxy, k, X, Y, Val(false))

    sum_columns!(K̃xx_e, Kxx)
    sum_columns!(K̃yy_e, Kyy)
    sum_columns!(Kxy_e, Kxy)
    sum_rows!(Kyx_e, Kxy)

    K̃xx_F2, K̃yy_F2, Kxy_F2 = frob_norm2(Kxx), frob_norm2(Kyy), frob_norm2(Kxy)
    e_K̃xx_e, e_K̃yy_e, e_Kxy_e = sum(Kxx), sum(Kyy), sum(Kxy)
    e_K̃xy_e = e_Kxy_e - tr(Kxy)
    e_K̃xx_Kxy_e, e_K̃yy_Kyx_e = dot(K̃xx_e, Kxy_e), dot(K̃yy_e, Kyx_e)
    K̃xx_e_F2, K̃yy_e_F2, Kxy_e_F2, Kyx_e_F2 = frob_norm2(K̃xx_e), frob_norm2(K̃yy_e), frob_norm2(Kxy_e), frob_norm2(Kyx_e)

    return MMDVarResults{T}(size(X,2), K̃xx_F2, K̃yy_F2, Kxy_F2, e_K̃xx_e, e_K̃yy_e, e_Kxy_e, e_K̃xy_e, K̃xx_e_F2, K̃yy_e_F2, Kxy_e_F2, Kyx_e_F2, e_K̃xx_Kxy_e, e_K̃yy_Kyx_e)
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
#### MMD using a generic kernel function
####

mmd(k::FunctionKernel, X::AbstractMatrix, Y::AbstractMatrix) = mmd(kernel_mmd_stats(k, X, Y))
mmdvar(k::FunctionKernel, X::AbstractMatrix, Y::AbstractMatrix) = mmdvar(kernel_mmdvar_stats(k, X, Y))
mmd_and_mmdvar(k::FunctionKernel, X::AbstractMatrix, Y::AbstractMatrix) = mmd_and_mmdvar(kernel_mmdvar_stats(k, X, Y))

function kernel_pairwise_sum(k::FunctionKernel, X::AbstractMatrix, Y::AbstractMatrix, ::Val{skipdiag} = Val(false)) where {skipdiag}
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

function kernel_mmd_stats(k::FunctionKernel, X::AbstractMatrix, Y::AbstractMatrix)
    # Ref: http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    @assert size(X) == size(Y)
    e_K̃xx_e = kernel_pairwise_sum(k, X, X, Val(true))
    e_K̃yy_e = kernel_pairwise_sum(k, Y, Y, Val(true))
    e_Kxy_e = kernel_pairwise_sum(k, X, Y, Val(true))
    Tk = promote_type(typeof(e_K̃xx_e), typeof(e_K̃yy_e), typeof(e_Kxy_e))
    return MMDResults{Tk}(size(X,2), e_K̃xx_e, e_K̃yy_e, e_Kxy_e)
end

function kernel_mmdvar_stats(k::FunctionKernel, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)
    m  = size(X, 2)
    Tk = typeof(k(zero(promote_type(eltype(X), eltype(Y)))))
    K̃xx_F2 = K̃yy_F2 = Kxy_F2 = e_K̃xx_e = e_K̃yy_e = e_Kxy_e = K̃xx_e_F2 = K̃yy_e_F2 = Kxy_e_F2 = Kyx_e_F2 = e_K̃xx_Kxy_e = e_K̃yy_Kyx_e = zero(Tk)

    @inbounds for j = 1:m
        K̃xx_e_j = K̃yy_e_j = Kxy_e_j = Kyx_e_j = zero(Tk)
        @inbounds @simd for i in 1:j-1
            kxx_ij = k(column_mse(X, X, i, j))
            kyy_ij = k(column_mse(Y, Y, i, j))
            kxy_ij = k(column_mse(X, Y, i, j))
            kyx_ij = k(column_mse(Y, X, i, j))
            K̃xx_F2 += 2 * kxx_ij * kxx_ij
            K̃yy_F2 += 2 * kyy_ij * kyy_ij
            Kxy_F2 += kxy_ij * kxy_ij
            e_K̃xx_e += 2 * kxx_ij
            e_K̃yy_e += 2 * kyy_ij
            e_Kxy_e += kxy_ij
            K̃xx_e_j += kxx_ij
            K̃yy_e_j += kyy_ij
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
            Kxy_F2 += kxy_ij * kxy_ij
            e_Kxy_e += kxy_ij
            K̃xx_e_j += kxx_ij
            K̃yy_e_j += kyy_ij
            Kxy_e_j += kyx_ij
            Kyx_e_j += kxy_ij
        end
        K̃xx_e_F2 += K̃xx_e_j * K̃xx_e_j
        K̃yy_e_F2 += K̃yy_e_j * K̃yy_e_j
        Kxy_e_F2 += Kxy_e_j * Kxy_e_j
        Kyx_e_F2 += Kyx_e_j * Kyx_e_j
        e_K̃xx_Kxy_e += K̃xx_e_j * Kxy_e_j
        e_K̃yy_Kyx_e += K̃yy_e_j * Kyx_e_j
    end

    return MMDVarResults{Tk}(m, K̃xx_F2, K̃yy_F2, Kxy_F2, e_K̃xx_e, e_K̃yy_e, e_Kxy_e, K̃xx_e_F2, K̃yy_e_F2, Kxy_e_F2, Kyx_e_F2, e_K̃xx_Kxy_e, e_K̃yy_Kyx_e)
end

####
#### MMD using sums of exponential kernels
####

mmd(k::DeepExponentialKernel, X::AbstractMatrix, Y::AbstractMatrix) = mmd(mmd_flux_u_statistic(mmd_flux_kernel_matrices(k.logσ, k.phi(X), k.phi(Y), Val(false))...))
mmdvar(k::DeepExponentialKernel, X::AbstractMatrix, Y::AbstractMatrix) = mmdvar(mmdvar_flux_u_statistic(mmd_flux_kernel_matrices(k.logσ, k.phi(X), k.phi(Y), Val(false))...))
mmd_and_mmdvar(k::DeepExponentialKernel, X::AbstractMatrix, Y::AbstractMatrix) = mmd_and_mmdvar(mmdvar_flux_u_statistic(mmd_flux_kernel_matrices(k.logσ, k.phi(X), k.phi(Y), Val(false))...))

function mmd_flux_u_statistic(Kxx, Kyy, Kxy)
    @assert size(Kxx) == size(Kyy) == size(Kxy)
    Tk = promote_type(eltype(Kxx), eltype(Kyy), eltype(Kxy))
    m = size(Kxx,1)
    e_K̃xx_e = sum(Kxx) - m # assumes k(0) == 1 --> tr(Kxx) = m
    e_K̃yy_e = sum(Kyy) - m # assumes k(0) == 1 --> tr(Kyy) = m
    e_K̃xy_e = sum(Kxy) - tr(Kxy)
    return MMDResults{Tk}(m, e_K̃xx_e, e_K̃yy_e, e_K̃xy_e)
end

function mmdvar_flux_u_statistic(Kxx, Kyy, Kxy)
    @assert size(Kxx) == size(Kyy) == size(Kxy)
    Tk = promote_type(eltype(Kxx), eltype(Kyy), eltype(Kxy))
    m = size(Kxx,1)

    e_K̃xx_e = sum(Kxx) - m # assumes k(0) == 1
    e_K̃yy_e = sum(Kyy) - m # assumes k(0) == 1
    e_Kxy_e = sum(Kxy)
    e_K̃xy_e = e_Kxy_e - tr(Kxy)
    K̃xx_F2 = sum(abs2, Kxx) - m # assumes k(0) == 1
    K̃yy_F2 = sum(abs2, Kyy) - m # assumes k(0) == 1
    Kxy_F2 = sum(abs2, Kxy)
    K̃xx_e = sum(Kxx; dims = 2) .- 1 # assumes k(0) == 1
    K̃yy_e = sum(Kyy; dims = 2) .- 1 # assumes k(0) == 1
    Kxy_e = sum(Kxy; dims = 2)
    Kyx_e = sum(Kxy; dims = 1)
    K̃xx_e_F2 = sum(abs2, K̃xx_e)
    K̃yy_e_F2 = sum(abs2, K̃yy_e)
    Kxy_e_F2 = sum(abs2, Kxy_e)
    Kyx_e_F2 = sum(abs2, Kyx_e)
    e_K̃xx_Kxy_e = dot(vec(K̃xx_e), vec(Kxy_e))
    e_K̃yy_Kyx_e = dot(vec(K̃yy_e), vec(Kyx_e))

    return MMDVarResults{Tk}(m, K̃xx_F2, K̃yy_F2, Kxy_F2, e_K̃xx_e, e_K̃yy_e, e_Kxy_e, e_K̃xy_e, K̃xx_e_F2, K̃yy_e_F2, Kxy_e_F2, Kyx_e_F2, e_K̃xx_Kxy_e, e_K̃yy_Kyx_e)
end

# Speed testing of mmd_flux_kernel_matrices
function _bench_mmd_and_mmdvar_cpu_vs_gpu(;T = Float32, n = 128, m = 2048)
    @assert CUDA.functional()
    cpu_and_gpu(x) = (Flux.cpu(x), Flux.gpu(x))
    _isapprox(x,y) = isapprox(Flux.cpu(x), Flux.cpu(y); rtol = sqrt(eps(T)), atol = sqrt(eps(T)))
    X, Xc = T(0.1) .* randn(T,n,m) |> cpu_and_gpu
    Y, Yc = T(2.0) .* randn(T,n,m) |> cpu_and_gpu

    for nchan in [n,1], nbw in [32,1], f in [mmdvar, mmd]
        @show f, n, m, nbw, nchan
        logσ, logσc = (nchan > 1 ? randn(T, nbw, nchan) : randn(T, nbw)) |> cpu_and_gpu
        k, kc = (logσ, logσc) .|> DeepExponentialKernel

        @assert _isapprox(@show(f(k,X,Y)), @show(f(kc,Xc,Yc)))

        y, back = Zygote.pullback((x,y) -> f(k,x,y), X, Y)
        dy1, dy2 = back(one(T))

        yc, backc = Zygote.pullback((xc,yc) -> f(kc,xc,yc), Xc, Yc)
        dy1c, dy2c = backc(one(T))

        @assert _isapprox(@show(y), @show(yc))
        @assert _isapprox(@show(norm(dy1)), @show(norm(dy1c))) && _isapprox(@show(norm(dy2)), @show(norm(dy2c)))
        @assert _isapprox(dy1, dy1c) && _isapprox(dy2, dy2c)

        print("cpu call:   "); @btime CUDA.@sync $f($k, $X, $Y)
        print("gpu call:   "); @btime CUDA.@sync $f($kc, $Xc, $Yc)
        print("cpu forward:"); _, back = @btime CUDA.@sync Zygote.pullback((x,y) -> $f($k,x,y), $X, $Y)
        print("gpu forward:"); _, back = @btime CUDA.@sync Zygote.pullback((xc,yc) -> $f($kc,xc,yc), $Xc, $Yc)
        print("cpu reverse:"); @btime CUDA.@sync $back($(one(T)))
        print("gpu reverse:"); @btime CUDA.@sync $backc($(one(T)))
    end
end;

####
#### Kernel matrices with generic kernel function
####

function mmd_flux_kernel_matrices(k::FunctionKernel, X::AbstractMatrix, Y::AbstractMatrix)
    @assert size(X) == size(Y)

    n = size(X,1)
    XX, XY, YY = X'X, X'Y, Y'Y
    xx, yy = batched_diag(XX), batched_diag(YY) # squared norms on diagonal
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

mmd_flux_kernel_matrices(logsigma::AbstractVector, args...) = mmd_flux_kernel_matrices(reshape(logsigma, 1, 1, length(logsigma)), args...) # reshape for broadcasting
mmd_flux_kernel_matrices(logsigma::AbstractMatrix, args...) = mmd_flux_kernel_matrices(reshape(permutedims(logsigma), size(logsigma,2), 1, :), args...) # reshape for broadcasting

function mmd_flux_kernel_matrices(logsigma::AbstractTensor3D, X::AbstractMatrix, Y::AbstractMatrix, batched::Val{true})
    @assert size(X) == size(Y)
    n, m = size(X)
    γ = @. inv(sqrt(2n * exp(2 * logsigma))) # γ = √(1/2n*sigma^2) = 1/√(2n*exp(2logsigma))
    return _mmd_flux_kernel_matrices(γ .* X, γ .* Y)
end

function mmd_flux_kernel_matrices(logsigma::AbstractTensor3D, X::AbstractMatrix, Y::AbstractMatrix, batched::Val{false})
    @assert size(X) == size(Y)
    n, m = size(X)
    nσ = size(logsigma, 3)

    # Compute matrices one slice at a time
    γ = @. inv(sqrt(2n * exp(2 * logsigma))) # γ = √(1/2n*sigma^2) = 1/√(2n*exp(2logsigma))
    γk = γ[:,1,1]
    Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices(γk .* X, γk .* Y)

    for k in 2:nσ
        γk = γ[:,1,k]
        _Kxx, _Kyy, _Kxy = _mmd_flux_kernel_matrices(γk .* X, γk .* Y)
        Kxx += _Kxx
        Kyy += _Kyy
        Kxy += _Kxy
    end

    Kxx /= nσ
    Kyy /= nσ
    Kxy /= nσ

    return @ntuple(Kxx, Kyy, Kxy)
end

# Speed testing of mmd_flux_kernel_matrices
function _bench_mmd_flux_kernel_matrices(;T = Float32, n = 128, m = 2048, gpu::Bool = false)
    maybegpu = gpu ? Flux.gpu : Flux.cpu
    X = randn(T,n,m) |> maybegpu
    Y = randn(T,n,m) |> maybegpu
    Δ = (rand(T,m,m) |> maybegpu for _ in 1:3) |> ((Kxx, Kyy, Kxy),) -> @ntuple(Kxx, Kyy, Kxy)

    for nbw in [1,8], nchan in [1] #[1,n]
        @show n, m, nbw, nchan
        logsigma = (nchan > 1 ? randn(T, nbw, nchan) : randn(T, nbw)) |> maybegpu

        @assert all(values(mmd_flux_kernel_matrices(logsigma,X,Y,Val(false))) .≈ values(mmd_flux_kernel_matrices(logsigma,X,Y,Val(true))))

        _y, _back = Zygote.pullback((_X,_Y) -> mmd_flux_kernel_matrices(logsigma,_X,_Y,Val(false)), X, Y)
        _dyA, _dyB = _back(Δ)

        y, back = Zygote.pullback((_X,_Y) -> mmd_flux_kernel_matrices(logsigma,_X,_Y,Val(true)), X, Y)
        dyA, dyB = back(Δ)

        @assert all(values(_y) .≈ values(y))
        @assert _dyA ≈ dyA
        @assert _dyB ≈ dyB

        for isbatched in [true, false]
            f = (_s,_X,_Y) -> mmd_flux_kernel_matrices(_s,_X,_Y,Val(isbatched))
            print("isbatched=$isbatched call:   "); @btime CUDA.@sync $f($logsigma, $X, $Y)
            print("isbatched=$isbatched forward:"); _, back = @btime CUDA.@sync Zygote.pullback((_X,_Y) -> $f($logsigma,_X,_Y), $X, $Y)
            print("isbatched=$isbatched reverse:"); @btime CUDA.@sync $back($Δ)
        end
    end
end;

#=
let # Consistency between vectors/matrices of logsigma
    logsigma1 = randn(4)
    logsigma2 = repeat(logsigma1, 1, 10)
    X, Y = randn(10,4), randn(10,4)
    out1 = mmd_flux_kernel_matrices(logsigma1, X, Y, Val(false))
    out2 = mmd_flux_kernel_matrices(logsigma2, X, Y, Val(false))
    isapprox.(values(out1), values(out2))
end
=#

####
#### Flux differentiable MMD kernel matrices
####

function _mmd_flux_kernel_matrices(X::AbstractMatrix, Y::AbstractMatrix)
    Kxx, Kyy, Kxy = X'X, Y'Y, X'Y
    xx, yy = batched_diag(Kxx), batched_diag(Kyy)
    Threads.@threads for j in 1:size(Kxx,2)
        @avx for i in 1:size(Kxx,1)
            Kxx[i,j] = exp(2 * Kxx[i,j] - xx[i] - xx[j])
            Kyy[i,j] = exp(2 * Kyy[i,j] - yy[i] - yy[j])
            Kxy[i,j] = exp(2 * Kxy[i,j] - xx[i] - yy[j])
        end
    end
    @ntuple(Kxx, Kyy, Kxy)
end

Zygote.@adjoint function _mmd_flux_kernel_matrices(X::AbstractMatrix, Y::AbstractMatrix)
    # Store kernel matrices for reverse pass
    @unpack Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices(X, Y)

    return @ntuple(Kxx, Kyy, Kxy), function(Δ)
        ΔKxx, ΔKyy, ΔKxy = Δ.Kxx, Δ.Kyy, Δ.Kxy

        # dK_dX
        @tullio Δ_buf[i,j] := @inbounds ΔKxx[i,j] * Kxx[i,j]
        @tullio Δ_buf[i,j] += @inbounds Δ_buf[j,i]
        Δ_buf_sumbuf = vec(sum(Δ_buf; dims = 1))
        mul_buf = X * Δ_buf
        @tullio dK_dX[i,j] := @inbounds 2 * (mul_buf[i,j] - X[i,j] * Δ_buf_sumbuf[j])

        # dK_dX/dK_dY cross-terms
        @tullio Δ_buf[i,j] = @inbounds ΔKxy[i,j] * Kxy[i,j]
        Δ_buf_sumbuf = vec(sum(Δ_buf; dims = 1))
        mul!(mul_buf, X, Δ_buf)
        @tullio dK_dY[i,j] := @inbounds 2 * (mul_buf[i,j] - Y[i,j] * Δ_buf_sumbuf[j])

        Δ_buf_sumbuf = vec(sum(Δ_buf; dims = 2))
        mul!(mul_buf, Y, Δ_buf')
        @tullio dK_dX[i,j] += @inbounds 2 * (mul_buf[i,j] - X[i,j] * Δ_buf_sumbuf[j])

        # dK_dY
        @tullio Δ_buf[i,j] = @inbounds ΔKyy[i,j] * Kyy[i,j]
        @tullio Δ_buf[i,j] += @inbounds Δ_buf[j,i]
        Δ_buf_sumbuf = vec(sum(Δ_buf; dims = 1))
        mul!(mul_buf, Y, Δ_buf)
        @tullio dK_dY[i,j] += @inbounds 2 * (mul_buf[i,j] - Y[i,j] * Δ_buf_sumbuf[j])

        return dK_dX, dK_dY
    end
end

function _mmd_flux_kernel_matrices(X::CuMatrix, Y::CuMatrix)
    Kxx, Kyy, Kxy = X'X, Y'Y, X'Y
    xx, yy = batched_diag(Kxx), batched_diag(Kyy)
    Kxx .= exp.(2 .* Kxx .- xx .- xx')
    Kyy .= exp.(2 .* Kyy .- yy .- yy')
    Kxy .= exp.(2 .* Kxy .- xx .- yy')
    @ntuple(Kxx, Kyy, Kxy)
end

Zygote.@adjoint function _mmd_flux_kernel_matrices(X::CuMatrix, Y::CuMatrix)
    # Store kernel matrices for reverse pass
    @unpack Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices(X, Y)

    return @ntuple(Kxx, Kyy, Kxy), function(Δ) # much faster + much less memory usage
        # dK_dX
        Δ_buf = Δ.Kxx .* Kxx
        Δ_buf .+= Δ_buf'
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        mul_buf = X * Δ_buf
        dK_dX = 2 .* (mul_buf .- X .* Δ_buf_rowsum)

        # dK_dX/dK_dY cross-terms
        Δ_buf .= Δ.Kxy .* Kxy
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        mul!(mul_buf, X, Δ_buf)
        dK_dY = 2 .* (mul_buf .- Y .* Δ_buf_rowsum)

        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        mul!(mul_buf, Y, Δ_buf')
        dK_dX .+= 2 .* (mul_buf .- X .* Δ_buf_colsum')

        # dK_dY
        Δ_buf .= Δ.Kyy .* Kyy
        Δ_buf .+= Δ_buf'
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        mul!(mul_buf, Y, Δ_buf)
        dK_dY .+= 2 .* (mul_buf .- Y .* Δ_buf_rowsum)

        return dK_dX, dK_dY
    end
end

#=
function _mmd_tullio_kernel_matrices end
function _∇mmd_tullio_kernel_matrices end
function _mmd_tullio_kernel_matrices_inner end

function _mmd_tullio_kernel_matrices(X::AbstractMatrix, Y::AbstractMatrix)
    Kxx, Kyy, Kxy = X'X, Y'Y, X'Y
    xx, yy = batched_diag(Kxx), batched_diag(Kyy)
    @tullio Kxx[i,j] = exp(2 * Kxx[i,j] - xx[i] - xx[j])
    @tullio Kyy[i,j] = exp(2 * Kyy[i,j] - yy[i] - yy[j])
    @tullio Kxy[i,j] = exp(2 * Kxy[i,j] - xx[i] - yy[j])
    @ntuple(Kxx, Kyy, Kxy)
end

function _∇mmd_tullio_kernel_matrices(Δ,X,Y,Kxx,Kyy,Kxy) # much faster + much less memory usage
    ΔKxx, ΔKyy, ΔKxy = Δ.Kxx, Δ.Kyy, Δ.Kxy
    # dK_dX
    @tullio Δ_buf[i,j] := @inbounds ΔKxx[i,j] * Kxx[i,j] # Δ_buf = Δ.Kxx .* Kxx
    @tullio Δ_buf[i,j] += @inbounds Δ_buf[j,i]
    Δ_buf_sumbuf = vec(sum(Δ_buf; dims = 1))
    mul_buf = X * Δ_buf
    @tullio dK_dX[i,j] := @inbounds 2 * (mul_buf[i,j] - X[i,j] * Δ_buf_sumbuf[j])

    # dK_dX/dK_dY cross-terms
    @tullio Δ_buf[i,j] = @inbounds ΔKxy[i,j] * Kxy[i,j]
    Δ_buf_sumbuf = vec(sum(Δ_buf; dims = 1))
    mul!(mul_buf, X, Δ_buf)
    @tullio dK_dY[i,j] := @inbounds 2 * (mul_buf[i,j] - Y[i,j] * Δ_buf_sumbuf[j])

    Δ_buf_sumbuf = vec(sum(Δ_buf; dims = 2))
    mul!(mul_buf, Y, Δ_buf')
    @tullio dK_dX[i,j] += @inbounds 2 * (mul_buf[i,j] - X[i,j] * Δ_buf_sumbuf[j])

    # dK_dY
    @tullio Δ_buf[i,j] = @inbounds ΔKyy[i,j] * Kyy[i,j]
    @tullio Δ_buf[i,j] += @inbounds Δ_buf[j,i]
    Δ_buf_sumbuf = vec(sum(Δ_buf; dims = 1))
    mul!(mul_buf, Y, Δ_buf)
    @tullio dK_dY[i,j] += @inbounds 2 * (mul_buf[i,j] - Y[i,j] * Δ_buf_sumbuf[j])

    return dK_dX, dK_dY
end

Zygote.@adjoint function _mmd_tullio_kernel_matrices(X::AbstractMatrix, Y::AbstractMatrix)
    # Store kernel matrices for reverse pass
    @unpack Kxx, Kyy, Kxy = _mmd_tullio_kernel_matrices(X, Y)
    return @ntuple(Kxx, Kyy, Kxy), Δ -> _∇mmd_tullio_kernel_matrices(Δ, X, Y, Kxx, Kyy, Kxy)
end

function _mmd_tullio_kernel_matrices_inner(X::AbstractTensor3D, Y::AbstractTensor3D)
    Kxx, Kyy, Kxy = NNlib.batched_mul(NNlib.batched_transpose(X), X), NNlib.batched_mul(NNlib.batched_transpose(Y), Y), NNlib.batched_mul(NNlib.batched_transpose(X), Y)
    xx, yy = batched_diag(Kxx), batched_diag(Kyy)
    Tullio.@tullio Kxx[i,j,k] = exp(2 * Kxx[i,j,k] - xx[i,1,k] - xx[j,1,k])
    Tullio.@tullio Kyy[i,j,k] = exp(2 * Kyy[i,j,k] - yy[i,1,k] - yy[j,1,k])
    Tullio.@tullio Kxy[i,j,k] = exp(2 * Kxy[i,j,k] - xx[i,1,k] - yy[j,1,k])
    @ntuple(Kxx, Kyy, Kxy)
end

function _mmd_tullio_kernel_matrices(X::AbstractTensor3D, Y::AbstractTensor3D)
    @unpack Kxx, Kyy, Kxy = _mmd_tullio_kernel_matrices_inner(X, Y)
    γ = inv(eltype(Kxx)(size(Kxx,3)))
    Tullio.@tullio _Kxx[i,j] := γ * Kxx[i,j,k]
    Tullio.@tullio _Kyy[i,j] := γ * Kyy[i,j,k]
    Tullio.@tullio _Kxy[i,j] := γ * Kxy[i,j,k]
    (Kxx = _Kxx, Kyy = _Kyy, Kxy = _Kxy)
end

Zygote.@adjoint function _mmd_tullio_kernel_matrices(X::AbstractTensor3D, Y::AbstractTensor3D)
    # Store kernel matrices for reverse pass
    @unpack Kxx, Kyy, Kxy = _mmd_tullio_kernel_matrices_inner(X, Y)
    γ = inv(eltype(Kxx)(size(Kxx,3)))
    Tullio.@tullio _Kxx[i,j] := γ * Kxx[i,j,k]
    Tullio.@tullio _Kyy[i,j] := γ * Kyy[i,j,k]
    Tullio.@tullio _Kxy[i,j] := γ * Kxy[i,j,k]
    out = (Kxx = _Kxx, Kyy = _Kyy, Kxy = _Kxy)
    return out, function(Δ) # much faster + much less memory usage
        ΔKxx, ΔKyy, ΔKxy = Δ.Kxx, Δ.Kyy, Δ.Kxy

        # dK_dX
        @tullio Δ_buf[i,j,k] := @inbounds γ * ΔKxx[i,j] * Kxx[i,j,k]
        @tullio Δ_buf[i,j,k] += @inbounds Δ_buf[j,i,k]
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        mul_buf = NNlib.batched_mul(X, Δ_buf)
        @tullio dK_dX[i,j,k] := @inbounds 2 * (mul_buf[i,j,k] - X[i,j,k] * Δ_buf_rowsum[1,j,k])

        # dK_dX/dK_dY cross-terms
        @tullio Δ_buf[i,j,k] = @inbounds γ * ΔKxy[i,j] * Kxy[i,j,k]
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        NNlib.batched_mul!(mul_buf, X, Δ_buf)
        @tullio dK_dY[i,j,k] := @inbounds 2 * (mul_buf[i,j,k] - Y[i,j,k] * Δ_buf_rowsum[1,j,k])

        # Δ_buf_colsum = permutedims(sum(Δ_buf; dims = 2), (2,1,3))
        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        NNlib.batched_mul!(mul_buf, Y, NNlib.batched_transpose(Δ_buf))
        @tullio dK_dX[i,j,k] += @inbounds 2 * (mul_buf[i,j,k] - X[i,j,k] * Δ_buf_colsum[j,1,k])

        # dK_dY
        @tullio Δ_buf[i,j,k] = @inbounds γ * ΔKyy[i,j] * Kyy[i,j,k]
        @tullio Δ_buf[i,j,k] += @inbounds Δ_buf[j,i,k]
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        NNlib.batched_mul!(mul_buf, Y, Δ_buf)
        @tullio dK_dY[i,j,k] += @inbounds 2 * (mul_buf[i,j,k] - Y[i,j,k] * Δ_buf_rowsum[1,j,k])

        return dK_dX, dK_dY
    end
end
=#

function _mmd_flux_kernel_matrices_inner(X::AbstractTensor3D, Y::AbstractTensor3D)
    Kxx = NNlib.batched_mul(NNlib.batched_transpose(X), X)
    Kyy = NNlib.batched_mul(NNlib.batched_transpose(Y), Y)
    Kxy = NNlib.batched_mul(NNlib.batched_transpose(X), Y)
    xx, yy = batched_diag(Kxx), batched_diag(Kyy)
    Threads.@sync for k in 1:size(Kxx,3) #TODO can be improved; k is usually small
        Threads.@spawn begin
            @avx for j in 1:size(Kxx,2), i in 1:size(Kxx,1)
                Kxx[i,j,k] = exp(2 * Kxx[i,j,k] - xx[i,1,k] - xx[j,1,k])
                Kyy[i,j,k] = exp(2 * Kyy[i,j,k] - yy[i,1,k] - yy[j,1,k])
                Kxy[i,j,k] = exp(2 * Kxy[i,j,k] - xx[i,1,k] - yy[j,1,k])
            end
        end
    end
    @ntuple(Kxx, Kyy, Kxy)
end

function _mmd_flux_kernel_matrices_inner(X::CuTensor3D, Y::CuTensor3D)
    Kxx = NNlib.batched_mul(NNlib.batched_transpose(X), X)
    Kyy = NNlib.batched_mul(NNlib.batched_transpose(Y), Y)
    Kxy = NNlib.batched_mul(NNlib.batched_transpose(X), Y)
    xx  = batched_diag(Kxx)
    yy  = batched_diag(Kyy)
    xxT = batched_transpose(xx)
    yyT = batched_transpose(yy)
    Kxx .= exp.(2 .* Kxx .- xx .- xxT)
    Kyy .= exp.(2 .* Kyy .- yy .- yyT)
    Kxy .= exp.(2 .* Kxy .- xx .- yyT)
    @ntuple(Kxx, Kyy, Kxy)
end

function _mmd_flux_kernel_matrices(X::AbstractTensor3D, Y::AbstractTensor3D)
    @unpack Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices_inner(X, Y)
    Kxx, Kyy, Kxy = mean3(Kxx), mean3(Kyy), mean3(Kxy)
    @ntuple(Kxx, Kyy, Kxy)
end

Zygote.@adjoint function _mmd_flux_kernel_matrices(X::AbstractTensor3D, Y::AbstractTensor3D)
    # Store kernel matrices for reverse pass
    @unpack Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices_inner(X, Y)
    out = (Kxx = mean3(Kxx), Kyy = mean3(Kyy), Kxy = mean3(Kxy))
    return out, function(Δ) # much faster + much less memory usage
        ΔKxx, ΔKyy, ΔKxy = Δ.Kxx, Δ.Kyy, Δ.Kxy
        γ = inv(eltype(Kxx)(size(Kxx,3)))

        # dK_dX
        @tullio Δ_buf[i,j,k] := @inbounds γ * ΔKxx[i,j] * Kxx[i,j,k]
        @tullio Δ_buf[i,j,k] += @inbounds Δ_buf[j,i,k]
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        mul_buf = NNlib.batched_mul(X, Δ_buf)
        @tullio dK_dX[i,j,k] := @inbounds 2 * (mul_buf[i,j,k] - X[i,j,k] * Δ_buf_rowsum[1,j,k])

        # dK_dX/dK_dY cross-terms
        @tullio Δ_buf[i,j,k] = @inbounds γ * ΔKxy[i,j] * Kxy[i,j,k]
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        NNlib.batched_mul!(mul_buf, X, Δ_buf)
        @tullio dK_dY[i,j,k] := @inbounds 2 * (mul_buf[i,j,k] - Y[i,j,k] * Δ_buf_rowsum[1,j,k])

        # Δ_buf_colsum = permutedims(sum(Δ_buf; dims = 2), (2,1,3))
        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        NNlib.batched_mul!(mul_buf, Y, NNlib.batched_transpose(Δ_buf))
        @tullio dK_dX[i,j,k] += @inbounds 2 * (mul_buf[i,j,k] - X[i,j,k] * Δ_buf_colsum[j,1,k])

        # dK_dY
        @tullio Δ_buf[i,j,k] = @inbounds γ * ΔKyy[i,j] * Kyy[i,j,k]
        @tullio Δ_buf[i,j,k] += @inbounds Δ_buf[j,i,k]
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        NNlib.batched_mul!(mul_buf, Y, Δ_buf)
        @tullio dK_dY[i,j,k] += @inbounds 2 * (mul_buf[i,j,k] - Y[i,j,k] * Δ_buf_rowsum[1,j,k])

        return dK_dX, dK_dY
    end
end

Zygote.@adjoint function _mmd_flux_kernel_matrices(X::CuTensor3D, Y::CuTensor3D)
    # Store kernel matrices for reverse pass
    @unpack Kxx, Kyy, Kxy = _mmd_flux_kernel_matrices_inner(X, Y)
    out = (Kxx = mean3(Kxx), Kyy = mean3(Kyy), Kxy = mean3(Kxy))
    return out, function(Δ) # much faster + much less memory usage
        T = NNlib.batched_transpose # lazy transpose for `batched_mul`
        P = batched_transpose # (possibly eager) permutation
        γ = inv(eltype(Kxx)(size(Kxx,3)))

        # dK_dX
        Δ_buf = γ .* Δ.Kxx .* Kxx
        add_transpose!(Δ_buf)
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        mul_buf = NNlib.batched_mul(X, Δ_buf)
        dK_dX = 2 .* (mul_buf .- X .* Δ_buf_rowsum)

        # dK_dX/dK_dY cross-terms
        Δ_buf .= γ .* Δ.Kxy .* Kxy
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        NNlib.batched_mul!(mul_buf, X, Δ_buf)
        dK_dY = 2 .* (mul_buf .- Y .* Δ_buf_rowsum)

        Δ_buf_colsum = sum(Δ_buf; dims = 2)
        NNlib.batched_mul!(mul_buf, Y, T(Δ_buf))
        dK_dX .+= 2 .* (mul_buf .- X .* P(Δ_buf_colsum))

        # dK_dY
        Δ_buf .= γ .* Δ.Kyy .* Kyy
        add_transpose!(Δ_buf)
        Δ_buf_rowsum = sum(Δ_buf; dims = 1)
        NNlib.batched_mul!(mul_buf, Y, Δ_buf)
        dK_dY .+= 2 .* (mul_buf .- Y .* Δ_buf_rowsum)

        return dK_dX, dK_dY
    end
end

function _test_mmd_flux_kernel_matrices(fs = [_mmd_flux_kernel_matrices], fcs = []; T = Float32, n = 10, m = 10, p = 0)
    X = (p <= 0 ? randn(T,n,m) : randn(T,n,m,p))
    Y = (p <= 0 ? randn(T,n,m) : randn(T,n,m,p))
    Xc = isempty(fcs) ? nothing : Flux.gpu(X)
    Yc = isempty(fcs) ? nothing : Flux.gpu(Y)

    function fwd_and_back(f,X,Y)
        out, back = Zygote.pullback((x,y) -> f(x,y), X, Y)
        grad = back(out)
        return @ntuple(out, grad, back)
    end

    function compare(val1, val2)
        for (k,v1,v2) in zip(keys(val1), values(val1), values(val2))
            cv1, cv2 = Flux.cpu(v1), Flux.cpu(v2)
            cmp = isapprox(cv1, cv2; rtol = sqrt(eps(T)), atol = 10 * eps(T))
            err = maximum(abs, cv1 .- cv2) # / max(sqrt(eps(T)), maximum(abs, cv1), maximum(abs, cv2))
            @show k, cmp, err
        end
    end

    fout = isempty(fs) ? [] : map(f -> fwd_and_back(f, X, Y), fs)
    fcout = isempty(fcs) ? [] : map(fc -> fwd_and_back(fc, Xc, Yc), fcs)
    allfs = vcat(fs, fcs)
    allfout = vcat(fout, fcout)

    for i in 1:length(allfout), j in i+1:length(allfout)
        labi, labj = (fout -> fout.out isa CuArray ? "gpu" : "cpu").((allfout[i], allfout[j]))
        println("$(allfs[i]) ($labi) vs. $(allfs[j]) ($labj)")
        compare(allfout[i].out, allfout[j].out)
        compare(allfout[i].grad, allfout[j].grad)
    end

    !isempty(fout) && map(zip(fs, fout)) do (f, (out, _, back))
        println("$f (cpu)")
        @btime $f($X, $Y)
        @btime $back($out)
    end

    !isempty(fcout) && map(zip(fcs, fcout)) do (fc, (out, _, back))
        println("$fc (gpu)")
        @btime CUDA.@sync $fc($Xc, $Yc)
        @btime CUDA.@sync $back($out)
    end

    return nothing
end

#=
# Testing adjoint for _mmd_flux_kernel_matrices
let
    Random.seed!(0)

    # Dummy version for Zygote to auto-diff through
    function _kernel_mats(X::AbstractMatrix, Y::AbstractMatrix)
        XX, YY, XY = X'X, Y'Y, X'Y
        xx, yy = batched_diag(XX), batched_diag(YY)
        Kxx = exp.(2 .* XX .- xx .- xx')
        Kyy = exp.(2 .* YY .- yy .- yy')
        Kxy = exp.(2 .* XY .- xx .- yy')
        @ntuple(Kxx, Kyy, Kxy)
    end
    function _kernel_mats(X::AbstractTensor3D, Y::AbstractTensor3D)
        XX = NNlib.batched_mul(NNlib.batched_transpose(X), X)
        YY = NNlib.batched_mul(NNlib.batched_transpose(Y), Y)
        XY = NNlib.batched_mul(NNlib.batched_transpose(X), Y)
        xx = batched_diag(XX)
        yy = batched_diag(YY)

        T = x -> permutedims(x, (2,1,3))
        Kxx = exp.(2 .* XX .- xx .- T(xx))
        Kyy = exp.(2 .* YY .- yy .- T(yy))
        Kxy = exp.(2 .* XY .- xx .- T(yy))
        return (Kxx = mean3(Kxx), Kyy = mean3(Kyy), Kxy = mean3(Kxy))
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
    @unpack Kxx, Kyy, Kxy = mmd_flux_kernel_matrices(kernelargs, X, Y, Val(true))
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
    for _ in 2:nsamples
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

        showplot && (p !== nothing) && display(p)
        return p
    catch e
        handleinterrupt(e; msg = "Error during permutation plot")
    end
end

#=
let m = 100, n = 2, nbw = 4, nperms = 128, nsamples = 100, ntrials = 10
    # γ = inv(2 * 2.0^2)
    # kernelargs = d -> exp(-γ * d)
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
#### Combinatorial bandwidth opt
####
function combinatorial_kernel_opt(k::DeepExponentialKernel, X, Y, σbucket; batchsize::Int, nsamples::Int, maxiters::Int, replace::Bool = true, verbose::Bool = false, pthresh = 1/batchsize)
    mmdsamples(knew) = map(1:nsamples) do _
        Xm, Ym = sample_columns(X, batchsize), sample_columns(Y, batchsize)
        batchsize * mmd(knew, Xm, Ym)
    end
    kbest = deepcopy(k)
    mmdbest = mmdsamples(kbest)
    for i in 1:maxiters
        inds = sample(eachindex(σbucket), length(logbandwidths(kbest)); replace)
        σnew = reshape(σbucket[inds], size(logbandwidths(kbest)))
        knew = DeepExponentialKernel(σnew)
        mmdnew = mmdsamples(knew)
        t = HypothesisTests.UnequalVarianceTTest(Float64.(mmdnew), Float64.(mmdbest))
        p = HypothesisTests.pvalue(t)
        if p < pthresh && mean(mmdnew) > mean(mmdbest)
            verbose && @info "i = $i: Updating... mmd = $(mean(mmdnew)) > mmd = $(mean(mmdbest)) (p = $p)"
            kbest = deepcopy(knew)
            mmdbest = copy(mmdnew)
        else
            verbose && @info "i = $i: mmd = $(mean(mmdnew)) <= mmd = $(mean(mmdbest)) (p = $p)"
        end
    end
    return kbest
end

####
#### Kernel bandwidth opt
####

function train_kernel!(k::MMDKernel{T}, X, Y, opt, Y2 = nothing; kernelloss::String, kernelsteps::Int, restrict! = nothing, verbose::Bool = false) where {T}
    # Kernel is trained with new optimizer for each X, Y; loss jumps too wildly
    isillposed = Ref(false)
    tstat_loss() = -tstat(k, X, Y, isillposed) # maximize tstat(X,Y)
    mmd_loss() = -mmd(k, X, Y) # maximize mmd(X,Y)
    mmd_diff_loss() = mmd(k, Y, Y2) - mmd(k, X, Y) # minimize mmd(Y1,Y2), maximize mmd(X,Y1)
    loss =
        (kernelloss == "mmd") ? mmd_loss :
        (kernelloss == "tstatistic") ? tstat_loss :
        (kernelloss == "mmd_diff") ? mmd_diff_loss :
        error("Unknown kernel loss: $kernelloss")

    # Training should not be performed using t-statistic loss function if isillposed[] == true
    ps = Flux.params(k)
    for i in 1:kernelsteps
        @timeit "forward" ℓ, back = Zygote.pullback(loss, ps)
        verbose && @info i, -ℓ, isillposed[]
        isillposed[] && break

        @timeit "reverse" ∇ℓ = back(one(T))
        Flux.Optimise.update!(opt, ps, ∇ℓ)

        (restrict! !== nothing) && restrict!(k)
    end

    # Let caller know if training was applied
    success = !isillposed[]
    return success
end
