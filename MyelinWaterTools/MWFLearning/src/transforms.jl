"""
Projection onto exponentials with tikhonov regularization, also known
as the inverse laplace transform.
For data `b` sampled at a rate η > 0, we write `b` as a sum of exponentials
with time constants `τ`:

    b_i = Σ_j exp(-η*i/τ_j) * x_j

where x_j are the unknown weights.
Since the problem is fundamentally ill-posed, Tikhonov regularization with
parameter α is performed, i.e. the output `x` is the minimizer of

    ||Ax - b||_2^2 + α ||x||_2^2

and is given by

    x = (A'A + α^2 * I)^{-1} A'b
"""
function ilaplace(b::AbstractVecOrMat, τ::AbstractVector, η::Number, α::Number = 1)
    # The below code is equivalent to the following (but is much faster):
    #   A = [exp(-ti/τj) for ti in t, τj in τ]
    #   x = (A'A + α^2*I)\(A'b)
    T = promote_type(eltype(b), eltype(τ), eltype(η), eltype(α))
    M, P = size(b)
    N = length(τ)
    t = η.*(1:M)
    bufs = (A = zeros(T, M, N), B = zeros(T, N, N), x = zeros(T, N, P))
    x = ilaplace!(bufs, b, τ, η, α)
    return copy(x)
end

function ilaplace!(bufs, b::AbstractVecOrMat, τ::AbstractVector, η::Number, α::Number = 1)
    M, P = size(b,1), size(b,2)
    N = length(τ)
    t = η.*(1:M)

    @unpack A, B, x = bufs
    @assert size(A) == (M, N) && size(B) == (N, N) && size(x,1) == N && size(x,2) == size(b,2)

    @inbounds for j in 1:N
        for i in 1:M
            A[i,j] = exp(-t[i]/τ[j]) # LHS matrix
        end
    end
    mul!(x, A', b) # RHS vector

    mul!(B, A', A)
    @inbounds for j in 1:N
        B[j,j] += α^2 # Tikhonov regularization
    end

    Bf = cholesky!(B)
    ldiv!(Bf, x) # Invert A'A + α^2*I onto A'b

    return x
end

"""
Discrete Laplace transform. For a stepsize η > 0, the discrete Laplace transform
of `x` is given by:
    L_η[x](s) = η * Σ_{k} exp(-s*k*η) * x[k]
"""
function dlt(x::AbstractVector, s::AbstractVector, η::Number)
    T = promote_type(eltype(x), eltype(s), typeof(η))
    y = zeros(T, length(s))
    @inbounds for j in eachindex(y,s)
        Σ = zero(T)
        ωj = exp(-η * s[j])
        ω = ωj
        for k in eachindex(x)
            Σ += ω * x[k]
            ω *= ωj
            # Σ += ω^k * x[k] # equivalent to above
        end
        y[j] = η * Σ
    end
    return y
end

"""
Mean displacement ratio (MDR) smoothing method.
If the data consists of N+1 discrete observations f_i, made at
equal increments of time Δt so that t_i = i Δt, the mean displaced
ratio is defined as
    
    Y_k = 1 / (N - L + 1) * Σ_{i = 0}^{N - L} f_{i+k} / f_i

where k in {0, 1, ..., L}. The degree of smoothing is controlled
by the choice of L, with increased smoothing achieved by
decreasing the L value. Ikossi-Anastasiou et al. and Dyson
et al. found that for the data with 5% - 10% noise, L = 0.9N
is a good choice.
"""
function mdr!(y::AbstractVector, x::AbstractVector)
    N = length(x) - 1
    L = length(y) - 1
    @assert L <= N
    @inbounds for k in 1:L+1
        Σ = zero(eltype(x))
        for i in 1:N-L+1
            Σ += x[i+k-1] / x[i]
        end
        y[k] = Σ / (N-L+1)
    end
    return y
end
mdr(x::AbstractVector, L::Int = floor(Int, 0.9 * length(x))) =
    mdr!(zeros(eltype(x), L+1), x)

"""
Provencher smoothing method.
If the data consists of N discrete observations f_i, made at
equal increments of time Δt so that t_i = i Δt, the Provencher
smoothing method is defined as
    
    Y(t_k) = Σ_{m = 1}^{L} f(t_m) * f(t_m + t_k - t_1) /
             Σ_{m = 1}^{L} f^2(t_m)

where k in {0, 1, ..., N-L+1}. Increased smoothing is associated
with increased L.
"""
function provencher!(y::AbstractVector, x::AbstractVector)
    N = length(x)
    L = N - length(y) + 1
    @assert L <= N
    Σ² = zero(eltype(x))
    @inbounds for m in 1:L
        Σ² += x[m]^2
    end
    @inbounds for k in 1:N-L+1
        Σ = zero(eltype(x))
        for m in 1:L
            Σ += x[m] * x[m+k-1]
        end
        y[k] = Σ / Σ²
    end
    return y
end
provencher(x::AbstractVector, L::Int = ceil(Int, 0.1 * length(x))) =
    provencher!(zeros(eltype(x), length(x)-L+1), x)

"""
Peel method

First buffer element is the output vector `z`, equal to the input `x` with two
exponential modes "peeled" off.
Second buffer element `y` is the biexponential sum which was peeled off.
Third buffer element is a temporary buffer not to be relied upon.
"""
function peel!(bufs::NTuple{3,A}, x::A, Nslow = 5, Nfast = 5) where {A <: AbstractVector}
    z, y, e = bufs
    copyto!(z, x)
    fill!(y, 0)
    tranges = [Nslow:length(z), 1:Nfast]
    
    function fitpeel(t)
        @unpack α, β = linfit(t, log.(abs.(z[t])))
        e .= exp.(α .+ β .* (1:length(z)))
        z .-= e # Subtract component off of original vector
        y .+= e # Add component to smoothed vector
        @ntuple(α, β)
    end

    return [fitpeel(t) for t in tranges]
end
peel(x::AbstractVector, args...; kwargs...) = peel!(ntuple(_->copy(x), 3), x, args...; kwargs...)

function linfit(x,y)
    β = cov(x,y) / var(x)
    α = mean(y) - β * mean(x)
    return @ntuple(α, β)
end

"""
Wavelet filtering
"""
function applydwt(
        b,
        nterms = length(b),
        wt = Wavelets.wavelet(Wavelets.WT.cdf97, Wavelets.WT.Lifting),
        # wt = Wavelets.wavelet(Wavelets.WT.db4),
    )
    w = Wavelets.dwt(b, wt)
    Wavelets.threshold!(w, Wavelets.BiggestTH(), nterms)
    wb = w[findall(!iszero, w)]
    return wb
end

function chopdwt(x, nterms)
    # n = length(x)
    # d = Vector{eltype(x)}(undef, nterms)
    # J = Wavelets.ndyadicscales(n)

    # @inbounds for j = 0:J-1, i in 1:Wavelets.dyadicdetailn(j)
    #     idx = Wavelets.dyadicdetailindex(j,i)
    #     @show idx
    #     (idx <= nterms) && (d[idx] = x[idx])
    # end
    # error("done")

    # return d

    return x[1:nterms]
end

function plotdwt(;
        # fname = nothing, TODO
        thresh = 0.1, # dots below `thresh` are zeroed
        # wt = Wavelets.wavelet(Wavelets.WT.cdf97, Wavelets.WT.Lifting),
        wt = Wavelets.wavelet(Wavelets.WT.db4),
        t = nothing,
        x = nothing,
        w = nothing,
    )

    (x == nothing && w == nothing) && (x = Wavelets.testfunction(512, "Bumps"))
    (x != nothing && w == nothing) && (w = Wavelets.dwt(x, wt))
    (x == nothing && w != nothing) && (x = Wavelets.idwt(w, wt))
    (t == nothing) && (t = 0:length(x)-1)

    @assert length(x) == length(w)

    n = length(w)
    J = floor(Int, log2(n))
    d, l = Wavelets.wplotdots(w, thresh, n)
    A = Wavelets.wplotim(w)

    xt, yt = 0:n-2, 0:J-1
    p1 = RecipesBase.plot(t, x; lc = :black, xlim = (1,length(x)), xtick = [1,length(x)], ylab = L"f(x)", leg = :none)
    for r in 1:2
        xr = Wavelets.idwt(chopdwt(w, length(w) ÷ 2^r), wt)
        RecipesBase.plot!(p1, 2^r .* t, xr)
    end
    p2 = RecipesBase.plot(d, l; st = :scatter, ytick = 0:J, xlim = (-0.5,n-1.5), ylab = L"level $j$", lab = "Thresh = $thresh", leg = :bottomright)
    p3 = RecipesBase.plot(xt, yt, A; st = :heatmap, ytick = 0:J, ylim = (-0.5, J-0.5), ylab = L"level $j$", cb = :none)

    p = RecipesBase.plot(p1,p2,p3; layout = (3,1))

    return p
end