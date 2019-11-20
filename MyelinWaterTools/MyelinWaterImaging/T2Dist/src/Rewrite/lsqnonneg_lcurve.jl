"""
LSQNONNEG_LCURVE(C, d) returns the regularized NNLS solution, X,
by minimizing (C*X-d).^2 + mu*(H*X).^2. H is the identity
matrix and mu is chosen by the L-curve theory using the Generalized 
Cross-Validation method. Chi2Factor is the ratio of the final Chi^2 to 
the minimum Chi^2 (mu=0).

Details of L-curve and GCV methods can be found in:
Hansen, P.C., 1992. Analysis of Discrete Ill-Posed Problems by Means of
the L-Curve. SIAM Review, 34(4), 561-580.

Created by Thomas Prasloski
email: tprasloski@gmail.com
Version 1.1, July 2012
"""
function lsqnonneg_lcurve_work(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @assert size(C,1) == length(d)
    d_backproj = zeros(T, length(d))
    resid = zeros(T, length(d))
    nnls_work = NNLSWorkspace(C, d)
    C_smooth = zeros(T, size(C,1) + size(C,2), size(C,2))
    d_smooth = zeros(T, length(d) + size(C,2))
    nnls_work_smooth = NNLSWorkspace(C_smooth, d_smooth)
    @views C_smooth_top, d_smooth_top = C_smooth[1:size(C,1), :], d_smooth[1:length(d)]
    @views C_smooth_bottom, d_smooth_bottom = C_smooth[size(C,1)+1:end, :], d_smooth[length(d)+1:end]
    A_mu = zeros(T, size(C,1), size(C,1))
    Ct_tmp = zeros(T, size(C,2), size(C,1))
    CtC_tmp = zeros(T, size(C,2), size(C,2))
    x = zeros(T, size(C,2))
    mu_opt = Ref(T(NaN))
    chi2fact_opt = Ref(T(NaN))
    return @ntuple(
        d_backproj, resid, nnls_work,
        C_smooth, d_smooth, nnls_work_smooth,
        C_smooth_top, C_smooth_bottom, d_smooth_bottom, d_smooth_top,
        A_mu, Ct_tmp, CtC_tmp,
        x, mu_opt, chi2fact_opt
    )
end

function lsqnonneg_lcurve(C, d)
    work = lsqnonneg_lcurve_work(C, d)
    lsqnonneg_lcurve!(work, C, d)
end

function lsqnonneg_lcurve!(work, C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    # Unpack workspace
    @unpack nnls_work, d_backproj, resid = work
    @unpack nnls_work_smooth, C_smooth, d_smooth = work
    @unpack C_smooth_top, C_smooth_bottom, d_smooth_bottom, d_smooth_top = work

    # Assign top and bottom of C_smooth, d_smooth
    @assert size(C,1) == length(d)
    @assert size(C_smooth,1) == length(d_smooth) == size(C,1) + size(C,2)
    C_smooth_top .= C; C_smooth_bottom .= 0
    d_smooth_top .= d; d_smooth_bottom .= 0

    # Find mu by minimizing the function G(mu) (GCV method)
    mu = @timeit_debug TIMER "L-curve Optimization" begin
        opt_result = Optim.optimize(μ -> lcurve_G(μ, C, d, work), T(0), T(0.1); abs_tol = T(1e-3), rel_tol = T(1e-3))
        Optim.minimizer(opt_result)
    end

    # Compute the regularized solution
    set_diag!(C_smooth_bottom, mu)
    @timeit_debug TIMER "Reg. lsqnonneg!" begin
        lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
    end
    mul!(d_backproj, C, nnls_work_smooth.x)
    resid .= d .- d_backproj
    chi2_final = sum(abs2, resid)

    # Find non-regularized solution
    @timeit_debug TIMER "Non-reg. lsqnonneg!" begin
        lsqnonneg!(nnls_work, C, d)
    end
    mul!(d_backproj, C, nnls_work.x)
    resid .= d .- d_backproj
    chi2_min = sum(abs2, resid)

    # Assign output
    work.x .= nnls_work_smooth.x
    work.mu_opt[] = mu
    work.chi2fact_opt[] = chi2_final/chi2_min

    return (x = work.x, mu = work.mu_opt[], Chi2FactorActual = work.chi2fact_opt[])
end

# function rdiv_lu!(A, B)
#     B_lu = lu!(B)
#     rdiv!(rdiv!(A, UpperTriangular(B_lu.factors)), UnitLowerTriangular(B_lu.factors))
#     _apply_inverse_ipiv_cols!(B_lu, A)
# end

function lcurve_G(mu, C, d, work)
    @unpack d_backproj, resid = work
    @unpack nnls_work_smooth, C_smooth, d_smooth = work
    @unpack C_smooth_top, C_smooth_bottom, d_smooth_bottom, d_smooth_top = work
    @unpack A_mu, Ct_tmp, CtC_tmp = work

    # C_smooth = [C; mu * Matrix(I, size(C, 2), size(C, 2))]
    # d_smooth = [d; zeros(eltype(d), size(C,2))]
    # resid = d - C * lsqnonneg(C_smooth, d_smooth)
    # A_mu = C * ((C' * C + mu * I) \ C')

    C_smooth_top .= C; C_smooth_bottom .= 0
    d_smooth_top .= d; d_smooth_bottom .= 0
    set_diag!(C_smooth_bottom, mu)
    
    @timeit_debug TIMER "lsqnonneg!" begin
        lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
        mul!(d_backproj, C, nnls_work_smooth.x)
        resid .= d .- d_backproj
    end

    # C is (m,n), Ct_tmp is (n,m), CtC_tmp is (n,n), A_mu is (m,m)
    mul!(CtC_tmp, C', C)
    @inbounds for i in 1:size(CtC_tmp,1) # CtC_tmp .+= mu * I
        CtC_tmp[i,i] += mu
    end
    @timeit_debug TIMER "ldiv!" begin
        ldiv!(Ct_tmp, lu!(CtC_tmp), C')
    end
    mul!(A_mu, C, Ct_tmp)
    trace = zero(eltype(A_mu))
    @inbounds for i in 1:size(A_mu,1) # tr(I - A_mu)
        trace += (1 - A_mu[i,i])
    end

    return sqrt(sum(abs2, resid)) / trace^2
end