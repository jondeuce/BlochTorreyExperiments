"""
x = LSQNONNEG_REG(C, d, Chi2Factor) returns the regularized NNLS solution x
that incurrs an increase in chi^2 by a factor of Chi2Factor.
"""
function lsqnonneg_reg(C, d, Chi2Factor)
    work = lsqnonneg_reg_work(C, d, Chi2Factor)
    lsqnonneg_reg!(work, C, d, Chi2Factor)
end

function lsqnonneg_reg_work(C, d, Chi2Factor)
    d_backproj = zeros(size(d))
    resid = zeros(size(d))
    nnls_work = NNLSWorkspace(C, d)
    C_smooth = [copy(C); zeros(size(C))]
    d_smooth = [copy(d); zeros(size(d))]
    nnls_work_smooth = NNLSWorkspace(C_smooth, d_smooth)
    x = zeros(size(C,2))
    mu_opt = Ref(NaN)
    chi2fact_opt = Ref(NaN)
    return @ntuple(
        d_backproj, resid, nnls_work,
        C_smooth, d_smooth, nnls_work_smooth,
        x, mu_opt, chi2fact_opt
    )
end

function lsqnonneg_reg!(work, C, d, Chi2Factor)
    # Unpack workspace
    @unpack nnls_work, d_backproj, resid = work
    @unpack nnls_work_smooth, C_smooth, d_smooth = work
    
    # Assign top and bottom of C_smooth, d_smooth
    @views C_smooth_top, d_smooth_top = C_smooth[1:end÷2, :], d_smooth[1:end÷2]
    @views C_smooth_bottom, d_smooth_bottom = C_smooth[end÷2+1:end, :], d_smooth[end÷2+1:end]
    C_smooth_top .= C
    d_smooth_top .= d
    C_smooth_bottom .= 0
    d_smooth_bottom .= 0

    # Find non-regularized solution
    lsqnonneg!(nnls_work, C, d)
    mul!(d_backproj, C, nnls_work.x)
    resid .= d .- d_backproj
    chi2_min = sum(abs2, resid)

    # Initialzation of various components
    mu_cache = [0.0]
    chi2_cache = [chi2_min]

    # Minimize energy of spectrum; loop to find largest mu that keeps chi-squared in desired range
    while chi2_cache[end] < Chi2Factor * chi2_min
        # Incrememt mu vector
        if mu_cache[end] > 0
            push!(mu_cache, 2*mu_cache[end])
        else
            push!(mu_cache, 0.001)
        end

        # Compute T2 distribution with smoothing
        set_diag!(C_smooth_bottom, mu_cache[end])
        lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
        
        # Find predicted curve and calculate residuals and chi-squared
        mul!(d_backproj, C, nnls_work_smooth.x)
        resid .= d .- d_backproj
        push!(chi2_cache, sum(abs2, resid))
    end
    
    # Smooth the chi2(mu) curve using a spline fit and find the mu value such
    # that chi2 increases by Chi2Factor, i.e. chi2(mu) = Chi2Factor * chi2_min
    mu = spline_root(mu_cache, chi2_cache, Chi2Factor * chi2_min)

    # Compute the regularized solution
    set_diag!(C_smooth_bottom, mu)
    lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
    mul!(d_backproj, C, nnls_work_smooth.x)
    resid .= d .- d_backproj
    chi2_final = sum(abs2, resid)
    
    # Verify actual chi2 increase factor
    Chi2FactorActual = chi2_final/chi2_min
    
    # Assign output
    work.x .= nnls_work_smooth.x
    work.mu_opt[] = mu
    work.chi2fact_opt[] = Chi2FactorActual

    return (x = work.x, mu = work.mu_opt, Chi2FactorActual = work.chi2fact_opt)
end
