"""
X = LSQNONNEG_REG(C, d, Chi2Factor) returns the regularized NNLS solution X
that incurrs an increase in chi^2 by a factor of Chi2Factor.
"""
function lsqnonneg_reg(C, d, Chi2Factor)
    work = lsqnonneg_reg_work(C, d, Chi2Factor)
    lsqnonneg_reg!(work, C, d, Chi2Factor)
end

function lsqnonneg_reg_work(C, d, Chi2Factor)
    d_backproj = similar(d)
    resid = similar(d)
    nnls_work = NNLSWorkspace(C, d)
    C_smooth = [copy(C); zeros(size(C))]
    d_smooth = [copy(d); zeros(size(d))]
    nnls_work_smooth = NNLSWorkspace(C_smooth, d_smooth)
    X = zeros(size(C,2))
    return @ntuple(d_backproj, resid, nnls_work, C_smooth, d_smooth, nnls_work_smooth, X)
end

function lsqnonneg_reg!(work, C, d, Chi2Factor)
    # Unpack workspace
    @unpack nnls_work, d_backproj, resid = work
    @unpack nnls_work_smooth, C_smooth, d_smooth, X = work
    @views C_smooth_bottom = C_smooth[end÷2+1:end, :]

    # Find non-regularized solution
    lsqnonneg!(nnls_work, C, d)
    mul!(d_backproj, C, nnls_work.x)
    resid .= d .- d_backproj
    chi2_min = sum(abs2, resid)

    # Initialzation of various components
    mu_cache = [0.0]
    chi2_cache = [chi2_min]

    # Minimize energy of spectrum; loop to find largest mu that keeps chi-squared in desired range
    while chi2_cache[end] < Chi2Factor*chi2_min
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
    
    # Smooth the chi2(mu) curve using spline fit
    mu_spline = 0:0.001:mu_cache[end]
    deg_spline = min(3, length(mu_cache)-1)
    chi2_spline = Spline1D(mu_cache, chi2_cache; k=deg_spline).(mu_spline)

    # Find the index of the minimum chi-squared satisfying the increase factor
    _, min_ind = findmin(abs.(chi2_spline .- Chi2Factor.*chi2_min))
    mu = mu_spline[min_ind]
    
    # Compute the regularized solution
    set_diag!(C_smooth_bottom, mu)
    lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
    mul!(d_backproj, C, nnls_work_smooth.x)
    resid .= d .- d_backproj
    chi2_final = sum(abs2, resid)
    
    # Verify actual chi2 increase factor
    Chi2FactorActual = chi2_final/chi2_min
    
    # Assign output
    X .= nnls_work_smooth.x

    return @ntuple(X, mu, Chi2FactorActual)
end
