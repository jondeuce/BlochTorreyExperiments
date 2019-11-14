"""
    Options structure for T2map_SEcorr
"""
@with_kw struct Options @deftype Float64
    TE::Union{Float64, String} = 0.010
    @assert TE isa Float64 ? 0.0001 <= TE <= 1.0 : TE == "variable"
    
    nTE::Int = 32
    @assert nTE >= 1
    
    vTEparam::Tuple{Float64,Float64,Int} = (0.01, 0.05, 16)
    @assert vTEparam[2] > vTEparam[1] && vTEparam[1] * round(Int, vTEparam[2]/vTEparam[1]) ≈ vTEparam[2] && vTEparam[3] < nTE
    
    T1 = 1.0
    @assert 0.001 <= T1 <= 10.0
    
    RefCon = 180.0
    @assert 1.0 <= RefCon <= 180.0
    
    Threshold = 200.0
    @assert Threshold >= 0.0
    
    Chi2Factor = 1.02
    @assert Chi2Factor > 1
    
    nT2::Int = 40
    @assert 10 <= nT2 <= 120
    
    T2Range::Tuple{Float64,Float64} = (0.015, 2.0)
    @assert 0.001 <= T2Range[1] < T2Range[2] <= 10.0
    
    MinRefAngle = 50.0
    @assert 1.0 < MinRefAngle < 180.0
    
    nAngles::Int = 8
    @assert nAngles > 1
    
    Reg::String = "chi2"
    @assert Reg ∈ ("no", "chi2", "lcurve")
    
    SetFlipAngle::Union{Float64,Nothing} = nothing
    @assert SetFlipAngle isa Nothing || 0.0 < SetFlipAngle <= 180.0
    
    nCores::Int = Threads.nthreads()
    @assert 1 <= nCores
    
    Save_regparam::String = "no"
    @assert Save_regparam ∈ ("yes", "no")
    
    Save_NNLS_basis::String = "no"
    @assert Save_NNLS_basis ∈ ("yes", "no")
    
    Waitbar::String = "no"
    @assert Waitbar == "no" # Not implemented
end

"""
[maps,distributions] = T2map_SEcorr(image, opts)

Description:
  Uses NNLS to compute T2 distributions in the presence of stimulated
  echos by optimizing the refocusing pulse flip angle.  Records parameter
  maps and T2 distributions for further partitioning.

Inputs:
  image: 4-D array with intensity data as (row,column,slice,echo)
  opts: A series of optional Property/Value pairs to modify settings.
    Defaults are given in brackets:
      "TE": Interecho spacing, usually set to one number, but may also
            be set to "variable" for sequences with 2 interecho spacings
            (see "vTEparam"). (0.01)
      "vTEparam": [TE1,TE2,number of echoes at TE1]. Only applied when
                  "TE" set to "variable". TE2 must be and integer
                  multiple of TE1. ([0.01,0.05,16])
      "nT2": Number of T2 times to use (40)
      "T2Range": Min and Max T2 values ([0.015,2.000])
      "T1": Assumed value of T1 (1)
      "Threshold": First echo intensity cutoff for empty voxels (200)
      "Reg": Regularization routine to use, options are:
             "no": do not regularize the solution
             "chi2": use Chi2Factor based regularization (default)
             "lcurve": use L-Curve based regularization
      "Chi2Factor": Constraint on chi^2 used for regularization (Reg must
                    be set to "chi2"!) (1.02)
      "RefCon": Refocusing Pulse Control Angle (180)
      "MinRefAngle": Minimum refocusing angle for EPG optimization (50)
      "nAngles": Number of angles used in EPG optimization (8)
      "SetFlipAngle": Instead of optimizing flip angle, uses this flip
                      angle for all voxels (not set)
      "nCores": Number of processor cores to use (6)
      "Save_regparam": yes/no option to include the regularization
                       paramter mu and the resulting chi^2 factor as
                       two outputs within the maps structure (mu=NaN and
                       chi2factor=1 if Reg=no) ("no")
      "Save_NNLS_basis": yes/no option to include a 5-D matrix of NNLS
                         basis matrices as another output within the maps
                         structure ("no")
      "Waitbar": yes/no option determining whether a progress bar is
                 generated.  Selecting "no" will also suppress any
                 mesages printed to the command window. ("yes")

Ouputs:
  maps: Structure containing 3D maps of the following parameters
      -gdn, general density
      -ggm, general geometric mean
      -gva, general variance
      -FNR, fit to noise ratio (gdn/stdev(residuals))
      -alpha, refocusing pulse flip angle
  distributions: 4-D matrix containing T2 distributions.

External Calls:
  EPGdecaycurve.m
  EPGdecaycurve_vTE.m
  lsqnonneg_reg.m
  lsqnonneg_lcurve.m

Created by Thomas Prasloski
email: tprasloski@gmail.com
Ver. 3.3, August 2013
"""
function T2map_SEcorr(image::Array{Float64,4}; kwargs...)
    reset_timer!(TIMER)
    out = @timeit_debug TIMER "T2map_SEcorr" begin
        _T2map_SEcorr(image, Options(nTE = size(image, 4), kwargs...))
    end
    if timeit_debug_enabled()
        println("\n"); show(TIMER); println("\n")
    end
    return out
end

function _T2map_SEcorr(image::Array{Float64,4}, opts::Options)
    # =========================================================================
    # Parse inputs and apply default values when necessary
    # =========================================================================
    @unpack TE, nTE, T1, RefCon, Threshold, Chi2Factor, nT2, T2Range, MinRefAngle, nAngles, Reg, nCores = opts
    TE1, TE2, nTE1 = TE == "variable" ? opts.vTEparam : (TE, TE, nTE÷2)
    faset = opts.SetFlipAngle
    savereg = opts.Save_regparam == "yes"
    saveNNLS = opts.Save_NNLS_basis == "yes"
    waitbar = opts.Waitbar == "yes"

    # =========================================================================
    # Initialize all the data
    # =========================================================================
    tstart = tic() # Start the clock
    
    # Initialize map matrices
    nrows, ncols, nslices, nechs = size(image)
    gdn = fill(NaN, nrows, ncols, nslices)
    ggm = fill(NaN, nrows, ncols, nslices)
    gva = fill(NaN, nrows, ncols, nslices)
    SNR = fill(NaN, nrows, ncols, nslices)
    FNR = fill(NaN, nrows, ncols, nslices)
    alpha = fill(NaN, nrows, ncols, nslices)
    distributions = fill(NaN, nrows, ncols, nslices, nT2)
    mu = savereg ? fill(NaN, nrows, ncols, nslices) : nothing
    chi2factor = savereg ? fill(NaN, nrows, ncols, nslices) : nothing
    NNLS_basis = saveNNLS ? fill(NaN, nrows, ncols, nslices, nechs, nT2) : nothing
    
    # =========================================================================
    # Find the basis matrices for each flip angle
    # =========================================================================
    
    # Read-only buffers
    T2_times = 10.0 .^ range(log10(T2Range[1]), log10(T2Range[2]), length = nT2)
    flip_angles = range(MinRefAngle, 180.0, length = nAngles)
    basis_angles = [zeros(nechs, nT2) for _ in 1:nAngles] # 1xnAngles vector that will contain the decay bases of each angle
    
    # Read/write buffers
    basis_decay = zeros(nechs, nT2)
    decay_data = zeros(nechs)
    decay_calc = zeros(nechs)
    residuals = zeros(nechs)

    basis_decay_work = calc_basis_decay_work(opts)
    opt_flip_angle_work = (nnls_work = lsqnonneg_work(basis_decay, decay_data), chi2_alpha = zeros(nAngles), decay_pred = zeros(nechs))
    t2_dist_work = t2_distribution_work(basis_decay, decay_data, opts)
    
    # Initialize parameters and variable for angle optimization
    @timeit_debug TIMER "Initialization" begin
        if faset === nothing
            # Loop to compute basis for each angle
            for i = 1:nAngles
                calc_basis_decay!(basis_decay_work, basis_angles[i], flip_angles[i], T2_times, opts)
            end
        else
            calc_basis_decay!(basis_decay_work, basis_decay, faset, T2_times, opts)
        end
    end

    # =========================================================================
    # Process all pixels
    # =========================================================================
    
    # Main triple for-loop to run through each pixel in the image
    for row = 1:nrows, col = 1:ncols, slice = 1:nslices
        
        if col == 1 && slice == 1
            @printf("Starting row %3d/%3d, -- Time: %2.0f hours, %2.0f minutes\n",
                row, nrows, floor(toc(tstart)/3600), (toc(tstart)/3600-floor(toc(tstart)/3600))*60)
        end
        
        # Skip low signal pixels
        if image[row,col,slice,1] < Threshold
            continue
        end
        
        # Extract decay curve from the pixel
        decay_data .= image[row,col,slice,:]
        
        # =====================================================
        # Find optimum flip angle
        # =====================================================
        alpha_opt = if faset === nothing
            @timeit_debug TIMER "Optimize Flip Angle" begin
                optimize_flip_angle!(opt_flip_angle_work, basis_angles, flip_angles, decay_data, T2_times, opts)
            end
        else
            faset
        end

        # =====================================================
        # Fit basis matrix using optimized alpha
        # =====================================================
        if faset === nothing
            @timeit_debug TIMER "Compute Final NNLS Basis" begin
                calc_basis_decay!(basis_decay_work, basis_decay, alpha_opt, T2_times, opts)
            end
        end

        # =========================================================
        # Calculate T2 distribution and global parameters
        # =========================================================
        
        # Find distribution depending on regularization routine
        T2_dis, mu_opt, chi2fact_opt = @timeit_debug TIMER "Calculate T2 Dist" begin
            fit_t2_distribution!(t2_dist_work, basis_decay, decay_data, opts)
        end

        # Save global values
        distributions[row,col,slice,:] .= T2_dis
        alpha[row,col,slice] = alpha_opt
        savereg && (mu[row,col,slice] = mu_opt)
        savereg && (chi2factor[row,col,slice] = chi2fact_opt)
        saveNNLS && (NNLS_basis[row,col,slice,:,:] .= basis_decay)
        
        # Compute parameters of distribution
        mul!(decay_calc, basis_decay, T2_dis)
        residuals .= decay_calc .- decay_data
        gdn[row,col,slice] = sum(T2_dis)
        ggm[row,col,slice] = exp(dot(T2_dis, log.(T2_times)) / sum(T2_dis))
        gva[row,col,slice] = exp(sum((log.(T2_times) .- log(ggm[row,col,slice])).^2 .* T2_dis) / sum(T2_dis)) - 1
        FNR[row,col,slice] = sum(T2_dis) / sqrt(var(residuals))
        SNR[row,col,slice] = maximum(decay_data) / sqrt(var(residuals))
    end

    # Assign outputs
    maps = @ntuple(gdn, ggm, gva, alpha, FNR, mu, chi2factor, NNLS_basis)

    return @ntuple(maps, distributions)
end

function calc_basis_decay_work(o::Options)
    return o.TE == "variable" ?
        EPGdecaycurve_vTE_work(o.nTE) :
        EPGdecaycurve_work(o.nTE)
end

function calc_basis_decay!(work, basis_decay, flip_angle, T2_times, o::Options)
    # Compute the NNLS basis over T2 space
    @assert length(T2_times) == size(basis_decay,2) == o.nT2
    @assert size(basis_decay,1) == o.nTE
    @inbounds for j = 1:o.nT2
        @timeit_debug TIMER "EPGdecaycurve!" begin
            if o.TE == "variable"
                EPGdecaycurve_vTE!(work, o.nTE, flip_angle, o.vTEparam..., T2_times[j], o.T1, o.RefCon)
            else
                EPGdecaycurve!(work, o.nTE, flip_angle, o.TE, T2_times[j], o.T1, o.RefCon)
            end
        end
        for i = 1:o.nTE
            basis_decay[i,j] = work.decay_curve[i]
        end
    end
    return basis_decay
end

function optimize_flip_angle!(work, basis_angles, flip_angles, decay_data, T2_times, o::Options)
    @timeit_debug TIMER "Fit each NNLS Basis" begin
        # Fit each basis and find chi-squared
        for i = 1:o.nAngles
            @timeit_debug TIMER "lsqnonneg!" begin
                lsqnonneg!(work.nnls_work, basis_angles[i], decay_data)
            end
            @timeit_debug TIMER "chi2_alpha[i]" begin
                T2_dis_ls = work.nnls_work.x
                mul!(work.decay_pred, basis_angles[i], T2_dis_ls)
                work.chi2_alpha[i] = sqeuclidean(decay_data, work.decay_pred)
            end
        end
    end
    
    alpha_opt, chi2_alpha_opt = @timeit_debug TIMER "Spline Opt" begin
        # Find the minimum chi-squared and the corresponding angle
        spline_opt(flip_angles, work.chi2_alpha)
    end

    return alpha_opt
end

function t2_distribution_work(basis_decay, decay_data, o::Options)
    if o.Reg == "no"
        # Fit T2 distribution using unregularized NNLS
        lsqnonneg_work(basis_decay, decay_data)
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2 based regularized NNLS
        lsqnonneg_reg_work(basis_decay, decay_data, o.Chi2Factor)
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using lcurve based regularization
        lsqnonneg_lcurve_work(basis_decay, decay_data)
    end
end

function fit_t2_distribution!(work, basis_decay, decay_data, o::Options)
    if o.Reg == "no"
        # Fit T2 distribution using unregularized NNLS
        lsqnonneg!(work, basis_decay, decay_data)
        (T2_dis = work.x, mu_opt = NaN, chi2fact_opt = 1.0)
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2 based regularized NNLS
        out = lsqnonneg_reg!(work, basis_decay, decay_data, o.Chi2Factor)
        (T2_dis = work.x, mu_opt = work.mu_opt, chi2fact_opt = work.chi2fact_opt)
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using lcurve based regularization
        lsqnonneg_lcurve!(work, basis_decay, decay_data)
        (T2_dis = work.x, mu_opt = work.mu_opt, chi2fact_opt = work.chi2fact_opt)
    end
end