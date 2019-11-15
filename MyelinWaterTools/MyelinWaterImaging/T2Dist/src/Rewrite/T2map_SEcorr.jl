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
    @assert nCores == Threads.nthreads()
    
    Save_regparam::Bool = false
    
    Save_NNLS_basis::Bool = false
    
    Waitbar::Bool = false
    @assert !Waitbar # Not implemented
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

    # =========================================================================
    # Initialize all the data
    # =========================================================================
    tstart = tic() # Start the clock
    
    # Initialize map matrices
    nrows, ncols, nslices, nechs = size(image)
    maps = (
        gdn = fill(NaN, nrows, ncols, nslices),
        ggm = fill(NaN, nrows, ncols, nslices),
        gva = fill(NaN, nrows, ncols, nslices),
        SNR = fill(NaN, nrows, ncols, nslices),
        FNR = fill(NaN, nrows, ncols, nslices),
        alpha = fill(NaN, nrows, ncols, nslices),
        mu = (opts.Save_regparam ? fill(NaN, nrows, ncols, nslices) : nothing),
        chi2factor = (opts.Save_regparam ? fill(NaN, nrows, ncols, nslices) : nothing),
        NNLS_basis = (opts.Save_NNLS_basis ? fill(NaN, nrows, ncols, nslices, nechs, nT2) : nothing),
    )
    distributions = fill(NaN, nrows, ncols, nslices, nT2)
    global_buffers = [global_buffer_maker(opts) for _ in 1:Threads.nthreads()]

    # =========================================================================
    # Find the basis matrices for each flip angle
    # =========================================================================
    
    # Initialize parameters and variable for angle optimization
    @timeit_debug TIMER "Initialization" begin
        initialize_basis_decay!.(global_buffers, Ref(opts))
    end

    # =========================================================================
    # Process all pixels
    # =========================================================================
    
    # Main triple for-loop to run through each pixel in the image
    Threads.@threads for row = 1:nrows
        global_buffer = global_buffers[Threads.threadid()]
        
        hour, min = floor(toc(tstart)/3600), (toc(tstart)/3600-floor(toc(tstart)/3600))*60
        println("Starting row $row/$nrows on thread $(Threads.threadid()) -- Time: $hour hours, $min minutes\n")
        
        for col = 1:ncols, slice = 1:nslices
            
            # Skip low signal pixels
            if image[row,col,slice,1] < Threshold
                continue
            end
            
            # Extract decay curve from the pixel
            global_buffer.decay_data .= image[row,col,slice,:]
            
            # =====================================================
            # Find optimum flip angle
            # =====================================================
            if opts.SetFlipAngle === nothing
                @timeit_debug TIMER "Optimize Flip Angle" begin
                    optimize_flip_angle!(global_buffer, opts)
                end
            end

            # =====================================================
            # Fit basis matrix using optimized alpha
            # =====================================================
            if opts.SetFlipAngle === nothing
                @timeit_debug TIMER "Compute Final NNLS Basis" begin
                    fit_basis_decay!(global_buffer, opts)
                end
            end

            # =========================================================
            # Calculate T2 distribution and global parameters
            # =========================================================
            @timeit_debug TIMER "Calculate T2 Dist" begin
                fit_t2_distribution!(global_buffer, opts)
            end

            # =========================================================
            # Save results
            # =========================================================
            save_results!(global_buffer, maps, distributions, opts, row, col, slice)
        end
    end

    return @ntuple(maps, distributions)
end

global_buffer_maker(o::Options) = (
    T2_times = (10.0 .^ range(log10(o.T2Range[1]), log10(o.T2Range[2]); length = o.nT2)),
    flip_angles = range(o.MinRefAngle, 180.0; length = o.nAngles),
    basis_angles = [zeros(o.nTE, o.nT2) for _ in 1:o.nAngles],
    basis_decay = zeros(o.nTE, o.nT2),
    decay_data = zeros(o.nTE),
    decay_calc = zeros(o.nTE),
    residuals = zeros(o.nTE),
    basis_decay_work = calc_epg_decay_work(o),
    opt_flip_angle_work = (
        nnls_work = lsqnonneg_work(zeros(o.nTE, o.nT2), zeros(o.nTE)),
        chi2_alpha = zeros(o.nAngles),
        decay_pred = zeros(o.nTE),
    ),
    t2_dist_work = t2_distribution_work(zeros(o.nTE, o.nT2), zeros(o.nTE), o),
    alpha_opt = (o.SetFlipAngle === nothing ? Ref(NaN) : Ref(o.SetFlipAngle)),
    chi2_alpha_opt = Ref(NaN),
    T2_dis = zeros(o.nT2),
    mu_opt = Ref(NaN),
    chi2fact_opt = Ref(NaN),
)

function calc_epg_decay_work(o::Options)
    return o.TE == "variable" ?
        EPGdecaycurve_vTE_work(o.nTE) :
        EPGdecaycurve_work(o.nTE)
end

function initialize_basis_decay!(global_buffer, o::Options)
    @unpack basis_decay_work, basis_angles, basis_decay, flip_angles, T2_times = global_buffer
    if o.SetFlipAngle === nothing
        # Loop to compute basis for each angle
        @inbounds for i = 1:o.nAngles
            calc_basis_decay!(basis_decay_work, basis_angles[i], flip_angles[i], T2_times, o)
        end
    else
        calc_basis_decay!(basis_decay_work, basis_decay, o.SetFlipAngle, T2_times, o)
    end
end

function fit_basis_decay!(global_buffer, o::Options)
    @unpack basis_decay_work, basis_decay, alpha_opt, T2_times = global_buffer
    calc_basis_decay!(basis_decay_work, basis_decay, alpha_opt[], T2_times, o)
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

function optimize_flip_angle!(global_buffer, o::Options)
    @unpack opt_flip_angle_work, basis_angles, flip_angles, decay_data, T2_times = global_buffer
    @unpack nnls_work, decay_pred, chi2_alpha = opt_flip_angle_work
    @unpack alpha_opt, chi2_alpha_opt = global_buffer

    @timeit_debug TIMER "Fit each NNLS Basis" begin
        # Fit each basis and find chi-squared
        for i = 1:o.nAngles
            @timeit_debug TIMER "lsqnonneg!" begin
                lsqnonneg!(nnls_work, basis_angles[i], decay_data)
            end
            T2_dis_ls = nnls_work.x
            mul!(decay_pred, basis_angles[i], T2_dis_ls)
            chi2_alpha[i] = sqeuclidean(decay_data, decay_pred)
        end
    end
    
    @timeit_debug TIMER "Spline Opt" begin
        # Find the minimum chi-squared and the corresponding angle
        alpha_opt[], chi2_alpha_opt[] = spline_opt(flip_angles, chi2_alpha)
    end

    return alpha_opt[]
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

function fit_t2_distribution!(global_buffer, o::Options)
    @unpack t2_dist_work, basis_decay, decay_data = global_buffer
    @unpack T2_dis, mu_opt, chi2fact_opt = global_buffer
    if o.Reg == "no"
        # Fit T2 distribution using unregularized NNLS
        lsqnonneg!(t2_dist_work, basis_decay, decay_data)
        T2_dis .= t2_dist_work.x
        mu_opt[] = NaN
        chi2fact_opt[] = 1.0
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2 based regularized NNLS
        out = lsqnonneg_reg!(t2_dist_work, basis_decay, decay_data, o.Chi2Factor)
        T2_dis .= t2_dist_work.x
        mu_opt[] = t2_dist_work.mu_opt[]
        chi2fact_opt[] = t2_dist_work.chi2fact_opt[]
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using lcurve based regularization
        lsqnonneg_lcurve!(t2_dist_work, basis_decay, decay_data)
        T2_dis .= t2_dist_work.x
        mu_opt[] = t2_dist_work.mu_opt[]
        chi2fact_opt[] = t2_dist_work.chi2fact_opt[]
    end
end

function save_results!(global_buffer, maps, distributions, opts, idx...)
    @unpack T2_dis, T2_times, decay_data, decay_calc, basis_decay, residuals, alpha_opt, mu_opt, chi2fact_opt = global_buffer
    
    # Save distribution
    distributions[idx...,:] .= T2_dis

    # Update buffers
    mul!(decay_calc, basis_decay, T2_dis)
    residuals .= decay_calc .- decay_data
    
    # Compute parameters of distribution
    maps.gdn[idx...] = sum(T2_dis)
    maps.ggm[idx...] = exp(dot(T2_dis, log.(T2_times)) / sum(T2_dis))
    maps.gva[idx...] = exp(sum((log.(T2_times) .- log(maps.ggm[idx...])).^2 .* T2_dis) / sum(T2_dis)) - 1
    maps.SNR[idx...] = maximum(decay_data) / sqrt(var(residuals))
    maps.FNR[idx...] = sum(T2_dis) / sqrt(var(residuals))
    maps.alpha[idx...] = alpha_opt[]
    opts.Save_regparam && (maps.mu[idx...] = mu_opt[])
    opts.Save_regparam && (maps.chi2factor[idx...] = chi2fact_opt[])
    opts.Save_NNLS_basis && (maps.NNLS_basis[idx...,:,:] .= basis_decay)

    return global_buffer
end
