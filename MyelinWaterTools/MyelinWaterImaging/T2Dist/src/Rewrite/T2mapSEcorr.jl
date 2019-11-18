"""
    T2mapOptions structure for T2mapSEcorr
"""
@with_kw struct T2mapOptions{T} @deftype T
    nTE::Int # required parameter
    @assert nTE > 1

    GridSize::NTuple{3,Int} # required parameter
    @assert all(GridSize .>= 1)

    TE::Union{T, Nothing} = 0.010
    @assert TE isa Nothing || 0.0001 <= TE <= 1.0

    TE1::Union{T, Nothing} = nothing
    TE2::Union{T, Nothing} = nothing
    nTE1::Union{Int, Nothing} = nothing
    @assert all((TE1, TE2, nTE1) .=== nothing) || all((TE1, TE2, nTE1) .!== nothing)
    @assert TE1 isa Nothing || (TE1 < TE2 && nTE1 < nTE && TE1 * round(Int, TE2/TE1) ≈ TE2)

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

    T2Range::Tuple{T,T} = (0.015, 2.0)
    @assert 0.001 <= T2Range[1] < T2Range[2] <= 10.0

    MinRefAngle = 50.0
    @assert 1.0 < MinRefAngle < 180.0

    nAngles::Int = 8
    @assert nAngles > 1

    Reg::String = "chi2"
    @assert Reg ∈ ("no", "chi2", "lcurve")

    SetFlipAngle::Union{T,Nothing} = nothing
    @assert SetFlipAngle isa Nothing || 0.0 < SetFlipAngle <= 180.0
    
    SaveRegParam::Bool = false
    
    SaveNNLSBasis::Bool = false

    # No longer used (set JULIA_NUM_THREADS externally)
    # nCores::Int = Threads.nthreads()
    # @assert nCores == Threads.nthreads()

    # Not implemented
    # Waitbar::Bool = false
    # @assert !Waitbar # Not implemented
end
T2mapOptions(args...; kwargs...) = T2mapOptions{Float64}(args...; kwargs...)

"""
maps, distributions = T2mapSEcorr(image; kwargs...)

Description:
  Uses NNLS to compute T2 distributions in the presence of stimulated
  echos by optimizing the refocusing pulse flip angle. Records parameter
  maps and T2 distributions for further partitioning.

Inputs:
  image: 4-D array with intensity data as (row, column, slice, echo)
  kwargs: A series of optional keyword argument settings.
    Defaults are given in brackets:
      "TE": Interecho spacing, usually set to one number, but may also
            be left unset for sequences with 2 interecho spacings
            (see "vTEparam"). (nothing)
      "vTEparam": (TE1, TE2, number of echoes at TE1). Only applied when
                  "TE" set to nothing. TE2 must be an integer
                  multiple of TE1. (nothing)
      "nT2": Number of T2 times to use (40)
      "T2Range": Min and Max T2 values ((0.015, 2.0))
      "T1": Assumed value of T1 (1.0)
      "Threshold": First echo intensity cutoff for empty voxels (200.0)
      "Reg": Regularization routine to use, options are:
             "no": do not regularize the solution
             "chi2": use Chi2Factor based regularization (default)
             "lcurve": use L-Curve based regularization
      "Chi2Factor": Constraint on chi^2 used for regularization (Reg must
                    be set to "chi2"!) (1.02)
      "RefCon": Refocusing Pulse Control Angle (180.0)
      "MinRefAngle": Minimum refocusing angle for EPG optimization (50.0)
      "nAngles": Number of angles used in EPG optimization (8)
      "SetFlipAngle": Instead of optimizing flip angle, uses this flip
                      angle for all voxels (nothing)
      "nCores": Number of processor cores to use (6)
      "SaveRegParam":  true/false option to include the regularization
                       paramter mu and the resulting chi^2 factor as
                       two outputs within the maps structure (mu=NaN and
                       chi2factor=1 if false) (false)
      "SaveNNLSBasis":   true/false option to include a 5-D array of NNLS
                         basis matrices as another output within the maps
                         structure (false)
      "Waitbar": true/false option determining whether a progress bar is
                 generated. Selecting false will also suppress any
                 mesages printed to the command window. (false)

Ouputs:
  maps: dictionary containing 3D maps with the following fields:
      "gdn": general density
      "ggm": general geometric mean
      "gva": general variance
      "SNR": signal to noise ratio (maximum(signal)/stdev(residuals))
      "FNR": fit to noise ratio (gdn/stdev(residuals))
      "alpha": refocusing pulse flip angle
      "mu": (optional) regularization parameter from NNLS fit
      "chi2factor": (optional) chi^2 increase factor from NNLS fit
      "NNLS_basis": (optional) decay basis from EPGdecaycurve
  distributions: 4-D array containing T2 distributions.

External Calls:
  EPGdecaycurve.m
  EPGdecaycurve_vTE.m
  lsqnonneg_reg.m
  lsqnonneg_lcurve.m

Created by Thomas Prasloski
email: tprasloski@gmail.com
Ver. 3.3, August 2013
"""
function T2mapSEcorr(image::Array{T,4}; kwargs...) where {T}
    reset_timer!(TIMER)
    out = @timeit_debug TIMER "T2mapSEcorr" begin
        _T2mapSEcorr(image, T2mapOptions{T}(;
            GridSize = size(image)[1:3],
            nTE = size(image, 4),
            kwargs...
        ))
    end
    if timeit_debug_enabled()
        println("\n"); show(TIMER); println("\n")
    end
    return out
end

function _T2mapSEcorr(image::Array{T,4}, opts::T2mapOptions{T}) where {T}
    # =========================================================================
    # Initialize output data structures and thread-local buffers
    # =========================================================================
    @assert size(image) == (opts.GridSize..., opts.nTE)
    maps = Dict{String, Array{T}}()
    maps["gdn"] = fill(T(NaN), opts.GridSize...)
    maps["ggm"] = fill(T(NaN), opts.GridSize...)
    maps["gva"] = fill(T(NaN), opts.GridSize...)
    maps["SNR"] = fill(T(NaN), opts.GridSize...)
    maps["FNR"] = fill(T(NaN), opts.GridSize...)
    maps["alpha"] = fill(T(NaN), opts.GridSize...)
    opts.SaveRegParam && (maps["mu"] = fill(T(NaN), opts.GridSize...))
    opts.SaveRegParam && (maps["chi2factor"] = fill(T(NaN), opts.GridSize...))
    opts.SaveNNLSBasis && (maps["NNLS_basis"] = fill(T(NaN), opts.GridSize..., opts.nTE, opts.nT2))
    distributions = fill(T(NaN), opts.GridSize..., opts.nT2)
    thread_buffers = [thread_buffer_maker(opts) for _ in 1:Threads.nthreads()]

    # =========================================================================
    # Find the basis matrices for each flip angle
    # =========================================================================
    
    # Initialize parameters and variable for angle optimization
    @timeit_debug TIMER "Initialization" begin
        for i in 1:length(thread_buffers)
            init_epg_decay_basis!(thread_buffers[i], opts)
        end
    end

    # =========================================================================
    # Process all pixels
    # =========================================================================
    loop_start_time = tic()
    
    Threads.@threads for row in 1:opts.GridSize[1]
        # Obtain thread-local buffer and print progress summary
        thread_buffer = thread_buffers[Threads.threadid()]
        update_progress!(thread_buffer, toc(loop_start_time), row, opts.GridSize[1])
        
        for col in 1:opts.GridSize[2], slice in 1:opts.GridSize[3]
            # Skip low signal voxels
            @inbounds if image[row,col,slice,1] < opts.Threshold
                continue
            end
            
            # Extract decay curve from the voxel
            @inbounds for i in 1:opts.nTE
                thread_buffer.decay_data[i] = image[row,col,slice,i]
            end
            
            # Find optimum flip angle
            if opts.SetFlipAngle === nothing
                @timeit_debug TIMER "Optimize Flip Angle" begin
                    optimize_flip_angle!(thread_buffer, opts)
                end
            end

            # Fit decay basis using optimized alpha
            if opts.SetFlipAngle === nothing
                @timeit_debug TIMER "Compute Final NNLS Basis" begin
                    fit_epg_decay_basis!(thread_buffer, opts)
                end
            end

            # Calculate T2 distribution and map parameters
            @timeit_debug TIMER "Calculate T2 Dist" begin
                fit_t2_distribution!(thread_buffer, opts)
            end

            # Save loop results to outputs
            save_results!(thread_buffer, maps, distributions, opts, row, col, slice)
        end
    end

    return @ntuple(maps, distributions)
end

# =========================================================
# Make thread-local buffers
# =========================================================
thread_buffer_maker(o::T2mapOptions{T}) where {T} = (
    row_counter = Ref(0),
    T2_times = logrange(o.T2Range..., o.nT2),
    flip_angles = range(o.MinRefAngle, T(180); length = o.nAngles),
    basis_angles = [zeros(T, o.nTE, o.nT2) for _ in 1:o.nAngles],
    decay_basis = zeros(T, o.nTE, o.nT2),
    decay_data = zeros(T, o.nTE),
    decay_calc = zeros(T, o.nTE),
    residuals = zeros(T, o.nTE),
    decay_basis_work = epg_decay_basis_work(o),
    flip_angle_work = optimize_flip_angle_work(o),
    t2_dist_work = t2_distribution_work(o),
    alpha_opt = (o.SetFlipAngle === nothing ? Ref(T(NaN)) : Ref(o.SetFlipAngle)),
    chi2_alpha_opt = Ref(T(NaN)),
    T2_dis = zeros(T, o.nT2),
    mu_opt = Ref(T(NaN)),
    chi2fact_opt = Ref(T(NaN)),
)

# =========================================================
# Progress printing
# =========================================================
function update_progress!(thread_buffer, time_elapsed, row, nrows)
    @unpack row_counter = thread_buffer
    est_complete = row_counter[] == 0 ? 0.0 :
        (nrows - row_counter[]) * (time_elapsed / row_counter[]) / Threads.nthreads()
    h0, m0, s0 = hour_min_sec(time_elapsed)
    h1, m1, s1 = hour_min_sec(max(est_complete - time_elapsed, 0.0))
    dr, dt = ndigits(nrows), ndigits(Threads.nthreads())
    println(join([
        "Row: $(lpad(row,dr))/$(lpad(nrows,dr)) (Thread $(lpad(Threads.threadid(),dt)))",
        "Elapsed Time: $(lpad(h0,2,"0"))h:$(lpad(m0,2,"0"))m:$(lpad(s0,2,"0"))s",
        "Time Remaining: $(lpad(h1,2,"0"))h:$(lpad(m1,2,"0"))m:$(lpad(s1,2,"0"))s",
    ], " -- "))
    row_counter[] += 1
end

# =========================================================
# EPG decay curve fitting
# =========================================================
epg_decay_basis_work(o::T2mapOptions{T}) where {T} =
    o.TE === nothing ?
        EPGdecaycurve_vTE_work(T, o.nTE) :
        EPGdecaycurve_work(T, o.nTE)

function init_epg_decay_basis!(thread_buffer, o::T2mapOptions)
    @unpack decay_basis_work, basis_angles, decay_basis, flip_angles, T2_times = thread_buffer
    
    if o.SetFlipAngle === nothing
        # Loop to compute basis for each angle
        @inbounds for i = 1:o.nAngles
            epg_decay_basis!(decay_basis_work, basis_angles[i], flip_angles[i], T2_times, o::T2mapOptions)
        end
    else
        epg_decay_basis!(decay_basis_work, decay_basis, o.SetFlipAngle, T2_times, o::T2mapOptions)
    end

    return nothing
end

function fit_epg_decay_basis!(thread_buffer, o::T2mapOptions)
    @unpack decay_basis_work, decay_basis, alpha_opt, T2_times = thread_buffer
    epg_decay_basis!(decay_basis_work, decay_basis, alpha_opt[], T2_times, o::T2mapOptions)
    return nothing
end

function epg_decay_basis!(work, decay_basis, flip_angle, T2_times, o::T2mapOptions)
    @assert length(T2_times) == size(decay_basis,2) == o.nT2
    @assert size(decay_basis,1) == o.nTE
    
    # Compute the NNLS basis over T2 space
    @inbounds for j = 1:o.nT2
        @timeit_debug TIMER "EPGdecaycurve!" begin
            if o.TE === nothing
                EPGdecaycurve_vTE!(work, o.nTE, flip_angle, o.TE1, o.TE2, o.nTE1, T2_times[j], o.T1, o.RefCon)
            else
                EPGdecaycurve!(work, o.nTE, flip_angle, o.TE, T2_times[j], o.T1, o.RefCon)
            end
        end
        for i = 1:o.nTE
            decay_basis[i,j] = work.decay_curve[i]
        end
    end

    return nothing
end

# =========================================================
# Flip angle optimization
# =========================================================
optimize_flip_angle_work(o::T2mapOptions{T}) where {T} = (
    nnls_work = lsqnonneg_work(zeros(T, o.nTE, o.nT2), zeros(T, o.nTE)),
    chi2_alpha = zeros(T, o.nAngles),
    decay_pred = zeros(T, o.nTE),
)

function optimize_flip_angle!(thread_buffer, o::T2mapOptions)
    @unpack flip_angle_work, basis_angles, flip_angles, decay_data, T2_times = thread_buffer
    @unpack nnls_work, decay_pred, chi2_alpha = flip_angle_work
    @unpack alpha_opt, chi2_alpha_opt = thread_buffer

    # Fit each decay basis and find chi-squared
    @timeit_debug TIMER "Fit each NNLS Basis" begin
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

    return nothing
end

# =========================================================
# T2-distribution fitting
# =========================================================
function t2_distribution_work(o::T2mapOptions{T}) where {T}
    decay_basis_buffer = zeros(T, o.nTE, o.nT2)
    decay_data_buffer = zeros(T, o.nTE)

    work = if o.Reg == "no"
        # Fit T2 distribution using unregularized NNLS
        lsqnonneg_work(decay_basis_buffer, decay_data_buffer)
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2 based regularized NNLS
        lsqnonneg_reg_work(decay_basis_buffer, decay_data_buffer)
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using lcurve based regularization
        lsqnonneg_lcurve_work(decay_basis_buffer, decay_data_buffer)
    end

    return work
end

function fit_t2_distribution!(thread_buffer, o::T2mapOptions{T}) where {T}
    @unpack t2_dist_work, decay_basis, decay_data = thread_buffer
    @unpack T2_dis, mu_opt, chi2fact_opt = thread_buffer

    if o.Reg == "no"
        # Fit T2 distribution using unregularized NNLS
        lsqnonneg!(t2_dist_work, decay_basis, decay_data)
        T2_dis .= t2_dist_work.x
        mu_opt[] = T(NaN)
        chi2fact_opt[] = one(T)
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2 based regularized NNLS
        out = lsqnonneg_reg!(t2_dist_work, decay_basis, decay_data, o.Chi2Factor)
        T2_dis .= t2_dist_work.x
        mu_opt[] = t2_dist_work.mu_opt[]
        chi2fact_opt[] = t2_dist_work.chi2fact_opt[]
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using lcurve based regularization
        lsqnonneg_lcurve!(t2_dist_work, decay_basis, decay_data)
        T2_dis .= t2_dist_work.x
        mu_opt[] = t2_dist_work.mu_opt[]
        chi2fact_opt[] = t2_dist_work.chi2fact_opt[]
    end

    return nothing
end

# =========================================================
# Save thread local results to output maps
# =========================================================
function save_results!(thread_buffer, maps, distributions, o::T2mapOptions, i...)
    @unpack T2_dis, T2_times, decay_data, decay_calc, decay_basis, residuals, alpha_opt, mu_opt, chi2fact_opt = thread_buffer
    
    # Update buffers
    mul!(decay_calc, decay_basis, T2_dis)
    residuals .= decay_calc .- decay_data
    
    # Compute and save parameters of distribution
    @unpack gdn, ggm, gva, SNR, FNR, alpha = maps
    gdn[i...] = sum(T2_dis)
    ggm[i...] = exp(dot(T2_dis, log.(T2_times)) / sum(T2_dis))
    gva[i...] = exp(sum((log.(T2_times) .- log(ggm[i...])).^2 .* T2_dis) / sum(T2_dis)) - 1
    SNR[i...] = maximum(decay_data) / sqrt(var(residuals))
    FNR[i...] = sum(T2_dis) / sqrt(var(residuals))
    alpha[i...] = alpha_opt[]

    # Save distribution
    @inbounds for j in 1:o.nT2
        distributions[i...,j] = T2_dis[j]
    end

    # Optionally save regularization parameters
    if o.SaveRegParam
        @unpack mu, chi2factor = maps
        mu[i...] = mu_opt[]
        chi2factor[i...] = chi2fact_opt[]
    end

    # Optionally save NNLS basis
    if o.SaveNNLSBasis
        @unpack NNLS_basis = maps
        @inbounds for j in 1:o.nTE, k in 1:o.nT2
            NNLS_basis[i...,j,k] .= decay_basis[j,k]
        end
    end

    return nothing
end
