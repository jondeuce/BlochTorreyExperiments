"""
    Options structure for T2map_SEcorr
"""
@with_kw struct Options @deftype Float64
    TE::Union{Float64, String} = 0.010
    @assert TE isa Float64 ? TE >= 0.0001 && TE <= 1.0 : TE == "variable"
    
    nTE::Int = 32
    @assert nTE >= 1
    
    vTEparam::Tuple{Float64,Float64,Int} = (0.01, 0.05, 16)
    @assert vTEparam[2] > vTEparam[1] && vTEparam[1] * round(Int, vTEparam[2]/vTEparam[1]) ≈ vTEparam[2] && vTEparam[3] < nTE
    
    T1 = 1.0
    @assert 0.001 <= T1 <= 10.0
    
    RefCon = 180.0
    @assert 1.0 <= RefCon <= 180.0
    
    Threshold = 200.0
    
    Chi2Factor = 1.02
    @assert Chi2Factor > 1
    
    nT2::Int = 40
    @assert 10 <= nT2 <= 120
    
    T2Range::Tuple{Float64,Float64} = (0.015, 2.0)
    @assert T2Range[2] > T2Range[1] && T2Range[1] >= 0.001 && T2Range[2] <= 10.0
    
    MinRefAngle = 50.0
    @assert 1.0 < MinRefAngle < 180.0
    
    nAngles::Int = 8
    @assert nAngles > 1
    
    Reg::String = "chi2"
    @assert Reg ∈ ["no", "chi2", "lcurve"]
    
    SetFlipAngle::Union{Float64,Nothing} = nothing
    
    nCores::Int = 6
    @assert 1 <= nCores <= 8
    
    Save_regparam::String = "no"
    @assert Save_regparam ∈ ["yes", "no"]
    
    Save_NNLS_basis::String = "no"
    @assert Save_NNLS_basis ∈ ["yes", "no"]
    
    Waitbar::String = "no"
    @assert Waitbar == "no"
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
function T2map_SEcorr(
        image::Array{Float64,4},
        opts = Options(nTE = size(image, 4))
    )
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
    timer = TimerOutput() # Start the clock
    
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
    decay_calc = zeros(nechs)
    residuals = zeros(nechs)
    chi2_alpha = zeros(nAngles)
    
    # Initialize parameters and variable for angle optimization
    @timeit timer "Initialization" begin
        if faset === nothing
            # Loop to compute each basis and assign them to a cell in the array
            for a = 1:nAngles
                for x = 1:nT2
                    basis_decay[:,x] .= TE == "variable" ?
                        EPGdecaycurve_vTE(nechs,flip_angles[a],TE1,TE2,nTE1,T2_times[x],T1,RefCon) :
                        EPGdecaycurve(nechs,flip_angles[a],TE,T2_times[x],T1,RefCon)
                end
                basis_angles[a] .= basis_decay
            end
        else
            for x = 1:nT2
                basis_decay[:,x] .= TE == "variable" ?
                    EPGdecaycurve_vTE(nechs,faset,TE1,TE2,nTE1,T2_times[x],T1,RefCon) :
                    EPGdecaycurve(nechs,faset,TE,T2_times[x],T1,RefCon)
            end
        end
    end

    # =========================================================================
    # Process all pixels
    # =========================================================================
    
    # Main triple for-loop to run through each pixel in the image
    for row = 1:nrows
        
        @printf("Starting row %3d/%3d, -- Time: %2.0f hours, %2.0f minutes\n",
            row, nrows, floor(toc(tstart)/3600), (toc(tstart)/3600-floor(toc(tstart)/3600))*60)
        
        rowloopstart = tic()
        rowloopcount = 0
        
        for col = 1:ncols

            # tfinish_est = nrows * ncols * nslices * (toc(rowloopstart) / rowloopcount)
            # @printf("Starting row %3d/%3d, column %3d/%3d -- Time: %2.0f hours, %2.0f minutes -- Estimated Finish: %2.0f hours, %2.0f minutes\n",
            #     row, nrows, col, ncols, floor(toc(tstart)/3600), (toc(tstart)/3600-floor(toc(tstart)/3600))*60, floor(tfinish_est/3600), (tfinish_est/3600-floor(tfinish_est/3600))*60)
            
            for slice = 1:nslices

                # Conditional loop to reject low signal pixels
                if image[row,col,slice,1] >= Threshold
                    
                    # Extract decay curve from the pixel
                    decay_data = image[row,col,slice,1:nechs]
                    
                    @timeit timer "Optimize Flip Angle" begin
                        if faset === nothing
                            # =====================================================
                            # Find optimum flip angle
                            # =====================================================
                            
                            # Fit each basis and find chi-squared
                            @timeit timer "Fit each NNLS Basis" begin
                                for a = 1:nAngles
                                    T2_dis_ls = lsqnonneg(basis_angles[a], decay_data)
                                    decay_pred = basis_angles[a]*T2_dis_ls
                                    chi2_alpha[a] = sqeuclidean(decay_data, decay_pred)
                                end
                            end
                            
                            # Find the minimum chi-squared and the corresponding angle
                            @timeit timer "Spline Opt" begin
                                deg_spline = min(3, length(flip_angles)-1)
                                spl = Spline1D(flip_angles, chi2_alpha; k = deg_spline)
                                # alpha_spline = flip_angles[1]:0.001:flip_angles[end]
                                # chi2_spline = spl.(alpha_spline)
                                # _, index = findmin(chi2_spline)
                                # alpha_opt_spline = alpha_spline[index]
                                alpha_opt_optim = Optim.minimizer(Optim.optimize(x->spl(x), flip_angles[1], flip_angles[end], Optim.Brent()))
                                alpha[row,col,slice] = alpha_opt_optim
                            end

                            # =====================================================
                            # Fit basis matrix using alpha
                            # =====================================================

                            # Compute the NNLS basis over T2 space
                            @timeit timer "Compute Final NNLS Basis" begin
                                for x = 1:nT2
                                    basis_decay[:,x] .= TE == "variable" ?
                                        EPGdecaycurve_vTE(nechs,alpha[row,col,slice],TE1,TE2,nTE1,T2_times[x],T1,RefCon) :
                                        EPGdecaycurve(nechs,alpha[row,col,slice],TE,T2_times[x],T1,RefCon)
                                end
                            end
                        else
                            alpha[row,col,slice] = faset
                        end
                    end

                    saveNNLS && (NNLS_basis[row,col,slice,:,:] .= basis_decay)

                    # =========================================================
                    # Calculate T2 distribution and global parameters
                    # =========================================================
                    
                    # Find distribution depending on regularization routine
                    T2_dis, mu_opt, chi2_opt = @timeit timer "Calculate T2 Dist" begin
                        if Reg == "no"
                            # Fit T2 distribution using unregularized NNLS
                            lsqnonneg(basis_decay, decay_data), NaN, 1.0
                        elseif Reg == "chi2"
                            # Fit T2 distribution using chi2 based regularized NNLS
                            lsqnonneg_reg(basis_decay, decay_data, Chi2Factor)
                        elseif Reg == "lcurve"
                            # Fit T2 distribution using lcurve based regularization
                            lsqnonneg_lcurve(basis_decay, decay_data)
                        end
                    end

                    # Save global values
                    distributions[row,col,slice,:] .= T2_dis
                    savereg && (mu[row,col,slice] = mu_opt)
                    savereg && (chi2factor[row,col,slice] = chi2_opt)
                    
                    # Compute parameters of distribution
                    mul!(decay_calc, basis_decay, T2_dis)
                    residuals .= decay_calc .- decay_data
                    gdn[row,col,slice] = sum(T2_dis)
                    ggm[row,col,slice] = exp(dot(T2_dis, log.(T2_times)) / sum(T2_dis))
                    gva[row,col,slice] = exp(sum((log.(T2_times) .- log(ggm[row,col,slice])).^2 .* T2_dis) / sum(T2_dis)) - 1
                    FNR[row,col,slice] = sum(T2_dis) / sqrt(var(residuals))
                    SNR[row,col,slice] = maximum(decay_data) / sqrt(var(residuals))
                end
                rowloopcount += 1
            end
            
            # error("erroring...")
        end
    end

    # # Print message on finish with total run time
    # @printf("Completed in %2.0f hours, %2.0f minutes\n", floor(toc(tstart)/3600), (toc(tstart)/3600-floor(toc(tstart)/3600))*60)
    show(timer)
    println("")

    # Assign outputs
    maps = @ntuple(gdn, ggm, gva, alpha, FNR, mu, chi2factor, NNLS_basis)

    return @ntuple(maps, distributions)
end
