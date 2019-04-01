# ---------------------------------------------------------------------------- #
# Myelin water fraction calculation
# ---------------------------------------------------------------------------- #

# Calculate exact mwf from grid
function getmwf(outer::VecOfCircles{2}, inner::VecOfCircles{2}, bdry::Rectangle)
    myelin_area = intersect_area(outer, bdry) - intersect_area(inner, bdry)
    total_area = area(bdry)
    return myelin_area/total_area
end

# Abstract interface for calculating mwf from measured signals
function getmwf(
        signals::Vector{V},
        modeltype::AbstractMWIFittingModel;
        kwargs...
    ) where {V <: Vec{2}}
    try
        _getmwf(modeltype, fitmwfmodel(signals, modeltype; kwargs...)...)
    catch e
        @warn "Error computing the myelin water fraction"
        @warn sprint(showerror, e, catch_backtrace())
        NaN
    end
end

# Abstract interface
function fitmwfmodel(
        signals::Vector{V},
        modeltype::AbstractMWIFittingModel;
        kwargs...
    ) where {V <: Vec{2}}
    try
        _fitmwfmodel(signals, modeltype; kwargs...)
    catch e
        @warn "Error fitting $modeltype model to signal."
        @warn sprint(showerror, e, catch_backtrace())
        nothing
    end
end

# MWI model data
mwimodeldata(modeltype::ThreePoolCplxToCplx, S::Vector{Vec{2,T}}) where {T} = copy(reinterpret(T, S[2:end]))
mwimodeldata(modeltype::ThreePoolMagnData, S::Vector{V}) where {V <: Vec{2}} = norm.(S[2:end])
mwimodeldata(modeltype::TwoPoolMagnData, S::Vector{V}) where {V <: Vec{2}} = norm.(S[2:end])

# NNLSRegression model
function _fitmwfmodel(
        signals::Vector{V},
        modeltype::NNLSRegression;
        TE = 10e-3, # First time point
        nTE = 32, # Number of echos
        T2Range = [8e-3, 2.0], # Min and Max T2 values used during fitting (typical for in-vivo)
        nT2 = 40, # Number of T2 bins used during fitting process, spaced logarithmically in `T2Range`
        Threshold = 0.0, # First echo intensity cutoff for empty voxels
        RefConAngle = 180.0, # Refocusing Pulse Control Angle (TODO: check value from scanner; old default is 165.0)
        spwin = [8e-3, 25e-3], # short peak window (typically 1.5X echospacing to 40ms)
        mpwin = [25e-3, 200e-3], # middle peak window
        PLOTDIST = false # plot resulting T2-distribution
    ) where {V <: Vec{2}}

    @assert length(signals) == nTE+1
    mag = norm.(signals[2:end]) # magnitude signal, discarding t=0 signal
    mag = reshape(mag, (1,1,1,length(mag))) # T2map_SEcorr expects 4D input

    MWImaps, MWIdist = mxcall(:T2map_SEcorr, 2, mag,
        "TE", TE,
        "nT2", nT2,
        "T2Range", T2Range,
        "Threshold", Threshold,
        "Reg", "chi2",
        "Chi2Factor", 1.02,
        "RefCon", RefConAngle,
        "Waitbar", "no",
        "Save_regparam", "yes"
    )
    MWIpart = mxcall(:T2part_SEcorr, 1, MWIdist,
        "T2Range", T2Range,
        "spwin", spwin,
        "mpwin", mpwin
    )

    if PLOTDIST
        logspace(start,stop,length) = exp10.(range(log10(start), stop=log10(stop), length=length))
        mwf = _getmwf(NNLSRegression(), MWImaps, MWIdist, MWIpart)

        mxcall(:figure, 0)
        mxcall(:semilogx, 0, 1e3 .* logspace(T2Range..., nT2), MWIdist[:])
        mxcall(:axis, 0, "on")
        mxcall(:xlim, 0, 1e3 .* [T2Range...])
        mxcall(:xlabel, 0, "T2 [ms]")
        mxcall(:title, 0, "T2 Distribution: nT2 = $nT2, mwf = $(round(mwf; digits=4))")
        mxcall(:hold, 0, "on")
        ylim = mxcall(:ylim, 1)
        for s in spwin; mxcall(:semilogx, 0, 1e3 .* [s,s], ylim, "r--"); end
        for m in mpwin; mxcall(:semilogx, 0, 1e3 .* [m,m], ylim, "g-."); end
    end

    MWImaps, MWIdist, MWIpart
end

_getmwf(modeltype::NNLSRegression, MWImaps, MWIdist, MWIpart) = MWIpart["sfr"]

# ----------------------- #
# ThreePoolCplxToCplx model
# ----------------------- #
function initialparams(modeltype::ThreePoolCplxToCplx, ts::AbstractVector{T}, S::Vector{Vec{2,T}}) where {T}
    S1, S2, SN = complex.(S[[2,3,end]]) # initial/final complex signals (S[1] is t=0 point)
    A1, AN, ϕ1, ϕ2 = abs(S1), abs(SN), angle(S1), angle(S2)
    t1, t2, tN = ts[2], ts[3], ts[end] # time points/differences
    Δt, ΔT = t2 - t1, tN - t1

    R = log(A1/AN)/ΔT
    A0 = A1*exp(R*t1) # Approximate initial total magnitude as mono-exponential
    # Δf = inv(2*(t2-t1)) # Assume a phase shift of π between S1 and S2
    Δf = (ϕ2-ϕ1)/(2π*Δt) # Use actual phase shift of π between S1 and S2 (negative sign cancelled by π pulse between t0 and t1)

    A_my, A_ax, A_ex = A0/10, 6*A0/10, 3*A0/10 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = T(10e-3), T(64e-3), T(48e-3) # T2* initial guesses
    Δf_bg_my, Δf_bg_ax, Δf_bg_ex = Δf, Δf, Δf # zero(T), zero(T), zero(T) # In continuous setting, initialize to zero #TODO (?)
    θ = ϕ1 # Initial phase (negative phase convention: -(-ϕ1) = ϕ1 from phase flip between t0 and t1)

    p  = T[A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex, Δf_bg_my,  Δf_bg_ax,  Δf_bg_ex,  θ    ]
    lb = T[0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3, Δf - 75.0, Δf - 25.0, Δf - 25.0, θ - π]
    ub = T[2*A0, 2*A0, 2*A0, 25e-3, 150e-3, 150e-3, Δf + 75.0, Δf + 25.0, Δf + 25.0, θ + π]

    p[4:6] = inv.(p[4:6]) # fit for R2 instead of T2
    lb[4:6], ub[4:6] = inv.(ub[4:6]), inv.(lb[4:6]) # swap bounds

    return p, lb, ub
end

function mwimodel(modeltype::ThreePoolCplxToCplx, t::AbstractVector, p::Vector)
    # A_my, A_ax, A_ex, T2_my, T2_ax, T2_ex, Δf_bg_my, Δf_bg_ax, Δf_bg_ex, θ = p
    # Γ_my, Γ_ax, Γ_ex = complex(1/T2_my, 2*pi*Δf_bg_my), complex(1/T2_ax, 2*pi*Δf_bg_ax), complex(1/T2_ex, 2*pi*Δf_bg_ex)

    A_my, A_ax, A_ex, R2_my, R2_ax, R2_ex, Δf_bg_my, Δf_bg_ax, Δf_bg_ex, θ = p
    Γ_my, Γ_ax, Γ_ex = complex(R2_my, 2π*Δf_bg_my), complex(R2_ax, 2π*Δf_bg_ax), complex(R2_ex, 2π*Δf_bg_ex)

    S = @. (A_my * exp(-Γ_my * t) + A_ax * exp(-Γ_ax * t) + A_ex * exp(-Γ_ex * t)) * cis(-θ)
    T = real(eltype(S)) # gives T s.t. eltype(S) <: Complex{T}
    S = copy(reinterpret(T, S)) # reinterpret as real array
    return S
end

# ThreePoolCplxToMagn model
function initialparams(modeltype::ThreePoolCplxToMagn, ts::AbstractVector{T}, S::Vector{Vec{2,T}}) where {T}
    A1, AN = norm(S[2]), norm(S[end]) # initial/final signal magnitudes (S[1] is t=0 point)
    t1, t2, tN = ts[2], ts[3], ts[end] # time points/differences
    Δt, ΔT = t2 - t1, tN - t1

    R = log(A1/AN)/ΔT
    A0 = A1*exp(R*t1) # Approximate initial total magnitude as mono-exponential

    A_my, A_ax, A_ex = A0/10, 6*A0/10, 3*A0/10 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = T(10e-3), T(64e-3), T(48e-3) # T2* initial guesses
    Δf_my_ex, Δf_ax_ex = T(5), zero(T) # In continuous setting, initialize to zero #TODO (?)

    p  = T[A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex, Δf_my_ex, Δf_ax_ex]
    lb = T[0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3,    -75.0,    -25.0]
    ub = T[2*A0, 2*A0, 2*A0, 25e-3, 150e-3, 150e-3,    +75.0,    +25.0]

    p[4:6] = inv.(p[4:6]) # fit for R2 instead of T2
    lb[4:6], ub[4:6] = inv.(ub[4:6]), inv.(lb[4:6]) # swap bounds

    return p, lb, ub
end

function mwimodel(modeltype::ThreePoolCplxToMagn, t::AbstractVector, p::Vector)
    # A_my, A_ax, A_ex, T2_my, T2_ax, T2_ex, Δf_my_ex, Δf_ax_ex = p
    # Γ_my, Γ_ax, Γ_ex = complex(1/T2_my, 2*pi*Δf_my_ex), complex(1/T2_ax, 2*pi*Δf_ax_ex), 1/T2_ex

    A_my, A_ax, A_ex, R2_my, R2_ax, R2_ex, Δf_my_ex, Δf_ax_ex = p
    Γ_my, Γ_ax, Γ_ex = complex(R2_my, 2π*Δf_my_ex), complex(R2_ax, 2π*Δf_ax_ex), R2_ex

    S = @. abs(A_my * exp(-Γ_my * t) + A_ax * exp(-Γ_ax * t) + A_ex * exp(-Γ_ex * t))
    return S
end

# ThreePoolMagnToMagn model
function initialparams(modeltype::ThreePoolMagnToMagn, ts::AbstractVector{T}, S::Vector{Vec{2,T}}) where {T}
    A1, AN = norm(S[2]), norm(S[end]) # initial/final signal magnitudes (S[1] is t=0 point)
    t1, t2, tN = ts[2], ts[3], ts[end] # time points/differences
    Δt, ΔT = t2 - t1, tN - t1

    R = log(A1/AN)/ΔT
    A0 = A1*exp(R*t1) # Approximate initial total magnitude as mono-exponential

    A_my, A_ax, A_ex = A0/10, 6*A0/10, 3*A0/10 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = T(10e-3), T(64e-3), T(48e-3) # T2* initial guesses

    p  = T[A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex]
    lb = T[0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3]
    ub = T[2*A0, 2*A0, 2*A0, 25e-3, 150e-3, 150e-3]

    p[4:6] = inv.(p[4:6]) # fit for R2 instead of T2
    lb[4:6], ub[4:6] = inv.(ub[4:6]), inv.(lb[4:6]) # swap bounds

    return p, lb, ub
end

function mwimodel(modeltype::ThreePoolMagnToMagn, t::AbstractVector, p::Vector)
    # A_my, A_ax, A_ex, T2_my, T2_ax, T2_ex = p
    # Γ_my, Γ_ax, Γ_ex = 1/T2_my, 1/T2_ax, 1/T2_ex
    A_my, A_ax, A_ex, Γ_my, Γ_ax, Γ_ex = p
    S = @. A_my * exp(-Γ_my * t) + A_ax * exp(-Γ_ax * t) + A_ex * exp(-Γ_ex * t)
    return S
end

# TwoPoolMagnToMagn model
function initialparams(modeltype::TwoPoolMagnToMagn, ts::AbstractVector{T}, S::Vector{Vec{2,T}}) where {T}
    A1, AN = norm(S[2]), norm(S[end]) # initial/final signal magnitudes (S[1] is t=0 point)
    t1, t2, tN = ts[2], ts[3], ts[end] # time points/differences
    Δt, ΔT = t2 - t1, tN - t1

    R = log(A1/AN)/ΔT
    A0 = A1*exp(R*t1) # Approximate initial total magnitude as mono-exponential

    A_my, A_ex = A0/3, 2*A0/3 # Relative magnitude initial guesses
    T2_my, T2_ex = T(10e-3), T(48e-3) # T2* initial guesses

    p  = T[A_my, A_ex, T2_my,  T2_ex]
    lb = T[0.0,  0.0,   3e-3,  25e-3]
    ub = T[2*A0, 2*A0, 25e-3, 150e-3]

    p[3:4] = inv.(p[3:4]) # fit for R2 instead of T2
    lb[3:4], ub[3:4] = inv.(ub[3:4]), inv.(lb[3:4]) # swap bounds

    return p, lb, ub
end

function mwimodel(modeltype::TwoPoolMagnToMagn, t::AbstractVector, p::Vector)
    # A_my, A_ax, A_ex, T2_my, T2_ax, T2_ex = p
    # Γ_my, Γ_ax, Γ_ex = 1/T2_my, 1/T2_ax, 1/T2_ex
    A_my, A_ex, Γ_my, Γ_ex = p
    S = @. A_my * exp(-Γ_my * t) + A_ex * exp(-Γ_ex * t)
    return S
end

# Fitting of general AbstractMWIFittingModel
function _fitmwfmodel(
        signals::Vector{V}, # signal vectors
        modeltype::AbstractMWIFittingModel = ThreePoolCplxToCplx();
        TE = 10e-3, # echo spacing
        fitmethod = :local
    ) where {V <: Vec{2}}

    nTE = length(signals)-1 # S[1] is t=0 point
    ts = TE.*(0:nTE) |> collect
    ydata = mwimodeldata(modeltype, signals) # model data
    tdata = ts[2:end] # ydata time points (first point is dropped)
    p0, lb, ub = initialparams(modeltype, ts, signals)

    model(t, p) = mwimodel(modeltype, t, p)

    return if fitmethod == :global
        # global optimization
        loss(p) = sum(abs2, ydata .- model(tdata, p))
        global_result = BlackBoxOptim.bboptimize(loss;
            SearchRange = collect(zip(lb, ub)),
            NumDimensions = length(p0),
            MaxSteps = 1e5,
            TraceMode = :silent
        )
        global_xbest = BlackBoxOptim.best_candidate(global_result)

        modelfit = (param = global_xbest,)
        errors = nothing

        modelfit, errors
    else
        # default to local optimization
        wrapped_model(p) = model(tdata, p)
        cfg = ForwardDiff.JacobianConfig(wrapped_model, p0, ForwardDiff.Chunk{length(p0)}())
        jac_model(t, p) = ForwardDiff.jacobian(wrapped_model, p, cfg)

        modelfit = LsqFit.curve_fit(model, jac_model, tdata, ydata, p0; lower = lb, upper = ub)
        errors = try
            LsqFit.margin_error(modelfit, 0.05) # 95% confidence errors
        catch e
            nothing
        end

        modelfit, errors
    end
end

function _getmwf(modeltype::ThreePoolModel, modelfit, errors)
    A_my, A_ax, A_ex = modelfit.param[1:3]
    return A_my/(A_my + A_ax + A_ex)
end

function _getmwf(modeltype::TwoPoolModel, modelfit, errors)
    A_my, A_ex = modelfit.param[1:2]
    return A_my/(A_my + A_ex)
end

function compareMWFmethods(sols, myelindomains, outercircles, innercircles, bdry)
    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    ts = tspan[1]:TE:tspan[2] # signal after each echo
    Stotal = calcsignal(sols, ts, myelindomains)

    mwfvalues = Dict(
        :exact => getmwf(outercircles, innercircles, bdry),
        :NNLSRegression => getmwf(Stotal, NNLSRegression(); TE = TE),
        :TwoPoolMagnToMagn => getmwf(Stotal, TwoPoolMagnToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolMagnToMagn => getmwf(Stotal, ThreePoolMagnToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolCplxToMagn => getmwf(Stotal, ThreePoolCplxToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolCplxToCplx => getmwf(Stotal, ThreePoolCplxToCplx(); TE = TE, fitmethod = :local)
    )

    return mwfvalues
end