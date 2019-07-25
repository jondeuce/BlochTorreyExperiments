# ---------------------------------------------------------------------------- #
# SpinEchoCallback
# ---------------------------------------------------------------------------- #

# Convention is that u₃ = M∞ - M₃; this convenience function converts between M and u
@inline shift_longitudinal(u::uType, M∞) where {uType <: Vec{3}} = uType((u[1], u[2], M∞ - u[3]))
shift_longitudinal!(u::AbstractVector{uType}, M∞) where {uType <: Vec{3}} = (u .= shift_longitudinal.(u, M∞))
shift_longitudinal!(u::AbstractVector{Tu}, M∞) where {Tu} = (_u = reinterpret(Vec{3,Tu}, u); shift_longitudinal!(_u, M∞); return u)
shift_longitudinal(u::AbstractVector, M∞) = shift_longitudinal!(copy(u), M∞)

@inline pi_flip(u::Complex) = conj(u)
@inline pi_flip(u::uType) where {uType <: Vec{2}} = uType((u[1], -u[2]))
@inline pi_flip(u::uType) where {uType <: Vec{3}} = uType((u[1], -u[2], -u[3]))
pi_pulse!(u::AbstractVector{uType}) where {uType <: FieldType} = (u .= pi_flip.(u))
pi_pulse(u::AbstractVector) = pi_pulse!(copy(u))

apply_pulse!(u::AbstractVector{Tu},    α, pulsetype::Symbol, ::Type{uType}) where {dim, Tu, uType <: Vec{dim,Tu}} = (_u = reinterpret(Vec{dim,Tu}, u); apply_pulse!(_u, α, pulsetype); return u)
apply_pulse!(u::AbstractVector{uType}, α, pulsetype::Symbol, ::Type{uType}) where {Tu, uType <: Complex{Tu}} = (_u = reinterpret(Vec{2,Tu}, u); apply_pulse!(_u, α, pulsetype); return u)
apply_pulse!(u::AbstractVector{uType}, α, pulsetype::Symbol, ::Type{uType}) where {uType <: Vec} = apply_pulse!(u, α, pulsetype)
apply_pulse!(u::AbstractVector{uType}, α, pulsetype::Symbol) where {Tu, uType <: Vec{3,Tu}} = (R = pulsemat3(Tu, α, pulsetype); u .= (x -> R ⋅ x).(u); return u)
apply_pulse!(u::AbstractVector{uType}, α, pulsetype::Symbol) where {Tu, uType <: Vec{2,Tu}} = (R = pulsemat2(Tu, α, pulsetype); u .= (x -> R ⋅ x).(u); return u)

function cpmg_savetimes(dt::T, TE::T, TR::T, nTE::Int, nTR::Int) where {T}
    ndt_TE = round(Int, TE/dt)
    @assert TE ≈ ndt_TE * dt && ndt_TE >= 2 && iseven(ndt_TE)
    @assert TR >= nTE * TE && nTE >= 1 && nTR >= 1
    
    ndt_TR = round(Int, TR/dt)
    !(ndt_TR * dt ≈ TR) && (ndt_TR * dt > TR) && (ndt_TR -= 1)

    savetimes = T[]
    for n in 0:nTR-1
        newsavetimes = n * TR .+ dt .* (0 : min(ndt_TR, (nTR-1-n) * ndt_TR + nTE * ndt_TE))
        (n < nTR-1) && (newsavetimes[end] ≈ (n+1) * TR) ?
            append!(savetimes, newsavetimes[1:end-1]) : # last point will be added next iteration
            append!(savetimes, newsavetimes)
    end
    savetimes[1]   = max(savetimes[1], zero(T)) # Enforce initial point
    savetimes[end] = min(savetimes[end], nTE * TE + (nTR-1) * TR) # Enforce final point

    return savetimes
end
function cpmg_savetimes(tspan::NTuple{2,T}, dt::T, TE::T, TR::T, nTE::Int, nTR::Int) where {T}
    @assert tspan[1] ≈ 0 && tspan[2] ≈ (nTR - 1) * TR + nTE * TE
    savetimes = cpmg_savetimes(dt, TE, TR, nTE, nTR)
    savetimes[1] = tspan[1] # Enforce initial point (correct floating point errors)
    savetimes[end] = tspan[2] # Enforce final point (correct floating point errors)
    return savetimes
end

"""
    CPMGCallback

Callback for applying pulses which simulate the (modified) Carr-Purcell-Meiboom-Gill (CPMG)
pulse sequence (Carr & Purcell 1954; Meiboom & Gill 1958).

Slice selective pulses with flip angle `sliceselectangle` are applied every TR, starting
at t = TR, repeating until the simulation end.
    
    NOTE: initial pulses at t = 0 must be handled outside of this callback.

Additionally, refocusing pulse trains of type `refocustype` are applied every `TE` from
`(n-1) * TR + TE/2` until `(n-1) * TR + nTE * TE - TE/2` for a total of `nTE` pulses,
where `n` is the repetition number.

`refocustype` is a symbol and may be one of three types (where α = `flipangle`):
    :x      Alternating RotX(+α), RotX(-α) pulses about the x-axis
    :y      Consecutive RotY(+α) pulses about the y-axis
    :xyx    Consecutive RotX(π/2) * RotY(+α) * RotX(π/2) composite block pulses,
            equivalent to consecutive +π-pulses about the axis u = [cos(α/2), sin(α/2), 0]

Finally, note that all pulses are shifted early by a very small configurable amount,
`pulseshift`, in order to ensure that sampling the signal at e.g. t = TE/2 is guaranteed
to occur after the pulse. By default, `pulseshift` is max(1ns, 10*eps).
"""
function CPMGCallback(
        ::Type{uType}, tspan::NTuple{2,T};
        TE::T = tspan[2] - tspan[1], # default to single echo
        TR::T = tspan[2] - tspan[1], # default to immediate repetition
        nTE::Int = 1, # default to single echo
        nTR::Int = 1, # default to single rep
        sliceselectangle::T = T(π)/2, # init pulse flip angle (default none)
        flipangle::T = T(π), # multi-echo flip angle
        refocustype::Symbol = :xyx, # Type of refocusing pulse
        pulseshift::T = max(T(1e-9), 10*eps(T)), # Pulse shift time (1ns by default)
        steadystate = 1, # steady state value for z-component of magnetization
        verbose = true, # verbose printing
        kwargs...
    ) where {T, uType <: FieldType}

    # Check parameters
    @assert nTE >= 1 && nTR >= 1 && TR >= nTE * TE
    @assert tspan[1] ≈ 0 && (tspan[2] ≈ nTE * TE + (nTR - 1) * TR)

    # Initial pulse, repeating every TR (not including t = 0 pulse, handled outside this callback)
    sliceselecttype = :x # Rotation about x-axis
    initpulse_fliptimes = [TR .* (1:nTR-1) .- pulseshift;]
    initpulse_callback = isempty(initpulse_fliptimes) ? CallbackSet() :
        EchoTrainCallback(uType, sliceselectangle, sliceselecttype, initpulse_fliptimes, steadystate, verbose)

    # Flip angle pulses, repeating nTE times every TE starting at TE/2, repeating every TR
    flipangle_callbacks = if refocustype == :x
        positive_flipangle_fliptimes = [[n*TR .+ TE/2 .+ TE .* (0:2:nTE-1) .- pulseshift;] for n in 0:nTR-1]
        negative_flipangle_fliptimes = [[n*TR .+ TE/2 .+ TE .* (1:2:nTE-1) .- pulseshift;] for n in 0:nTR-1]
        CallbackSet(
            (EchoTrainCallback(uType, +flipangle, refocustype, fliptimes, steadystate, verbose) for fliptimes in positive_flipangle_fliptimes)...,
            (EchoTrainCallback(uType, -flipangle, refocustype, fliptimes, steadystate, verbose) for fliptimes in negative_flipangle_fliptimes)...)
    else
        flipangle_fliptimes = [[n*TR .+ TE/2 .+ TE .* (0:nTE-1) .- pulseshift;] for n in 0:nTR-1]
        CallbackSet((EchoTrainCallback(uType, flipangle, refocustype, fliptimes, steadystate, verbose) for fliptimes in flipangle_fliptimes)...)
    end

    return CallbackSet(initpulse_callback, flipangle_callbacks)
end

"""
    EchoTrainCallback

Apply pulse of type `pulsetype` with flip angle `flipangle` at time points `fliptimes`.

    NOTE: `fliptimes` cannot obtain t = tspan[1]; flipping of the initial state
          must be handled outside of this callback.
"""
function EchoTrainCallback(
        ::Type{uType},
        flipangle::T, # Flip angle
        pulsetype::Symbol, # Pulse type
        fliptimes::AbstractVector{T}, # Flip times
        steadystate = 1, # Steady state value for z-component of magnetization
        verbose = true, # Verbose printing
        kwargs...
    ) where {T, uType <: FieldType{T}}

    # Collect tstops
    tstops = copy(collect(fliptimes))
    if isempty(tstops)
        return nothing
    end

    # Return next pulse time/next pulse type
    time_choice(integrator) = isempty(tstops) ? typemax(T) : popfirst!(tstops)
    
    # Apply appropriate pulse
    function user_affect!(integrator)
        if verbose
            anglestr, timestr = @sprintf("%6.1f", rad2deg(flipangle)), @sprintf("%.1f ms", 1000 * integrator.t)
            println("$anglestr degree pulse of type $pulsetype applied at t = $timestr")
        end
        (uType <: Vec{3}) && shift_longitudinal!(integrator.u, steadystate) # Convert from u-space to M-space
        apply_pulse!(integrator.u, flipangle, pulsetype, uType) # Apply rotation in M-space
        (uType <: Vec{3}) && shift_longitudinal!(integrator.u, steadystate) # Convert back to u-space
        return integrator
    end

    # Return IterativeCallback
    callback = IterativeCallback(time_choice, user_affect!, T; initial_affect = false, kwargs...)
    return callback
end