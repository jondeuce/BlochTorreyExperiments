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

# Our convention is that M∞ points in the +z-direction. This means that our typical initial condition,
# M₀ = [0, M∞, 0], is actually a rotation of [0, 0, M∞] by -π/2 about the x-axis, not of +π/2.
# To be consistent, we apply all general rotations by -α (which is equivalent to +α when α = π)
apply_pulse!(u::AbstractVector{Tu}, α, ::Type{uType}) where {dim, Tu, uType <: Vec{dim,Tu}} = (_u = reinterpret(Vec{dim,Tu}, u); apply_pulse!(_u, α); return u)
apply_pulse!(u::AbstractVector{uType}, α, ::Type{uType}) where {Tu, uType <: Complex{Tu}} = (_u = reinterpret(Vec{2,Tu}, u); apply_pulse!(_u, α); return u)
apply_pulse!(u::AbstractVector{uType}, α, ::Type{uType}) where {uType <: Vec} = apply_pulse!(u, α)
apply_pulse!(u::AbstractVector{uType}, α) where {Tu, uType <: Vec{3,Tu}} = (R = pulsemat(Tu, α); u .= (x -> R ⋅ x).(u); return u)
apply_pulse!(u::AbstractVector{uType}, α) where {uType <: Vec{2}} = (@assert α ≈ π; pi_pulse!(u); return u)

# Flip times are equispaced pulses every TE starting at t = TE/2 until t = nTE * TE,
# repeating every TR, with an additional pulse at multiple of t = TR, not including t = 0.
# All pulses are shifted early by √eps() so that sampling the signal at e.g. t = TE/2
# is guaranteed to be after the pulse; pulses are therefore contained in the closed interval
# [TE/2 - √eps(), nTR * TR + nTE * TE - TE/2 - √eps()]
function init_fliptimes(TE::T, TR::T, nTE::Int = 1, nTR::Int = 1) where {T}
    @assert TR >= nTE * TE && nTE >= 1 && nTR >= 1
    fliptimes = if TR ≈ nTE * TE
        [TE/2 .+ TE .* (0 : round(Int, nTR * TR / TE) - 1);]
    else
        fliptimes = T[]
        for n in 0:nTR-1
            (n > 0) && push!(fliptimes, n*TR)
            append!(fliptimes, n*TR .+ TE/2 .+ TE .* (0:nTE-1))
        end
        fliptimes
    end
    fliptimes .-= √eps(TE)
    return fliptimes
end

function init_savetimes(TE::T, TR::T, nTE::Int = 1, nTR::Int = 1) where {T}
    @assert TR >= nTE * TE && nTE >= 1 && nTR >= 1
    if TR ≈ nTE * TE
        [TE/2 .* (0 : round(Int, 2 * nTR * TR / TE));]
    else
        savetimes = T[]
        for n in 0:nTR - 1
            append!(savetimes, n*TR .+ TE/2 .* (0 : 2 * nTE))
            (n < nTR - 1) && append!(savetimes, n*TR .+ TE/2 .* (2 * nTE + 1 : ceil(Int, 2 * TR / TE) - 1))
        end
        savetimes
    end
end

# NOTE: This constructor works and is more robust than the below version, but
#       requires callbacks to be initialized, which Sundials doesn't support.
function MultiSpinEchoCallback(
        ::Type{uType}, tspan::NTuple{2,T};
        TE::T = tspan[2] - tspan[1], # default to single echo
        TR::T = tspan[2] - tspan[1], # default to immediate repetition
        nTE::Int = 1, # default to single echo
        nTR::Int = 1, # default to single rep
        fliptimes::AbstractVector{T} = init_fliptimes(TE, TR, nTE, nTR), # default pulsetrain
        initpulse::T = T(π)/2, # init pulse flip angle (default none)
        flipangle::T = T(π), # multi-echo flip angle
        steadystate = 1, # steady state value for z-component of magnetization
        verbose = true, # verbose printing
        kwargs...
    ) where {T, uType <: FieldType}

    # Check parameters
    @assert nTE >= 1 && nTR >= 1 && TR >= TE
    @assert tspan[1] ≈ 0 && (tspan[2] ≈ nTE * TE + (nTR - 1) * TR)
    
    tstops = copy(collect(fliptimes))
    isinitpulse = reduce(vcat, [true; falses(nTE)] for _ in 1:nTR-1; init = falses(nTE))
    
    @assert !isempty(tstops)
    @assert !(tstops[1] ≈ 0) # Initial t=0 pulse must be handled outside this callback

    # Return next pulse time/next pulse type
    isinitpulse_choice() = isempty(isinitpulse) ? false : popfirst!(isinitpulse)
    time_choice(integrator) = isempty(tstops) ? typemax(T) : popfirst!(tstops)
    
    # Apply appropriate pulse
    function user_affect!(integrator)
        α = isinitpulse_choice() ? initpulse : flipangle
        verbose && println("$(round(rad2deg(α); digits=3)) degree pulse at t = $(round(1000*integrator.t; digits=3)) ms")
        (uType <: Vec{3}) && shift_longitudinal!(integrator.u, steadystate) # convert from u-space to M-space
        apply_pulse!(integrator.u, α, uType) # apply rotation in M-space
        (uType <: Vec{3}) && shift_longitudinal!(integrator.u, steadystate) # convert back to u-space
        return integrator
    end

    # Return IterativeCallback
    callback = IterativeCallback(time_choice, user_affect!, T; initial_affect = false, kwargs...)
    return callback
end

# TODO: Update below alternative constructor
# NOTE: This constructor works but is fragile, in particular `tstops` must be
#       passed to `solve` to ensure that no pulses are missed; the above
#       constructor should be preferred when not using Sundials

# function MultiSpinEchoCallback(
#         tspan::NTuple{2,T};
#         TE::T = (tspan[2] - tspan[1]), # default to single echo
#         verbose = true, # verbose printing
#         kwargs...
#     ) where {T}
# 
#     # Pulse times start at tspan[1] + TE/2 and repeat every TE
#     tstops = collect(tspan[1] + TE/2 : TE : tspan[2])
#     next_pulse() = isempty(tstops) ? typemax(T) : popfirst!(tstops) # next pulse time function
#     next_pulse(integrator) = next_pulse() # next_pulse is integrator independent
# 
#     # Discrete condition
#     tnext = Ref(typemin(T)) # initialize to -Inf
#     function condition(u, t, integrator)
#         # TODO this initialization hack is due to Sundials not supporting initializing callbacks
#         # NOTE this can fail if the first time step is bigger than TE/2! need to set tstops in `solve` explicitly to be sure
#         (tnext[] == typemin(T)) && (tnext[] = next_pulse(); add_tstop!(integrator, tnext[]))
#         t == tnext[]
#     end
# 
#     # # Discrete condition
#     # tnext = Ref(next_pulse()) # initialize to tstops[1]
#     # function condition(u, t, integrator)
#     #     t == tnext[]
#     # end
# 
#     # Apply π-pulse
#     function apply_pulse!(integrator)
#         verbose && println("π-pulse at t = $(1000*integrator.t) ms")
#         pi_pulse!(integrator.u)
#         return integrator
#     end
# 
#     # Affect function
#     function affect!(integrator)
#         apply_pulse!(integrator)
#         tnext[] = next_pulse()
#         (tnext[] <= tspan[2]) && add_tstop!(integrator, tnext[])
#         return integrator
#     end
# 
#     # Create DiscreteCallback
#     callback = DiscreteCallback(condition, affect!; kwargs...)
# 
#     # Return CallbackSet
#     return callback
# end