# ---------------------------------------------------------------------------- #
# SpinEchoCallback
# ---------------------------------------------------------------------------- #

# Apply pi-pulse to u
pi_pulse!(u::AbstractVector) = (@views u[2:2:end] .= -u[2:2:end]; return u) # This assumes u represents an array of Vec{2}'s
pi_pulse!(u::AbstractVector{Vec{2,T}}) where {T} = (_u = reinterpret(T, u); pi_pulse!(_u); return u)
pi_pulse!(u::AbstractVector{Tc}) where {Tc<:Complex} = (u .= conj.(u); return u)

# NOTE: This constructor works and is more robust than the below version, but
#       requires callbacks to be initialized, which Sundials doesn't support.
function MultiSpinEchoCallback(
        tspan::NTuple{2,T};
        TE::T = (tspan[2] - tspan[1]), # default to single echo
        verbose = true, # verbose printing
        pulsetimes::AbstractVector{T} = tspan[1] + TE/2 : TE : tspan[2], # default equispaced π-pulses every TE starting at TE/2
        kwargs...
    ) where {T}

    if isempty(pulsetimes)
        @warn "No pulsetimes given; returning `nothing` for callback"
        return nothing
    end

    # If first pulse time is at tspan[1], set initial_affect true and pop tstops[1]
    tstops = copy(collect(pulsetimes))
    initial_affect = (tstops[1] == tspan[1]) # tstops is non-empty from above check
    initial_affect && popfirst!(tstops) # first pulse is handled by initial_affect

    # Apply π-pulse/return next π-pulse time
    time_choice(integrator) = isempty(tstops) ? typemax(T) : popfirst!(tstops)
    function user_affect!(integrator)
        verbose && println("π-pulse at t = $(1000*integrator.t) ms")
        pi_pulse!(integrator.u)
        return integrator
    end

    # Return IterativeCallback
    callback = IterativeCallback(time_choice, user_affect!, T; initial_affect = initial_affect, kwargs...)
    return callback
end

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