# ---------------------------------------------------------------------------- #
# SpinEchoCallback
# ---------------------------------------------------------------------------- #

# Convention is that u₃ = M∞ - M₃; this convenience function converts between M and u
@inline shift_uz(u::uType, M∞) where {uType <: Vec{3}} = uType((u[1], u[2], M∞ - u[3]))

@inline pi_flip(u::Complex) = conj(u)
@inline pi_flip(u::uType) where {uType <: Vec{2}} = uType((u[1], -u[2]))
@inline pi_flip(u::uType) where {uType <: Vec{3}} = uType((u[1], -u[2], -u[3]))
pi_pulse!(u::AbstractVector{uType}) where {uType <: FieldType} = (u .= pi_flip.(u); return u)

apply_pulse!(u::AbstractVector{Tu}, α, M∞, uDim) where {Tu} = (apply_pulse!(reinterpret(Vec{uDim,Tu}, u), α, M∞); return u)
apply_pulse!(u::AbstractVector{uType}, α, M∞) where {uType <: Complex} = (@assert α ≈ π; pi_pulse!(u); return u)
apply_pulse!(u::AbstractVector{uType}, α, M∞) where {uType <: Vec{2}} = (@assert α ≈ π; pi_pulse!(u); return u)
function apply_pulse!(u::AbstractVector{uType}, α, M∞) where {Tu, uType <: Vec{3,Tu}}
    u .= shift_uz.(u, M∞)
    if α ≈ π
        pi_pulse!(u)
    else
        # Our convention is that M∞ points in the +z-direction. This means that our typical initial condition,
        # M₀ = [0, M∞, 0], is actually a rotation of [0, 0, M∞] by -π/2 about the x-axis, not of +π/2.
        # To be consistent, we apply all general rotations by -α (which is equivalent to +α when α = π)
        R = Tensor{2,3,Tu}(RotX(-α))
        u .= (x -> R ⋅ x).(u)
    end
    u .= shift_uz.(u, M∞)
    return u
end

# NOTE: This constructor works and is more robust than the below version, but
#       requires callbacks to be initialized, which Sundials doesn't support.
function MultiSpinEchoCallback(
        u0::FieldType, tspan::NTuple{2,T};
        TE::T = (tspan[2] - tspan[1]), # default to single echo
        flipangle = π, # flip angle
        steadystate = 1, # steady state value for z-component of magnetization
        verbose = true, # verbose printing
        pulsetimes::AbstractVector{T} = tspan[1] + TE/2 : TE : tspan[2], # default to equispaced pulses every TE starting at TE/2
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

    # Apply pulse and return next pulse time
    time_choice(integrator) = isempty(tstops) ? typemax(T) : popfirst!(tstops)
    function user_affect!(integrator)
        verbose && println("$(rad2deg(flipangle)) degree pulse at t = $(1000*integrator.t) ms")
        apply_pulse!(integrator.u, flipangle, steadystate, fielddim(u0))
        return integrator
    end

    # Return IterativeCallback
    callback = IterativeCallback(time_choice, user_affect!, T; initial_affect = initial_affect, kwargs...)
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