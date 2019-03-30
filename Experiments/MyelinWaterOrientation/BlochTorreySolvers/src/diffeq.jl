# ---------------------------------------------------------------------------- #
# ODEProblem constructor for ParabolicDomain's and MyelinDomain's
# ---------------------------------------------------------------------------- #

# Create an `ODEProblem` from a `ParabolicDomain` representing either
#   du/dt = (M\K)*u   [invertmass = true], or
#   M*du/dt = K*u     [invertmass = false]
function OrdinaryDiffEq.ODEProblem(d::ParabolicDomain, u0, tspan; invertmass = true)
    if !invertmass
        @warn "invertmass = false not yet supported; setting invertmass = true."
        invertmass = true
    end

    f!(du,u,p,t) = mul!(du, p[1], u) # RHS action of ODE for general matrix A stored in p[1]
    A = ParabolicLinearMap(d) # ParabolicLinearMap returns a subtype of LinearMap which acts onto u as (M\K)*u.
    p = (A,) # ODEProblem parameter tuple
    return ODEProblem(f!, u0, tspan, p)

    # if invertmass
    #     # ParabolicLinearMap returns a linear operator which acts by (M\K)*u.
    #     A = ParabolicLinearMap(d) # subtype of LinearMap
    #     p = (DiffEqParabolicLinearMapWrapper(A),) # wrap LinearMap in an AbstractArray wrapper
    #     F! = ODEFunction{true,true}(f!; # represents M*du/dt = K*u system
    #         mass_matrix = I, # mass matrix
    #         jac = (J,u,p,t) -> J, # Jacobian is constant (DiffEqParabolicLinearMapWrapper)
    #         jac_prototype = p[1]
    #     )
    #     return ODEProblem(F!, u0, tspan, p)
    # else
    #     K, M = getstiffness(d), getmass(d)
    #     p = (K, M)
    #     F! = ODEFunction{true,true}(f!; # represents M*du/dt = K*u system
    #         mass_matrix = M, # mass matrix
    #         jac = (J,u,p,t) -> J, # Jacobian is constant (stiffness matrix)
    #         jac_prototype = K
    #     )
    #     return ODEProblem(F!, u0, tspan, p)
    # end
end
OrdinaryDiffEq.ODEProblem(m::MyelinDomain, u0, tspan; kwargs...) = ODEProblem(getdomain(m), u0, tspan; kwargs...)

# ---------------------------------------------------------------------------- #
# SpinEchoCallback
# ---------------------------------------------------------------------------- #

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
        @views integrator.u[2:2:end] .= -integrator.u[2:2:end]
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
#         @views integrator.u[2:2:end] .= -integrator.u[2:2:end]
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

# ---------------------------------------------------------------------------- #
# AbstractExpmv utilities
# ---------------------------------------------------------------------------- #

abstract type AbstractExpmvAlgorithm <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end

OrdinaryDiffEq.isfsal(alg::AbstractExpmvAlgorithm) = true # NOTE: this is default; "first same as last" property
OrdinaryDiffEq.alg_order(alg::AbstractExpmvAlgorithm) = 2 # TODO: order of expmv? used for interpolation; likely unimportant?
OrdinaryDiffEq.alg_cache(alg::AbstractExpmvAlgorithm,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{true}})  = ExpmvCache(u,uprev,similar(u),similar(u),zero(rate_prototype),zero(rate_prototype))
OrdinaryDiffEq.alg_cache(alg::AbstractExpmvAlgorithm,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{false}}) = ExpmvConstantCache()

struct ExpmvCache{uType,rateType} <: OrdinaryDiffEq.OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    b1::uType
    b2::uType
    k::rateType
    fsalfirst::rateType
end
struct ExpmvConstantCache <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache end

OrdinaryDiffEq.u_cache(c::ExpmvCache) = ()
OrdinaryDiffEq.du_cache(c::ExpmvCache) = (c.k,c.fsalfirst)

# function OrdinaryDiffEq.initialize!(integrator,cache::ExpmvConstantCache)
#   integrator.kshortsize = 2
#   integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
#   integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
#
#   # Avoid undefined entries if k is an array of arrays
#   integrator.fsallast = zero(integrator.fsalfirst)
#   integrator.k[1] = integrator.fsalfirst
#   integrator.k[2] = integrator.fsallast
# end

function OrdinaryDiffEq.initialize!(integrator,cache::ExpmvCache)
    integrator.kshortsize = 2
    @unpack k,fsalfirst = cache
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t) # For the interpolation, needs k at the updated point
end

# ---------------------------------------------------------------------------- #
# ExpokitExpmv stepper
# ---------------------------------------------------------------------------- #

@with_kw struct ExpokitExpmv{T,F} <: AbstractExpmvAlgorithm
    m::Int = 30
    anorm::T = 1.0
    norm::F = LinearAlgebra.norm
end
ExpokitExpmv(A; kwargs...) = ExpokitExpmv(;kwargs..., anorm = LinearAlgebra.opnorm(A, Inf))

# function OrdinaryDiffEq.perform_step!(integrator,cache::ExpokitExpmvConstantCache,repeat_step=false)
#   @unpack t,dt,uprev,u,f,p = integrator
#   @muladd u = uprev + dt*integrator.fsalfirst
#   k = f(u, p, t+dt) # For the interpolation, needs k at the updated point
#   integrator.fsallast = k
#   integrator.k[1] = integrator.fsalfirst
#   integrator.k[2] = integrator.fsallast
#   integrator.u = u
# end

function OrdinaryDiffEq.perform_step!(
        integrator::OrdinaryDiffEq.ODEIntegrator{Alg},
        cache::Cache,
        repeat_step = false
    ) where {Alg <: ExpokitExpmv, Cache <: ExpmvCache}

    @unpack t,dt,uprev,u,f,p = integrator
    A = p[1] # matrix being exponentiated

    Expokit.expmv!(u, dt, A, uprev;
        tol = integrator.opts.reltol,
        m = integrator.alg.m,
        norm = integrator.alg.norm,
        anorm = eltype(A)(integrator.alg.anorm)
    )
    integrator.fsallast = u
end

# ---------------------------------------------------------------------------- #
# ExpmV stepper
# ---------------------------------------------------------------------------- #

@with_kw struct HighamExpmv{T,F1,F2} <: AbstractExpmvAlgorithm
    M::Union{Nothing,Matrix{T}} = nothing
    b_columns::Int = 1
    precision::Symbol = :double
    shift::Bool = true
    full_term::Bool = false
    prnt::Bool = false
    check_positive::Bool = false
    force_estm::Bool = false
    norm::F1 = LinearAlgebra.norm
    opnorm::F2 = LinearAlgebra.opnorm
    m_max::Int = 55
    p_max::Int = 8
end
function HighamExpmv(
        A, b_columns = 1, norm = LinearAlgebra.norm, opnorm = LinearAlgebra.opnorm;
        precision = :double
    )
    M = ExpmV.select_taylor_degree(A, b_columns, opnorm; precision = precision, force_estm = true, check_positive = false, shift = true)[1]
    return HighamExpmv(
        M = M, b_columns = b_columns, norm = norm, opnorm = opnorm,
        precision = precision, force_estm = true, check_positive = false, shift = true, full_term = false)
end

# function OrdinaryDiffEq.perform_step!(integrator,cache::HighamExpmvConstantCache,repeat_step=false)
#   @unpack t,dt,uprev,u,f,p = integrator
#   @muladd u = uprev + dt*integrator.fsalfirst
#   k = f(u, p, t+dt) # For the interpolation, needs k at the updated point
#   integrator.fsallast = k
#   integrator.k[1] = integrator.fsalfirst
#   integrator.k[2] = integrator.fsallast
#   integrator.u = u
# end

function OrdinaryDiffEq.perform_step!(
        integrator::OrdinaryDiffEq.ODEIntegrator{Alg},
        cache::Cache,
        repeat_step = false
    ) where {Alg <: HighamExpmv, Cache <: ExpmvCache}

    @unpack t,dt,uprev,u,f,p,alg = integrator
    A = p[1] # matrix being exponentiated

    ExpmV.expmv!(
        u, dt, A, uprev,
        alg.M, alg.norm, alg.opnorm, cache.b1, cache.b2;
        precision = alg.precision,
        shift = alg.shift,
        full_term = alg.full_term,
        check_positive = alg.check_positive,
    )
    integrator.fsallast = u
end