# ---------------------------------------------------------------------------- #
# AbstractExpmv
# ---------------------------------------------------------------------------- #

abstract type AbstractExpmvAlgorithm <: OrdinaryDiffEq.OrdinaryDiffEqExponentialAlgorithm end

OrdinaryDiffEq.isfsal(alg::AbstractExpmvAlgorithm) = true # "first same as last" property (default)
OrdinaryDiffEq.alg_order(alg::AbstractExpmvAlgorithm) = 1 # Interpolation order (same as OrdinaryDiffEq.LinearExponential)

# TODO
# OrdinaryDiffEq.u_cache(c::ExpokitExpmvCache) = ()
# OrdinaryDiffEq.du_cache(c::ExpokitExpmvCache) = (c.k,c.fsalfirst)
# OrdinaryDiffEq.u_cache(c::HighamExpmvCache) = ()
# OrdinaryDiffEq.du_cache(c::HighamExpmvCache) = (c.k,c.fsalfirst)

# ---------------------------------------------------------------------------- #
# ExpokitExpmv
# ---------------------------------------------------------------------------- #

@with_kw struct ExpokitExpmv{aType,F1,F2} <: AbstractExpmvAlgorithm
    m::Int = 30
    anorm::aType = nothing
    norm::F1 = LinearAlgebra.norm
    opnorm::F2 = LinearAlgebra.opnorm
end

struct ExpokitExpmvCache{uType,rateType,aType} <: OrdinaryDiffEq.OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    rtmp::rateType
    anorm::aType
end

function OrdinaryDiffEq.alg_cache(alg::ExpokitExpmv,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{true}})
    @unpack anorm, norm, opnorm = alg
    A = f.f # assume f to be an ODEFunction wrapped around a linear operator
    if anorm == nothing
        anorm = opnorm(A, Inf)
    end
    ExpokitExpmvCache(u, uprev, similar(u), zero(rate_prototype), anorm)
end

function OrdinaryDiffEq.initialize!(integrator,cache::ExpokitExpmvCache)
    # Pre-start fsal
    integrator.fsalfirst = zero(cache.rtmp)
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t)
    integrator.destats.nf += 1
    integrator.fsallast = zero(integrator.fsalfirst)

    # Initialize interpolation derivatives
    integrator.kshortsize = 2
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
end

function OrdinaryDiffEq.perform_step!(
        integrator::OrdinaryDiffEq.ODEIntegrator{Alg},
        cache::Cache,
        repeat_step = false
    ) where {Alg <: ExpokitExpmv, Cache <: ExpokitExpmvCache}

    @unpack t, dt, uprev, u, f, p = integrator
    @unpack tmp, anorm = cache
    A = f.f # assume f to be an ODEFunction wrapped around a linear operator

    Expokit.expmv!(tmp, dt, A, uprev;
        tol = integrator.opts.reltol,
        m = integrator.alg.m,
        norm = integrator.alg.norm,
        anorm = anorm)

    # Update integrator state
    u .= tmp
    f(integrator.fsallast, u, p, t + dt)
    integrator.destats.nf += 1
    # integrator.k is automatically set due to aliasing
end

# struct ExpokitExpmvConstantCache{aType} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
#     anorm::aType
# end
# OrdinaryDiffEq.alg_cache(alg::ExpokitExpmv,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{false}}) = ExpokitExpmvConstantCache()
#
# function OrdinaryDiffEq.perform_step!(integrator,cache::ExpokitExpmvConstantCache,repeat_step=false)
#     @unpack t,dt,uprev,u,f,p = integrator
#     @muladd u = uprev + dt*integrator.fsalfirst
#     k = f(u, p, t+dt) # For the interpolation, needs k at the updated point
#     integrator.fsallast = k
#     integrator.k[1] = integrator.fsalfirst
#     integrator.k[2] = integrator.fsallast
#     integrator.u = u
# end
#
# function OrdinaryDiffEq.initialize!(integrator,cache::ExpokitExpmvConstantCache)
#     integrator.kshortsize = 2
#     integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
#     integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
#
#     # Avoid undefined entries if k is an array of arrays
#     integrator.fsallast = zero(integrator.fsalfirst)
#     integrator.k[1] = integrator.fsalfirst
#     integrator.k[2] = integrator.fsallast
# end

# ---------------------------------------------------------------------------- #
# HighamExpmv
# ---------------------------------------------------------------------------- #

@with_kw struct HighamExpmv{MType,F1,F2} <: AbstractExpmvAlgorithm
    M::MType = nothing
    b_columns::Int = 1
    precision::Symbol = :double
    shift::Bool = true
    full_term::Bool = false
    prnt::Bool = false
    check_positive::Bool = false
    force_estm::Bool = true
    norm::F1 = LinearAlgebra.norm
    opnorm::F2 = LinearAlgebra.opnorm
    m_max::Int = 55
    p_max::Int = 8
end

struct HighamExpmvCache{uType,rateType,MType} <: OrdinaryDiffEq.OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    b1::uType
    b2::uType
    tmp::uType
    rtmp::rateType
    M::MType
end
function OrdinaryDiffEq.alg_cache(alg::HighamExpmv,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{true}})
    @unpack M, b_columns, opnorm, precision, force_estm, check_positive, shift = alg
    A = f.f # assume f to be an ODEFunction wrapped around a MatrixFreeOperator
    if M == nothing
        M = ExpmV.select_taylor_degree(A, b_columns, opnorm;
            precision = precision, force_estm = force_estm,
            check_positive = check_positive, shift = shift)[1]
    end
    HighamExpmvCache(u, uprev, similar(u), similar(u), similar(u), zero(rate_prototype), M)
end

function OrdinaryDiffEq.initialize!(integrator,cache::HighamExpmvCache)
    # Pre-start fsal
    integrator.fsalfirst = zero(cache.rtmp)
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t)
    integrator.destats.nf += 1
    integrator.fsallast = zero(integrator.fsalfirst)

    # Initialize interpolation derivatives
    integrator.kshortsize = 2
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
end

function OrdinaryDiffEq.perform_step!(
        integrator::OrdinaryDiffEq.ODEIntegrator{Alg},
        cache::Cache,
        repeat_step = false
    ) where {Alg <: HighamExpmv, Cache <: HighamExpmvCache}

    @unpack t, dt, uprev, u, f, p, alg = integrator
    @unpack norm, opnorm, precision, shift, full_term, check_positive = alg
    @unpack tmp, b1, b2, M = cache
    A = f.f # assume f to be an ODEFunction wrapped around a linear operator

    ExpmV.expmv!(tmp, dt, A, uprev, M, norm, opnorm, b1, b2;
        precision = precision,
        shift = shift,
        full_term = full_term,
        check_positive = check_positive)

    # Update integrator state
    u .= tmp
    f(integrator.fsallast, u, p, t + dt)
    integrator.destats.nf += 1
    # integrator.k is automatically set due to aliasing
end

# struct HighamExpmvConstantCache{aType} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
#     anorm::aType
# end
# OrdinaryDiffEq.alg_cache(alg::HighamExpmv,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{false}}) = HighamExpmvConstantCache()
#
# function OrdinaryDiffEq.initialize!(integrator,cache::HighamExpmvConstantCache)
#     integrator.kshortsize = 2
#     integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
#     integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
#
#     # Avoid undefined entries if k is an array of arrays
#     integrator.fsallast = zero(integrator.fsalfirst)
#     integrator.k[1] = integrator.fsalfirst
#     integrator.k[2] = integrator.fsallast
# end
#
# function OrdinaryDiffEq.perform_step!(integrator,cache::HighamExpmvConstantCache,repeat_step=false)
#     @unpack t,dt,uprev,u,f,p = integrator
#     @muladd u = uprev + dt*integrator.fsalfirst
#     k = f(u, p, t+dt) # For the interpolation, needs k at the updated point
#     integrator.fsallast = k
#     integrator.k[1] = integrator.fsalfirst
#     integrator.k[2] = integrator.fsallast
#     integrator.u = u
# end
