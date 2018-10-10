complex# ---------------------------------------------------------------------------- #
# BlochTorreySolvers
# ---------------------------------------------------------------------------- #

module BlochTorreySolvers

using GeometryUtils
using BlochTorreyUtils
using MeshUtils
using Expmv
using Expokit

using LinearAlgebra
using JuAFEM
using LinearMaps, ForwardDiff, Interpolations
# using DifferentialEquations # really kills compile time
using DiffEqBase, OrdinaryDiffEq, DiffEqCallbacks # DiffEqOperators, Sundials
# using ApproxFun # really kills compile time
using Tensors
using LsqFit
using BlackBoxOptim
using MATLAB
using TimerOutputs
using Parameters: @with_kw, @pack, @unpack

# export SignalIntegrator
# export gettime, getsignal, numsignals, signalnorm, complexsignal, relativesignalnorm, relativesignal

export AbstractMWIFittingModel, NNLSRegression, TwoPoolMagnToMagn, ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx
# export AbstractTestProblem, SingleAxonTestProblem
# export testproblem

# export diffeq_solver, expokit_solver, expmv_solver
export getmwf, fitmwfmodel, mwimodel, initialparams

export MultiSpinEchoCallback
export ExpokitExpmv, HighamExpmv

# ---------------------------------------------------------------------------- #
# ODEProblem constructor for ParabolicDomain's and MyelinDomain's
# ---------------------------------------------------------------------------- #

function OrdinaryDiffEq.ODEProblem(d::ParabolicDomain, u0, tspan)
    A = ParabolicLinearMap(d)
    p = (A,)
    f!(du,u,p,t) = LinearAlgebra.mul!(du, p[1], u)
    return ODEProblem(f!, u0, tspan, p)
end
OrdinaryDiffEq.ODEProblem(m::MyelinDomain, u0, tspan) = ODEProblem(getdomain(m), u0, tspan)

# ---------------------------------------------------------------------------- #
# SpinEchoCallback
# ---------------------------------------------------------------------------- #

function MultiSpinEchoCallback(
        tspan::NTuple{2,T};
        TE::T = (tspan[2]-tspan[1]), # default to single echo
        TE0::T = tspan[1] + TE/2, # default to first echo at TE/2
        kwargs...
    ) where {T}

    # Apply π-pulse/return next π-pulse time
    time_choice(integrator) = (t = integrator.t; return ifelse(t == tspan[1], TE0, t + TE))
    user_affect!(integrator) = @views(integrator.u[2:2:end] .= -integrator.u[2:2:end])

    # Return IterativeCallback
    return IterativeCallback(time_choice, user_affect!, T; initial_affect = false, kwargs...)
end

# ---------------------------------------------------------------------------- #
# AbstractExpmv utilities
# ---------------------------------------------------------------------------- #

abstract type AbstractExpmvAlgorithm <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end

OrdinaryDiffEq.isfsal(alg::AbstractExpmvAlgorithm) = true # NOTE: this is default; "first same as last" property
OrdinaryDiffEq.alg_order(alg::AbstractExpmvAlgorithm) = 2 # TODO: order of expmv? used for interpolation; likely unimportant?
OrdinaryDiffEq.alg_cache(alg::AbstractExpmvAlgorithm,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{true}})  = ExpmvCache(u,uprev,similar(u),zero(rate_prototype),zero(rate_prototype))
OrdinaryDiffEq.alg_cache(alg::AbstractExpmvAlgorithm,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{false}}) = ExpmvConstantCache()

struct ExpmvCache{uType,rateType} <: OrdinaryDiffEq.OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
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
ExpokitExpmv(A; kwargs...) = ExpokitExpmv(;kwargs..., anorm = LinearAlgebra.norm(A, Inf))

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
# ExpokitExpmv stepper
# ---------------------------------------------------------------------------- #

@with_kw struct HighamExpmv{T,F1,F2} <: AbstractExpmvAlgorithm
    M::Matrix{T} = Matrix{T}(undef, 0, 0)
    prec::String = "double"
    shift::Bool = true
    full_term::Bool = false
    prnt::Bool = false
    m_max::Int = 55
    p_max::Int = 8
    force_estm::Bool = false
    norm::F1 = LinearAlgebra.norm
    opnorm::F2 = LinearAlgebra.opnorm
end
HighamExpmv(A, b; kwargs...) = HighamExpmv(;kwargs..., M = _select_taylor_degree(A, b; kwargs...))
function _select_taylor_degree(A, b; m_max = 55, p_max = 8, prec = "double", shift = false, force_estm = false, opnorm = LinearAlgebra.opnorm, kwargs...)
    M = select_taylor_degree(A, b; m_max = m_max, p_max = p_max, prec = prec, shift = shift, force_estm = force_estm, opnorm = opnorm)[1]
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

    Expmv.expmv!(u, dt, A, uprev, cache.tmp;
        M = alg.M,
        prec = alg.prec,
        shift = alg.shift,
        full_term = alg.full_term,
        prnt = alg.prnt,
        norm = alg.norm,
        opnorm = alg.opnorm
    )
    integrator.fsallast = u
end
# Expmv.normAm(A::LinearMap, p::Real, t::Int = 10) = (normest1_norm(A^p, 1, t), 0) # no mv-product estimate

# function diffeq_solver(domain;
#                        alg = CVODE_BDF(linear_solver = :GMRES),
#                        abstol = 1e-8,
#                        reltol = 1e-8,
#                        solverargs...)
#     #solve(odeprob, ETDRK4(krylov=true); dt = 1e-3);
#
#     function solver!(U, A, tspan, U0)
#         signal, callbackfun = IntegrationCallback(U0, tspan[1], domain)
#
#         A_wrap = DiffEqParabolicLinearMapWrapper(A);
#         A_op = DiffEqArrayOperator(A_wrap);
#         f! = ODEFunction{true,true}(A_op; jac_prototype = A_op);
#         odeprob = ODEProblem(f!, U0, tspan);
#
#         # odeprob = ODEProblem((du,u,p,t)->mul!(du,p[1],u), U0, tspan, (A,));
#         sol = solve(odeprob, alg;
#                     saveat = tspan,
#                     abstol = abstol,
#                     reltol = reltol,
#                     alg_hints = :stiff,
#                     callback = callbackfun,
#                     solverargs...)
#
#         !(U == nothing) && copyto!(U, sol.u[end])
#
#         return sol, signal
#     end
#     return solver!
# end

# ---------------------------------------------------------------------------- #
# Setup up problem and domains for a single axon for testing
# ---------------------------------------------------------------------------- #
abstract type AbstractTestProblem end
struct SingleAxonTestProblem <: AbstractTestProblem end

# function testproblem(::SingleAxonTestProblem, btparams = BlochTorreyParameters{Float64}())
#     rs = [btparams.R_mu] # one radius of average size
#     os = zeros(Vec{2}, 1) # one origin at the origin
#     outer_circles = GeometryUtils.Circle.(os, rs)
#     inner_circles = scale_shape.(outer_circles, btparams.g_ratio)
#     bcircle = scale_shape(outer_circles[1], 1.5)
#
#     h0 = 0.2 * btparams.R_mu * (1.0 - btparams.g_ratio) # fraction of size of average torus width
#     eta = 5.0 # approx ratio between largest/smallest edges
#
#     mxcall(:figure,0); mxcall(:hold,0,"on")
#     @time grid = circle_mesh_with_tori(bcircle, inner_circles, outer_circles, h0, eta)
#     @time exteriorgrid, torigrids, interiorgrids = form_tori_subgrids(grid, bcircle, inner_circles, outer_circles)
#
#     all_tori = form_subgrid(grid, getcellset(grid, "tori"), getnodeset(grid, "tori"), getfaceset(grid, "boundary"))
#     all_int = form_subgrid(grid, getcellset(grid, "interior"), getnodeset(grid, "interior"), getfaceset(grid, "boundary"))
#     mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(exteriorgrid); sleep(0.5)
#     mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(all_tori); sleep(0.5)
#     mxcall(:figure,0); mxcall(:hold,0,"on"); mxplot(all_int)
#
#     prob = MyelinProblem(btparams)
#     domains = MyelinDomain(grid, outer_circles, inner_circles, bcircle,
#         exteriorgrid, torigrids, interiorgrids;
#         quadorder = 3, funcinterporder = 1)
#
#     doassemble!(prob, domains)
#     factorize!(domains)
#
#     return prob, domains
# end

# ---------------------------------------------------------------------------- #
# DiscreteCallback for integrating magnetization at each step
# ---------------------------------------------------------------------------- #
# struct SignalIntegrator{Tt,Tu,uDim,gDim,T,Nd,Nf} #TODO
#     time::Vector{Tt}
#     signal::Vector{Vec{uDim,Tu}}
#     domain::ParabolicDomain{gDim,Nd,T,Nf}
# end
# function (p::SignalIntegrator)(int)
#     push!(p.signal, BlochTorreyUtils.integrate(int.u, p.domain))
#     push!(p.time, int.t)
#     u_modified!(int, false)
# end
# function IntegrationCallback(u0, t0, domain)
#     intial_signal = BlochTorreyUtils.integrate(u0, domain)
#     signalintegrator! = SignalIntegrator([t0], [intial_signal], domain)
#     discretecallback = DiscreteCallback((u,t,int) -> true, signalintegrator!, save_positions = (false, false))
#     return signalintegrator!, discretecallback
# end
#
# gettime(p::SignalIntegrator) = p.time
# getsignal(p::SignalIntegrator) = p.signal
# numsignals(p::SignalIntegrator) = length(p.signal)
# signalnorm(p::SignalIntegrator) = norm.(p.signal)
# complexsignal(p::SignalIntegrator{Tt,Tu,2}) where {Tt,Tu} = reinterpret(Complex{Tu}, p.signal)
# relativesignalnorm(p::SignalIntegrator) = signalnorm(p)./norm(getsignal(p)[1])
# relativesignal(p::SignalIntegrator) = (S = getsignal(p); return S./norm(S[1]))
# function reset!(p::SignalIntegrator)
#     !isempty(p.time) && (resize!(p.time, 1))
#     !isempty(p.signal) && (resize!(p.signal, 1))
#     return p
# end
#
# # function ApproxFun.Fun(p::SignalIntegrator)
# #     t = gettime(p) # grid of time points
# #     v = complexsignal(p) # values
# #
# #     m = 100
# #     tol = 1e-8
# #     S = ApproxFun.Chebyshev(ApproxFun.Interval(t[1], t[end]))
# #     V = Array{eltype(v)}(undef, numsignals(p), m) # Create a Vandermonde matrix by evaluating the basis at the grid
# #     for k = 1:m
# #         V[:,k] = ApproxFun.Fun(S, [zeros(k-1); 1]).(t)
# #     end
# #     f = ApproxFun.Fun(S, V\v)
# #     f = ApproxFun.chop(f, tol)
# #
# #     return f
# # end
# # ApproxFun.Fun(ps::Vector{S}) where {S<:SignalIntegrator} = sum(ApproxFun.Fun.(ps))
#
# function Interpolations.interpolate(p::SignalIntegrator)
#     t = gettime(p) # grid of time points
#     v = complexsignal(p) # values
#     f = Interpolations.interpolate((t,), v, Interpolations.Gridded(Interpolations.Linear()))
#     return f
# end
# function Interpolations.interpolate(ps::Vector{S}) where {S<:SignalIntegrator}
#     fs = Interpolations.interpolate.(ps)
#     return t -> sum(f -> f(t), fs)
# end
#
# function Base.show(io::IO, p::SignalIntegrator)
#     compact = get(io, :compact, false)
#     nsignals = length(p.signal)
#     ntimes = length(p.time)
#     plural_s = nsignals == 1 ? "" : "s"
#     print(io, "$(typeof(p))")
#     if compact || !compact
#         print(io, " with $nsignals stored signal", plural_s)
#     else
#         print(io, "\n    time: $ntimes-element ", typeof(p.time))
#         print(io, "\n  signal: $nsignals-element ", typeof(p.signal))
#         print(io, "\n  domain: "); show(IOContext(io, :compact => true), p.domain)
#     end
# end

# ---------------------------------------------------------------------------- #
# Myelin water fraction calculation
# ---------------------------------------------------------------------------- #

abstract type AbstractMWIFittingModel end
struct NNLSRegression <: AbstractMWIFittingModel end
struct TwoPoolMagnToMagn <: AbstractMWIFittingModel end
struct ThreePoolMagnToMagn <: AbstractMWIFittingModel end
struct ThreePoolCplxToMagn <: AbstractMWIFittingModel end
struct ThreePoolCplxToCplx <: AbstractMWIFittingModel end
const TwoPoolMagnData = TwoPoolMagnToMagn
const TwoPoolModel = TwoPoolMagnToMagn
const ThreePoolMagnData = Union{ThreePoolMagnToMagn, ThreePoolCplxToMagn}
const ThreePoolModel = Union{ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx}

# Convenience conversion between a Vec{2} and a complex number
@inline Base.complex(x::Vec{2}) = complex(x[1], x[2])

# True myelin water fraction
# function getmwf(m::MyelinDomain)
#     A_total = area(m)
#     A_myelin = sum(area, getmyelindomains(m))
#     return A_myelin/A_total
# end

# Abstract interface
function getmwf(
        signals::Vector{V},
        modeltype::AbstractMWIFittingModel;
        kwargs...
        ) where {V <: Vec{2}}
    return getmwf(modeltype, fitmwfmodel(signals, modeltype; kwargs...)...)
end

# MWI model data
mwimodeldata(modeltype::ThreePoolCplxToCplx, S::Vector{Vec{2,T}}) where {T} = copy(reinterpret(T, S[2:end]))
mwimodeldata(modeltype::ThreePoolMagnData, S::Vector{V}) where {V <: Vec{2}} = norm.(S[2:end])
mwimodeldata(modeltype::TwoPoolMagnData, S::Vector{V}) where {V <: Vec{2}} = norm.(S[2:end])

# NNLSRegression model
function fitmwfmodel(
        signals::Vector{V},
        modeltype::NNLSRegression;
        TE = 10e-3, # First time point
        nTE = 32, # Number of echos
        T2Range = [15e-3, 2.0], # Min and Max T2 values used during fitting (typical for in-vivo)
        nT2 = 40, # Number of T2 bins used during fitting process, spaced logarithmically in `T2Range`
        Threshold = 0.0, # First echo intensity cutoff for empty voxels
        RefConAngle = 165.0, # Refocusing Pulse Control Angle (TODO: check value from scanner)
        spwin = [1.5*TE, 40e-3], # short peak window (typically 1.5X echospacing to 40ms)
        mpwin = [40e-3, 200e-3], # middle peak window
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
        mwf = getmwf(NNLSRegression(), MWImaps, MWIdist, MWIpart)

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

getmwf(modeltype::NNLSRegression, MWImaps, MWIdist, MWIpart) = MWIpart["sfr"]

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
    𝛷₀ = ϕ1 # Initial phase (negative phase convention: -(-ϕ1) = ϕ1 from phase flip between t0 and t1)

    p  = T[A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex, Δf_bg_my,  Δf_bg_ax,  Δf_bg_ex,  𝛷₀    ]
    lb = T[0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3, Δf - 75.0, Δf - 25.0, Δf - 25.0, 𝛷₀ - π]
    ub = T[2*A0, 2*A0, 2*A0, 25e-3, 150e-3, 150e-3, Δf + 75.0, Δf + 25.0, Δf + 25.0, 𝛷₀ + π]

    p[4:6] = inv.(p[4:6]) # fit for R2 instead of T2
    lb[4:6], ub[4:6] = inv.(ub[4:6]), inv.(lb[4:6]) # swap bounds

    return p, lb, ub
end

function mwimodel(modeltype::ThreePoolCplxToCplx, t::AbstractVector, p::Vector)
    # A_my, A_ax, A_ex, T2_my, T2_ax, T2_ex, Δf_bg_my, Δf_bg_ax, Δf_bg_ex, 𝛷₀ = p
    # Γ_my, Γ_ax, Γ_ex = complex(1/T2_my, 2*pi*Δf_bg_my), complex(1/T2_ax, 2*pi*Δf_bg_ax), complex(1/T2_ex, 2*pi*Δf_bg_ex)

    A_my, A_ax, A_ex, R2_my, R2_ax, R2_ex, Δf_bg_my, Δf_bg_ax, Δf_bg_ex, 𝛷₀ = p
    Γ_my, Γ_ax, Γ_ex = complex(R2_my, 2π*Δf_bg_my), complex(R2_ax, 2π*Δf_bg_ax), complex(R2_ex, 2π*Δf_bg_ex)

    S = @. (A_my * exp(-Γ_my * t) + A_ax * exp(-Γ_ax * t) + A_ex * exp(-Γ_ex * t)) * cis(-𝛷₀)
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
function fitmwfmodel(
        signals::Vector{V}, # signal vectors
        modeltype::AbstractMWIFittingModel = ThreePoolCplxToCplx();
        TE = 10e-3, # echo spacing
        fitmethod = :local
    ) where {V <: Vec{2}}

    # # Scplx = sum(ApproxFun.Fun.(signals))
    # # d = ApproxFun.domain(ApproxFun.space(Scplx))
    # # tspan = (first(d), last(d))
    #
    # Scplx = Interpolations.interpolate(signals)
    # tspan = (gettime(signals[1])[1], gettime(signals[1])[end])
    # tdata = range(tspan[1], stop = tspan[2], length = npts)

    nTE = length(signals)-1 # S[1] is t=0 point
    ts = TE.*(0:nTE) |> collect
    ydata = mwimodeldata(modeltype, signals) # model data
    tdata = ts[2:end] # ydata time points (first point is dropped)
    p0, lb, ub = initialparams(modeltype, ts, signals)

    model(t, p) = mwimodel(modeltype, t, p)

    if fitmethod == :global
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
    else
        # default to local optimization
        wrapped_model(p) = model(tdata, p)
        cfg = ForwardDiff.JacobianConfig(wrapped_model, p0, ForwardDiff.Chunk{length(p0)}())
        jac_model(t, p) = ForwardDiff.jacobian(wrapped_model, p, cfg)

        modelfit = curve_fit(model, jac_model, tdata, ydata, p0; lower = lb, upper = ub)
        errors = try
            margin_error(modelfit, 0.05) # 95% confidence errors
        catch e
            nothing
        end
    end

    return modelfit, errors
end

function getmwf(modeltype::ThreePoolModel, modelfit, errors)
    A_my, A_ax, A_ex = modelfit.param[1:3]
    return A_my/(A_my + A_ax + A_ex)
end

function getmwf(modeltype::TwoPoolModel, modelfit, errors)
    A_my, A_ex = modelfit.param[1:2]
    return A_my/(A_my + A_ex)
end

end # module BlochTorreySolvers

nothing
