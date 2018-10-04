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
using DiffEqBase, OrdinaryDiffEq, DiffEqOperators, Sundials
# using ApproxFun # really kills compile time
using Tensors
using LsqFit
using BlackBoxOptim
using MATLAB
using TimerOutputs
using Parameters: @with_kw

# export SignalIntegrator
# export gettime, getsignal, numsignals, signalnorm, complexsignal, relativesignalnorm, relativesignal

export AbstractMWIFittingModel, NNLSRegression, TwoPoolMagnToMagn, ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx
# export AbstractTestProblem, SingleAxonTestProblem
# export testproblem

# export diffeq_solver, expokit_solver, expmv_solver
export getmwf, fitmwfmodel, mwimodel, initialparams

export ExpokitExpmv

# ---------------------------------------------------------------------------- #
# expmv and related functions
# ---------------------------------------------------------------------------- #

struct ExpokitExpmv <: OrdinaryDiffEqAlgorithm end

function DiffEqBase.ODEProblem(
        m::MyelinDomain,
        u0,
        tspan
    )
    A = ParabolicLinearMap(getdomain(m))
    p = (A,)
    f!(du,u,p,t) = LinearAlgebra.mul!(du, p[1], u)
    return ODEProblem(f!, u0, tspan, p)
end

function DiffEqBase.solve(
        prob::ODEProblem{uType,tType,isinplace},
        alg::ExpokitExpmv;
        reltol = 1e-8,
        saveat = eltype(tType)[],
        tstops = eltype(tType)[],
        dt = nothing,
        m::Int = 30,
        norm = LinearAlgebra.norm,
        opnorm = LinearAlgebra.opnorm,
        anorm = opnorm(prob.p[1], Inf), # p[1] is the matrix `A`
        affect! = identity,
        verbose = true
    ) where {uType,tType,isinplace}

    # Timer
    to = TimerOutput()

    # Problem parameters
    A, tspan = prob.p[1], prob.tspan

    if !(dt == nothing)
        nsteps = round(Int, (tspan[2]-tspan[1])/dt)
        @assert tspan[2]-tspan[1] ≈ dt * nsteps

        # `saveat` points
        for t in range(tspan[1], stop=tspan[2], length=nsteps+1)
            push!(saveat, t)
        end

        # `tstops` points
        for t in (tspan[1] .+ dt/2 .+ dt .* (0:nsteps-1))
            push!(tstops, t)
        end
    end

    # Ensure that tspan[2] is included in saveat, and tspan[1] is not
    saveat = sort(collect(saveat))
    !(saveat[1] == tspan[1]) && pushfirst!(saveat, tspan[1])
    !(saveat[end] == tspan[2]) && push!(saveat, tspan[2])

    # Form tcheckpoints
    tstops = sort(collect(tstops))
    tcheckpoints = sort(vcat(saveat, tstops)) # all checkpoints

    # Check that all checkpoints are strictly within tspan
    # @assert (tcheckpoints[1] > tspan[1]) && (tcheckpoints[end] < tspan[2])
    # push!(tcheckpoints, tspan[2]) # now, safely add last checkpoint to mean simulation is finished

    t = [tspan[1]]
    u = uType[copy(prob.u0)] # for _ in 1:length(t)]

    (tcheckpoints[1] == tspan[1]) && popfirst!(tcheckpoints) # already saved
    @timeit to "Total Time" begin
        # for i = 1:length(t)-1
        while !isempty(tcheckpoints)
            tc = popfirst!(tcheckpoints)
            tstep = tc - t[end]

            # Check for tstop
            if tc ∈ tstops
                push!(t, tc)
                push!(u, affect!(copy(u[end])))
            end

            push!(t, tc)
            push!(u, similar(prob.u0))

            # @timeit to "Step $i/$(length(t)-1)" begin
            Expokit.expmv!(u[end], tstep, A, u[end-1];
                tol = reltol,
                m = m,
                norm = norm,
                anorm = anorm
            )
            # end
        end
    end
    verbose && print_timer(to)

    # Keep only `saveat` points
    b = t .∈ [saveat]
    t, u = t[b], u[b]

    # Build and return solution
    sol = DiffEqBase.build_solution(prob, alg, t, u)

    return sol
end

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
#
# function expokit_solver(domain; tol = 1e-8, m = 30, opnorm = normest1_norm)
#     function solver!(U, A, tspan, U0)
#         anorm = opnorm(A, Inf)
#         Expokit.expmv!(U, tspan[end], A, U0;
#             anorm = anorm,
#             tol = tol,
#             m = m)
#         return U
#     end
#     return solver!
# end
#
# function expmv_solver(domain; prec = "single", opnorm = normest1_norm)
#     function solver!(U, A, tspan, U0)
#         M = Expmv.select_taylor_degree(A, U0; opnorm = opnorm)[1]
#         Expmv.expmv!(U, tspan[end], A, U0;
#             prec = prec,
#             M = M,
#             opnorm = opnorm)
#         return U
#     end
#     return solver!
# end
# Expmv.normAm(A::LinearMap, p::Real, t::Int = 10) = (normest1_norm(A^p, 1, t), 0) # no mv-product estimate
#
# function DiffEqBase.solve(prob, domains, tspan = (0.0,1e-3);
#                           abstol = 1e-8,
#                           reltol = 1e-8,
#                           linear_solver = :GMRES)
#     to = TimerOutput()
#     N = numsubdomains(domains)
#     u0 = Vec{2}((0.0, 1.0))
#     sols = Vector{ODESolution}(undef, N)
#     signals = Vector{SignalIntegrator}(undef, N)
#
#     # @timeit to "Assembly" doassemble!(prob, domains)
#     # @timeit to "Factorization" factorize!(domains)
#     @timeit to "Interpolation" U0 = BlochTorreyUtils.interpolate(u0, domains) # π/2-pulse at each node
#
#     @timeit to "Solving on subdomains" begin
#         Threads.@threads for i = 1:N
#             subdomain = getsubdomain(domains, i)
#             # print("Subdomain $i/$(numsubdomains(domains)): ")
#             @time @timeit to "Subdomain $i/$(numsubdomains(domains))" begin
#                 A = ParabolicLinearMap(subdomain)
#                 solver! = diffeq_solver(subdomain; abstol = abstol, reltol = reltol, linear_solver = linear_solver)
#                 sols[i], signals[i] = solver!(nothing, A, tspan, U0[i])
#             end
#             flush(stdout)
#         end
#     end
#
#     print_timer(to)
#     return sols, signals
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
        nT2 = 120, # Number of T2 bins used during fitting process, spaced logarithmically in `T2Range`
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
        # logspace(start,stop,length) = exp10.(range(log10(start), stop=log10(stop), length=length))
        mxcall(:figure, 0)
        mxcall(:hold, 0, "on")
        mxcall(:plot, 0, MWIdist[:])
        # mxcall(:xlabel, 0, "T2 [s]")
        # mxcall(:title, 0, "T2 Distribution")
    end

    MWImaps, MWIdist, MWIpart
end

getmwf(modeltype::NNLSRegression, MWImaps, MWIdist, MWIpart) = MWIpart["sfr"]

# ----------------------- #
# ThreePoolCplxToCplx model
# ----------------------- #
function initialparams(modeltype::ThreePoolCplxToCplx, ts::AbstractVector{T}, S::Vector{Vec{2,T}}) where {T}
    S1, S2 = complex(S[2]), complex(S[end]) # initial/final complex signals (S[1] is t=0 point)
    ΔT = ts[end] - ts[2] # time difference between S1 and S2
    A1, ϕ1, ϕ2 = abs(S1), angle(S1), angle(S2)
    Δϕ = ϕ2 - ϕ1
    Δf = -Δϕ/(2π*ΔT) # negative phase convention

    A_my, A_ax, A_ex = A1/10, 6*A1/10, 3*A1/10 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = T(10e-3), T(64e-3), T(48e-3) # T2* initial guesses
    Δf_bg_my, Δf_bg_ax, Δf_bg_ex = Δf, Δf, Δf # zero(T), zero(T), zero(T) # In continuous setting, initialize to zero #TODO (?)
    𝛷₀ = -ϕ1 # zero(T) # Initial phase (negative phase convention)

    p  = T[A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex, Δf_bg_my,  Δf_bg_ax,  Δf_bg_ex,  𝛷₀]
    lb = T[0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3, Δf - 75.0, Δf - 25.0, Δf - 25.0, -π]
    ub = T[2*A1, 2*A1, 2*A1, 25e-3, 150e-3, 150e-3, Δf + 75.0, Δf + 25.0, Δf + 25.0,  π]

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
    A1 = norm(S[2]) # initial magnitude (S[1] is t=0 point)

    A_my, A_ax, A_ex = A1/10, 6*A1/10, 3*A1/10 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = T(10e-3), T(64e-3), T(48e-3) # T2* initial guesses
    Δf_my_ex, Δf_ax_ex = T(5), zero(T) # zero(T), zero(T) # In continuous setting, initialize to zero #TODO (?)

    p  = T[A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex, Δf_my_ex,  Δf_ax_ex]
    lb = T[0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3,    -75.0,     -25.0]
    ub = T[2*A1, 2*A1, 2*A1, 25e-3, 150e-3, 150e-3,    +75.0,     +25.0]

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
    A1 = norm(S[2]) # initial magnitude (S[1] is t=0 point)

    A_my, A_ax, A_ex = A1/10, 6*A1/10, 3*A1/10 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = T(10e-3), T(64e-3), T(48e-3) # T2* initial guesses

    p  = T[A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex]
    lb = T[0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3]
    ub = T[2*A1, 2*A1, 2*A1, 25e-3, 150e-3, 150e-3]

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
    A1 = norm(S[2]) # initial magnitude (S[1] is t=0 point)

    A_my, A_ex = A1/3, 2*A1/3 # Relative magnitude initial guesses
    T2_my, T2_ex = T(10e-3), T(48e-3) # T2* initial guesses

    p  = T[A_my, A_ex, T2_my,  T2_ex]
    lb = T[0.0,  0.0,   3e-3,  25e-3]
    ub = T[2*A1, 2*A1, 25e-3, 150e-3]

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
