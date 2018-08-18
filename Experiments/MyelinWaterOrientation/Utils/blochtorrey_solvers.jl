# ---------------------------------------------------------------------------- #
# BlochTorreySolvers
# ---------------------------------------------------------------------------- #

module BlochTorreySolvers

using ..GeometryUtils
using ..BlochTorreyUtils
using ..Expmv
using LinearAlgebra
using LinearMaps, DifferentialEquations, ForwardDiff, Interpolations
using Tensors
using LsqFit
using MATLAB
using TimerOutputs
# using ApproxFun # really kills compile time

export SignalIntegrator, AbstractMWIFittingModel, NNLSRegression, ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx

export diffeq_solver, expokit_solver, expmv_solver
export getmwf, fitmwfmodel, mwimodel, initialparams
export gettime, getsignal, numsignals, signalnorm, complexsignal, relativesignalnorm, relativesignal

# ---------------------------------------------------------------------------- #
# expmv and related functions
# ---------------------------------------------------------------------------- #

function diffeq_solver(domain; abstol = 1e-8, reltol = 1e-8, linear_solver = :GMRES)
    function solver!(U, A, tspan, U0)
        signal, callbackfun = IntegrationCallback(U0, tspan[1], domain)
        prob = ODEProblem((du,u,p,t)->mul!(du,p[1],u), U0, tspan, (A,));
        sol = solve(prob, CVODE_BDF(linear_solver = linear_solver);
                    abstol = abstol,
                    reltol = reltol,
                    saveat = tspan,
                    alg_hints = :stiff,
                    callback = callbackfun)
        !(U == nothing) && copyto!(U, sol.u[end])
        return sol, signal
    end
    return solver!
end

function expokit_solver(domain; tol = 1e-8, m = 30, opnorm = normest1_norm)
    function solver!(U, A, tspan, U0)
        anorm = opnorm(A, Inf)
        Expokit.expmv!(U, tspan[end], A, U0;
            anorm = anorm,
            tol = tol,
            m = m)
        return U
    end
    return solver!
end

function expmv_solver(domain; prec = "single", opnorm = normest1_norm)
    function solver!(U, A, tspan, U0)
        M = Expmv.select_taylor_degree(A, U0; opnorm = opnorm)[1]
        Expmv.expmv!(U, tspan[end], A, U0;
            prec = prec,
            M = M,
            opnorm = opnorm)
        return U
    end
    return solver!
end
Expmv.normAm(A::LinearMap, p::Real, t::Int = 10) = (normest1_norm(A^p, 1, t), 0) # no mv-product estimate

function DiffEqBase.solve(prob, domains, tspan = (0.0,1e-3);
                          abstol = 1e-8,
                          reltol = 1e-8,
                          linear_solver = :GMRES)
    to = TimerOutput()
    u0 = Vec{2}((0.0, 1.0))
    sols = ODESolution[]
    signals = SignalIntegrator[]

    @timeit to "Assembly" doassemble!(prob, domains)
    @timeit to "Factorization" factorize!(domains)
    @timeit to "Interpolation" U0 = BlochTorreyUtils.interpolate(u0, domains) # π/2-pulse at each node

    @timeit to "Solving on subdomains" for (i, subdomain) in enumerate(getsubdomains(domains))
        print("Subdomain $i/$(numsubdomains(domains)): ")
        @time @timeit to "Subdomain $i/$(numsubdomains(domains))" begin
            A = ParabolicLinearMap(subdomain)
            solver! = diffeq_solver(subdomain; abstol = abstol, reltol = reltol, linear_solver = linear_solver)
            sol, signal = solver!(nothing, A, tspan, U0[i])
        end
        push!(sols, sol)
        push!(signals, signal)
        flush(stdout)
    end

    print_timer(to)
    return sols, signals
end

# ---------------------------------------------------------------------------- #
# DiscreteCallback for integrating magnetization at each step
# ---------------------------------------------------------------------------- #
struct SignalIntegrator{Tt,Tu,uDim,gDim,T,Nd,Nf} #TODO
    time::Vector{Tt}
    signal::Vector{Vec{uDim,Tu}}
    domain::ParabolicDomain{gDim,Nd,T,Nf}
end
function (p::SignalIntegrator)(int)
    push!(p.signal, BlochTorreyUtils.integrate(int.u, p.domain))
    push!(p.time, int.t)
    u_modified!(int, false)
end
function IntegrationCallback(u0, t0, domain)
    intial_signal = BlochTorreyUtils.integrate(u0, domain)
    signalintegrator! = SignalIntegrator([t0], [intial_signal], domain)
    discretecallback = DiscreteCallback((u,t,int) -> true, signalintegrator!, save_positions = (false, false))
    return signalintegrator!, discretecallback
end

gettime(p::SignalIntegrator) = p.time
getsignal(p::SignalIntegrator) = p.signal
numsignals(p::SignalIntegrator) = length(p.signal)
signalnorm(p::SignalIntegrator) = norm.(p.signal)
complexsignal(p::SignalIntegrator{Tt,Tu,2}) where {Tt,Tu} = reinterpret(Complex{Tu}, p.signal)
relativesignalnorm(p::SignalIntegrator) = signalnorm(p)./norm(getsignal(p)[1])
relativesignal(p::SignalIntegrator) = (S = getsignal(p); return S./norm(S[1]))

# function ApproxFun.Fun(p::SignalIntegrator)
#     t = gettime(p) # grid of time points
#     v = complexsignal(p) # values
#
#     m = 100
#     tol = 1e-8
#     S = ApproxFun.Chebyshev(ApproxFun.Interval(t[1], t[end]))
#     V = Array{eltype(v)}(undef, numsignals(p), m) # Create a Vandermonde matrix by evaluating the basis at the grid
#     for k = 1:m
#         V[:,k] = ApproxFun.Fun(S, [zeros(k-1); 1]).(t)
#     end
#     f = ApproxFun.Fun(S, V\v)
#     f = ApproxFun.chop(f, tol)
#
#     return f
# end
# ApproxFun.Fun(ps::Vector{S}) where {S<:SignalIntegrator} = sum(ApproxFun.Fun.(ps))

function Interpolations.interpolate(p::SignalIntegrator)
    t = gettime(p) # grid of time points
    v = complexsignal(p) # values
    f = Interpolations.interpolate((t,), v, Interpolations.Gridded(Interpolations.Linear()))
    return f
end
function Interpolations.interpolate(ps::Vector{S}) where {S<:SignalIntegrator}
    fs = Interpolations.interpolate.(ps)
    return t -> sum(f -> f(t), fs)
end

function Base.show(io::IO, p::SignalIntegrator)
    compact = get(io, :compact, false)
    nsignals = length(p.signal)
    ntimes = length(p.time)
    plural_s = nsignals == 1 ? "" : "s"
    print(io, "$(typeof(p))")
    if compact || !compact
        print(io, " with $nsignals stored signal", plural_s)
    else
        print(io, "\n    time: $ntimes-element ", typeof(p.time))
        print(io, "\n  signal: $nsignals-element ", typeof(p.signal))
        print(io, "\n  domain: "); show(IOContext(io, :compact => true), p.domain)
    end
end

# ---------------------------------------------------------------------------- #
# Myelin water fraction calculation
# ---------------------------------------------------------------------------- #

abstract type AbstractMWIFittingModel end
struct NNLSRegression <: AbstractMWIFittingModel end
struct ThreePoolMagnToMagn <: AbstractMWIFittingModel end
struct ThreePoolCplxToMagn <: AbstractMWIFittingModel end
struct ThreePoolCplxToCplx <: AbstractMWIFittingModel end
const ThreePoolMagnData = Union{ThreePoolMagnToMagn, ThreePoolCplxToMagn}
const ThreePoolModel = Union{ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx}

# True myelin water fraction
function getmwf(m::MyelinDomain)
    A_total = area(m)
    A_myelin = sum(area, getmyelindomains(m))
    return A_myelin/A_total
end

# Abstract interface
function getmwf(
        signals::Vector{S},
        modeltype::AbstractMWIFittingModel;
        kwargs...
        ) where {S <: SignalIntegrator}
    return getmwf(modeltype, fitmwfmodel(signals, modeltype; kwargs...)...)
end

# MWI model data
function mwimodeldata(modeltype::ThreePoolCplxToCplx, t, Scplx::Function)
    ydata = Scplx.(t)
    ydata = reinterpret(eltype(real(ydata[1])), ydata) # reinterpret as real array
    return ydata
end
mwimodeldata(modeltype::ThreePoolMagnData, t, Scplx::Function) = abs.(Scplx.(t))

# NNLSRegression model
function fitmwfmodel(
        signals::Vector{S},
        modeltype::NNLSRegression;
        TE0 = 1e-3, # First time point
        nTE = 32, # Number of echos
        T2Range = [15e-3,2.000], # Min and Max T2 values
        nT2 = 120, # Number of T2 times to use in fitting process
        Threshold = 0.0, # First echo intensity cutoff for empty voxels
        spwin = [14e-3, 40e-3], # small pool window
        plotdist = true # plot resulting T2-distribution
        ) where {S <: SignalIntegrator}
    # t = linspace(2.1e-3,61.93e-3,32);

    # # ApproxFun representation the magnitude signal
    # fmag = abs(sum(ApproxFun.Fun.(signals)))
    # d = ApproxFun.domain(ApproxFun.space(fmag))

    fcplx = Interpolations.interpolate(signals)
    fmag = t -> abs(fcplx(t))

    tspan = (TE0, gettime(signals[1])[end])
    t = range(tspan[1], stop = tspan[2], length = nTE)
    mag = reshape(fmag.(t), (1,1,1,nTE))
    TE = step(t)

    MWImaps, MWIdist = mxcall(:T2map_SEcorr, 2, mag,
        "TE", TE, "T2Range", T2Range, "Threshold", Threshold,
        "nT2", nT2, "Waitbar", "no", "Save_regparam", "yes")
    MWIpart = mxcall(:T2part_SEcorr, 1, MWIdist, "spwin", spwin)

    if plotdist
        mxcall(:figure, 0)
        mxcall(:hold, 0, "on")
        mxcall(:plot, 0, MWIdist[:])
        mxcall(:title, 0, "T2* Distribution")
    end

    MWImaps, MWIdist, MWIpart
end

getmwf(modeltype::NNLSRegression, MWImaps, MWIdist, MWIpart) = MWIpart["sfr"]

# ThreePoolCplxToCplx model
function initialparams(modeltype::ThreePoolCplxToCplx, tspan, Scplx::Function)
    S1, S2 = Scplx(tspan[1]), Scplx(tspan[2]) # initial/final complex signals
    A1, ϕ1 = abs(S1), angle(S1)
    ΔTE = tspan[2] - tspan[1]
    Δϕ = angle(S2) - angle(S1)
    Δf = -Δϕ/(2π*ΔTE) # negative phase convention

    A_my, A_ax, A_ex = 0.1*A1, 0.6*A1, 0.3*A1 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = 10e-3, 64e-3, 48e-3 # T2* initial guesses
    Δf_bg_my, Δf_bg_ax, Δf_bg_ex = Δf, Δf, Δf # In continuous setting, initialize to zero #TODO (?)
    𝛷₀ = -ϕ1 # Initial phase (negative phase convention)

    p  = [A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex, Δf_bg_my,  Δf_bg_ax,  Δf_bg_ex,  𝛷₀]
    lb = [0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3, Δf - 75.0, Δf - 25.0, Δf - 25.0, -π]
    ub = [2*A1, 2*A1, 2*A1, 25e-3, 150e-3, 150e-3, Δf + 75.0, Δf + 25.0, Δf + 25.0,  π]

    p[4:6] = inv.(p[4:6]) # fit for R2 instead of T2
    lb[4:6], ub[4:6] = inv.(ub[4:6]), inv.(lb[4:6]) # swap bounds

    return p, lb, ub
end

function mwimodel(modeltype::ThreePoolCplxToCplx, t, p)
    # A_my, A_ax, A_ex, T2_my, T2_ax, T2_ex, Δf_bg_my, Δf_bg_ax, Δf_bg_ex, 𝛷₀ = p
    # Γ_my, Γ_ax, Γ_ex = Complex(1/T2_my, 2*pi*Δf_bg_my), Complex(1/T2_ax, 2*pi*Δf_bg_ax), Complex(1/T2_ex, 2*pi*Δf_bg_ex)

    A_my, A_ax, A_ex, R2_my, R2_ax, R2_ex, Δf_bg_my, Δf_bg_ax, Δf_bg_ex, 𝛷₀ = p
    Γ_my, Γ_ax, Γ_ex = Complex(R2_my, 2*pi*Δf_bg_my), Complex(R2_ax, 2*pi*Δf_bg_ax), Complex(R2_ex, 2*pi*Δf_bg_ex)

    S = @. (A_my * exp(-Γ_my * t) + A_ax * exp(-Γ_ax * t) + A_ex * exp(-Γ_ex * t)) * cis(-𝛷₀)
    S = reinterpret(eltype(real(S[1])), S) # reinterpret as real array
    return S
end

# ThreePoolCplxToMagn model
function initialparams(modeltype::ThreePoolCplxToMagn, tspan, Scplx::Function)
    A1 = abs(Scplx(tspan[1])) # initial magnitude

    A_my, A_ax, A_ex = 0.1*A1, 0.6*A1, 0.3*A1 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = 10e-3, 64e-3, 48e-3 # T2* initial guesses
    Δf_my_ex, Δf_ax_ex = 5.0, 0.0 # In continuous setting, initialize to zero #TODO (?)

    p  = [A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex, Δf_my_ex,  Δf_ax_ex]
    lb = [0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3,    -75.0,     -25.0]
    ub = [2*A1, 2*A1, 2*A1, 25e-3, 150e-3, 150e-3,    +75.0,     +25.0]

    p[4:6] = inv.(p[4:6]) # fit for R2 instead of T2
    lb[4:6], ub[4:6] = inv.(ub[4:6]), inv.(lb[4:6]) # swap bounds

    return p, lb, ub
end

function mwimodel(modeltype::ThreePoolCplxToMagn, t, p)
    # A_my, A_ax, A_ex, T2_my, T2_ax, T2_ex, Δf_my_ex, Δf_ax_ex = p
    # Γ_my, Γ_ax, Γ_ex = Complex(1/T2_my, 2*pi*Δf_my_ex), Complex(1/T2_ax, 2*pi*Δf_ax_ex), 1/T2_ex

    A_my, A_ax, A_ex, R2_my, R2_ax, R2_ex, Δf_my_ex, Δf_ax_ex = p
    Γ_my, Γ_ax, Γ_ex = Complex(R2_my, 2*pi*Δf_my_ex), Complex(R2_ax, 2*pi*Δf_ax_ex), R2_ex

    S = @. abs(A_my * exp(-Γ_my * t) + A_ax * exp(-Γ_ax * t) + A_ex * exp(-Γ_ex * t))
    return S
end

# ThreePoolMagnToMagn model
function initialparams(modeltype::ThreePoolMagnToMagn, tspan, Scplx::Function)
    A1 = abs(Scplx(tspan[1])) # initial magnitude

    A_my, A_ax, A_ex = 0.1*A1, 0.6*A1, 0.3*A1 # Relative magnitude initial guesses
    T2_my, T2_ax, T2_ex = 10e-3, 64e-3, 48e-3 # T2* initial guesses

    p  = [A_my, A_ax, A_ex, T2_my,  T2_ax,  T2_ex]
    lb = [0.0,  0.0,  0.0,   3e-3,  25e-3,  25e-3]
    ub = [2*A1, 2*A1, 2*A1, 25e-3, 150e-3, 150e-3]

    p[4:6] = inv.(p[4:6]) # fit for R2 instead of T2
    lb[4:6], ub[4:6] = inv.(ub[4:6]), inv.(lb[4:6]) # swap bounds

    return p, lb, ub
end

function mwimodel(modeltype::ThreePoolMagnToMagn, t, p)
    # A_my, A_ax, A_ex, T2_my, T2_ax, T2_ex = p
    # Γ_my, Γ_ax, Γ_ex = 1/T2_my, 1/T2_ax, 1/T2_ex
    A_my, A_ax, A_ex, Γ_my, Γ_ax, Γ_ex = p
    S = @. A_my * exp(-Γ_my * t) + A_ax * exp(-Γ_ax * t) + A_ex * exp(-Γ_ex * t)
    return S
end

# Fitting of general AbstractMWIFittingModel
function fitmwfmodel(
        signals::Vector{S},
        modeltype::AbstractMWIFittingModel = ThreePoolCplxToCplx();
        npts = 100
        ) where {S <: SignalIntegrator}

    # Scplx = sum(ApproxFun.Fun.(signals))
    # d = ApproxFun.domain(ApproxFun.space(Scplx))
    # tspan = (first(d), last(d))

    Scplx = Interpolations.interpolate(signals)
    tspan = (gettime(signals[1])[1], gettime(signals[1])[end])
    tdata = range(tspan[1], stop = tspan[2], length = npts)
    ydata = mwimodeldata(modeltype, tdata, Scplx)
    p0, lb, ub = initialparams(modeltype, tspan, Scplx)

    model(t, p) = mwimodel(modeltype, t, p)
    wrapped_model(p) = model(tdata, p)
    cfg = ForwardDiff.JacobianConfig(wrapped_model, p0, ForwardDiff.Chunk{length(p0)}())
    jac_model(t, p) = ForwardDiff.jacobian(wrapped_model, p, cfg)

    modelfit = curve_fit(model, jac_model, tdata, ydata, p0; lower = lb, upper = ub)
    errors = try
        margin_error(modelfit, 0.05) # 95% confidence errors
    catch e
        nothing
    end

    return modelfit, errors
end

function getmwf(modeltype::ThreePoolModel, modelfit, errors)
    A_my, A_ax, A_ex = modelfit.param[1:3]
    return A_my/(A_my + A_ax + A_ex)
end

end # module BlochTorreySolvers

nothing
