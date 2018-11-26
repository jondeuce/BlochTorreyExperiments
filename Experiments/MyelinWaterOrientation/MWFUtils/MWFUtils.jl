module MWFUtils

using GeometryUtils
using CirclePackingUtils
using MeshUtils
using BlochTorreyUtils
using BlochTorreySolvers
using ExpmvHigham
import EnergyCirclePacking
import GreedyCirclePacking

using JuAFEM
using MATLAB
using OrdinaryDiffEq, DiffEqOperators, Sundials
using BenchmarkTools
using Parameters: @with_kw, @unpack
using IterableTables, DataFrames, CSV, Dates

export creategrids, createdomains
export calcomegas, calcomega
export calcsignals, calcsignal
export solveblochtorrey, default_algfun, get_algfun
export plotmagnitude, plotSEcorr, plotbiexp
export compareMWFmethods
export savefig, getnow

export MWFResults

@with_kw struct MWFResults{T}
    metadata::Dict{Symbol,Any}               = Dict()
    params::Vector{BlochTorreyParameters{T}} = []
    sols::Vector{Vector{ODESolution}}        = []
    mwfvalues::Vector{Dict{Symbol,T}}        = []
end

function creategrids(btparams::BlochTorreyParameters{T}; fname = nothing) where {T}
    Dim = 2
    Ncircles = 30

    η = btparams.AxonPDensity # goal packing density
    ϵ = 0.1 * btparams.R_mu # overlap occurs when distance between circle edges is ≤ ϵ
    α = 1e-1 # covariance penalty weight (enforces circular distribution)
    β = 1e-6 # mutual distance penalty weight
    λ = 1.0 # overlap penalty weight (or lagrange multiplier for constrained version)

    rs = rand(radiidistribution(btparams), Ncircles)
    @time initial_circles = GreedyCirclePacking.pack(rs; goaldensity = 1.0, iters = 100)
    @show minimum(radius.(initial_circles))
    @show estimate_density(initial_circles)
    @show minimum_signed_edge_distance(initial_circles)
    @show estimate_density(initial_circles)

    @time outercircles = EnergyCirclePacking.pack(initial_circles;
        autodiff = true,
        secondorder = false,
        setcallback = false,
        goaldensity = η,
        distancescale = btparams.R_mu,
        weights = [α, β, λ],
        epsilon = ϵ
    )
    innercircles = scale_shape.(outercircles, btparams.g_ratio)

    dmin = minimum_signed_edge_distance(outercircles)
    @show covariance_energy(outercircles)
    @show estimate_density(outercircles)
    @show is_any_overlapping(outercircles)
    @show (dmin, ϵ, dmin > ϵ)

    h0 = minimum(radius.(outercircles))*(1-btparams.g_ratio) # fraction of size of minimum torus width
    h_min = 1.0*h0 # minimum edge length
    h_max = 5.0*h0 # maximum edge length
    h_range = 10.0*h0 # distance over which h increases from h_min to h_max
    h_rate = 0.6 # rate of increase of h from circle boundaries (power law; smaller = faster radial increase)

    bdry, _ = opt_subdomain(outercircles)
    @time exteriorgrids, torigrids, interiorgrids, parentcircleindices = disjoint_rect_mesh_with_tori(
        bdry, innercircles, outercircles, h_min, h_max, h_range, h_rate;
        CIRCLESTALLITERS = 500, EXTERIORSTALLITERS = 1000, plotgrids = false, exterior_tiling = (1, 1)
    )

    cell_area_mismatch = sum(area.(exteriorgrids)) + sum(area.(torigrids)) + sum(area.(interiorgrids)) - area(bdry)
    @show cell_area_mismatch

    allgrids = vcat(exteriorgrids[:], torigrids[:], interiorgrids[:])
    simpplot(allgrids; newfigure = true, axis = mxaxis(bdry))

    !(fname == nothing) && savefig(fname)

    return exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry
end

function createdomains(
        btparams::BlochTorreyParameters{T},
        exteriorgrids::AbstractArray{G},
        torigrids::AbstractArray{G},
        interiorgrids::AbstractArray{G},
        outercircles::AbstractArray{C},
        innercircles::AbstractArray{C}
    ) where {T, G<:JuAFEM.Grid{2}, C<:Circle{2,T}}

    myelinprob = MyelinProblem(btparams)
    myelinsubdomains = createmyelindomains(exteriorgrids[:], torigrids[:], interiorgrids[:], outercircles[:], innercircles[:])

    @time doassemble!.(myelinsubdomains, Ref(myelinprob))
    @time factorize!.(getdomain.(myelinsubdomains))
    @time combinedmyelindomain = MyelinDomain(PermeableInterfaceRegion(), myelinprob, myelinsubdomains)
    @time factorize!(combinedmyelindomain)
    myelindomains = [combinedmyelindomain]

    return myelinprob, myelinsubdomains, myelindomains
end

function calcomegas(myelinprob, myelinsubdomains)
    omegavalues = omegamap.(Ref(myelinprob), myelinsubdomains)
    return omegavalues
end
calcomega(myelinprob, myelinsubdomains) = reduce(vcat, calcomegas(myelinprob, myelinsubdomains))

# Vector of signals on each domain.
# NOTE: This are integrals over the region, so the signals are already weighted
#       for the relative size of the region; the total signal is the sum of the
#       signals returned here
function calcsignals(sols, ts, myelindomains)
    Signals = map(sols, myelindomains) do s, m
        [integrate(s(t), m) for t in ts]
    end
    return Signals
end

# Sum signals over all domains
calcsignal(sols, ts, myelindomains) = sum(calcsignals(sols, ts, myelindomains))

function solveblochtorrey(myelinprob, myelindomains, algfun = default_algfun())
    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    ts = tspan[1]:TE/2:tspan[2] # tstops, which includes π-pulse times
    u0 = Vec{2}((0.0, 1.0)) # initial π/2 pulse

    probs = [ODEProblem(m, interpolate(u0, m), tspan; invertmass = true) for m in myelindomains]
    sols = Vector{ODESolution}()

    @time for (i,prob) in enumerate(probs)
        println("i = $i/$(length(probs)): ")
        cb = MultiSpinEchoCallback(tspan; TE = TE)
        alg = algfun(prob)
        sol = @time solve(prob, alg;
            dense = false, # don't save all intermediate time steps
            saveat = ts, # timepoints to save solution at
            tstops = ts, # ensure stopping at all ts points
            dt = TE,
            reltol = 1e-4,
            callback = cb
        )
        push!(sols, sol)
    end

    return sols
end

function get_algfun(algtype = :ExpokitExpmv)
    algfun = if algtype == :CVODE_BDF
        prob -> CVODE_BDF(;method = :Functional)
    else
        default_algfun()
    end
    return algfun
end
default_algfun() = prob -> ExpokitExpmv(prob.p[1]; m = 30) # first parameter is A in du/dt = A*u

function plotmagnitude(sols, btparams, myelindomains, bdry; titlestr = "Magnitude", fname = nothing)
    Umagn = reduce(vcat, norm.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    simpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = Umagn)
    mxcall(:title, 0, titlestr)

    !(fname == nothing) && savefig(fname)

    # Uphase = reduce(vcat, angle.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    # simpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = Uphase)
    # mxcall(:title, 0, "Phase")

    nothing
end

function plotbiexp(sols, btparams, myelindomains, outercircles, innercircles, bdry; titlestr = "Signal Magnitude vs. Time", fname = nothing)
    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    ts = tspan[1]:TE:tspan[2] # signal after each echo
    Stotal = calcsignal(sols, ts, myelindomains)

    myelin_area = intersect_area(outercircles, bdry) - intersect_area(innercircles, bdry)
    total_area = area(bdry)
    ext_area = total_area - myelin_area

    # In the high diffusion & highly permeable membrane limit, spins are equally
    # likely to be anywhere on the grid, hence experience a decay rate R2_mono
    # on the average, where R2_mono is the area averaged R2 of each compartment
    R2_mono = (btparams.R2_sp * myelin_area + btparams.R2_lp * ext_area) / total_area
    y_monoexp = @. total_area * exp(-ts * R2_mono)

    # In the low diffusion OR impermeable membrane limit, spins are confined to
    # their separate regions and experience their compartment R2 only
    y_biexp = @. ext_area * exp(-ts * btparams.R2_lp) + myelin_area * exp(-ts * btparams.R2_sp)

    mxcall(:figure, 0)
    mxcall(:plot, 0, collect(1000.0.*ts), [norm.(Stotal) y_biexp])
    mxcall(:legend, 0, "Simulated", "Bi-Exponential")
    mxcall(:title, 0, titlestr)
    mxcall(:xlabel, 0, "Time [ms]")
    mxcall(:xlim, 0, 1000.0 .* [tspan...])
    mxcall(:ylabel, 0, "S(t) Magnitude")

    !(fname == nothing) && savefig(fname)

    nothing
end

function plotSEcorr(sols, btparams, myelindomains; fname = nothing)
    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    ts = tspan[1]:TE:tspan[2] # signal after each echo
    Stotal = calcsignal(sols, ts, myelindomains)

    MWImaps, MWIdist, MWIpart = fitmwfmodel(Stotal, NNLSRegression();
        T2Range = [8e-3, 2.0],
        spwin = [8e-3, 24.75e-3],
        mpwin = [25.25e-3, 200e-3],
        nT2 = 32,
        RefConAngle = 165.0,
        PLOTDIST = true
    )

    !(fname == nothing) && savefig(fname)

    return MWImaps, MWIdist, MWIpart
end

# Save plot
function savefig(fname)
    datedfname = "$(getnow())__$fname"
    mxcall(:savefig, 0, datedfname * ".fig")
    mxcall(:export_fig, 0, datedfname, "-png")
    mxcall(:close, 0)
end

function compareMWFmethods(sols, myelindomains, outercircles, innercircles, bdry)
    tspan = (0.0, 320.0e-3)
    TE = 10e-3
    ts = tspan[1]:TE:tspan[2] # signal after each echo
    Stotal = calcsignal(sols, ts, myelindomains)

    mwfvalues = Dict(
        :exact => getmwf(outercircles, innercircles, bdry),
        :TwoPoolMagnToMagn => getmwf(Stotal, TwoPoolMagnToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolMagnToMagn => getmwf(Stotal, ThreePoolMagnToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolCplxToMagn => getmwf(Stotal, ThreePoolCplxToMagn(); TE = TE, fitmethod = :local),
        :ThreePoolCplxToCplx => getmwf(Stotal, ThreePoolCplxToCplx(); TE = TE, fitmethod = :local)
    )
    return mwfvalues
end

function CSV.write(results::MWFResults, i)
    for (j,sol) in enumerate(results.sols[i])
        df = DataFrame(sol)
        fname = "$(getnow())__sol_$(i)__region_$(j).csv"
        CSV.write(fname, df)
    end
    return nothing
end

function CSV.write(results::MWFResults)
    for i in 1:length(results.sols)
        CSV.write(results, i)
    end
    return nothing
end

# Standard date format
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")

end # module MWFUtils
