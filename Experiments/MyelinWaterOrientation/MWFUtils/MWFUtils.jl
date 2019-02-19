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
using MATLABPlots
using OrdinaryDiffEq, DiffEqOperators, Sundials
using BenchmarkTools
using Parameters: @with_kw, @unpack
using IterableTables, DataFrames, CSV, Dates

export packcircles, creategrids, createdomains
export calcomegas, calcomega
export calcsignals, calcsignal
export solveblochtorrey, default_algfun, get_algfun
export plotmagnitude, plotphase, plotSEcorr, plotbiexp
export compareMWFmethods
export mxsavefig, getnow

export MWFResults

@with_kw struct MWFResults{T}
    metadata::Dict{Symbol,Any}               = Dict()
    params::Vector{BlochTorreyParameters{T}} = []
    sols::Vector{Vector{ODESolution}}        = []
    mwfvalues::Vector{Dict{Symbol,T}}        = []
end

function packcircles(btparams::BlochTorreyParameters{T};
        N = 20, # number of circles
        η = btparams.AxonPDensity, # goal packing density
        ϵ = 0.1 * btparams.R_mu, # overlap occurs when distance between circle edges is ≤ ϵ
        α = 1e-1, # covariance penalty weight (enforces circular distribution)
        β = 1e-6, # mutual distance penalty weight
        λ = 1.0, # overlap penalty weight (or lagrange multiplier for constrained version)
        it = 100, # maximum iterations for greedy packing
        maxiter = 5 # maximum attempts for sampling radii + greedy packing + energy packing
    ) where {T}
    
    local circles
    η_best = 0.0
    
    for i in 1:maxiter
        println("\nPacking... (attempt $i/$maxiter)\n")
        rs = rand(radiidistribution(btparams), N) # Initial radii distribution
        
        print("GreedyCirclePacking: ")
        @time greedycircles = GreedyCirclePacking.pack(rs; goaldensity = 1.0, iters = it)

        print("EnergyCirclePacking: ")
        @time energycircles = EnergyCirclePacking.pack(greedycircles;
            autodiff = true,
            secondorder = false,
            setcallback = false,
            goaldensity = 1.0, #η # pack as much as possible, scale to goal density after
            distancescale = btparams.R_mu,
            weights = [α, β, λ],
            epsilon = ϵ # pack as much as possible, penalizing packing tighter than distance ϵ
        )

        scaledcircles = CirclePackingUtils.scale_to_density(energycircles, η, ϵ)

        println("")
        println("Distance threshold: $ϵ")
        println("Minimum myelin thickness: $(minimum(radius.(scaledcircles))*(1-btparams.g_ratio))")
        println("Minimum circles distance: $(minimum_signed_edge_distance(scaledcircles))")
        println("")
        println("GreedyCirclePacking density:  $(estimate_density(greedycircles))")
        println("EnergyCirclePacking density:  $(estimate_density(energycircles))")
        println("Final scaled circles density: $(estimate_density(scaledcircles))")
        
        η_curr = estimate_density(scaledcircles)
        (η_curr ≈ η) && (circles = scaledcircles; break)
        (η_curr > η_best) && (η_best = η_curr; circles = scaledcircles)
    end

    return circles
end

function creategrids(btparams::BlochTorreyParameters{T};
        fname = nothing, # filename for saving
        N = 20, # number of circles
        η = btparams.AxonPDensity, # goal packing density
        ϵ = 0.1 * btparams.R_mu, # overlap occurs when distance between circle edges is ≤ ϵ
        maxpackiter = 10,
        outercircles = packcircles(btparams; N=N,η=η,ϵ=ϵ,α=1e-1,β=1e-6,λ=1.0,it=100,maxiter=maxpackiter), # outer circles
        bdry = opt_subdomain(outercircles)[1], # default boundary is automatically determined in packcircles
        FORCEDENSITY = false, # If this flag is true, an error is thrown if the reached packing density is not η
        FORCEAREA = false, # If this flag is true, an error is thrown if the resulting grid area doesn't match the bdry area
        RESOLUTION = 1.0,
        CIRCLESTALLITERS = 1000, #DEBUG
        EXTERIORSTALLITERS = 1000, #DEBUG
        PLOT = true
    ) where {T}

    if FORCEDENSITY
        η_curr = estimate_density(outercircles)
        !(η_curr ≈ η) && error("Packing density not reached: goal density was $η, reached $η_curr.")
    end

    mindist = minimum_signed_edge_distance(outercircles)
    h_min = RESOLUTION * T(0.5 * mindist)
    h_max = RESOLUTION * T(1.0 * mindist)
    h_range = RESOLUTION * T(2.0 * mindist)
    h_rate = T(1.0)

    innercircles = scale_shape.(outercircles, btparams.g_ratio)

    @time exteriorgrids, torigrids, interiorgrids, parentcircleindices = disjoint_rect_mesh_with_tori(
        bdry, innercircles, outercircles, h_min, h_max, h_range, h_rate;
        plotgrids = PLOT, exterior_tiling = (1, 1), # DEBUG
        CIRCLESTALLITERS = CIRCLESTALLITERS, EXTERIORSTALLITERS = EXTERIORSTALLITERS
    )

    grid_area = sum(area.(exteriorgrids)) + sum(area.(torigrids)) + sum(area.(interiorgrids))
    bdry_area = area(bdry)
    cell_area_mismatch = bdry_area - grid_area
    if FORCEAREA
        !(grid_area ≈ bdry_area) && error("Grid area not matched with boundary area; relative error is $(cell_area_mismatch/bdry_area).")
    end
    @show cell_area_mismatch

    PLOT && mxsimpplot(vcat(exteriorgrids[:], torigrids[:], interiorgrids[:]); newfigure = true, axis = mxaxis(bdry))
    (fname != nothing) && mxsavefig(fname) #DEBUG

    # PLOT && simpplot(vcat(exteriorgrids[:], torigrids[:], interiorgrids[:]); color = :cyan) |> display

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

    println("Assembling...")
    @time doassemble!.(myelinsubdomains, Ref(myelinprob))
    @time factorize!.(getdomain.(myelinsubdomains))
    @time combinedmyelindomain = MyelinDomain(PermeableInterfaceRegion(), myelinprob, myelinsubdomains)
    @time factorize!(combinedmyelindomain)
    myelindomains = [combinedmyelindomain]

    # error("breakpoint10")
    # error("breakpoint15")

    return myelinprob, myelinsubdomains, myelindomains
end

calcomegas(myelinprob, myelinsubdomains) = omegamap.(Ref(myelinprob), myelinsubdomains)
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

function solveblochtorrey(myelinprob, myelindomains, algfun = default_algfun();
        tspan = (0.0, 320.0e-3),
        TE = 10e-3,
        ts = tspan[1]:TE/2:tspan[2], # tstops, which includes π-pulse times
        u0 = Vec{2}((0.0, 1.0)), # initial π/2 pulse
        cb = MultiSpinEchoCallback(tspan; TE = TE),
        reltol = 1e-4,
        abstol = 1e-12,
    )
    probs = [ODEProblem(m, interpolate(u0, m), tspan; invertmass = true) for m in myelindomains]
    sols = Vector{ODESolution}()

    @time for (i,prob) in enumerate(probs)
        println("i = $i/$(length(probs)): ")
        alg = algfun(prob)
        sol = @time solve(prob, alg;
            dense = false, # don't save all intermediate time steps
            saveat = ts, # timepoints to save solution at
            tstops = ts, # ensure stopping at all ts points
            dt = TE,
            reltol = reltol,
            abstol = abstol,
            callback = cb
        )
        push!(sols, sol)
    end

    return sols
end

function get_algfun(algtype = :ExpokitExpmv)
    algfun = if algtype == :CVODE_BDF
        prob -> CVODE_BDF(;method = :Functional)
    elseif algtype isa DiffEqBase.AbstractODEAlgorithm
        prob -> algtype # given an DiffEqBase.AbstractODEAlgorithm algorithm directly
    elseif algtype == ExpokitExpmv
        expokit_algfun()
    else
        default_algfun()
    end
    return algfun
end
expokit_algfun() = prob -> ExpokitExpmv(prob.p[1]; m = 30) # first parameter is A in du/dt = A*u
default_algfun() = expokit_algfun()

function plotmagnitude(sols, btparams, myelindomains, bdry; titlestr = "Magnitude", fname = nothing)
    Umagn = reduce(vcat, norm.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = Umagn)
    mxcall(:title, 0, titlestr)

    # allgrids = vcat(exteriorgrids[:], torigrids[:], interiorgrids[:])
    # mxsimpplot(allgrids; newfigure = true, axis = mxaxis(bdry))

    !(fname == nothing) && mxsavefig(fname)

    nothing
end

function plotphase(sols, btparams, myelindomains, bdry; titlestr = "Phase", fname = nothing)
    Uphase = reduce(vcat, angle.(reinterpret(Vec{2,Float64}, s.u[end])) for s in sols)
    mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = Uphase)
    mxcall(:title, 0, titlestr)

    !(fname == nothing) && mxsavefig(fname)

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

    !(fname == nothing) && mxsavefig(fname)

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

    !(fname == nothing) && mxsavefig(fname)

    return MWImaps, MWIdist, MWIpart
end

# Save plot
function mxsavefig(fname)
    fname = getnow() * "__" * fname
    mxcall(:savefig, 0, fname * ".fig")
    mxcall(:export_fig, 0, fname, "-png")
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
        fname = getnow() * "__sol_$(i)__region_$(j).csv"
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
