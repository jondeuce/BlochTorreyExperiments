module MWFUtils

using LinearAlgebra, Statistics
using GeometryUtils
using CirclePackingUtils
using MeshUtils
using DistMesh
using BlochTorreyUtils
using BlochTorreySolvers
import EnergyCirclePacking
import GreedyCirclePacking

using JuAFEM
using OrdinaryDiffEq, DiffEqOperators, Sundials
using BenchmarkTools
using Parameters: @with_kw, @unpack
using IterableTables, DataFrames, BSON, CSV, Dates

# Plotting
using StatsPlots, MATLABPlots

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

        scaledcircles, domain, _ = CirclePackingUtils.scale_to_density(energycircles, η, ϵ; MODE = :corners)

        println("")
        println("Distance threshold: $ϵ")
        println("Minimum myelin thickness: $(minimum(radius.(scaledcircles))*(1-btparams.g_ratio))")
        println("Minimum circles distance: $(minimum_signed_edge_distance(scaledcircles))")
        println("")
        println("GreedyCirclePacking density:  $(estimate_density(greedycircles, domain))")
        println("EnergyCirclePacking density:  $(estimate_density(energycircles, domain))")
        println("Final scaled circles density: $(estimate_density(scaledcircles, domain))")
        
        η_curr = estimate_density(scaledcircles, domain)
        (η_curr ≈ η) && (circles = scaledcircles; break)
        (η_curr > η_best) && (η_best = η_curr; circles = scaledcircles)
    end

    return circles
end

function creategrids(btparams::BlochTorreyParameters{T};
        fname = nothing, # filename for saving
        Ncircles = 20, # number of circles
        goaldensity = btparams.AxonPDensity, # goal packing density
        overlapthresh = 0.1, # overlap occurs when distance between circle edges is ≤ overlapthresh * btparams.R_mu
        maxpackiter = 10,
        outercircles = packcircles(btparams;
            N = Ncircles, maxiter = maxpackiter,
            η = goaldensity, ϵ = overlapthresh * btparams.R_mu),
        alpha = 0.5, #DEBUG
        beta = 0.5, #DEBUG
        gamma = 1.0, #DEBUG
        RESOLUTION = 1.25, #DEBUG
        FORCEDENSITY = false, # If this flag is true, an error is thrown if the reached packing density is not goaldensity
        FORCEAREA = false, # If this flag is true, an error is thrown if the resulting grid area doesn't match the bdry area
        FORCEQUALITY = false, # If this flag is true, an error is thrown if the resulting grid doesn't have high enough quality
        QMIN = 0.3, #DEBUG
        MAXITERS = 1000, #DEBUG
        FIXPOINTSITERS = 250, #DEBUG
        FIXSUBSITERS = 200, #DEBUG
        PLOT = true
    ) where {T}

    # Initial set of circles
    innercircles = scale_shape.(outercircles, btparams.g_ratio)
    allcircles = collect(Iterators.flatten(zip(innercircles, outercircles)))
    
    # Optimize the rectangular subdomain 
    bdry, _ = opt_subdomain(allcircles; MODE = :corners)
    outercircles, bdry, α_best = scale_to_density(outercircles, bdry, btparams.AxonPDensity)
    innercircles = scale_shape.(outercircles, btparams.g_ratio)
    allcircles = collect(Iterators.flatten(zip(innercircles, outercircles)))
    
    if FORCEDENSITY
        density = estimate_density(outercircles, bdry)
        !(density ≈ goaldensity) && error("Packing density not reached: goal density was $goaldensity, reached $density.")
    end
    
    mincircdist = minimum_signed_edge_distance(outercircles)
    h0 = gamma * mincircdist
    dmax = beta * btparams.R_mu
    bbox = [xmin(bdry) ymin(bdry); xmax(bdry) ymax(bdry)]
    pfix = [Vec{2,T}[corners(bdry)...]; reduce(vcat, intersection_points(c,bdry) for c in allcircles)]
    
    # Increase resolution by a factor RESOLUTION
    h0 /= RESOLUTION
    alpha /= RESOLUTION
    
    # Signed distance function
    fd(x) = drectangle0(x, bdry)

    # Relative edge length function
    function fh(x::Vec{2,T}) where {T}
        douter = MeshUtils.dcircles(x, outercircles)
        dinner = MeshUtils.dcircles(x, innercircles)
        hallcircles = min(abs(douter), abs(dinner))/T(dmax)
        return alpha + min(hallcircles, one(T))
    end
    
    # Region and sub-region definitions. Order of `allcircles` is important, as we want to project
    # inner circle points first, followed by outer circle points. Note also that zipping the circles
    # together allows for the anonymous function in the comprehension to be well typed.
    fsubs = [x->dcircle(x,c) for c in allcircles]

    p, t = kmg2d(fd, fsubs, fh, h0, bbox, 1, 0, pfix;
        QMIN = QMIN,
        MAXITERS = MAXITERS,
        FIXPOINTSITERS = FIXPOINTSITERS,
        FIXSUBSITERS = FIXSUBSITERS,
        VERBOSE = true,
        DETERMINISTIC = true,
        PLOT = false,
        PLOTLAST = false
    );

    if FORCEQUALITY
        Qmesh = DistMesh.mesh_quality(p,t)
        !(Qmesh >= QMIN) && error("Grid quality not high enough; Q = $Qmesh < $QMIN.")
    end

    text  = [NTuple{3,Int}[] for _ in 1:1]
    tint  = [NTuple{3,Int}[] for _ in 1:length(outercircles)]
    ttori = [NTuple{3,Int}[] for _ in 1:length(outercircles)]
    for t in t
        @inbounds pmid = (p[t[1]] + p[t[2]] + p[t[3]])/3
        isfound = false
        for j in 1:length(outercircles)
            (fsubs[2j  ](pmid) < 0) && (push!(tint[j],  t); isfound = true; break) # check interior first
            (fsubs[2j-1](pmid) < 0) && (push!(ttori[j], t); isfound = true; break) # then tori
        end
        isfound && continue
        push!(text[1], t) # otherwise, exterior
    end
    
    function reorder(p, t)
        isempty(t) && (return eltype(p)[], eltype(t)[])
        idx = reinterpret(Int, t) |> copy |> sort! |> unique!
        d = Dict{Int,Int}(idx .=> 1:length(idx))
        return p[idx], [(d[t[1]], d[t[2]], d[t[3]]) for t in t]
    end

    G = Grid{2,3,T,3}
    exteriorgrids = G[Grid(reorder(p,t)...) for t in text]
    torigrids     = G[Grid(reorder(p,t)...) for t in ttori]
    interiorgrids = G[Grid(reorder(p,t)...) for t in tint]

    grid_area = sum(area.(exteriorgrids)) + sum(area.(torigrids)) + sum(area.(interiorgrids))
    bdry_area = area(bdry)
    cell_area_mismatch = bdry_area - grid_area
    if FORCEAREA
        !(grid_area ≈ bdry_area) && error("Grid area is not matched with boundary area; error is $(cell_area_mismatch).")
        dA_max = maximum(1:length(innercircles)) do i
            gin, gout = interiorgrids[i], torigrids[i]
            Ain, Aout = area(gin), area(gout)
            cin, cout = innercircles[i], outercircles[i]
            NCin, NCout = getncells(gin), getncells(gout)
            ain0 = NCin == 0 ? zero(T) : mean(c->area(gin,c), 1:NCin)
            aout0 = NCout == 0 ? zero(T) : mean(c->area(gout,c), 1:NCout)
            Ain0 = intersect_area(cin, bdry)
            Aout0 = intersect_area(cout, bdry) - Ain0
            dAin = ain0 ≈ 0 ? Ain0/(h0^2/2) : (Ain-Ain0)/ain0
            dAout = aout0 ≈ 0 ? Aout0/(h0^2/2) : (Aout-Aout0)/aout0
            return max(abs(dAin), abs(dAout))
        end
        !(dA_max < one(T)) && error("Grid subregion areas are not close to analytical circle areas; error relative to average triangle area is $(dA_max).")
    end

    if PLOT
        fig = plot(bdry; aspectratio = :equal);
        for c in allcircles; plot!(fig, c); end
        display(fig)
        (fname != nothing) && savefig(fig, fname * "__circles.pdf")
    end

    if PLOT
        numtri = sum(JuAFEM.getncells, exteriorgrids) + sum(JuAFEM.getncells, torigrids) + sum(JuAFEM.getncells, interiorgrids)
        numpts = sum(JuAFEM.getnnodes, exteriorgrids) + sum(JuAFEM.getnnodes, torigrids) + sum(JuAFEM.getnnodes, interiorgrids)
        fig = simpplot(exteriorgrids; colour = :cyan)
        simpplot!(fig, torigrids; colour = :yellow)
        simpplot!(fig, interiorgrids; colour = :red)
        title!("Disjoint Grids: $numtri total triangles, $numpts total points")
        display(fig)
        (fname != nothing) && savefig(fig, fname * "__grid.pdf")
    end

    if fname != nothing
        try
            BSON.bson(fname * "__structs.bson", Dict(
                :exteriorgrids => exteriorgrids,
                :torigrids     => torigrids,
                :interiorgrids => interiorgrids,
                :outercircles  => outercircles,
                :innercircles  => innercircles,
                :bdry => bdry))
        catch e
            @warn "Error saving geometries"
        end
    end
    
    return exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry
end

# function creategrids(btparams::BlochTorreyParameters{T};
#         fname = nothing, # filename for saving
#         Ncircles = 20, # number of circles
#         goaldensity = btparams.AxonPDensity, # goal packing density
#         overlapthresh = 0.1, # overlap occurs when distance between circle edges is ≤ overlapthresh * btparams.R_mu
#         maxpackiter = 10,
#         outercircles = packcircles(btparams;
#             N = Ncircles, η = goaldensity, ϵ = overlapthresh * btparams.R_mu,
#             α = 1e-1, β = 1e-6, λ = 1.0, it = 100, maxiter = maxpackiter), # outer circles
#         bdry = opt_subdomain(outercircles; MODE = :corners)[1], # default boundary is automatically determined in packcircles
#         h_min = 0.5, # minimum bar length, relative to the minimum distance between outer circles 
#         h_max = 1.0, # maximum bar length, relative to the minimum distance between outer circles 
#         h_range = 2.0, # range over which bar lengths vary, relative to the minimum distance between outer circles 
#         h_rate = 1.0, # rate of increase of bar lengths; 1.0 is linear, 0.5 is sqrt, etc.
#         RESOLUTION = 1.0, # Multiplicative factor to easily uniformly increase/decrease bar lengths
#         FORCEDENSITY = false, # If this flag is true, an error is thrown if the reached packing density is not goaldensity
#         FORCEAREA = false, # If this flag is true, an error is thrown if the resulting grid area doesn't match the bdry area
#         CIRCLESTALLITERS = 1000, #DEBUG
#         EXTERIORSTALLITERS = 1000, #DEBUG
#         PLOT = true
#     ) where {T}
# 
#     if FORCEDENSITY
#         density = estimate_density(outercircles; MODE = :corners)
#         !(density ≈ goaldensity) && error("Packing density not reached: goal density was $goaldensity, reached $density.")
#     end
# 
#     mindist = minimum_signed_edge_distance(outercircles)
#     h_min = T(RESOLUTION * h_min * mindist)
#     h_max = T(RESOLUTION * h_max * mindist)
#     h_range = T(RESOLUTION * h_range * mindist)
#     h_rate = T(h_rate)
# 
#     innercircles = scale_shape.(outercircles, btparams.g_ratio)
# 
#     @time exteriorgrids, torigrids, interiorgrids, parentcircleindices = disjoint_rect_mesh_with_tori(
#         bdry, innercircles, outercircles, h_min, h_max, h_range, h_rate;
#         plotgrids = PLOT, exterior_tiling = (1,1), # DEBUG
#         CIRCLESTALLITERS = CIRCLESTALLITERS, EXTERIORSTALLITERS = EXTERIORSTALLITERS
#     )
# 
#     grid_area = sum(area.(exteriorgrids)) + sum(area.(torigrids)) + sum(area.(interiorgrids))
#     bdry_area = area(bdry)
#     cell_area_mismatch = bdry_area - grid_area
#     if FORCEAREA
#         !(grid_area ≈ bdry_area) && error("Grid area not matched with boundary area; error is $(cell_area_mismatch).")
#     end
#     @show cell_area_mismatch
# 
#     PLOT && mxsimpplot(vcat(exteriorgrids[:], torigrids[:], interiorgrids[:]); newfigure = true, axis = mxaxis(bdry))
#     # PLOT && simpplot(vcat(exteriorgrids[:], torigrids[:], interiorgrids[:]); color = :cyan) |> display
# 
#     if fname != nothing
#         PLOT && mxsavefig(fname)     
#         try
#             BSON.bson(fname, Dict(
#                 :exteriorgrids => exteriorgrids,
#                 :torigrids     => torigrids,
#                 :interiorgrids => interiorgrids,
#                 :outercircles  => outercircles,
#                 :innercircles  => innercircles,
#                 :bdry => bdry))
#         catch e
#             @warn "Error saving geometries"
#         end
#     end
# 
#     return exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry
# end

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
    probs = [ODEProblem(m, interpolate(u0, m), tspan) for m in myelindomains]
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
    algfun = if algtype isa DiffEqBase.AbstractODEAlgorithm
        prob -> algtype # given an DiffEqBase.AbstractODEAlgorithm algorithm directly
    elseif algtype == :CVODE_BDF
        prob -> CVODE_BDF(;method = :Functional)
    elseif algtype == :ExpokitExpmv
        expokit_algfun()
    elseif algtype == :HighamExpmV
        higham_algfun()
    else
        default_algfun()
    end
    return algfun
end
expokit_algfun() = prob -> ExpokitExpmv(prob.p[1]; m = 30) # first parameter is A in du/dt = A*u
higham_algfun() = prob -> HighamExpmv(prob.p[1]) # first parameter is A in du/dt = A*u
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
function mxsavefig(fname; fig = true, png = true, pdf = true, eps = true)
    fig && mxcall(:savefig, 0, fname * ".fig")
    png && mxcall(:export_fig, 0, fname, "-png")
    pdf && mxcall(:export_fig, 0, fname, "-dpdf")
    eps && mxcall(:export_fig, 0, fname, "-eps")
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
    curr_date = getnow()
    for (j,sol) in enumerate(results.sols[i])
        fname = curr_date * "__sol_$(i)__region_$(j).csv"
        CSV.write(fname, DataFrame(sol))
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
