# Initialization of packages
include(joinpath(@__DIR__, "init.jl")) # call "init.jl", located in the same directory as this file
mxcall(:cd, 0, pwd()) # change MATLAB path to current path for saving outputs

# Creating backup
if !isfile("reproduce.jl")
    include(joinpath(@__DIR__, "make_reproduce.jl"))
    open("reproduce.jl", "a") do io
        str =
            """
            include("BlochTorreyExperiments/Experiments/MyelinWaterOrientation/MWF-run.jl")
            """
        write(io, str)
    end
end

using BSON, CSV, Dates, Printf
using Plots, MATLABPlots
gr(size=(1200,1200))
# gr(size=(1200,1200), leg = false, grid = false, xticks = nothing, yticks = nothing)

# Precomputed geometries
geomfilename = if !isfile("geom.bson")
    joinpath(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/kmg_geom_sweep_3",
        "2019-03-28-T-15-24-11-877__N-10_g-0.7500_p-0.7500__structs.bson" # 1.3k triangles, 1.2k points, Qmin = 0.3
        # "2019-03-28-T-15-26-44-544__N-10_g-0.8000_p-0.8300__structs.bson" # 4.7k triangles, 3.2k points, Qmin = 0.3
        # "2019-03-28-T-15-27-56-042__N-20_g-0.7500_p-0.7000__structs.bson" # 3.1k triangles, 2.6k points, Qmin = 0.3
        # "2019-03-28-T-15-33-59-628__N-20_g-0.8000_p-0.8000__structs.bson" #13.3k triangles, 9.2k points, Qmin = 0.3
    )
    # geomfilename = joinpath(
    #     "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/kmg_geom_sweep_4",
    #     "2019-03-28-T-16-19-20-218__N-40_g-0.7500_p-0.8000__structs.bson" # 11.0k triangles, 8.6k points, Qmin = 0.3
    # )
    # geomfilename = joinpath(
    #     "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/kmg_geom_sweep_6",
    #     # "2019-03-29-T-10-47-05-945__N-40_g-0.7500_p-0.7000__structs.bson" #10k triangles,  8k points, Qmin = 0.4
    #     # "2019-03-29-T-12-19-17-694__N-40_g-0.8370_p-0.7500__structs.bson" #13k triangles, 10k points, Qmin = 0.4
    #     "2019-03-29-T-12-15-03-265__N-40_g-0.8000_p-0.8300__structs.bson" #28k triangles, 19k points, Qmin = 0.4
    # )
    geomfilename = cp(geomfilename, "geom.bson")
else
    "geom.bson"
end

function MWF!(results, params, geom)
    # save current parameters
    push!(results.params, params)

    # unpack geometry and create myelin domains
    domains, omegas = results.metadata[:domains], results.metadata[:omegas]
    exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = geom
    myelinprob, myelinsubdomains, myelindomains = createdomains(params, exteriorgrids, torigrids, interiorgrids, outercircles, innercircles)
    domain = (myelinprob = myelinprob, myelinsubdomains = myelinsubdomains, myelindomains = myelindomains)
    push!(domains, domain)

    omega = calcomega(myelinprob, myelinsubdomains)
    push!(omegas, omega)

    # mxsimpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = omega)

    sols = solveblochtorrey(myelinprob, myelindomains)
    push!(results.sols, sols)

    titleparamstr = "theta = $(rad2deg(params.theta)) deg, D = $(params.D_Tissue) um2/s, K = $(params.K_perm) um/s"
    curr_date = getnow()
    plotmagnitude(sols, params, myelindomains, bdry; titlestr = "Magnitude: " * titleparamstr, fname = "$(curr_date)__magnitude")
    plotSEcorr(sols, params, myelindomains, fname = "$(curr_date)__SEcorr")
    plotbiexp(sols, params, myelindomains, outercircles, innercircles, bdry; titlestr = "Signal: " * titleparamstr, fname = "$(curr_date)__signal")

    mwfvalues = compareMWFmethods(sols, myelindomains, outercircles, innercircles, bdry)
    push!(results.mwfvalues, mwfvalues)

    return nothing
end

function main(geomfilename = "geom.bson")
    # Load geometries
    geom = loadgeometry(geomfilename)
    
    # Params to sweep over
    # thetarange = range(0.0, stop = π/2, length = 5)
    # Krange = [0.05, 0.1, 0.5, 1.0]
    # Drange = [100.0, 500.0, 1000.0]
    thetarange = range(0.0, stop = π/2, length = 3)
    Krange = [0.1]
    Drange = [50.0]
    
    # Default parameters
    default_btparams = BlochTorreyParameters{Float64}(
        theta = π/2,
        AxonPDensity = 0.8,
        g_ratio = 0.8,
        D_Tissue = 500.0, #0.5, # [μm²/s]
        D_Sheath = 50.0, #0.5, # [μm²/s]
        D_Axon = 500.0, #0.5, # [μm²/s]
        K_perm = 1.0 #0.0 # [μm/s]
    )
    
    # Labels
    numfibres = length(geom.innercircles)
    to_str(x) = @sprintf "%.4f" x
    params_to_str(θ,κ,D) = "N-$(numfibres)_theta-$(to_str(rad2deg(θ)))_K-$(to_str(κ))_D-$(to_str(D))"

    # Parameter sweep
    results = MWFResults{Float64}()
    results.metadata[:geom]       = geom
    results.metadata[:domains]    = []
    results.metadata[:omegas]     = []
    results.metadata[:TE]         = 10e-3
    results.metadata[:numfibres]  = numfibres
    results.metadata[:thetarange] = thetarange
    results.metadata[:Krange]     = Krange
    results.metadata[:Drange]     = Drange

    paramlist = Iterators.product(thetarange, Krange, Drange)
    for (count,params) in enumerate(paramlist)
        theta, K, D = params
        paramstr = params_to_str(theta,K,D)

        try
            println("\n\n---- SIMULATION $count/$(length(paramlist)): $(Dates.now()): $paramstr ----\n\n")

            # Create new set of parameters
            btparams = BlochTorreyParameters(default_btparams;
                theta = theta,
                K_perm = K,
                D_Tissue = D,
                D_Sheath = D/10,
                D_Axon = D
            )
            MWF!(results, btparams, geom)

            BSON.bson(getnow() * "__" * paramstr * "__btparams.bson", Dict(:btparams => btparams))
            CSV.write(results, count)
        catch e
            if e isa InterruptException
                @warn "Parameter sweep interrupted by user. Breaking out of loop and saving current results..."
                break
            else    
                @warn "Error running simulation $count/$(length(paramlist))"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    # Save results
    try
        BSON.bson(getnow() * "__results.bson", Dict(:results => deepcopy(results)))
    catch e
        @warn "Error saving results!"
        @warn sprint(showerror, e, catch_backtrace())
    end

    return results
end

results = main(geomfilename)