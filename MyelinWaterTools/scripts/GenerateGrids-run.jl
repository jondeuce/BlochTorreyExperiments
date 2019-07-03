# Activate project and load packages for this script
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "../initpaths.jl"))

# NOTE: must load pyplot backend BEFORE loading MATLAB in init.jl
using StatsPlots
pyplot(size=(1200,900), leg = false, grid = false, labels = nothing)
using GlobalUtils
using MWFUtils

# Initialize project packages
include(joinpath(@__DIR__, "../init.jl"))
make_reproduce(
    """
    include("BlochTorreyExperiments/MyelinWaterTools/scripts/GenerateGrids-run.jl")
    """
)

gitdir() = realpath(joinpath(DrWatson.projectdir(), "..")) * "/"

function runcreategeometry(params; numreps = 5)
    # BlochTorreyParameters
    @unpack numfibres, gratio, density, mwf = params
    btparams = BlochTorreyParameters{Float64}(AxonPDensity = density, g_ratio = gratio, MWF = mwf)
    
    # Attempt to generate `numreps` grids for given parameter set
    for _ in 1:numreps
        try
            geom = creategeometry(PeriodicPackedFibres(), btparams;
                Ncircles = numfibres, # number of fibres to pack (resulting grid will have less due to cropping)
                maxpackiter = 10, # number of radii distributions to attempt packing
                overlapthresh = 0.05, # overlap relative to btparams.R_mu
                alpha = 0.4,
                beta = 0.5,
                gamma = 1.0,
                QMIN = 0.4, #DEBUG
                MAXITERS = 2000, #DEBUG
                FIXPOINTSITERS = 1000, #DEBUG
                FIXSUBSITERS = 950, #DEBUG
                FORCEDENSITY = true, #DEBUG # error if desired density isn't reached
                FORCEAREA = true, #DEBUG # error if the resulting grid area doesn't match the bdry area
                FORCEQUALITY = true) #DEBUG # error if the resulting grid area have high enough quality

            # Common filename without suffix
            params[:ntri] = sum(JuAFEM.getncells, geom.exteriorgrids) + sum(JuAFEM.getncells, geom.torigrids) + sum(JuAFEM.getncells, geom.interiorgrids)
            params[:npts] = sum(JuAFEM.getnnodes, geom.exteriorgrids) + sum(JuAFEM.getnnodes, geom.torigrids) + sum(JuAFEM.getnnodes, geom.interiorgrids)
            fname = DrWatson.savename(MWFUtils.getnow(), params)

            # Plot circles and grid
            plotcircles([geom.innercircles; geom.outercircles], geom.bdry; fname = "plots/" * fname * ".circles")
            plotgrids(geom.exteriorgrids, geom.torigrids, geom.interiorgrids; fname = "plots/" * fname * ".grids")

            # Save generated geometry, tagging file with git commit
            DrWatson.@tagsave(
                "geom/" * fname * ".geom.bson",
                Dict(pairs(geom)),
                true, # safe saving (don't overwrite existing files)
                gitdir())
        catch e
            if e isa InterruptException
                # User interrupt; rethrow and catch in main()
                rethrow(e)
            else
                # Grid generation; print error and continue
                @warn "Error generating grid with param string: " * DrWatson.savename("", params; connector = ", ")
                @warn sprint(showerror, e, catch_backtrace())
                continue
            end
        end

        # Grid successfully generated; exit loop
        break
    end

    return nothing
end

function main()
    # Make subfolders
    mkpath.(("plots", "geom"))

    # Parameters to sweep over
    sweep_params = Dict{Symbol,Any}(
        :numfibres => collect(6:30),
        :mwf       => collect(0.15:0.01:0.3))

    all_params = DrWatson.dict_list(sweep_params)
    all_params = sort(all_params; by = d -> (d[:numfibres], d[:mwf]))
    for (i,params) in enumerate(all_params)
        try
            @info "Generating geometry $i/$(length(all_params)) at $(Dates.now()): $(DrWatson.savename("", params; connector = ", "))"
            @unpack g_ratio, AxonPDensity = BlochTorreyUtils.optimal_g_ratio_packdensity_gridsearch(params[:mwf])
            geomparams = deepcopy(params)
            geomparams[:gratio] = g_ratio
            geomparams[:density] = AxonPDensity
            runcreategeometry(geomparams)
        catch e
            if e isa InterruptException
                @warn "Parameter sweep interrupted by user. Breaking out of loop and returning..."
                break
            else
                @warn "Error running simulation $i/$(length(all_params))"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return nothing
end

main()