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
    main()
    """
)

gitdir() = realpath(joinpath(DrWatson.projectdir(), "..")) * "/"

function runcreategeometry(params; numreps = 5)
    @unpack numfibres, gratio, density = params
    btparams = BlochTorreyParameters{Float64}(AxonPDensity = density, g_ratio = gratio)
    
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
                MAXITERS = 1000, #DEBUG
                FIXPOINTSITERS = 250,
                FIXSUBSITERS = 200,
                FORCEDENSITY = true, #DEBUG # error if desired density isn't reached
                FORCEAREA = true, #DEBUG # error if the resulting grid area doesn't match the bdry area
                FORCEQUALITY = true, #DEBUG # error if the resulting grid area have high enough quality
            )

            # Common filename without suffix
            fname = DrWatson.savename(MWFUtils.getnow(), params)

            # Plot circles and grid
            plotcircles([geom.innercircles; geom.outercircles], geom.bdry; fname = "plots/" * fname * ".circles")
            plotgrids(geom.exteriorgrids, geom.torigrids, geom.interiorgrids; fname = "plots/" * fname * ".grids")

            # Save generated geometry, tagging file with git commit
            DrWatson.@tagsave(
                "geom/" * fname * ".geom.bson",
                Dict(pairs(geom)),
                true, # safe saving (don't overwrite existing files)
                gitdir()
            )
        catch e
            @warn "Error generating grid with param string: " * DrWatson.savename("", params; connector = ", ")
            @warn sprint(showerror, e, catch_backtrace())
        end
    end

    return nothing
end

function main()
    # Make subfolders
    mkpath.(("plots", "geom"))

    # Parameters to sweep over
    sweep_params = Dict{Symbol,Any}(
        :numfibres  => [5:5:50;],
        :gratio     => [0.75, 0.78],
        :density    => [0.78, 0.8]
    )

    all_params = DrWatson.dict_list(sweep_params)
    all_params = sort(all_params; by = d -> (d[:numfibres], d[:gratio], d[:density]))
    for (i,params) in enumerate(all_params)
        params = convert(Dict{Symbol,Any}, params)
        @info "Generating geometry $i/$(length(all_params)) at $(Dates.now()): $(DrWatson.savename("", params; connector = ", "))"
        runcreategeometry(params)
    end

    return nothing
end