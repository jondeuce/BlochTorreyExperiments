# Initialize packages, and create backup file
include(joinpath(@__DIR__, "init.jl"))
make_reproduce(
    """
    include("BlochTorreyExperiments/Experiments/MyelinWaterOrientation/GenerateGrids-run.jl")
    main()
    """
)

using BSON, Dates, Printf
using MWFUtils, StatsPlots
gr(size=(1200,900), leg = false, grid = false, labels = nothing)

import DrWatson
using DrWatson: @dict, @ntuple
DrWatson.default_prefix(c) = MWFUtils.getnow()
gitdir() = realpath(joinpath(DrWatson.projectdir(), "../..")) * "/"

function runcreategeometry(params)
    @unpack numfibres, gratio, density = params
    btparams = BlochTorreyParameters{Float64}(AxonPDensity = density, g_ratio = gratio)
    
    numreps = 5 # number of grid generation attempts per parameter set
    for _ in 1:numreps
        try
            geom = creategeometry(btparams;
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
            fname = DrWatson.savename(params)
            
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
            @warn "Error generating grid with param string: " * DrWatson.savename("", params)
            @warn sprint(showerror, e, catch_backtrace())
        end
    end

    return nothing
end

function main()
    # Make subfolders
    mkpath.(("plots", "geom"))

    # Parameters to sweep over
    general_params = Dict{Symbol,Any}(
        :numfibres  => [5:5:50;],
        :gratio     => [0.75, 0.78],
        :density    => [0.78, 0.8]
    )

    all_params = DrWatson.dict_list(general_params)
    for (i,params) in enumerate(all_params)
        params = convert(Dict{Symbol,Any}, params)
        @info "Generating geometry $i/$(length(all_params)) at $(Dates.now()): $(DrWatson.savename("", params))"
        runcreategeometry(params)
    end

    return nothing
end