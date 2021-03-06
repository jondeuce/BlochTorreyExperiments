# Activate project and load packages for this script
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using MWFUtils
using GlobalUtils
make_reproduce(
    """
    include("BlochTorreyExperiments/MyelinWaterTools/scripts/GenerateGrids-run.jl")
    """
)

pyplot(size=(1200,900), leg = false, grid = false, labels = nothing)
gitdir() = realpath(DrWatson.projectdir("..")) # DrWatson package for tagged saving

function runcreategeometry(params; numreps = 5)
    # BlochTorreyParameters
    @unpack numfibres, gratio, density, mvf, mwf = params
    btparams = BlochTorreyParameters{Float64}(
        PD_lp = 1.0,
        PD_sp = 0.5,
        g_ratio = gratio,
        AxonPDensity = density,
        MVF = mvf,
        MWF = mwf,
    )

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
                VERBOSE = true,
                FORCEDENSITY = true, #DEBUG # error if desired density isn't reached
                FORCEAREA = true, #DEBUG # error if the resulting grid area doesn't match the bdry area
                FORCEQUALITY = true) #DEBUG # error if the resulting grid area have high enough quality

            # Common filename without suffix
            params[:Ntri] = sum(JuAFEM.getncells, geom.exteriorgrids) + sum(JuAFEM.getncells, geom.torigrids) + sum(JuAFEM.getncells, geom.interiorgrids)
            params[:Npts] = sum(JuAFEM.getnnodes, geom.exteriorgrids) + sum(JuAFEM.getnnodes, geom.torigrids) + sum(JuAFEM.getnnodes, geom.interiorgrids)
            fname = DrWatson.savename(MWFUtils.getnow(), params)

            # Plot circles and grid
            plotcircles([geom.innercircles; geom.outercircles], geom.bdry; fname = "plots/circles/" * fname * ".circles")
            plotgrids(geom.exteriorgrids, geom.torigrids, geom.interiorgrids; fname = "plots/grids/" * fname * ".grids")

            # Save generated geometry, tagging file with git commit
            DrWatson.@tagsave(
                "geom/" * fname * ".geom.bson",
                Dict(:params => params, pairs(geom)...),
                safe = true, # safe saving (don't overwrite existing files)
                gitpath = gitdir())
        catch e
            if e isa InterruptException
                # User interrupt; rethrow and catch in main()
                rethrow(e)
            else
                # Grid generation error; print error and try again
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

function main(;iters = 1000, randomorder = false)
    # Make subfolders
    map(mkpath, ("plots/circles", "plots/grids", "geom"))

    # Parameter sampler
    sweep_params_sampler() = Dict{Symbol,Union{Float64,Int}}(
        :numfibres => rand(5:30),
        :mvf       => 0.025 + (0.40 - 0.025) * rand(),
    )

    # Parameters to sweep over
    sweep_params = (sweep_params_sampler() for _ in 1:iters)
    if !randomorder
        sweep_params = sort(collect(sweep_params); by = d -> (d[:numfibres], d[:mvf]))
    end

    for (i,params) in enumerate(sweep_params)
        try
            @info "Generating geometry $i/$(length(sweep_params)) at $(Dates.now()): $(DrWatson.savename("", params; connector = ", "))"
            @unpack g_ratio, AxonPDensity = BlochTorreyUtils.optimal_g_ratio_packdensity(
                params[:mvf];
                g_ratio_bounds = (0.60, 0.92), # mvf = (1-g^2)*η. These bounds permit mvf solutions
                density_bounds = (0.15, 0.82), # approximately in the range [2.5%, 50%]
                solution_choice = :random,
            )
            geomparams = deepcopy(params)
            geomparams[:gratio] = g_ratio
            geomparams[:density] = AxonPDensity
            geomparams[:mvf] = (1 - geomparams[:gratio]^2) * geomparams[:density] # mvf to be simulated
            geomparams[:mwf] = geomparams[:mvf] / (2 - geomparams[:mvf]) # assumes relative proton density == 1/2
            runcreategeometry(geomparams)
        catch e
            if e isa InterruptException
                @warn "Parameter sweep interrupted by user. Breaking out of loop and returning..."
                break
            else
                @warn "Error running simulation $i/$(length(sweep_params))"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return nothing
end

main(;iters = typemax(Int), randomorder = true)

# # Renaming + resaving mislabeled geometries
# geomdata = []
# geomfiles = joinpath.("geom", filter(s->endswith(s, ".bson"), readdir("geom")))
# 
# for (i,geomfile) in enumerate(geomfiles)
#     geom = BSON.load(geomfile)
# 
#     A_out = sum(c->intersect_area(c,geom[:bdry]), geom[:outercircles])
#     A_in = sum(c->intersect_area(c,geom[:bdry]), geom[:innercircles])
#     A_bdry = area(geom[:bdry])
#     mvf = (A_out - A_in) / A_bdry
#     mwf = mvf / (2 - mvf)
#     geom[:params][:mvf] = mvf
#     geom[:params][:mwf] = mwf
#     
#     # Save generated geometry, tagging file with git commit
#     fname = DrWatson.savename(basename(geomfile)[1:25], geom[:params])
#     DrWatson.@tagsave(
#         "geom-renamed/" * fname * ".geom.bson",
#         geom,
#         safe = true, # safe saving (don't overwrite existing files)
#         gitpath = gitdir())
#     
#     push!(geomdata, geom)
# end