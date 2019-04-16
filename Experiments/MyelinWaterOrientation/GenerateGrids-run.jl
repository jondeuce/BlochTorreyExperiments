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
gr(size=(1200,900), leg = false, grid = false, labels = nothing) #xticks = nothing, yticks = nothing

function main()
    default_btparams = BlochTorreyParameters{Float64}()
    numreps = 2 # number of grids per parameter set

    numfibres = 10:10:50
    g_ratios = [0.75, 0.78, 0.80, default_btparams.g_ratio]
    packing_densities = [0.7, 0.75, 0.8, default_btparams.AxonPDensity]

    to_str(x) = @sprintf "%.4f" x
    params_to_str(p,g,n) = "N-$(n)_g-$(to_str(g))_p-$(to_str(p))"

    paramlist = Iterators.product(packing_densities, g_ratios, numfibres)

    for (i,params) in enumerate(paramlist)
        p, g, n = params
        paramstr = params_to_str(p,g,n)
        
        println(""); @info "Generating geometry $i/$(length(paramlist)), $(Dates.now()): $paramstr"

        btparams = BlochTorreyParameters(default_btparams; AxonPDensity = p, g_ratio = g)
        for _ in 1:numreps
            try
                creategeometry(btparams;
                    fname = getnow() * "__" * paramstr, # filename for saving MATLAB figure
                    Ncircles = n, # number of fibres to pack (resulting grid will have less due to cropping)
                    maxpackiter = 10, # number of radii distributions to attempt packing
                    overlapthresh = 0.01, # overlap relative to btparams.R_mu
                    alpha = 0.4,
                    beta = 0.5,
                    gamma = 1.0,
                    QMIN = 0.4,
                    RESOLUTION = 1.0,
                    MAXITERS = 2000,
                    FIXPOINTSITERS = 250,
                    FIXSUBSITERS = 200,
                    FORCEDENSITY = true, # error if desired density isn't reached
                    FORCEAREA = true, # error if the resulting grid area doesn't match the bdry area
                    FORCEQUALITY = true, # error if the resulting grid area have high enough quality
                    PLOT = true
                )
            catch e
                @warn "Error generating grid with param string: " * paramstr
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return nothing
end