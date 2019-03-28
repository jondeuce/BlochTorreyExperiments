include(joinpath(@__DIR__, "init.jl")) # call "init.jl", located in the same directory as this file

using BSON, Dates, Printf
using StatsPlots
gr(size=(1200,1200), leg = false, grid = false, labels = nothing) #xticks = nothing, yticks = nothing

function generate_and_save_geometry(btparams; kwargs...)
    exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = creategrids(btparams; kwargs...)
    return exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry
end

function main()
    default_btparams = BlochTorreyParameters{Float64}()
    numreps = 1 # number of grids per parameter set

    numfibres = 10:10:50
    g_ratios = [0.75, 0.78, 0.80, default_btparams.g_ratio]
    packing_densities = [0.7, 0.75, 0.8, default_btparams.AxonPDensity]
    # numfibres = [20]
    # g_ratios = [0.80]
    # packing_densities = [0.7]

    to_str(x) = @sprintf "%.4f" x
    params_to_str(p,g,n) = "N-$(n)_g-$(to_str(g))_p-$(to_str(p))"

    paramlist = Iterators.product(packing_densities, g_ratios, numfibres)

    for (i,params) in enumerate(paramlist)
        p, g, n = params
        paramstr = params_to_str(p,g,n)
        
        println("\n\n---- Generating geometry $i/$(length(paramlist)), $(Dates.now()): $paramstr ----\n\n")

        btparams = BlochTorreyParameters(default_btparams; AxonPDensity = p, g_ratio = g)
        for _ in 1:numreps
            try
                generate_and_save_geometry(btparams;
                    fname = getnow() * "__" * paramstr, # filename for saving MATLAB figure
                    Ncircles = n, # number of fibres to pack (resulting grid will have less due to cropping)
                    maxpackiter = 10, # number of radii distributions to attempt packing
                    overlapthresh = 0.05, # overlap relative to btparams.R_mu
                    alpha = 0.5,
                    beta = 0.5,
                    gamma = 1.0,
                    QMIN = 0.3,
                    RESOLUTION = 1.0,
                    MAXITERS = 1000,
                    FIXPOINTSITERS = 250,
                    FIXSUBSITERS = 200,
                    FORCEDENSITY = true, # error if desired density isn't reached
                    FORCEAREA = true, # error if the resulting grid area doesn't match the bdry area
                    FORCEQUALITY = true, # error if the resulting grid area have high enough quality
                    PLOT = true
                )
            catch e
                @warn "error generating grid with param string: " * paramstr
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return nothing
end

main()