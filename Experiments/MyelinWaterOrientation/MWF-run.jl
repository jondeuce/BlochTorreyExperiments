include(joinpath(@__DIR__, "init.jl")) # call "init.jl", located in the same directory as this file
mxcall(:cd, 0, pwd()) # change MATLAB path to current path for saving outputs

using BSON, CSV, Dates, Printf
using Plots, MATLABPlots
gr(size=(1200,1200), leg = false, grid = false, xticks = nothing, yticks = nothing)

function load_geometry(fname)
    d = BSON.load(fname)
    G = Grid{2,3,Float64,3} # 2D triangular grid
    C = Circle{2,Float64}
    R = Rectangle{2,Float64}

    # Ensure proper typing of grids
    return convert(Vector{G}, d[:exteriorgrids][:]),
           convert(Vector{G}, d[:torigrids][:]),
           convert(Vector{G}, d[:interiorgrids][:]),
           convert(Vector{C}, d[:outercircles][:]),
           convert(Vector{C}, d[:innercircles][:]),
           convert(R, d[:bdry])
end

function MWF!(
        results, domains, omegas, # modified in-place
        params,
        exteriorgrids,
        torigrids,
        interiorgrids,
        outercircles,
        innercircles,
        bdry
    )
    # save current parameters
    push!(results.params, params)

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

function main()
    # Load geometries
    fname = "2019-02-15-T-14-57-53-542__N-20_g-0.8000_p-0.7500__grids.bson"
    exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = load_geometry(fname)
    numfibres = length(outercircles)

    # Default parameters
    thetarange = range(0.0, stop = π/2, length = 5)
    Krange = [0.1, 0.5, 1.0]
    Drange = [100.0, 500.0]

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
    to_str(x) = @sprintf "%.4f" x
    params_to_str(θ,κ,D) = "N-$(numfibres)_alpha-$(to_str(rad2deg(θ)))_K-$(to_str(κ))_D-$(to_str(D))"

    # Parameter sweep
    results = MWFResults{Float64}(metadata = Dict(:TE => 10e-3))
    domains = []
    omegas = []

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
            MWF!(results, domains, omegas, btparams, exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry)

            BSON.bson(getnow() * "__" * paramstr * "__btparams.bson", Dict(:btparams => btparams))
            CSV.write(results, count)
        catch e
            @warn "error running simulation $count/$(length(paramlist))"; @warn e
        end
    end

    all_results = Dict(
        :results => results,
        :domains => domains,
        :omegas => omegas,
        :numfibres => numfibres,
        :thetarange => thetarange,
        :Krange => Krange,
        :Drange => Drange,
        :exteriorgrids => exteriorgrids,
        :torigrids => torigrids,
        :interiorgrids => interiorgrids,
        :outercircles => outercircles,
        :innercircles => innercircles,
        :bdry => bdry
    )

    return all_results
end

out = main()