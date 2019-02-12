include(joinpath(@__DIR__, "init.jl")) # call "init.jl", located in the same directory as this file
mxcall(:cd, 0, pwd()) # change MATLAB path to current path for saving outputs

using Dates, BSON, CSV

btparams = BlochTorreyParameters{Float64}(
    theta = π/2,
    AxonPDensity = 0.8,
    g_ratio = 0.8,
    D_Tissue = 500.0, #0.5, # [μm²/s]
    D_Sheath = 50.0, #0.5, # [μm²/s]
    D_Axon = 500.0, #0.5, # [μm²/s]
    K_perm = 1.0) #0.0 # [μm/s]

exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = creategrids(
    btparams;
    N = 50, # number of fibres
    fname = "gridplot" # file name prefix for saving MATLAB figure
)
mxcall(:close, 0, "all") # close all figures

# # TODO: For some reason, BSON has a bug with saving sets... just remove them for now
# let exterior, tori, interior = deepcopy(exteriorgrids), deepcopy(torigrids), deepcopy(interiorgrids)
#     for g in Iterators.flatten((exterior, tori, interior))
#         g.cellsets = Dict(); g.nodesets = Dict(); g.facesets = Dict() # remove sets
#     end

#     BSON.bson("$(getnow())__grids.bson", Dict(
#         :exteriorgrids => exterior,
#         :torigrids => tori,
#         :interiorgrids => interior,
#         :outercircles => outercircles,
#         :innercircles => innercircles,
#         :bdry => bdry
#     ))
# end

# Temporary BSON fix implemented
try
    BSON.bson("$(getnow())__grids.bson", Dict(
        :exteriorgrids => exteriorgrids,
        :torigrids => torigrids,
        :interiorgrids => interiorgrids,
        :outercircles => outercircles,
        :innercircles => innercircles,
        :bdry => bdry
    ))
catch e
    @warn "Error saving geometries"
end

function MWF!(results, domains, omegas, # modified in-place
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

    # simpplot(getgrid.(myelindomains); newfigure = true, axis = mxaxis(bdry), facecol = omega)

    sols = solveblochtorrey(myelinprob, myelindomains)
    push!(results.sols, sols)

    paramstr = "theta = $(rad2deg(params.theta)) deg, D = $(params.D_Tissue) um2/s, K = $(params.K_perm) um/s"
    plotmagnitude(sols, params, myelindomains, bdry; titlestr = "Magnitude: " * paramstr, fname = "magnitude")
    plotSEcorr(sols, params, myelindomains, fname = "SEcorr")
    plotbiexp(sols, params, myelindomains, outercircles, innercircles, bdry; titlestr = "Signal: " * paramstr, fname = "signal")

    mwfvalues = compareMWFmethods(sols, myelindomains, outercircles, innercircles, bdry)
    push!(results.mwfvalues, mwfvalues)

    return results, domains, omegas
end

function run_MWF!(Drange, Krange, thetarange, btparams, results, domains, omegas, exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry)
    count = 0
    totalcount = length(Drange) * length(Krange) * length(thetarange)

    for D in Drange for K in Krange for theta in thetarange
        count += 1

        try
            println("\n\n")
            println("---- SIMULATION $count/$totalcount: $(Dates.now()) ----")
            @show rad2deg(theta), D, K
            println("\n\n")

            # Create new set of parameters
            params = BlochTorreyParameters(btparams;
                theta = theta,
                K_perm = K,
                D_Tissue = D,
                D_Sheath = D/10,
                D_Axon = D)
            MWF!(results, domains, omegas, params, exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry)

            BSON.bson("$(getnow())__params_$count.bson", Dict(:params => params))
            CSV.write(results, count)
        catch e
            @warn "error running simulation $count/$totalcount"; @warn e
        end

    end end end

    nothing
end

results = MWFResults{Float64}(metadata = Dict(:TE => 10e-3))
domains = []
omegas = []
# thetarange = [π/2]
thetarange = range(0.0, stop = π/2, length = 5)
# Krange = [0.0]
# Krange = [0.0, 0.1, 1.0, 10.0, 50.0]
Krange = [0.1, 0.5, 1.0]
# Krange = [0.0, 10.0.^(-2:3)...]
# Krange = [0.0, 1e-3, 5e-3, 10e-3, 25e-3, 50e-3, 100e-3, 250e-3, 500e-3, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
# Drange = [1.0]
# Drange = [1e-10]
# Drange = [10.0.^(0:3)..., 2000.0]
Drange = [100.0, 500.0]
# Drange = [1e-2, 1e-1, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]

run_MWF!(Drange, Krange, thetarange, btparams, # loop params
    results, domains, omegas, # modified in-place
    exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry) # other args
