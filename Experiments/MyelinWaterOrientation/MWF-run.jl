using Dates, BSON, CSV
include(joinpath(@__DIR__, "init.jl"))
mxcall(:cd, 0, pwd()) # change MATLAB path to current path

btparams = BlochTorreyParameters{Float64}(
    theta = π/2,
    AxonPDensity = 0.7,
    g_ratio = 0.8,
    D_Tissue = 10.0, #0.5, # [μm²/s]
    D_Sheath = 10.0, #0.5, # [μm²/s]
    D_Axon = 10.0, #0.5, # [μm²/s]
    K_perm = 1.0) #0.0 # [μm/s]

exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = creategrids(btparams; fname = "gridplot")

# TODO: For some reason, BSON has a bug with saving sets... just remove them for now
exterior, tori, interior = deepcopy(exteriorgrids), deepcopy(torigrids), deepcopy(interiorgrids)
for g in Iterators.flatten((exterior, tori, interior))
    g.cellsets = Dict(); g.nodesets = Dict(); g.facesets = Dict()
end

BSON.bson("$(getnow())__grids.bson", Dict(
    :exteriorgrids => exterior,
    :torigrids => tori,
    :interiorgrids => interior,
    :outercircles => outercircles,
    :innercircles => innercircles,
    :bdry => bdry
))

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

    # simpplot(getgrid.(myelindomains);
    #     newfigure = true,
    #     axis = mxaxis(bdry),
    #     facecol = omega)

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
                D_Sheath = D,
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
thetarange = range(0.0, stop = π/2, length = 2)
# Krange = [0.0, 10.0.^(-2:3)...]
# Drange = [10.0.^(0:3)..., 2000.0]
# Krange = [0.0, 1e-3, 5e-3, 10e-3, 25e-3, 50e-3, 100e-3, 250e-3, 500e-3, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
# Drange = [1e-2, 1e-1, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]
Krange = [1000.0]
Drange = [1e-3]

run_MWF!(Drange, Krange, thetarange, btparams, # loop params
    results, domains, omegas, # modified in-place
    exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry # other args
)
