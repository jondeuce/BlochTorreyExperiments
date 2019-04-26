# Initialization of packages
include(joinpath(@__DIR__, "init.jl")) # call "init.jl", located in the same directory as this file
mxcall(:cd, 0, pwd()) # change MATLAB path to current path for saving outputs
mxcall(:figure, 0) # bring up MATLAB figure gui
make_reproduce( # Creating backup file
    """
    include("BlochTorreyExperiments/Experiments/MyelinWaterOrientation/MWF-run.jl")
    """
)

using BSON, Dates
using StatsPlots
pyplot(size=(800,600), leg = false, grid = false, labels = nothing)
using MATLABPlots # NOTE: must use MATLABPlots AFTER loading pyplot backend

import DrWatson
using DrWatson: @dict, @ntuple
DrWatson.default_prefix(c) = MWFUtils.getnow()
gitdir() = realpath(joinpath(DrWatson.projectdir(), "../..")) * "/"

####
#### Parameters to sweep over
####

const sweepparams = Dict{Symbol,Any}(
    :theta => Vector(range(0.0, 90.0, length = 7)),
    :K     => [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    :D     => [10.0, 50.0, 100.0, 500.0]
)

####
#### Load geometries
####

storedgeomfile = "geom.bson"; # Default: look for "geom.bson" in current directory
geomfile = storedgeomfile;

if !isfile(geomfile)
    # storedgeomfile = joinpath(
    #     "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/kmg_geom_sweep_3",
    #     "2019-03-28-T-15-24-11-877__N-10_g-0.7500_p-0.7500__structs.bson" # 1.3k triangles, 1.2k points, Qmin = 0.3
    #     # "2019-03-28-T-15-26-44-544__N-10_g-0.8000_p-0.8300__structs.bson" # 4.7k triangles, 3.2k points, Qmin = 0.3
    #     # "2019-03-28-T-15-27-56-042__N-20_g-0.7500_p-0.7000__structs.bson" # 3.1k triangles, 2.6k points, Qmin = 0.3
    #     # "2019-03-28-T-15-33-59-628__N-20_g-0.8000_p-0.8000__structs.bson" #13.3k triangles, 9.2k points, Qmin = 0.3
    # )
    # storedgeomfile = joinpath(
    #     "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/kmg_geom_sweep_4",
    #     "2019-03-28-T-16-19-20-218__N-40_g-0.7500_p-0.8000__structs.bson" # 11.0k triangles, 8.6k points, Qmin = 0.3
    # )
    # storedgeomfile = joinpath(
    #     "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/kmg_geom_sweep_6",
    #     # "2019-03-29-T-10-47-05-945__N-40_g-0.7500_p-0.7000__structs.bson" #10k triangles, 8k points, Qmin = 0.4
    #     # "2019-03-29-T-12-19-17-694__N-40_g-0.8370_p-0.7500__structs.bson" #13k triangles, 10k points, Qmin = 0.4
    #     "2019-03-29-T-12-15-03-265__N-40_g-0.8000_p-0.8300__structs.bson" #28k triangles, 19k points, Qmin = 0.4
    # )
    # storedgeomfile = joinpath(
    #     "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/drwatson_geom_sweep_1/geom",
    #     "2019-04-24-T-18-33-57-731_density=0.75_gratio=0.78_numfibres=20.geom.bson" #12.8k triangles, 9.6k points, Qmin = 0.4
    #     # "2019-04-24-T-21-16-38-329_density=0.75_gratio=0.78_numfibres=35.geom.bson" #36.7k triangles, 25.3k points, Qmin = 0.4
    #     # "2019-04-24-T-17-54-24-004_density=0.75_gratio=0.78_numfibres=5.geom.bson" #3.4k triangles, 2.5k points, Qmin = 0.4
    # )
    storedgeomfile = joinpath(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/drwatson_geom_sweep_2/geom",
        # "2019-04-25-T-11-05-25-221_density=0.78_gratio=0.78_numfibres=10.geom.bson" #11.4k triangles, 7.8k points, Qmin = 0.4
        # "2019-04-25-T-11-59-59-400_density=0.78_gratio=0.75_numfibres=20.geom.bson" #20.8k triangles, 14.5k points, Qmin = 0.4
        "2019-04-25-T-15-13-27-738_density=0.78_gratio=0.75_numfibres=30.geom.bson" #38.7k triangles, 25.9k points, Qmin = 0.4
    )
    geomfile = cp(storedgeomfile, "geom.bson")
end

const geom = loadgeometry(geomfile);

####
#### Solver parameters and MWF models
####

const TE = 10e-3; # Echotime
const nTE = 32; # Number of echoes
const solverparams = Dict(
    :u0     => 1.0im,                   # Initial π/2 pulse (Default: Vec{2}((0.0,1.0)))
    :TE     => TE,                      # Echotime for MultiSpinEchoCallback (Default: 10e-3)
    :tspan  => TE .* (0, nTE),          # Solver time span (Default: (0.0, 320e-3); must start at zero)
    :reltol => 1e-8,
    :abstol => 0.0);

const nnlsparams = Dict(
    :TE          => TE,                 # Echotime of signal
    :nTE         => nTE,                # Number of echoes
    :nT2         => 120,                # Number of T2 points used for fitting the distribution
    :Threshold   => 0.0,                # Intensity threshold below which signal is ignored (zero for simulation)
    :RefConAngle => 180.0,              # Flip angle parameter for reconstruction
    :T2Range     => [5e-3, 2.0],        # Range of T2 values used
    :SPWin       => [10e-3, 40e-3],     # Small pool window, i.e. myelin peak
    :MPWin       => [40e-3, 200e-3],    # Intra/extracelluar water peak window
    :PlotDist    => false);             # Plot resulting distribution in MATLAB

const mwfmodels = [
    NNLSRegression(;nnlsparams...),
    TwoPoolMagnToMagn(TE = TE, nTE = nTE, fitmethod = :local),
    ThreePoolMagnToMagn(TE = TE, nTE = nTE, fitmethod = :local),
    ThreePoolCplxToMagn(TE = TE, nTE = nTE, fitmethod = :local),
    ThreePoolCplxToCplx(TE = TE, nTE = nTE, fitmethod = :local)];

####
#### Default BlochTorreyParameters
####

const defaultbtparams = BlochTorreyParameters{Float64}(
    theta = π/2,
    AxonPDensity = 0.8,
    g_ratio = 0.8,
    D_Tissue = 500.0, # [μm²/s]
    D_Sheath = 500.0, # [μm²/s]
    D_Axon = 500.0, # [μm²/s]
    K_perm = 0.5, # [μm/s]
    # ChiI = 10 * BlochTorreyParameters{Float64}().ChiI, # amplify myelin susceptibility for testing
    # ChiA = 10 * BlochTorreyParameters{Float64}().ChiA, # amplify myelin susceptibility for testing
);

####
#### Save metadata
####

DrWatson.@tagsave(
    MWFUtils.getnow() * ".metadata.bson",
    deepcopy(@dict(sweepparams, defaultbtparams, storedgeomfile, TE, nTE, solverparams, nnlsparams, mwfmodels)),
    true, gitdir())

####
#### Simulation functions
####

function runsolve(btparams)
    # Unpack geometry, create myelin domains, and create omegafield
    exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = geom
    ferritins = Vec{3,floattype(bdry)}[]
    
    myelinprob, myelinsubdomains, myelindomains = createdomains(btparams,
        exteriorgrids, torigrids, interiorgrids,
        outercircles, innercircles, ferritins, typeof(solverparams[:u0]))
        
    # Solve Bloch-Torrey equation and plot
    sols = solveblochtorrey(myelinprob, myelindomains; solverparams...)
    
    return @ntuple(sols, myelinprob, myelinsubdomains, myelindomains)
end

function runsimulation!(results, params)
    @unpack theta, K, D = params
    btparams = BlochTorreyParameters(defaultbtparams;
        theta = deg2rad(theta),
        K_perm = K,
        D_Tissue = D, D_Sheath = D, D_Axon = D)
    sols, myelinprob, myelinsubdomains, myelindomains = runsolve(btparams)

    # Common filename without suffix
    fname = DrWatson.savename(params)
    titleparamstr = DrWatson.savename("", params; connector = ", ")
    
    # Save btparams and ode solutions
    try
        DrWatson.@tagsave(
            "sol/" * fname * ".btparams.bson",
            deepcopy(@dict(btparams)),
            true, gitdir())
    catch e
        @warn "Error saving BlochTorreyParameters"
        @warn sprint(showerror, e, catch_backtrace())
    end

    for (i,sol) in enumerate(sols)
        try
            DrWatson.@tagsave(
                "sol/" * fname * ".region$(i).odesolution.bson",
                deepcopy(@dict(sol)),
                true, gitdir())
        catch e
            @warn "Error saving ODE solution in region #$(i)"
            @warn sprint(showerror, e, catch_backtrace())
        end
    end

    # Compute MWF values
    mwfvalues, signals = compareMWFmethods(sols, myelindomains,
        geom.outercircles, geom.innercircles, geom.bdry;
        models = mwfmodels)

    # Update results struct and return
    push!(results[:params], btparams)
    push!(results[:myelinprobs], myelinprob)
    push!(results[:myelinsubdomains], myelinsubdomains)
    push!(results[:myelindomains], myelindomains)
    push!(results[:omegas], calcomega(myelinprob, myelinsubdomains))
    push!(results[:sols], sols)
    push!(results[:signals], signals)
    push!(results[:mwfvalues], mwfvalues)

    # Plot and save various figures
    try
        mxplotomega(myelinprob, myelindomains, myelinsubdomains, geom.bdry;
            titlestr = "Frequency Map (theta = $(round(theta; digits=3)) deg)",
            fname = "omega/" * fname * ".omega")
    catch e
        @warn "Error plotting omega"
        @warn sprint(showerror, e, catch_backtrace())
    end

    try
        mxplotmagnitude(sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Magnitude (" * titleparamstr * ")",
            fname = "mag/" * fname * ".magnitude")
    catch e
        @warn "Error plotting magnetization magnitude"
        @warn sprint(showerror, e, catch_backtrace())
    end
    
    try
        nnlsindex = findfirst(m->m isa NNLSRegression, mwfmodels)
        if !(nnlsindex == nothing)
            plotSEcorr(sols, btparams, myelindomains;
                mwftrue = getmwf(geom.outercircles, geom.innercircles, geom.bdry),
                opts = mwfmodels[nnlsindex], fname = "t2dist/" * fname * ".t2dist.SEcorr")
        end
    catch e
        @warn "Error plotting SEcorr T2 distribution"
        @warn sprint(showerror, e, catch_backtrace())
    end

    try
        if !isempty(mwfmodels)
            plotbiexp(sols, btparams, myelindomains,
                geom.outercircles, geom.innercircles, geom.bdry;
                titlestr = "Signal Magnitude (" * titleparamstr * ")",
                opts = mwfmodels[1], fname = "sig/" * fname * ".signalmag")
        end
    catch e
        @warn "Error plotting biexponential"
        @warn sprint(showerror, e, catch_backtrace())
    end

    return results
end

function main()
    # Make subfolders
    mkpath.(("mag", "t2dist", "sig", "omega", "mwfplots", "sol"))

    # Initialize results
    results = blank_results_dict()
    results[:geom] = geom

    all_params = DrWatson.dict_list(sweepparams)
    all_params = sort(all_params; by = d -> (d[:D], d[:K], d[:theta]))
    for (i,params) in enumerate(all_params)
        params = convert(Dict{Symbol,Any}, params)
        try
            @info "Running simulation $i/$(length(all_params)) at $(Dates.now()): $(DrWatson.savename("", params; connector = ", "))"
            runsimulation!(results, params)
        catch e
            if e isa InterruptException
                @warn "Parameter sweep interrupted by user. Breaking out of loop and returning current results..."
                break
            else
                @warn "Error running simulation $i/$(length(all_params))"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return results
end

####
#### Run sweep
####

results = main();
@unpack sols, myelindomains, params, signals, mwfvalues, geom, myelinsubdomains, myelinprobs, omegas = results;
@unpack exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = geom;

####
#### Plot and save derived quantities from results
####

try
    plotMWF(results; disp = false, fname = "mwfplots/" * MWFUtils.getnow() * ".mwf")
catch e
    @warn "Error plotting MWF."
    @warn sprint(showerror, e, catch_backtrace())
end

try
    @unpack params = results
    BSON.bson(MWFUtils.getnow() * ".allbtparams.bson", deepcopy(@dict(params)))
catch e
    @warn "Error saving all BlochTorreyParameter's"
    @warn sprint(showerror, e, catch_backtrace())
end

try
    @unpack signals, mwfvalues = results
    BSON.bson(MWFUtils.getnow() * ".measurables.bson", deepcopy(@dict(signals, mwfvalues)))
catch e
    @warn "Error saving measurables"
    @warn sprint(showerror, e, catch_backtrace())
end