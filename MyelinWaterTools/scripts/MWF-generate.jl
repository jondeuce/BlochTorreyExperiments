# NOTE: must load pyplot backend BEFORE loading MATLAB in init.jl
using StatsPlots, BSON, Dates
pyplot(size=(1200,900))
# gr(size=(1200,900)) #TODO

# Initialize project packages
include(joinpath(@__DIR__, "../init.jl")) # call "init.jl", located in the same directory as this file
mxcall(:cd, 0, pwd()) # change MATLAB path to current path for saving outputs
mxcall(:figure, 0) # bring up MATLAB figure gui
make_reproduce( # Creating backup file
    """
    include("BlochTorreyExperiments/MyelinWaterTools/scripts/MWF-generate.jl")
    """
)

import DrWatson
using DrWatson: @dict, @ntuple
gitdir() = realpath(joinpath(DrWatson.projectdir(), "..")) * "/"

####
#### Geometries to sweep over
####

geomfiles = vcat(
    joinpath.(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/kmg_geom_sweep_3",
        [
            "2019-03-28-T-15-24-11-877__N-10_g-0.7500_p-0.7500__structs.bson" # 1.3k triangles, 1.2k points, Qmin = 0.3
            "2019-03-28-T-15-26-44-544__N-10_g-0.8000_p-0.8300__structs.bson" # 4.7k triangles, 3.2k points, Qmin = 0.3
            "2019-03-28-T-15-27-56-042__N-20_g-0.7500_p-0.7000__structs.bson" # 3.1k triangles, 2.6k points, Qmin = 0.3
            "2019-03-28-T-15-33-59-628__N-20_g-0.8000_p-0.8000__structs.bson" #13.3k triangles, 9.2k points, Qmin = 0.3
        ]
    ),
    joinpath.(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/kmg_geom_sweep_4",
        [
            "2019-03-28-T-16-19-20-218__N-40_g-0.7500_p-0.8000__structs.bson" # 11.0k triangles, 8.6k points, Qmin = 0.3
        ]
    ),
    joinpath.(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/kmg_geom_sweep_6",
        [
            "2019-03-29-T-10-47-05-945__N-40_g-0.7500_p-0.7000__structs.bson" #10k triangles, 8k points, Qmin = 0.4
            "2019-03-29-T-12-19-17-694__N-40_g-0.8370_p-0.7500__structs.bson" #13k triangles, 10k points, Qmin = 0.4
            "2019-03-29-T-12-15-03-265__N-40_g-0.8000_p-0.8300__structs.bson" #28k triangles, 19k points, Qmin = 0.4
        ]
    ),
    joinpath.(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/drwatson_geom_sweep_1/geom",
        [
            "2019-04-24-T-18-33-57-731_density=0.75_gratio=0.78_numfibres=20.geom.bson" #12.8k triangles, 9.6k points, Qmin = 0.4
            "2019-04-24-T-21-16-38-329_density=0.75_gratio=0.78_numfibres=35.geom.bson" #36.7k triangles, 25.3k points, Qmin = 0.4
            "2019-04-24-T-17-54-24-004_density=0.75_gratio=0.78_numfibres=5.geom.bson" #3.4k triangles, 2.5k points, Qmin = 0.4
        ]
    ),
    joinpath.(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/drwatson_geom_sweep_2/geom",
        [
            "2019-04-25-T-11-05-25-221_density=0.78_gratio=0.78_numfibres=10.geom.bson" #11.4k triangles, 7.8k points, Qmin = 0.4
            "2019-04-25-T-11-59-59-400_density=0.78_gratio=0.75_numfibres=20.geom.bson" #20.8k triangles, 14.5k points, Qmin = 0.4
            "2019-04-25-T-15-13-27-738_density=0.78_gratio=0.75_numfibres=30.geom.bson" #38.7k triangles, 25.9k points, Qmin = 0.4
        ]
    )
)

function copy_and_load_geomfiles(storedgeomfilenames)
    mkpath("geom")
    geoms = []
    for (i,geomfile) in enumerate(storedgeomfilenames)
        # load geom file and store locally
        geom = loadgeometry(geomfile)
        DrWatson.@tagsave(
            "geom/" * MWFUtils.getnow() * ".geom$i.bson",
            deepcopy(@dict(geomfile, geom)),
            true, gitdir())
        push!(geoms, geom)
    end
    return geoms
end
const geometries = copy_and_load_geomfiles(geomfiles);

####
#### Default solver parameters and MWF models
####

const default_TE = 10e-3; # Echotime
const default_nTE = 32; # Number of echoes
const default_solverparams_dict = Dict(
    :u0     => 1.0im,                          # Initial π/2 pulse (Default: Vec{2}((0.0,1.0)))
    :TE     => default_TE,                     # Echotime for MultiSpinEchoCallback (Default: 10e-3)
    :tspan  => default_TE .* (0, default_nTE), # Solver time span (Default: (0.0, 320e-3); must start at zero)
    :reltol => 1e-8,
    :abstol => 0.0);

const default_nnlsparams_dict = Dict(
    :TE          => default_TE,      # Echotime of signal
    :nTE         => default_nTE,     # Number of echoes
    :nT2         => 120,             # Number of T2 points used for fitting the distribution
    :Threshold   => 0.0,             # Intensity threshold below which signal is ignored (zero for simulation)
    :RefConAngle => 180.0,           # Flip angle parameter for reconstruction
    :T2Range     => [5e-3, 2000e-3], # Range of T2 values used
    :SPWin       => [10e-3, 40e-3],  # Small pool window, i.e. myelin peak
    :MPWin       => [40e-3, 200e-3], # Intra/extracelluar water peak window
    :PlotDist    => false);          # Plot resulting distribution in MATLAB

const default_mwfmodels = [
    #NNLSRegression(;default_nnlsparams_dict...),
    TwoPoolMagnToMagn(TE = default_TE, nTE = default_nTE, fitmethod = :local),
    ThreePoolMagnToMagn(TE = default_TE, nTE = default_nTE, fitmethod = :local),
    ThreePoolCplxToMagn(TE = default_TE, nTE = default_nTE, fitmethod = :local),
    ThreePoolCplxToCplx(TE = default_TE, nTE = default_nTE, fitmethod = :local)];
const default_mwfmodels_dict = Dict(map(m -> Symbol(typeof(m)) => Dict(m), default_mwfmodels)...)

####
#### Default BlochTorreyParameters
####

const default_btparams = BlochTorreyParameters{Float64}(
    theta = π/2,
    D_Tissue = 500.0, # [μm²/s]
    D_Sheath = 500.0, # [μm²/s]
    D_Axon = 500.0, # [μm²/s]
    K_perm = 0.5, # [μm/s]
);
const default_btparams_dict = Dict(default_btparams)

####
#### Parameters to sweep over
####

linearsampler(a,b) = a + rand() * (b - a)
unitrangesampler(a,b) = rand(a:b)
log10sampler(a,b) = 10^linearsampler(log10(a), log10(b))
acossampler() = rad2deg(acos(rand()))

const paramsampler_settings = Dict{Symbol,Any}(
    :theta => (sampler = :acossampler,      args = ()),
    :K     => (sampler = :log10sampler,     args = (lb = 1e-3, ub = 1.0)),#0.05)),
    :Dtiss => (sampler = :log10sampler,     args = (lb = 10.0, ub = 500.0)),#25.0)),
    :Dmye  => (sampler = :log10sampler,     args = (lb = 10.0, ub = 500.0)),#25.0)),
    :Dax   => (sampler = :log10sampler,     args = (lb = 10.0, ub = 500.0)),#25.0)),
    :TE    => (sampler = :linearsampler,    args = (lb = 5e-3, ub = 15e-3)),
    :nTE   => (sampler = :unitrangesampler, args = (lb = 24,   ub = 48)),
)
paramsampler() = Dict{Symbol,Union{Float64,Int}}(
    k => eval(Expr(:call, v.sampler, v.args...))
    for (k,v) in paramsampler_settings)

####
#### Save metadata
####

DrWatson.@tagsave(
    MWFUtils.getnow() * ".metadata.bson",
    deepcopy(@dict(paramsampler_settings, geomfiles, default_mwfmodels_dict, default_btparams_dict, default_solverparams_dict, default_nnlsparams_dict, default_TE, default_nTE)),
    true, gitdir())

####
#### Simulation functions
####

function runsolve(btparams, params, geom)
    # Unpack solver settings
    solverparams_dict = copy(default_solverparams_dict)
    solverparams_dict[:TE] = params[:TE]
    solverparams_dict[:nTE] = params[:nTE]
    solverparams_dict[:tspan] = params[:TE] .* (0, params[:nTE])
    
    # @show solverparams_dict #TODO

    # Unpack geometry, create myelin domains, and create omegafield
    exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = geom
    ferritins = Vec{3,floattype(bdry)}[]
    
    myelinprob, myelinsubdomains, myelindomains = createdomains(btparams,
        exteriorgrids, torigrids, interiorgrids,
        outercircles, innercircles, ferritins, typeof(solverparams_dict[:u0]))
    
    # Solve Bloch-Torrey equation and plot
    sols = solveblochtorrey(myelinprob, myelindomains; solverparams_dict...)
    
    return @ntuple(sols, myelinprob, myelinsubdomains, myelindomains)
end

function runsimulation!(results, params, geom)
    @unpack theta, K, Dtiss, Dmye, Dax = params
    density = intersect_area(geom.outercircles, geom.bdry) / area(geom.bdry)
    gratio = radius(geom.innercircles[1]) / radius(geom.outercircles[1])

    # @show density, gratio #TODO

    btparams = BlochTorreyParameters(default_btparams;
        theta = deg2rad(theta),
        K_perm = K,
        D_Tissue = Dtiss,
        D_Sheath = Dmye,
        D_Axon = Dax,
        AxonPDensity = density,
        g_ratio = gratio,
    )
    sols, myelinprob, myelinsubdomains, myelindomains = runsolve(btparams, params, geom)

    # @show params #TODO
    # @show sols[1].t ./ params[:TE] #TODO

    # Common filename without suffix
    fname = DrWatson.savename(MWFUtils.getnow(), params)
    titleparamstr = DrWatson.savename("", params; connector = ", ")
    
    # Save btparams and ode solutions
    try
        DrWatson.@tagsave(
            "btparams/" * fname * ".btparams.bson",
            deepcopy(@dict(btparams)),
            true, gitdir())
    catch e
        @warn "Error saving BlochTorreyParameters"
        @warn sprint(showerror, e, catch_backtrace())
    end

    # Compute MWF values
    mwfmodels = map(default_mwfmodels) do model
        typeof(model)(model; TE = params[:TE], nTE = params[:nTE])
    end
    # @show mwfmodels #TODO
    mwfvalues, signals = compareMWFmethods(sols, myelindomains,
        geom.outercircles, geom.innercircles, geom.bdry;
        models = mwfmodels)

    
    # Update results struct and return
    push!(results[:params], btparams)
    push!(results[:signals], signals)
    push!(results[:mwfvalues], mwfvalues)

    # Plot and save various figures
    # try
    #     mxplotomega(myelinprob, myelindomains, myelinsubdomains, geom.bdry;
    #         titlestr = "Frequency Map (theta = $(round(theta; digits=3)) deg)",
    #         fname = "omega/" * fname * ".omega")
    # catch e
    #     @warn "Error plotting omega"
    #     @warn sprint(showerror, e, catch_backtrace())
    # end

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

function main(;iters::Int = typemax(Int))
    # Make subfolders
    mkpath.(("mag", "t2dist", "sig", "omega", "mwfplots", "btparams"))

    # Initialize results
    results = blank_results_dict()

    all_params = (paramsampler() for _ in 1:iters)
    for (i,params) in enumerate(all_params)
        params = convert(Dict{Symbol,Any}, params)

        # geomnumber = 1 #TODO
        geomnumber = rand(1:length(geometries))
        geom = geometries[geomnumber]
        try
            @info "Running simulation $i/$(length(all_params)) at $(Dates.now()):"
            @info "    Sweep parameters:  " * DrWatson.savename("", params; connector = ", ")
            @info "    Geometry info:     Geom #$geomnumber - " * geomfiles[geomnumber]
            
            tic = Dates.now()
            runsimulation!(results, params, geom)
            toc = Dates.now()
            Δt = Dates.canonicalize(Dates.CompoundPeriod(toc - tic))

            @info "Elapsed simulation time: $Δt"
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

results = main(); #iters = 25 TODO
@unpack sols, myelindomains, params, signals, mwfvalues, geom, myelinsubdomains, myelinprobs, omegas = results;

####
#### Plot and save derived quantities from results
####

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

try
    plotMWFvsMethod(results; disp = false, fname = "mwfplots/" * MWFUtils.getnow() * ".mwf")
catch e
    @warn "Error plotting MWF."
    @warn sprint(showerror, e, catch_backtrace())
end
