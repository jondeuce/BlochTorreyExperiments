# Activate project and load packages for this script
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "../initpaths.jl"))
Pkg.instantiate()

# NOTE: must load pyplot backend BEFORE loading MATLAB in init.jl
using StatsPlots
pyplot(size=(1200,900))
using GlobalUtils
using MWFUtils
mxcall(:cd, 0, pwd()) # Set MATLAB path (Note: pwd(), not @__DIR__)
const SIM_START_TIME = MWFUtils.getnow()

# Create reproduce file
make_reproduce( # Creating backup file
    """
    include("BlochTorreyExperiments/MyelinWaterTools/scripts/MWF-generate.jl")
    """;
    fname = SIM_START_TIME * ".reproduce.jl"
)

# DrWatson package for tagged saving
gitdir() = realpath(joinpath(DrWatson.projectdir(), "..")) * "/"

####
#### Geometries to sweep over
####

function copy_and_load_geomfiles!(
        geomfiles::AbstractVector{String},
        maxnnodes::Int = typemax(Int)
    )
    mkpath("geom")
    geoms = []
    storedgeomfilenames = filter(s->endswith(s, ".bson"), readdir("geom"))

    skipped_geoms = Int[]
    for (i,geomfile) in enumerate(geomfiles)
        # load geom file and store locally
        geom = loadgeometry(geomfile)
        nnodes = sum(JuAFEM.getnnodes, geom.exteriorgrids) +
                 sum(JuAFEM.getnnodes, geom.torigrids) +
                 sum(JuAFEM.getnnodes, geom.interiorgrids)
        if nnodes > maxnnodes
            # Geometry is too large; skip it
            push!(skipped_geoms, i)
            continue
        end
        if basename(geomfile) ∉ storedgeomfilenames
            DrWatson.@tagsave(
                "geom/" * basename(geomfile),
                deepcopy(@dict(geomfile, geom)),
                true, gitdir())
        end
        push!(geoms, geom)
    end

    # Update geomfiles, removing skipped geometries from list
    deleteat!(geomfiles, skipped_geoms)

    return geoms
end

# Load geometries with at most `maxnnodes` number of nodes to avoid exceedingly long simulations
const geombasepath = "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterLearning/geometries/periodic-packed-fibres-1/geom"
const geomfiles = realpath.(joinpath.(geombasepath, readdir(geombasepath)))
const maxnnodes = 5000; #TODO
const geometries = copy_and_load_geomfiles!(geomfiles, 5000);

####
#### Default solver parameters and MWF models
####

const default_TE = 10e-3; # Echotime
const default_TR = 1000e-3; # Repetition time
const default_nTE = 32; # Number of echoes
const default_nTR = 1; # Number of repetitions
const default_tspan = (0.0, default_nTE * default_TE + (default_nTR - 1) * default_TR); # timespan
const default_solverparams_dict = Dict(
    :u0          => Vec3d((0,0,1)),  # Initial magnetization; should be [0,-1] for 2D (π/2 pulse) or [0,0,1] for 3D (steady-state)
    :flipangle   => Float64(π),      # Flip angle for CPMGCallback
    :refocustype => :x,              # Refocusing pulse type (Default: :xyx)
    # :refocustype => :y,            # Refocusing pulse type (Default: :xyx)
    # :refocustype => :xyx,          # Refocusing pulse type (Default: :xyx)
    :TE          => default_TE,      # Echotime for CPMGCallback (Default: 10e-3)
    :TR          => default_TR,      # Repetition time for CPMGCallback (Default: 1000e-3)
    :nTE         => default_nTE,     # Number of echoes for CPMGCallback (Default: 32)
    :nTR         => default_nTR,     # Number of repetitions for CPMGCallback (Default: 1)
    :tspan       => default_tspan,   # Solver time span (Default: (0.0, 320e-3); must start at zero)
    :reltol      => 1e-8,
    :abstol      => 0.0);

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
    # NNLSRegression(;default_nnlsparams_dict...), #TODO
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
rangesampler(a,b,s=1) = rand(a:s:b)
log10sampler(a,b) = 10^linearsampler(log10(a), log10(b))
acossampler(a,b) = acosd(linearsampler(cosd(b), cosd(a)))

const sweepparamsampler_settings = Dict{Symbol,Any}(
    :theta  => (sampler = :acossampler,   args = (lb = 0.0,     ub = 90.0)),
    :alpha  => (sampler = :linearsampler, args = (lb = 140.0,   ub = 180.0)),
    :K      => (sampler = :log10sampler,  args = (lb = 1e-3,    ub = 10.0)),
    :Dtiss  => (sampler = :log10sampler,  args = (lb = 100.0,   ub = 500.0)),
    :Dmye   => (sampler = :log10sampler,  args = (lb = 100.0,   ub = 500.0)),
    :Dax    => (sampler = :log10sampler,  args = (lb = 100.0,   ub = 500.0)),
    :FRD    => (sampler = :linearsampler, args = (lb = 0.5,     ub = 0.5)), #TODO
    :TE     => (sampler = :linearsampler, args = (lb = 5e-3,    ub = 15e-3)),
    :TR     => (sampler = :linearsampler, args = (lb = 800e-3,  ub = 1200e-3)), #TODO
    :nTE    => (sampler = :rangesampler,  args = (lb = 24, ub = 48, step = 2)), #Must be an even number
    :nTR    => (sampler = :rangesampler,  args = (lb = 1,       ub = 1)),
    :T2sp   => (sampler = :linearsampler, args = (lb = 10e-3,   ub = 20e-3)),
    :T2lp   => (sampler = :linearsampler, args = (lb = 50e-3,   ub = 80e-3)),
    :T2tiss => (sampler = :linearsampler, args = (lb = 50e-3,   ub = 80e-3)),
    :T1sp   => (sampler = :linearsampler, args = (lb = 150e-3,  ub = 250e-3)),
    :T1lp   => (sampler = :linearsampler, args = (lb = 949e-3,  ub = 1219e-3)), #3-sigma range for T1 = 1084 +/- 45
    :T1tiss => (sampler = :linearsampler, args = (lb = 949e-3,  ub = 1219e-3)), #3-sigma range for T1 = 1084 +/- 45
)
# const sweepparamsampler_settings = Dict{Symbol,Any}( #TODO testing settings
#     :Dax    => (sampler = :log10sampler,  args = (lb = 345.087, ub = 345.087)),
#     :Dmye   => (sampler = :log10sampler,  args = (lb = 242.866, ub = 242.866)),
#     :Dtiss  => (sampler = :log10sampler,  args = (lb = 300.332, ub = 300.332)),
#     :FRD    => (sampler = :linearsampler, args = (lb = 0.306,   ub = 0.306)),
#     :K      => (sampler = :log10sampler,  args = (lb = 0.010,   ub = 0.010)),
#     :T1lp   => (sampler = :linearsampler, args = (lb = 1.134,   ub = 1.134)),
#     :T1sp   => (sampler = :linearsampler, args = (lb = 0.230,   ub = 0.230)),
#     :T1tiss => (sampler = :linearsampler, args = (lb = 1.075,   ub = 1.075)),
#     :T2lp   => (sampler = :linearsampler, args = (lb = 0.051,   ub = 0.051)),
#     :T2sp   => (sampler = :linearsampler, args = (lb = 0.013,   ub = 0.013)),
#     :T2tiss => (sampler = :linearsampler, args = (lb = 0.068,   ub = 0.068)),
#     :TE     => (sampler = :linearsampler, args = (lb = 0.009,   ub = 0.009)),
#     :TR     => (sampler = :linearsampler, args = (lb = 0.977,   ub = 0.977)),
#     :alpha  => (sampler = :linearsampler, args = (lb = 180.0,   ub = 180.0)),
#     :nTE    => (sampler = :rangesampler,  args = (lb =    45,   ub =    45)),
#     :nTR    => (sampler = :rangesampler,  args = (lb =     1,   ub =     1)),
#     :theta  => (sampler = :acossampler,   args = (lb = 81.04,   ub = 81.04)),
# )
sweepparamsampler() = Dict{Symbol,Union{Float64,Int}}(
    k => eval(Expr(:call, v.sampler, v.args...))
    for (k,v) in sweepparamsampler_settings)

####
#### Save metadata
####

DrWatson.@tagsave(
    SIM_START_TIME * ".metadata.bson",
    deepcopy(@dict(sweepparamsampler_settings, geomfiles, default_mwfmodels_dict, default_btparams_dict, default_solverparams_dict, default_nnlsparams_dict, default_TE, default_nTE)),
    true, gitdir())

####
#### Simulation functions
####

function runsolve(btparams, sweepparams, geom)
    # Unpack solver settings
    solverparams_dict = copy(default_solverparams_dict)
    solverparams_dict[:TE] = sweepparams[:TE]
    solverparams_dict[:TR] = sweepparams[:TR]
    solverparams_dict[:nTE] = sweepparams[:nTE]
    solverparams_dict[:nTR] = sweepparams[:nTR]
    solverparams_dict[:tspan] = (0.0, sweepparams[:nTE] * sweepparams[:TE] + (sweepparams[:nTR] - 1) * sweepparams[:TR])
    solverparams_dict[:flipangle] = deg2rad(sweepparams[:alpha])

    # Unpack geometry, create myelin domains, and create omegafield
    @unpack exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = geom
    ferritins = Vec{3,floattype(bdry)}[]
    
    @unpack myelinprob, myelinsubdomains, myelindomains = createdomains(btparams,
        exteriorgrids, torigrids, interiorgrids,
        outercircles, innercircles, ferritins, typeof(solverparams_dict[:u0]))
    
    # Solve Bloch-Torrey equation and plot
    sols = solveblochtorrey(myelinprob, myelindomains; solverparams_dict...)
    
    return @ntuple(sols, myelinprob, myelinsubdomains, myelindomains, solverparams_dict)
end

function runsimulation!(results, sweepparams, geom)
    # @unpack alpha, theta, K, Dtiss, Dmye, Dax, FRD, TE, TR, nTE, nTR, T2sp, T2lp, T2tiss, T1sp, T1lp, T1tiss = sweepparams
    density = intersect_area(geom.outercircles, geom.bdry) / area(geom.bdry)
    gratio = radius(geom.innercircles[1]) / radius(geom.outercircles[1])

    btparams = BlochTorreyParameters(default_btparams;
        theta = deg2rad(sweepparams[:theta]),
        K_perm = sweepparams[:K],
        D_Tissue = sweepparams[:Dtiss],
        D_Sheath = sweepparams[:Dmye],
        D_Axon = sweepparams[:Dax],
        FRD_Sheath = sweepparams[:FRD],
        R2_sp = inv(sweepparams[:T2sp]),
        R2_lp = inv(sweepparams[:T2lp]),
        R2_Tissue = inv(sweepparams[:T2tiss]),
        R1_sp = inv(sweepparams[:T1sp]),
        R1_lp = inv(sweepparams[:T1lp]),
        R1_Tissue = inv(sweepparams[:T1tiss]),
        AxonPDensity = density,
        g_ratio = gratio,
    )
    sols, myelinprob, myelinsubdomains, myelindomains, solverparams_dict = runsolve(btparams, sweepparams, geom)
    
    # Sample solution signals
    dt = sweepparams[:TE]/10 # TODO
    tpoints = cpmg_savetimes(sols[1].prob.tspan, dt, sweepparams[:TE], sweepparams[:TR], sweepparams[:nTE], sweepparams[:nTR]) #TODO should be exported
    signals = calcsignal(sols, tpoints, myelindomains)

    # Common filename without suffix
    curr_time = MWFUtils.getnow()
    fname = DrWatson.savename(curr_time, sweepparams)
    titleparamstr = wrap_string(DrWatson.savename("", sweepparams; connector = ", "), 50, ", ")
    
    # Compare MWF values
    mwfvalues, mwfmodels = nothing, nothing
    try
        mwfmodels = map(default_mwfmodels) do model
            if model isa NNLSRegression
                typeof(model)(model; TE = sweepparams[:TE], nTE = sweepparams[:nTE], RefConAngle = sweepparams[:alpha])
            else
                typeof(model)(model; TE = sweepparams[:TE], nTE = sweepparams[:nTE])
            end
        end
        mwfvalues, _ = compareMWFmethods(sols, myelindomains,
            geom.outercircles, geom.innercircles, geom.bdry;
            models = mwfmodels)
    catch e
        @warn "Error comparing MWF methods"
        @warn sprint(showerror, e, catch_backtrace())
    end

    # Update results struct and return
    push!(results[:btparams], btparams)
    push!(results[:solverparams_dict], solverparams_dict)
    push!(results[:sweepparams], sweepparams)
    push!(results[:tpoints], tpoints)
    push!(results[:signals], signals)
    # push!(results[:sols], sols) #TODO
    # push!(results[:myelinprob], myelinprob) #TODO
    # push!(results[:myelinsubdomains], myelinsubdomains) #TODO
    # push!(results[:myelindomains], myelindomains) #TODO
    push!(results[:mwfvalues], mwfvalues)

    # Save measurables
    try
        btparams_dict = Dict(btparams)
        DrWatson.@tagsave(
            "measurables/" * fname * ".measurables.bson",
            deepcopy(@dict(btparams_dict, solverparams_dict, sweepparams, tpoints, signals, mwfvalues)),
            true, gitdir())
    catch e
        @warn "Error saving measurables"
        @warn sprint(showerror, e, catch_backtrace())
    end

    # # Save solution as vtk file
    # try
    #     #TODO don't save these for full sweep
    #     vtkfilepath = mkpath(joinpath("vtk/", fname))
    #     saveblochtorrey(myelindomains, sols; timepoints = tpoints, filename = joinpath(vtkfilepath, "vtksolution"))
    # catch e
    #     @warn "Error saving solution to vtk file"
    #     @warn sprint(showerror, e, catch_backtrace())
    # end

    # # Plot and save various figures
    # try
    #     mxplotomega(myelinprob, myelindomains, myelinsubdomains, geom.bdry;
    #         titlestr = "Frequency Map (theta = $(round(sweepparams[:theta]; digits=3)) deg)",
    #         fname = "omega/" * fname * ".omega")
    # catch e
    #     @warn "Error plotting omega"
    #     @warn sprint(showerror, e, catch_backtrace())
    # end
    
    try
        mxplotmagnitude(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Magnitude (" * titleparamstr * ")",
            fname = "mag/" * fname * ".magnitude")
        # mxgifmagnitude(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
        #     titlestr = "Field Magnitude (" * titleparamstr * ")", totaltime = (2*sweepparams[:nTR]-1) * 10.0,
        #     fname = "mag/" * fname * ".magnitude.gif")
    catch e
        @warn "Error plotting magnetization magnitude"
        @warn sprint(showerror, e, catch_backtrace())
    end

    try
        mxplotphase(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Phase (" * titleparamstr * ")",
            fname = "phase/" * fname * ".phase")
        # mxgifphase(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
        #     titlestr = "Field Phase (" * titleparamstr * ")", totaltime = (2*sweepparams[:nTR]-1) * 10.0,
        #     fname = "phase/" * fname * ".phase.gif")
    catch e
        @warn "Error plotting magnetization phase"
        @warn sprint(showerror, e, catch_backtrace())
    end
    
    try
        #TODO only for 3D
        mxplotlongitudinal(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Longitudinal (" * titleparamstr * ")",
            fname = "long/" * fname * ".longitudinal")
        # mxgiflongitudinal(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
        #     titlestr = "Longitudinal (" * titleparamstr * ")", totaltime = (2*sweepparams[:nTR]-1) * 10.0,
        #     fname = "long/" * fname * ".longitudinal.gif")
    catch e
        @warn "Error plotting longitudinal magnetization"
        @warn sprint(showerror, e, catch_backtrace())
    end
    
    try
        if mwfmodels != nothing && !isempty(mwfmodels)
            nnlsindex = findfirst(m->m isa NNLSRegression, mwfmodels)
            if !(nnlsindex == nothing)
                plotSEcorr(sols, btparams, myelindomains;
                    mwftrue = getmwf(geom.outercircles, geom.innercircles, geom.bdry),
                    opts = mwfmodels[nnlsindex], fname = "t2dist/" * fname * ".t2dist.SEcorr")
            end
        end
    catch e
        @warn "Error plotting SEcorr T2 distribution"
        @warn sprint(showerror, e, catch_backtrace())
    end

    try
        if mwfmodels != nothing && !isempty(mwfmodels)
            plotbiexp(sols, btparams, myelindomains,
                geom.outercircles, geom.innercircles, geom.bdry;
                titlestr = "Signal Magnitude (" * titleparamstr * ")",
                opts = mwfmodels[1], fname = "sig/" * fname * ".biexp")
        end
        plotsignal(tpoints, signals;
            timeticks = cpmg_savetimes(sols[1].prob.tspan, sweepparams[:TE]/2, sweepparams[:TE], sweepparams[:TR], sweepparams[:nTE], sweepparams[:nTR]), #TODO should be exported
            titlestr = "Magnetization Signal (" * titleparamstr * ")",
            apply_pi_correction = false,
            fname = "sig/" * fname * ".signal")
    catch e
        @warn "Error plotting signal"
        @warn sprint(showerror, e, catch_backtrace())
    end

    return results
end

function main(;iters::Int = typemax(Int))
    # Make subfolders
    mkpath.(("vtk", "mag", "phase", "long", "t2dist", "sig", "omega", "mwfplots", "measurables"))

    # Initialize results
    results = Dict{Symbol,Any}(
        :sweepparams        => [],
        :btparams           => [],
        :solverparams_dict  => [],
        :tpoints            => [],
        :signals            => [],
        # :sols               => [], #TODO
        # :myelinprob         => [], #TODO
        # :myelinsubdomains   => [], #TODO
        # :myelindomains      => [], #TODO
        :mwfvalues          => [])

    all_sweepparams = (sweepparamsampler() for _ in 1:iters)
    for (i,sweepparams) in enumerate(all_sweepparams)
        geomnumber = rand(1:length(geometries))
        geom = geometries[geomnumber]
        tspan = (0.0, sweepparams[:nTE] * sweepparams[:TE] + (sweepparams[:nTR] - 1) * sweepparams[:TR])
        try
            println("\n")
            @info "Running simulation $i/$(length(all_sweepparams)) at $(Dates.now()):"
            @info "    Sweep parameters:    " * DrWatson.savename("", sweepparams; connector = ", ")
            @info "    Geometry info:       Geom #$geomnumber - " * basename(geomfiles[geomnumber])
            @info "    Simulation timespan: (0.0 ms, $(round(1000 .* tspan[2]; digits=3)) ms)"
            
            tic = Dates.now()
            runsimulation!(results, sweepparams, geom)
            toc = Dates.now()
            Δt = Dates.canonicalize(Dates.CompoundPeriod(toc - tic))

            @info "Elapsed simulation time: $Δt"
        catch e
            if e isa InterruptException
                @warn "Parameter sweep interrupted by user. Breaking out of loop and returning current results..."
                break
            else
                @warn "Error running simulation $i/$(length(all_sweepparams))"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return results
end

####
#### Run sweep
####

results = main() #TODO
@unpack sweepparams, btparams, solverparams_dict, tpoints, signals, mwfvalues = results;
# @unpack sols, myelinprob, myelinsubdomains, myelindomains = results; #TODO
btparams_dict = Dict.(btparams);

####
#### Plot and save derived quantities from results
####

try
    BSON.bson(SIM_START_TIME * ".allparams.bson", deepcopy(@dict(sweepparams, btparams_dict)))
catch e
    @warn "Error saving all BlochTorreyParameter's"
    @warn sprint(showerror, e, catch_backtrace())
end

try
    BSON.bson(SIM_START_TIME * ".allmeasurables.bson", deepcopy(@dict(tpoints, signals, mwfvalues)))
catch e
    @warn "Error saving measurables"
    @warn sprint(showerror, e, catch_backtrace())
end

try
    plotMWFvsMethod(results; disp = false, fname = "mwfplots/" * SIM_START_TIME * ".mwf")
catch e
    @warn "Error plotting MWF."
    @warn sprint(showerror, e, catch_backtrace())
end
