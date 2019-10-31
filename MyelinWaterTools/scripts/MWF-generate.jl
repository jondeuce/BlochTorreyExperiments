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
gitdir() = realpath(DrWatson.projectdir(".."))

####
#### Geometries to sweep over
####

function copy_and_load_geomfiles!(
        geomfiles::AbstractVector{String};
        maxnnodes::Int = typemax(Int)
    )
    mkpath("geom")
    geomdata = []
    storedgeomfilenames = filter(s->endswith(s, ".bson"), readdir("geom"))

    skipped_geoms = Int[]
    for (i,geomfile) in enumerate(geomfiles)
        geom = BSON.load(geomfile)
        geom[:originalfile] = geomfile
        if geom[:params][:Npts] > maxnnodes
            push!(skipped_geoms, i) # Geometry is too large; skip it
            continue
        end
        if basename(geomfile) ∉ storedgeomfilenames
            DrWatson.@tagsave(
                joinpath("geom", basename(geomfile)),
                deepcopy(geom),
                true, gitdir())
        end
        push!(geomdata, geom)
    end

    # Update geomfiles, removing skipped geometries from list
    deleteat!(geomfiles, skipped_geoms)

    return geomdata
end
function copy_and_load_random_geom(
        geomdirs::Vector{String};
        maxnnodes::Int = typemax(Int), # max allowable nodes
        binwidth = 2.5/100, # mwf bin width
    )
    mkpath("geom")
    parse_mwf(g) = parse(Float64, match(r"mwf=(0.[0-9]+)", g).captures[1]) # extract mwf from filename
    parse_Npts(g) = parse(Int, match(r"Npts=([0-9]+)", g).captures[1]) # extract Npts from filename
    
    # Parse geometry files from given directory and choose a random file uniformly based on the mwf
    geomfiles = reduce(vcat, [map(g -> joinpath(gdir, g), filter(s -> endswith(s, ".bson"), readdir(gdir))) for gdir in geomdirs]) # read geom filenames
    data = [(mwf = parse_mwf(g), geomfile = g) for g in geomfiles if parse_Npts(g) <= maxnnodes] # filter filenames
    data_binned = [filter(d -> lb ≤ d.mwf < lb + binwidth, data) for lb in 0.0:binwidth:1.0] # bin filenames by mwf value
    filter!(!isempty, data_binned) # remove empty bins
    geomfile = rand(rand(data_binned)).geomfile # sample uniformly randomly within a randomly chosen bin
    
    # Load the geometry, and save a copy of it in the current directory
    geom = BSON.load(geomfile) # load geometry
    geom[:originalfile] = geomfile # tag geometry with original filename
    BSON.bson(joinpath("geom", basename(geomfile)), deepcopy(geom))    
    return geom
end
copy_and_load_random_geom(geomdir::String; kwargs...) = copy_and_load_random_geom([geomdir]; kwargs...)

# Load geometries with at most `maxnnodes` number of nodes to avoid exceedingly long simulations
const geombasepaths = [
    realpath("./geom"),
    # "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterLearning/geometries/periodic-packed-fibres-3/geom",
    # "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterLearning/geometries/periodic-packed-fibres-4/geom",
]
const geomfiles = reduce(vcat, realpath.(joinpath.(gp, readdir(gp))) for gp in geombasepaths)
const maxnnodes = 15_000
# const geomdata = copy_and_load_geomfiles!(geomfiles; maxnnodes = maxnnodes);

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
    :abstol      => 0.0,
);

const default_nnlsparams_dict = Dict(
    :TE          => default_TE,      # Echotime of signal
    :nTE         => default_nTE,     # Number of echoes
    :nT2         => 120,             # Number of T2 points used for fitting the distribution
    :Threshold   => 0.0,             # Intensity threshold below which signal is ignored (zero for simulation)
    :RefConAngle => 180.0,           # Flip angle parameter for reconstruction
    :T2Range     => [5e-3, 2000e-3], # Range of T2 values used
    :SPWin       => [10e-3, 40e-3],  # Small pool window, i.e. myelin peak
    :MPWin       => [40e-3, 200e-3], # Intra/extracelluar water peak window
    :PlotDist    => false,           # Plot resulting distribution in MATLAB
);

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
    :theta  => (sampler = :acossampler,   args = (lb = 0.0,     ub = 90.0)), # Uniformly random orientations => cosθ ~ Uniform(0,1)
    :alpha  => (sampler = :linearsampler, args = (lb = 120.0,   ub = 180.0)),
    :K      => (sampler = :log10sampler,  args = (lb = 1e-3,    ub = 10.0)),
    :Dtiss  => (sampler = :log10sampler,  args = (lb = 500.0,   ub = 500.0)), # Diffusion fixed relatively small for faster simulations
    :Dmye   => (sampler = :log10sampler,  args = (lb = 500.0,   ub = 500.0)), # Diffusion fixed relatively small for faster simulations
    :Dax    => (sampler = :log10sampler,  args = (lb = 500.0,   ub = 500.0)), # Diffusion fixed relatively small for faster simulations
    :FRD    => (sampler = :linearsampler, args = (lb = 0.5,     ub = 0.5)),
    :TE     => (sampler = :linearsampler, args = (lb = 10e-3,   ub = 10e-3)), # NOTE: Fixed time scale; only e.g. T2/TE is learned
    :nTE    => (sampler = :rangesampler,  args = (lb = 64, ub = 64, step = 2)), # Simulate many echoes; can chop later
    :TR     => (sampler = :linearsampler, args = (lb = 1000e-3, ub = 1000e-3)), # Irrelevant when nTR = 1 (set below)
    :nTR    => (sampler = :rangesampler,  args = (lb = 1,       ub = 1)), # Only simulate from t = 0 to t = nTE * TE
    :T2sp   => (sampler = :linearsampler, args = (lb = 10e-3,   ub = 70e-3)), # Bounds based on T2sp/TE when TE = 10ms (above), T2sp ∈ [10ms, 35ms], TE ∈ [5ms, 10ms]
    :T2lp   => (sampler = :linearsampler, args = (lb = 50e-3,   ub = 180e-3)), # Bounds based on T2lp/TE when TE = 10ms (above), T2lp ∈ [50ms, 90ms], TE ∈ [5ms, 10ms]
    #:T2tiss=> (sampler = :linearsampler, args = (lb = 50e-3,   ub = 90e-3)),
    :T1sp   => (sampler = :linearsampler, args = (lb = 150e-3,  ub = 250e-3)),
    :T1lp   => (sampler = :linearsampler, args = (lb = 949e-3,  ub = 1219e-3)), #3-sigma range for T1 = 1084 +/- 45
    #:T1tiss=> (sampler = :linearsampler, args = (lb = 949e-3,  ub = 1219e-3)), #3-sigma range for T1 = 1084 +/- 45
)
sweepparamsample() = Dict{Symbol,Union{Float64,Int}}(
    k => eval(Expr(:call, v.sampler, v.args...))
    for (k,v) in sweepparamsampler_settings)
sweepparamconstraints(d) = d[:T2lp] ≥ 1.5*d[:T2sp] # Extreme T2sp and T2lp ranges above require this additional constraint to make sure each sample is realistic
function sweepparamsampler()
    while true
        d = sweepparamsample()
        sweepparamconstraints(d) && return d
    end
end

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
    mvf = (1-gratio^2) * density # myelin volume fraction
    mwf = mvf/(2-mvf) # myelin water fraction, assuming relative myelin proton density of 1/2

    btparams = BlochTorreyParameters(default_btparams;
        theta = deg2rad(sweepparams[:theta]),
        K_perm = sweepparams[:K],
        D_Tissue = sweepparams[:Dtiss],
        D_Sheath = sweepparams[:Dmye],
        D_Axon = sweepparams[:Dax],
        FRD_Sheath = sweepparams[:FRD],
        R2_sp = inv(sweepparams[:T2sp]),
        R2_lp = inv(sweepparams[:T2lp]),
        R2_Tissue = inv(sweepparams[:T2lp]), #Force equal (else, set to inv(sweepparams[:T2tiss]))
        R1_sp = inv(sweepparams[:T1sp]),
        R1_lp = inv(sweepparams[:T1lp]),
        R1_Tissue = inv(sweepparams[:T1lp]), #Force equal (else, set to inv(sweepparams[:T1tiss]))
        AxonPDensity = density,
        g_ratio = gratio,
        PD_lp = 1.0,
        PD_sp = 0.5,
        MVF = mvf,
        MWF = mwf,
    )
    sols, myelinprob, myelinsubdomains, myelindomains, solverparams_dict = runsolve(btparams, sweepparams, geom)
    
    # Sample solution signals
    dt = sweepparams[:TE]/20 # TODO how many samples to save?
    tpoints = cpmg_savetimes(sols[1].prob.tspan, dt, sweepparams[:TE], sweepparams[:TR], sweepparams[:nTE], sweepparams[:nTR])
    subregion_names = string.(typeof.(getregion.(myelindomains)))
    subregion_signals = calcsignals(sols, tpoints, btparams, myelindomains)
    signals = sum(subregion_signals)

    # Common filename without suffix
    curr_time = MWFUtils.getnow()
    fname = DrWatson.savename(curr_time, sweepparams)
    titleparamstr = wrap_string(DrWatson.savename("", sweepparams; connector = ", "), 50, ", ")

    # Compare MWF values
    # mwfvalues, mwfmodels = nothing, nothing
    # try
    #     mwfmodels = map(default_mwfmodels) do model
    #         if model isa NNLSRegression
    #             typeof(model)(model; TE = sweepparams[:TE], nTE = sweepparams[:nTE], RefConAngle = sweepparams[:alpha])
    #         else
    #             typeof(model)(model; TE = sweepparams[:TE], nTE = sweepparams[:nTE])
    #         end
    #     end
    #     mwfvalues, _ = compareMWFmethods(sols, myelindomains, btparams,
    #         geom.outercircles, geom.innercircles, geom.bdry;
    #         models = mwfmodels)
    # catch e
    #     @warn "Error comparing MWF methods"
    #     @warn sprint(showerror, e, catch_backtrace())
    # end

    # Compute exact MWF only
    mwfmodels = nothing
    mwfvalues = Dict(
        :exact => getmwf(geom.outercircles, geom.innercircles, geom.bdry)
    )

    # Update results struct and return
    push!(results[:btparams], btparams)
    push!(results[:solverparams_dict], solverparams_dict)
    push!(results[:sweepparams], sweepparams)
    push!(results[:tpoints], tpoints)
    push!(results[:signals], signals)
    push!(results[:mwfvalues], mwfvalues)
    # push!(results[:subregion_names], subregion_names) #TODO
    # push!(results[:subregion_signals], subregion_signals) #TODO
    # push!(results[:sols], sols) #TODO
    # push!(results[:myelinprob], myelinprob) #TODO
    # push!(results[:myelinsubdomains], myelinsubdomains) #TODO
    # push!(results[:myelindomains], myelindomains) #TODO

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

    # # Save solution as vtk file (NOTE: should only be uncommented for testing)
    # try
    #     vtkfilepath = mkpath(joinpath("vtk/", fname))
    #     saveblochtorrey(myelindomains, sols; timepoints = tpoints, filename = joinpath(vtkfilepath, "vtksolution"))
    # catch e
    #     @warn "Error saving solution to vtk file"
    #     @warn sprint(showerror, e, catch_backtrace())
    # end

    # # Plot frequency/field map
    # try
    #     mxplotomega(myelinprob, myelindomains, myelinsubdomains, geom.bdry;
    #         titlestr = "Frequency Map (theta = $(round(sweepparams[:theta]; digits=3)) deg)",
    #         fname = "omega/" * fname * ".omega")
    # catch e
    #     @warn "Error plotting omega"
    #     @warn sprint(showerror, e, catch_backtrace())
    # end
    
    # Plot solution magnitude on mesh
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

    # Plot solution phase on mesh
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
    
    # Plot solution longitudinal component on mesh
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
    
    # Plot SEcorr results, if used
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

    # Plot multiexponential signal compared with true signal
    try
        if mwfmodels != nothing && !isempty(mwfmodels)
            plotmultiexp(sols, btparams, myelindomains,
                geom.outercircles, geom.innercircles, geom.bdry;
                titlestr = "Signal Magnitude (" * titleparamstr * ")",
                opts = mwfmodels[1], fname = "sig/" * fname * ".multiexp")
        end
        plotsignal(tpoints, signals;
            timeticks = cpmg_savetimes(sols[1].prob.tspan, sweepparams[:TE]/2, sweepparams[:TE], sweepparams[:TR], sweepparams[:nTE], sweepparams[:nTR]),
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
    map(mkpath, ("vtk", "mag", "phase", "long", "t2dist", "sig", "omega", "mwfplots", "measurables"))

    # Initialize results
    results = Dict{Symbol,Any}(
        :sweepparams        => [],
        :btparams           => [],
        :solverparams_dict  => [],
        :tpoints            => [],
        :signals            => [],
        :mwfvalues          => [],
        # :subregion_names    => [], #TODO: no point saving this for PermeableInterfaceRegion simulations
        # :subregion_signals  => [], #TODO: no point saving this for PermeableInterfaceRegion simulations
        # :sols               => [], #TODO: prefer not to save custom types / large memory footprint
        # :myelinsubdomains   => [], #TODO: prefer not to save custom types / large memory footprint
        # :myelindomains      => [], #TODO: prefer not to save custom types / large memory footprint
        # :myelinprob         => [], #TODO: prefer not to save custom types
    )

    all_sweepparams = (sweepparamsampler() for _ in 1:iters)
    for (i,sweepparams) in enumerate(all_sweepparams)
        geom = copy_and_load_random_geom(geombasepaths; maxnnodes = maxnnodes)
        # geom = rand(geomdata)
        tspan = (0.0, sweepparams[:nTE] * sweepparams[:TE] + (sweepparams[:nTR] - 1) * sweepparams[:TR])
        try
            println("\n")
            @info "Running simulation $i/$(length(all_sweepparams)) at $(Dates.now()):"
            @info "    Sweep parameters:    " * DrWatson.savename("", sweepparams; connector = ", ")
            @info "    Geometry filename:   " * basename(geom[:originalfile])
            @info "    Simulation timespan: (0.0 ms, $(round(1000 .* tspan[2]; digits=3)) ms)"
            
            tic = Dates.now()
            runsimulation!(results, sweepparams, geometrytuple(geom))
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

results = main()
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
