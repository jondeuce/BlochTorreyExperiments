# Activate project and load packages for this script
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MWFUtils
using GlobalUtils
const SIM_START_TIME = MWFUtils.getnow()
make_reproduce( # Creating backup file
    """
    include("BlochTorreyExperiments/MyelinWaterTools/scripts/MWF-generate.jl")
    """;
    fname = SIM_START_TIME * ".reproduce.jl"
)

pyplot(size=(1200,900))
mxcall(:cd, 0, pwd()) # Set MATLAB path (Note: pwd(), not @__DIR__)
gitdir() = realpath(DrWatson.projectdir("..")) # DrWatson package for tagged saving

####
#### Geometries to sweep over
####

function perturb_geom!(geom, min_rel_gratio = 0.9)
    # Perturb inner circle radii + corresponding grids
    @unpack exteriorgrids, torigrids, interiorgrids, outercircles, innercircles, bdry = geom
    @assert 0 < min_rel_gratio <= 1
    for i in 1:length(outercircles)
        cout, cin = outercircles[i], innercircles[i]
        if is_inside(cout, bdry)
            r1, r2 = radius(cin), radius(cout)
            o1, o2 = origin(cin), origin(cout)
            r̄ = (min_rel_gratio + rand() * (1 - min_rel_gratio)) * r1
            linscale(r, a, b, ā, b̄) = ((b̄ - ā) * (r - a) / (b - a) + ā) / r
            JuAFEM.transform!(interiorgrids[i], x -> (r̄ / r1) * (x - o1) + o1)
            JuAFEM.transform!(torigrids[i], x -> linscale(norm(x-o1), r1, r2, r̄, r2) * (x - o1) + o1)
            innercircles[i] = scale_shape(cin, r̄ / r1)
        end
    end
    return geom
end
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
                safe = true, gitpath = gitdir())
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
    # realpath("./geom"), #TODO
    # "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterLearning/geometries/periodic-packed-fibres-3/geom",
    # "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterLearning/geometries/periodic-packed-fibres-4/geom",
    "/project/st-arausch-1/jcd1994/ismrm2020/experiments/Fall-2019/diff-med-1-input-data/geom", #TODO
]
const geomfiles = reduce(vcat, realpath.(joinpath.(gp, readdir(gp))) for gp in geombasepaths)
const maxnnodes = 15_000

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
    :reltol      => 1e-4,
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
    B0 = -3.0,
    theta = π/2,
    D_Tissue = 1000.0, # [μm²/s]
    D_Sheath = 1000.0, # [μm²/s]
    D_Axon = 1000.0, # [μm²/s]
    K_perm = 0.1, # [μm/s]
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
    :K      => (sampler = :log10sampler,  args = (lb = 1e-3,    ub = 0.1)),
    :Dtiss  => (sampler = :log10sampler,  args = (lb = 1000.0,  ub = 1000.0)),
    :Dmye   => (sampler = :log10sampler,  args = (lb = 1000.0,  ub = 1000.0)),
    :Dax    => (sampler = :log10sampler,  args = (lb = 1000.0,  ub = 1000.0)),
    :FRD    => (sampler = :linearsampler, args = (lb = 0.5,     ub = 0.5)), # Fractional radial diffusivity (0.5 is isotropic, 1.0 fully radial, 0.0 fully axial)
    :TE     => (sampler = :linearsampler, args = (lb = 10e-3,   ub = 10e-3)), # NOTE: Fixed time scale (only e.g. T2/TE is learned)
    :nTE    => (sampler = :rangesampler,  args = (lb = 64, ub = 64, step = 2)), # Simulate many echoes (can chop later)
    :TR     => (sampler = :linearsampler, args = (lb = 1000e-3, ub = 1000e-3)), # Irrelevant when nTR = 1 (set below)
    :nTR    => (sampler = :rangesampler,  args = (lb = 1,       ub = 1)), # nTR = 1 (only simulate from t = 0 to t = nTE * TE)
    :T2sp   => (sampler = :linearsampler, args = (lb = 10e-3,   ub = 70e-3)), # Bounds based on T2sp/TE when TE = 10ms (above), T2sp ∈ [10ms, 35ms], TE ∈ [5ms, 10ms]
    :T2lp   => (sampler = :linearsampler, args = (lb = 50e-3,   ub = 180e-3)), # Bounds based on T2lp/TE when TE = 10ms (above), T2lp ∈ [50ms, 90ms], TE ∈ [5ms, 10ms]
    #:T2tiss=> (sampler = :linearsampler, args = (lb = 50e-3,   ub = 90e-3)),
    :T1sp   => (sampler = :linearsampler, args = (lb = 150e-3,  ub = 250e-3)),
    :T1lp   => (sampler = :linearsampler, args = (lb = 949e-3,  ub = 1219e-3)), #3-sigma range for T1 = 1084 +/- 45
    #:T1tiss=> (sampler = :linearsampler, args = (lb = 949e-3,  ub = 1219e-3)), #3-sigma range for T1 = 1084 +/- 45
)
sweepparamsample() = Dict{Symbol,Union{Float64,Int}}(k => eval(Expr(:call, v.sampler, v.args...)) for (k,v) in sweepparamsampler_settings)
sweepparamconstraints(d) = d[:T2lp] ≥ 1.5*d[:T2sp] # Extreme T2sp and T2lp ranges above require this additional constraint to make sure each sample is realistic
sweepparamsampler() = (while true; d = sweepparamsample(); sweepparamconstraints(d) && return d; end)

####
#### Save metadata
####

DrWatson.@tagsave(
    SIM_START_TIME * ".metadata.bson",
    deepcopy(@dict(sweepparamsampler_settings, geomfiles, default_mwfmodels_dict, default_btparams_dict, default_solverparams_dict, default_nnlsparams_dict, default_TE, default_nTE)),
    safe = true, gitpath = gitdir())

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
    density = intersect_area(geom.outercircles, geom.bdry) / area(geom.bdry)
    # gratio = radius(geom.innercircles[1]) / radius(geom.outercircles[1]) # uniform g-ratio
    gratio = sqrt(sum(area, geom.innercircles) / sum(area, geom.outercircles)) # area weighted g-ratio
    mvf = (1-gratio^2) * density # myelin volume fraction (for periodically tired circles)
    mwf = mvf/(2-mvf) # myelin water fraction, assuming relative myelin proton density of 1/2

    geomparams_dict = Dict{Symbol,Any}(
        :gratio    => gratio,
        :density   => density,
        :mvf       => mvf,
        :mwf       => mwf,
        :Ntri      => sum(JuAFEM.getncells, geom.exteriorgrids) + sum(JuAFEM.getncells, geom.torigrids) + sum(JuAFEM.getncells, geom.interiorgrids),
        :Npts      => sum(JuAFEM.getnnodes, geom.exteriorgrids) + sum(JuAFEM.getnnodes, geom.torigrids) + sum(JuAFEM.getnnodes, geom.interiorgrids),
        :numfibres => length(geom.outercircles),
    )

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

    TimerOutputs.reset_timer!(BlochTorreyUtils.TIMER)
    @timeit BlochTorreyUtils.TIMER "runsolve" begin
        @unpack sols, myelinprob, myelinsubdomains, myelindomains, solverparams_dict = runsolve(btparams, sweepparams, geom)
    end
    timer_buf = IOBuffer(); TimerOutputs.print_timer(timer_buf, BlochTorreyUtils.TIMER)
    @info "\n" * String(take!(timer_buf))
    solve_time = TimerOutputs.tottime(BlochTorreyUtils.TIMER)
    
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

    # Compute exact MWF only
    mwfmodels = nothing
    mwfvalues = Dict(:exact => getmwf(geom.outercircles, geom.innercircles, geom.bdry))

    #=
    mwfvalues, mwfmodels = nothing, nothing
    tryshow("Error comparing MWF methods") do
        mwfmodels = map(default_mwfmodels) do model
            if model isa NNLSRegression
                typeof(model)(model; TE = sweepparams[:TE], nTE = sweepparams[:nTE], RefConAngle = sweepparams[:alpha])
            else
                typeof(model)(model; TE = sweepparams[:TE], nTE = sweepparams[:nTE])
            end
        end
        mwfvalues, _ = compareMWFmethods(sols, myelindomains, btparams,
            geom.outercircles, geom.innercircles, geom.bdry;
            models = mwfmodels)
    end
    =#

    # Update results struct and return
    # push!(results[:btparams], btparams) #TODO
    # push!(results[:solverparams_dict], solverparams_dict) #TODO
    # push!(results[:sweepparams], sweepparams) #TODO
    # push!(results[:tpoints], tpoints) #TODO
    # push!(results[:signals], signals) #TODO
    # push!(results[:mwfvalues], mwfvalues) #TODO
    # push!(results[:subregion_names], subregion_names) #TODO
    # push!(results[:subregion_signals], subregion_signals) #TODO
    # push!(results[:sols], sols) #TODO
    # push!(results[:myelinprob], myelinprob) #TODO
    # push!(results[:myelinsubdomains], myelinsubdomains) #TODO
    # push!(results[:myelindomains], myelindomains) #TODO

    # Save measurables
    tryshow("Error saving measurables") do
        btparams_dict = Dict(btparams)
        DrWatson.@tagsave(
            "measurables/" * fname * ".measurables.bson",
            deepcopy(@dict(btparams_dict, solverparams_dict, geomparams_dict, sweepparams, tpoints, signals, mwfvalues, solve_time)),
            safe = true, gitpath = gitdir())
    end

    #=
    tryshow("Error saving solution to vtk file") # NOTE: should only be uncommented for testing
        vtkfilepath = mkpath(joinpath("vtk/", fname))
        saveblochtorrey(myelindomains, sols; timepoints = tpoints, filename = joinpath(vtkfilepath, "vtksolution"))
    end
    =#

    #=
    tryshow("Error plotting omega map")
        mxplotomega(myelinprob, myelindomains, myelinsubdomains, geom.bdry;
            titlestr = "Frequency Map (theta = $(round(sweepparams[:theta]; digits=3)) deg)",
            fname = "omega/" * fname * ".omega")
    end
    =#
    
    #=
    tryshow("Error plotting magnetization magnitude") #TODO FIXME
        mxplotmagnitude(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Magnitude (" * titleparamstr * ")",
            fname = "mag/" * fname * ".magnitude")
        mxgifmagnitude(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Magnitude (" * titleparamstr * ")", totaltime = (2*sweepparams[:nTR]-1) * 10.0,
            fname = "mag/" * fname * ".magnitude.gif")
    end
    =#

    #=
    tryshow("Error plotting magnetization phase") #TODO FIXME
        mxplotphase(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Phase (" * titleparamstr * ")",
            fname = "phase/" * fname * ".phase")
        mxgifphase(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Phase (" * titleparamstr * ")", totaltime = (2*sweepparams[:nTR]-1) * 10.0,
            fname = "phase/" * fname * ".phase.gif")
    end
    =#
    
    #=
    tryshow("Error plotting longitudinal magnetization") #TODO FIXME (only for 3D)
        mxplotlongitudinal(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Longitudinal (" * titleparamstr * ")",
            fname = "long/" * fname * ".longitudinal")
        mxgiflongitudinal(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Longitudinal (" * titleparamstr * ")", totaltime = (2*sweepparams[:nTR]-1) * 10.0,
            fname = "long/" * fname * ".longitudinal.gif")
    end
    =#
    
    #=
    tryshow("Error plotting SEcorr T2 distribution")
        if mwfmodels != nothing && !isempty(mwfmodels)
            nnlsindex = findfirst(m->m isa NNLSRegression, mwfmodels)
            if !(nnlsindex == nothing)
                plotSEcorr(sols, btparams, myelindomains;
                    mwftrue = getmwf(geom.outercircles, geom.innercircles, geom.bdry),
                    opts = mwfmodels[nnlsindex], fname = "t2dist/" * fname * ".t2dist.SEcorr")
            end
        end
    end
    =#

    #=
    tryshow("Error plotting multiexponential signal compared with true signal")
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
    end
    =#

    return results
end

function main(;
        iters::Int = typemax(Int),
        perturb_geometry = true,
    )
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
        geomtuple = geometrytuple(geom)
        perturb_geometry && perturb_geom!(geomtuple, 1.0)
        tspan = (0.0, sweepparams[:nTE] * sweepparams[:TE] + (sweepparams[:nTR] - 1) * sweepparams[:TR])
        try
            println("\n")
            @info "Running simulation $i/$(length(all_sweepparams)) at $(Dates.now()):"
            @info "    Sweep parameters:    " * DrWatson.savename("", sweepparams; connector = ", ")
            @info "    Geometry filename:   " * basename(geom[:originalfile])
            @info "    Simulation timespan: (0.0 ms, $(round(1000 .* tspan[2]; digits=3)) ms)"
            
            tic = Dates.now()
            runsimulation!(results, sweepparams, geomtuple)
            toc = Dates.now()
    
            @info "Elapsed simulation time: " * string(Dates.canonicalize(Dates.CompoundPeriod(toc - tic)))
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

main() #TODO

####
#### Plot and save derived quantities from results
####

#=
tryshow("Error saving all BlochTorreyParameter's")
    BSON.bson(SIM_START_TIME * ".allparams.bson", deepcopy(@dict(sweepparams, btparams_dict)))
end
tryshow("Error saving all measurables")
    BSON.bson(SIM_START_TIME * ".allmeasurables.bson", deepcopy(@dict(tpoints, signals, mwfvalues)))
end
tryshow("Error plotting MWF vs method")
    plotMWFvsMethod(results; disp = false, fname = "mwfplots/" * SIM_START_TIME * ".mwf")
end
=#

#=
function plot_geom(geom, fname = "tmp.png")
    p = plot();
    map(g -> simpplot!(p,g), geom.exteriorgrids);
    map(g -> simpplot!(p,g), geom.torigrids);
    map(g -> simpplot!(p,g), geom.interiorgrids);
    map(c -> !is_outside(c, geom.bdry) ? plot!(p,c; line=(3,:black)) : nothing, geom.outercircles)
    map(c -> !is_outside(c, geom.bdry) ? plot!(p,c; line=(3,:black)) : nothing, geom.innercircles)
    savefig(p, joinpath(@__DIR__, fname));
end
geom = geometrytuple(copy_and_load_random_geom(geombasepaths; maxnnodes = maxnnodes));
plot_geom(geom, "tmp.png");
plot_geom(perturb_geom!(geom, 0.5), "tmp-rescaled.png");
=#

nothing
