# Activate project and load packages for this script
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MWFUtils
using GlobalUtils
const SIM_START_TIME = MWFUtils.getnow()
make_reproduce( # Creating backup file
    """
    include("BlochTorreyExperiments/MyelinWaterTools/scripts/MWF-orientation-generate.jl")
    """;
    fname = SIM_START_TIME * ".reproduce.jl"
)

pyplot(size=(1200,900))
mxcall(:cd, 0, pwd()) # Set MATLAB path (Note: pwd(), not @__DIR__)
gitdir() = realpath(DrWatson.projectdir("..")) # DrWatson package for tagged saving

####
#### Environment variables
####

const MAX_ITERS = parse(Int, get(ENV, "MAX_ITERS", "$(typemax(Int))"))
const CHI_FACT  = parse(Float64, get(ENV, "CHI_FACT", "1.0"))
const R_MU      = parse(Float64, get(ENV, "R_MU", "0.46"))
const R_MU_FACT = R_MU / 0.46

####
#### Geometry
####

const geomfile = BSON.load("/project/st-arausch-1/jcd1994/ismrm2020/experiments/Fall-2019/diff-med-1-input-data/geom-newgridtype/2019-10-04-T-16-01-11-467_Npts=6684_Ntri=8881_density=0.481_gratio=0.79_mvf=0.181_mwf=0.1_numfibres=17.geom.bson");

# Scale geometry by R_MU_FACT
foreach([:interiorgrids, :torigrids, :exteriorgrids]) do f
    foreach(geomfile[f]) do g
        o = origin(geomfile[:bdry])
        JuAFEM.transform!(g, x -> R_MU_FACT * (x - o) + o)
    end
end;
geomfile[:bdry] = scale_shape(geomfile[:bdry], origin(geomfile[:bdry]), R_MU_FACT);
geomfile[:innercircles] = map(c -> scale_shape(c, origin(geomfile[:bdry]), R_MU_FACT), geomfile[:innercircles]);
geomfile[:outercircles] = map(c -> scale_shape(c, origin(geomfile[:bdry]), R_MU_FACT), geomfile[:outercircles]);

####
#### Default solver parameters and MWF models
####

const default_TE = 8e-3; # Echotime
const default_TR = 1000e-3; # Repetition time
const default_nTE = 48; # Number of echoes
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
    :nT2         => 40,              # Number of T2 points used for fitting the distribution
    :Threshold   => 0.0,             # Intensity threshold below which signal is ignored (zero for simulation)
    :RefConAngle => 180.0,           # Flip angle parameter for reconstruction
    :T2Range     => [8e-3, 2000e-3], # Range of T2 values used
    :SPWin       => [8e-3, 25e-3],   # Small pool window, i.e. myelin peak
    :MPWin       => [25e-3, 200e-3], # Intra/extracelluar water peak window
    :PlotDist    => false,           # Plot resulting distribution in MATLAB
);

####
#### Default BlochTorreyParameters
####

const default_btparams = BlochTorreyParameters{Float64}(
    B0 = -3.0,
    theta = π/2,
    D_Tissue = 1000.0, # [μm²/s]
    D_Sheath = 1000.0, # [μm²/s]
    D_Axon = 1000.0, # [μm²/s]
    K_perm = 0.0, # [μm/s]
    PD_lp = 1.0, # [a.u.] large pool relative proton density
    PD_sp = 0.5, # [a.u.] small pool relative proton density
    ChiI = CHI_FACT * -60e-9, # [a.u.] isotropic suceptibility of myelin
    ChiA = CHI_FACT * -120e-9, # [a.u.] anisotropic suceptibility of myelin
    E = 0.0, # [a.u.] susceptibility exchange component
    R_mu = R_MU, # [um] mean radius
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
    :theta  => (sampler = :linearsampler, args = (lb = 0.0,     ub = 90.0)), # Uniformly random orientations => cosθ ~ Uniform(0,1)
    :alpha  => (sampler = :linearsampler, args = (lb = 165.0,   ub = 180.0)),
    :K      => (sampler = :linearsampler, args = (lb = 0.0,     ub = 0.0)),
    :Dtiss  => (sampler = :log10sampler,  args = (lb = 1000.0,  ub = 1000.0)),
    :Dmye   => (sampler = :log10sampler,  args = (lb = 1000.0,  ub = 1000.0)),
    :Dax    => (sampler = :log10sampler,  args = (lb = 1000.0,  ub = 1000.0)),
    :FRD    => (sampler = :linearsampler, args = (lb = 0.5,     ub = 0.5)), # Fractional radial diffusivity (0.5 is isotropic, 1.0 fully radial, 0.0 fully axial)
    :TE     => (sampler = :linearsampler, args = (lb = 8e-3,    ub = 8e-3)),
    :nTE    => (sampler = :rangesampler,  args = (lb = 48, ub = 48, step = 2)), # Simulate many echoes (can chop later)
    :TR     => (sampler = :linearsampler, args = (lb = 1000e-3, ub = 1000e-3)), # Irrelevant when nTR = 1 (set below)
    :nTR    => (sampler = :rangesampler,  args = (lb = 1,       ub = 1)),
    :T2sp   => (sampler = :linearsampler, args = (lb = 15e-3,   ub = 15e-3)),
    :T2lp   => (sampler = :linearsampler, args = (lb = 70e-3,   ub = 70e-3)),
    #:T2tiss=> (sampler = :linearsampler, args = (lb = 50e-3,   ub = 90e-3)),
    :T1sp   => (sampler = :linearsampler, args = (lb = 1000e-3, ub = 1000e-3)),
    :T1lp   => (sampler = :linearsampler, args = (lb = 1000e-3, ub = 1000e-3)),
    #:T1tiss=> (sampler = :linearsampler, args = (lb = 1000e-3, ub = 1000e-3)),
)
sweepparamsample() = Dict{Symbol,Union{Float64,Int}}(k => eval(Expr(:call, v.sampler, v.args...)) for (k,v) in sweepparamsampler_settings)
sweepparamconstraints(d) = d[:T2lp] ≥ 1.5*d[:T2sp] # Extreme T2sp and T2lp ranges above require this additional constraint to make sure each sample is realistic
sweepparamsampler() = (while true; d = sweepparamsample(); sweepparamconstraints(d) && return d; end)

####
#### Save metadata
####

DrWatson.@tagsave(
    SIM_START_TIME * ".metadata.bson",
    deepcopy(@dict(sweepparamsampler_settings, default_btparams_dict, default_solverparams_dict, default_nnlsparams_dict, default_TE, default_nTE)),
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

function runsimulation(sweepparams, geom)
    density = intersect_area(geom.outercircles, geom.bdry) / area(geom.bdry)
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

    # Save measurables
    tryshow("Error saving measurables") do
        btparams_dict = Dict(btparams)
        DrWatson.@tagsave(
            "measurables/" * fname * ".measurables.bson",
            deepcopy(@dict(btparams_dict, solverparams_dict, geomparams_dict, sweepparams, tpoints, signals, mwfvalues, solve_time)),
            safe = true, gitpath = gitdir())
    end

    return nothing
end

function main(;
        iters::Int = typemax(Int),
        perturb_geometry = true,
    )
    # Make subfolders
    map(mkpath, ("vtk", "mag", "phase", "long", "t2dist", "sig", "omega", "mwfplots", "measurables"))

    all_sweepparams = (sweepparamsampler() for _ in 1:iters)
    for (i,sweepparams) in enumerate(all_sweepparams)
        geomtuple = geometrytuple(geomfile)
        tspan = (0.0, sweepparams[:nTE] * sweepparams[:TE] + (sweepparams[:nTR] - 1) * sweepparams[:TR])
        try
            println("\n")
            @info "Running simulation $i/$(length(all_sweepparams)) at $(Dates.now()):"
            @info "    Sweep parameters:    " * DrWatson.savename("", sweepparams; connector = ", ")
            @info "    Geometry filename:   " * basename(geomfile[:originalfile])
            @info "    Simulation timespan: (0.0 ms, $(round(1000 .* tspan[2]; digits=3)) ms)"

            tic = Dates.now()
            runsimulation(sweepparams, geomtuple)
            toc = Dates.now()

            @info "Elapsed simulation time: " * string(Dates.canonicalize(Dates.CompoundPeriod(toc - tic)))
        catch e
            if e isa InterruptException
                @warn "Parameter sweep interrupted by user. Breaking out of loop..."
                break
            else
                @warn "Error running simulation $i/$(length(all_sweepparams))"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
    
    return nothing
end

####
#### Run sweep
####

main(iters = MAX_ITERS)

nothing
