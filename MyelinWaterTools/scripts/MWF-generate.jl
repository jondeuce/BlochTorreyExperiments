# Activate project and load packages for this script
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "../initpaths.jl"))

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
            # "2019-03-29-T-12-15-03-265__N-40_g-0.8000_p-0.8300__structs.bson" #28k triangles, 19k points, Qmin = 0.4
        ]
    ),
    joinpath.(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/drwatson_geom_sweep_1/geom",
        [
            "2019-04-24-T-18-33-57-731_density=0.75_gratio=0.78_numfibres=20.geom.bson" #12.8k triangles, 9.6k points, Qmin = 0.4
            # "2019-04-24-T-21-16-38-329_density=0.75_gratio=0.78_numfibres=35.geom.bson" #36.7k triangles, 25.3k points, Qmin = 0.4
            "2019-04-24-T-17-54-24-004_density=0.75_gratio=0.78_numfibres=5.geom.bson" #3.4k triangles, 2.5k points, Qmin = 0.4
        ]
    ),
    joinpath.(
        "/home/jdoucette/Documents/code/BlochTorreyResults/Experiments/MyelinWaterOrientation/Geometries/drwatson_geom_sweep_2/geom",
        [
            "2019-04-25-T-11-05-25-221_density=0.78_gratio=0.78_numfibres=10.geom.bson" #11.4k triangles, 7.8k points, Qmin = 0.4
            "2019-04-25-T-11-24-46-840_density=0.78_gratio=0.78_numfibres=15.geom.bson" #20.2k triangles, 13.6k points, Qmin = 0.4
            "2019-04-25-T-11-46-05-769_density=0.8_gratio=0.75_numfibres=15.geom.bson" #19.4k triangles, 13.1k points, Qmin = 0.4
            "2019-04-25-T-11-59-59-400_density=0.78_gratio=0.75_numfibres=20.geom.bson" #20.8k triangles, 14.5k points, Qmin = 0.4
            # "2019-04-25-T-15-13-27-738_density=0.78_gratio=0.75_numfibres=30.geom.bson" #38.7k triangles, 25.9k points, Qmin = 0.4
            # "2019-04-25-T-20-10-44-038_density=0.8_gratio=0.75_numfibres=35.geom.bson" #62.8k triangles, 40.7k points, Qmin = 0.4
        ]
    ),
)

function copy_and_load_geomfiles(geomfilenames)
    mkpath("geom")
    geoms = []
    storedgeomfilenames = filter(s->endswith(s, ".bson"), readdir("geom"))

    for (i,geomfile) in enumerate(geomfilenames)
        # load geom file and store locally
        geom = loadgeometry(geomfile)
        if basename(geomfile) ∉ storedgeomfilenames
            DrWatson.@tagsave(
                "geom/" * basename(geomfile),
                deepcopy(@dict(geomfile, geom)),
                true, gitdir())
        end
        push!(geoms, geom)
    end
    return geoms
end
const geometries = copy_and_load_geomfiles(geomfiles);

####
#### Default solver parameters and MWF models
####

const default_TE = 10e-3; # Echotime
const default_TR = 1000e-3; # Repetition time
const default_nTE = 32; # Number of echoes
const default_nTR = 1; # Number of repetitions
const default_tspan = (0.0, default_nTE * default_TE + (default_nTR - 1) * default_TR); # timespan
const default_solverparams_dict = Dict(
    :u0        => 1.0im,                   # Initial magnetization; should be [0,1] for 2D (π/2 pulse) or [0,0,1] for 3D (steady-state)
    :flipangle => Float64(π),              # Flip angle for MultiSpinEchoCallback
    :TE        => default_TE,              # Echotime for MultiSpinEchoCallback (Default: 10e-3)
    :TR        => default_TR,              # Repetition time for MultiSpinEchoCallback (Default: 1000e-3)
    :nTE       => default_nTE,             # Number of echoes for MultiSpinEchoCallback (Default: 32)
    :nTR       => default_nTR,             # Number of repetitions for MultiSpinEchoCallback (Default: 1)
    :tspan     => default_tspan,           # Solver time span (Default: (0.0, 320e-3); must start at zero)
    :reltol    => 1e-8,
    :abstol    => 0.0);

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
    :alpha  => (sampler = :linearsampler, args = (lb = 180.0,   ub = 180.0)),
    :K      => (sampler = :log10sampler,  args = (lb = 1e-3,    ub = 10.0)),
    :Dtiss  => (sampler = :log10sampler,  args = (lb = 100.0,   ub = 500.0)),
    :Dmye   => (sampler = :log10sampler,  args = (lb = 100.0,   ub = 500.0)),
    :Dax    => (sampler = :log10sampler,  args = (lb = 100.0,   ub = 500.0)),
    :FRD    => (sampler = :linearsampler, args = (lb = 0.0,     ub = 0.5)),
    :TE     => (sampler = :linearsampler, args = (lb = 5e-3,    ub = 15e-3)),
    :TR     => (sampler = :linearsampler, args = (lb = 800e-3,  ub = 1200e-3)),
    :nTE    => (sampler = :rangesampler,  args = (lb = 24,      ub = 48)),
    :nTR    => (sampler = :rangesampler,  args = (lb = 1,       ub = 1)),
    :T2sp   => (sampler = :linearsampler, args = (lb = 10e-3,   ub = 20e-3)),
    :T2lp   => (sampler = :linearsampler, args = (lb = 50e-3,   ub = 80e-3)),
    :T2tiss => (sampler = :linearsampler, args = (lb = 50e-3,   ub = 80e-3)),
    :T1sp   => (sampler = :linearsampler, args = (lb = 150e-3,  ub = 250e-3)),
    :T1lp   => (sampler = :linearsampler, args = (lb = 949e-3,  ub = 1219e-3)), #3-sigma range for T1 = 1084 +/- 45
    :T1tiss => (sampler = :linearsampler, args = (lb = 949e-3,  ub = 1219e-3)), #3-sigma range for T1 = 1084 +/- 45
)
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
    @unpack alpha, theta, K, Dtiss, Dmye, Dax, FRD, TE, TR, nTE, nTR, T2sp, T2lp, T2tiss, T1sp, T1lp, T1tiss = sweepparams
    density = intersect_area(geom.outercircles, geom.bdry) / area(geom.bdry)
    gratio = radius(geom.innercircles[1]) / radius(geom.outercircles[1])

    btparams = BlochTorreyParameters(default_btparams;
        theta = deg2rad(theta),
        K_perm = K,
        D_Tissue = Dtiss,
        D_Sheath = Dmye,
        D_Axon = Dax,
        FRD_Sheath = FRD,
        R2_sp = inv(T2sp),
        R2_lp = inv(T2lp),
        R2_Tissue = inv(T2tiss),
        R1_sp = inv(T1sp),
        R1_lp = inv(T1lp),
        R1_Tissue = inv(T1tiss),
        AxonPDensity = density,
        g_ratio = gratio,
    )
    sols, myelinprob, myelinsubdomains, myelindomains, solverparams_dict = runsolve(btparams, sweepparams, geom)
    
    # Sample solution signals, ensuring that each sample is taken after the pulse occurs
    dt = TE/4 # TODO
    tpoints = multispinecho_savetimes(sols[1].prob.tspan, dt, TE, TR, nTE, nTR)
    signals = calcsignal(sols, tpoints, myelindomains)

    # Common filename without suffix
    curr_time = MWFUtils.getnow()
    fname = DrWatson.savename(curr_time, sweepparams)
    titleparamstr = wrap_string(DrWatson.savename("", sweepparams; connector = ", "), 50, ", ")
    
    # Compute MWF values
    mwfmodels = map(default_mwfmodels) do model
        if model isa NNLSRegression
            typeof(model)(model; TE = TE, nTE = nTE, RefConAngle = alpha)
        else
            typeof(model)(model; TE = TE, nTE = nTE)
        end
    end
    mwfvalues, _ = compareMWFmethods(sols, myelindomains,
        geom.outercircles, geom.innercircles, geom.bdry;
        models = mwfmodels)

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
    #     vtkfilepath = mkpath(joinpath("vtk/", fname))
    #     saveblochtorrey(myelindomains, sols; timepoints = tpoints, filename = joinpath(vtkfilepath, "vtksolution"))
    # catch e
    #     @warn "Error saving solution to vtk file"
    #     @warn sprint(showerror, e, catch_backtrace())
    # end

    try
        mxplotmagnitude(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Magnitude (" * titleparamstr * ")",
            fname = "mag/" * fname * ".magnitude")
        # mxgifmagnitude(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
        #     titlestr = "Field Magnitude (" * titleparamstr * ")", totaltime = (2*nTR-1) * 10.0,
        #     fname = "mag/" * fname * ".magnitude.gif")
    catch e
        @warn "Error plotting magnetization magnitude"
        @warn sprint(showerror, e, catch_backtrace())
    end

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
        mxplotmagnitude(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
            titlestr = "Field Magnitude (" * titleparamstr * ")",
            fname = "mag/" * fname * ".magnitude")
        # mxgifmagnitude(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
        #     titlestr = "Field Magnitude (" * titleparamstr * ")", totaltime = (2*nTR-1) * 10.0,
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
        #     titlestr = "Field Phase (" * titleparamstr * ")", totaltime = (2*nTR-1) * 10.0,
        #     fname = "phase/" * fname * ".phase.gif")
    catch e
        @warn "Error plotting magnetization phase"
        @warn sprint(showerror, e, catch_backtrace())
    end
    
    # try
    #     mxplotlongitudinal(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
    #         titlestr = "Longitudinal (" * titleparamstr * ")",
    #         fname = "long/" * fname * ".longitudinal")
    #     # mxgiflongitudinal(typeof(default_solverparams_dict[:u0]), sols, btparams, myelindomains, geom.bdry;
    #     #     titlestr = "Longitudinal (" * titleparamstr * ")", totaltime = (2*nTR-1) * 10.0,
    #     #     fname = "long/" * fname * ".longitudinal.gif")
    # catch e
    #     @warn "Error plotting longitudinal magnetization"
    #     @warn sprint(showerror, e, catch_backtrace())
    # end
    
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
                opts = mwfmodels[1], fname = "sig/" * fname * ".biexp")
        end
        plotsignal(tpoints, signals;
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
        :sols               => [], #TODO
        :myelinprob         => [], #TODO
        :myelinsubdomains   => [], #TODO
        :myelindomains      => [], #TODO
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

results = main()
@unpack sweepparams, btparams, solverparams_dict, tpoints, signals, mwfvalues = results;
@unpack sols, myelinprob, myelinsubdomains, myelindomains = results; #TODO
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
