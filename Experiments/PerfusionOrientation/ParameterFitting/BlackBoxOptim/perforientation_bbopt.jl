using Distributed

# Instantiate worker environments
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere Pkg.instantiate()

@everywhere using Dates, Logging, Glob, ReadableRegex, MATLAB, MAT, BSON, BlackBoxOptim, NLopt, Plots, LaTeXStrings

# Helpers
@everywhere @eval homedir() = $(get(ENV, "JOB_DIR", pwd()))
@everywhere @eval maxtime() = $(3600.0 * parse(Float64, get(ENV, "MAX_TIME_HOURS", "47")))
@everywhere cdall(path) = (cd(path); mxcall(:cd, 0, path); nothing)
@everywhere cdhome() = cdall(homedir())
@everywhere getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
@everywhere function logger(f, prefix; active = true)
    !active && return f()
    local ret = nothing
    open(prefix * ".log", "a") do log
        open(prefix * ".out", "a") do out
            open(prefix * ".err", "a") do err
                with_logger(SimpleLogger(log)) do
                    redirect_stdout(out) do
                        redirect_stderr(err) do
                            ret = f()
                        end
                    end
                end
            end
        end
    end
    return ret
end

# Start MATLAB
@everywhere function init_mx_workers()
    btpath = match(r"(.*?)/BlochTorreyExperiments", @__DIR__)[1] |> string
    repobranch = match(r"BlochTorreyExperiments-(.*?)/", @__DIR__)[1] |> string
    reponame = "BlochTorreyExperiments-" * repobranch |> string
    cdall(btpath)
    mat"""
    disp('Hello from Matlab (worker #$(myid()))');
    cd($btpath);
    addpath($btpath);
    addpath(btpathdef);
    addpath(genpath($reponame));
    setbtpath($repobranch, false);
    maxNumCompThreads($(Threads.nthreads()));
    rng(0);
    """
    cdhome()
end
@everywhere init_mx_workers()

# Optimization
@everywhere function f(x::AbstractVector{Float64})
    if myid() ∉ workers()
        return NaN64 # Only workers should be working
    end
    try
        logdir = joinpath(homedir(), getnow() * "-worker-$(myid())")
        mkpath(logdir)
        cdall(logdir)
        logger(joinpath(logdir, "Diary")) do
            ℓ = mxcall(:perforientation_bbopt_caller, 1, convert(Vector{Float64}, x), homedir()) |> Float64
            println("loss = $ℓ")
            return ℓ
        end
    finally
        cdhome()
    end
end

@everywhere maybefire(file) = (isf = isfile(file); if isf; mv(file, file * ".fired"; force = true); end; return isf)
@everywhere function cb(ctrl::BlackBoxOptim.OptRunController)
    if maybefire(joinpath(homedir(), "stop"))
        # error("Halting optimizer: found file `stop`")
        BlackBoxOptim.shutdown_optimizer!(ctrl)
    end
end

# Initial solve using global optimizer
function bbopt()
    params = MAT.matread(joinpath(homedir(), "Params0.mat"))["Params0"]
    bounds = vec(tuple.(params["lb"], params["ub"]))

    #= toy problem
    bounds = tuple.(-ones(5), ones(5))
    local f(x) = (ℓ = sum(abs2, x); @show(ℓ); sleep(1.0); return ℓ)
    =#

    res = bboptimize(bbsetup(
        f;
        Method = :dxnes,
        SearchRange = bounds,
        MaxTime = maxtime(),
        CallbackFunction = cb,
        CallbackInterval = 0.0,
        Workers = workers(),
    ))

    # Save results
    MAT.matwrite(joinpath(homedir(), "BBOptResults.mat"), Dict(
        "x" => deepcopy(best_candidate(res)),
        "loss" => best_fitness(res),
    ))

    BSON.bson(joinpath(homedir(), "BBOptResults.bson"), Dict(
        "results" => deepcopy(res),
    ))
end

# Cleanup: save one large file instead of many small files
function cleanup_bbopt()
    # Safest to read + re-save results within Matlab to ensure collect handling of class objects, function handles, etc.
    let
        files = readdir(glob"**/*.mat", homedir()) .|> string
        outfile = joinpath(homedir(), "AllIterationsResults.mat") |> string
        savetime = @elapsed mat"""
        AllIterationsResults = cell($(length(files)), 1);
        for ii = 1:numel(AllIterationsResults)
            AllIterationsResults{ii} = load($files{ii});
        end
        save($outfile, 'AllIterationsResults');
        """
        println("$(basename(homedir())): collecting iteration results... ($(round(savetime, digits = 1)) s)")
    end

    # Zip all iteration results together
    let
        iterdirs = readdir(glob"*-worker-*", homedir()) .|> string
        iterfiles = readdir(glob"*-worker-**/*", homedir()) .|> string
        outfile = joinpath(homedir(), "AllIterationsResults.zip") |> string

        ziptime = @elapsed try
            run(`zip -rq $outfile $iterdirs`; wait = true) # zip files

            zipinfo = read(`zipinfo -t $(homedir())/AllIterationsResults.zip`, String) |> chomp
            numfiles = parse(Int, match(r"(\d+) files", zipinfo)[1])

            @assert length(iterfiles) + length(iterdirs) == numfiles # ensure all files accounted for
            run(`rm -rf $iterdirs`; wait = true)
        catch e
            println(sprint(showerror, e, catch_backtrace()))
        end
        println("$(basename(homedir())): zipping iteration results... ($(round(ziptime, digits = 1)) s)")
    end

    return nothing
end

logger(joinpath(homedir(), "Diary")) do
    mxcall(:perforientation_bbopt_init, 0)
    bbopt()
    cleanup_bbopt()
end

#= Refine solution using local optimizer
function nlopt()
    # TODO
end
logger(joinpath(homedir(), "Diary")) do
    mxcall(:perforientation_bbopt_init, 0)
    nlopt()
end
=#

nothing
