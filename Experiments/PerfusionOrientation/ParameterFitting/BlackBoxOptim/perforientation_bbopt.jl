using Distributed

# Instantiate worker environments
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere Pkg.instantiate()

@everywhere using Dates, Logging, Glob, MATLAB, MAT, BSON, BlackBoxOptim, NLopt, Plots, LaTeXStrings

# Helpers
@everywhere @eval homedir() = $(get(ENV, "JOB_DIR", pwd()))
@everywhere @eval maxtime() = $(3600.0 * parse(Float64, get(ENV, "MAX_TIME_HOURS", "47")))
@everywhere cdall(path) = (cd(path); mxcall(:cd, 0, path); nothing)
@everywhere cdhome() = cdall(homedir())
@everywhere getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
@everywhere function logger(f, prefix)
    local ret = nothing
    open(prefix * ".log", "a") do io
        open(prefix * ".out", "a") do out
            open(prefix * ".err", "a") do err
                with_logger(SimpleLogger(io)) do
                    redirect_stdout(out) do
                        redirect_stderr(err) do
                            ret = f()
                        end
                    end
                end
                foreach(flush, (io, out, err))
            end
        end
    end
    return ret
end

# Start MATLAB
@everywhere function init_mx_workers()
    repobranch = (@__DIR__)[match(r"BlochTorreyExperiments-", @__DIR__).offset + 23 : match(r"/Experiments/PerfusionOrientation", @__DIR__).offset - 1]
    reponame = "BlochTorreyExperiments-" * repobranch
    cdall("/project/st-arausch-1/jcd1994/code")
    mat"""
    disp('Hello from Matlab (worker #$(myid()))');
    cd('/project/st-arausch-1/jcd1994/code');
    addpath('/project/st-arausch-1/jcd1994/code');
    addpath(btpathdef);
    addpath(genpath($reponame));
    setbtpath($repobranch);
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
        BlackBoxOptim.shutdown_optimizer!(ctrl)
    end
    foreach(flush, (stdout, stdin))
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
logger(joinpath(homedir(), "Diary")) do
    mxcall(:perforientation_bbopt_init, 0)
    bbopt()
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
