using Distributed

# Instantiate worker environments
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere Pkg.instantiate()

@everywhere using Dates, Glob, MAT, BSON, MATLAB, BlackBoxOptim, Plots, LaTeXStrings

# Helpers
@everywhere @eval homedir() = $(get(ENV, "JOB_DIR", pwd()))
@everywhere cdall(path) = (cd(path); mxcall(:cd, 0, path); nothing)
@everywhere cdhome() = cdall(homedir())
@everywhere getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
@everywhere function logger(f, prefix)
    open(prefix * ".log", "a") do out
        open(prefix * ".err", "a") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    return f()
                end
            end
        end
    end
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

# Initialization
logger(joinpath(homedir(), "Diary")) do
    mxcall(:perforientation_bbopt_init, 0)
end

# Optimization
@everywhere function f(x)
    myid() == 1 && return NaN
    try
        logdir = joinpath(homedir(), getnow() * "-worker-$(myid())")
        mkpath(logdir)
        cdall(logdir)

        logger(joinpath(logdir, "Diary")) do
            ℓ = mxcall(:perforientation_bbopt_caller, 1, x, homedir())
            println("loss = $ℓ")
            return ℓ
        end
    finally
        cdhome()
    end
end

# Solve
params = MAT.matread(joinpath(homedir(), "Params0.mat"))["Params0"]
bounds = vec(tuple.(params["lb"], params["ub"]))
res = bboptimize(bbsetup(
    f;
    Method = :dxnes,
    SearchRange = bounds,
    MaxTime = 47 * 3600,
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

# Plot results
resfiles = readdir(glob"2020**/*.mat", homedir())
results = matread.(resfiles)
diaries = joinpath.(dirname.(resfiles), "Diary.log")
times = diaries .|> dirname .|> basename .|> s -> DateTime(s[1:25], "yyyy-mm-dd-T-HH-MM-SS-sss")
losses = diaries .|> f -> readlines(f)[end] .|> l -> parse(Float64, l[8:end])
