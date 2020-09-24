using Distributed

# Instantiate worker environments
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere Pkg.instantiate()

@everywhere using Dates, Logging, Glob, ReadableRegex, MATLAB, Plots, LaTeXStrings
@everywhere import MAT, BSON, BlackBoxOptim, NLopt

# Helpers
@everywhere @eval jobdir() = $(get(ENV, "JOB_DIR", pwd()))
@everywhere @eval maxtime() = $(3600.0 * parse(Float64, get(ENV, "OPT_TIME_HOURS", "0")))
@everywhere @eval initguessdir() = $(get(ENV, "INIT_GUESS_DIR", ""))
@everywhere cdall(path) = (cd(path); mxcall(:cd, 0, path); nothing)
@everywhere cdhome() = cdall(jobdir())
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
        logdir = joinpath(jobdir(), getnow() * "-worker-$(myid())")
        mkpath(logdir)
        cdall(logdir)
        logger(joinpath(logdir, "Diary")) do
            ℓ = mxcall(:perforientation_bbopt_caller, 1, convert(Vector{Float64}, x), jobdir()) |> Float64
            println("loss = $ℓ")
            return ℓ
        end
    finally
        cdhome()
    end
end

@everywhere maybefire(file) = (isf = isfile(file); if isf; mv(file, file * ".fired"; force = true); end; return isf)
@everywhere cb_bbopt(ctrl::BlackBoxOptim.OptRunController) = maybefire(joinpath(jobdir(), "stop")) ? BlackBoxOptim.shutdown_optimizer!(ctrl) : nothing
@everywhere cb_nlopt(args...) = maybefire(joinpath(jobdir(), "stop")) ? throw(NLopt.ForcedStop()) : nothing

# Initial solve using global optimizer
function bbopt()
    params = MAT.matread(joinpath(jobdir(), "Params0.mat"))["Params0"]
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
        CallbackFunction = cb_bbopt,
        CallbackInterval = 0.0,
        Workers = workers(),
    ))

    # Save results
    MAT.matwrite(joinpath(jobdir(), "BBOptResults.mat"), Dict(
        "x" => deepcopy(best_candidate(res)),
        "loss" => best_fitness(res),
    ))

    BSON.bson(joinpath(jobdir(), "BBOptResults.bson"), Dict(
        "results" => deepcopy(res),
    ))

    return nothing
end

# Refine solution using local optimizer
function nlopt()
    lb, ub = let
        @assert isfile(joinpath(jobdir(), "Params0.mat"))
        params = MAT.matread(joinpath(jobdir(), "Params0.mat"))["Params0"]
        Float64.(vec(params["lb"])), Float64.(vec(params["ub"]))
    end
    x0 = let
        @assert !isempty(initguessdir()) && isdir(initguessdir()) && isfile(joinpath(initguessdir(), "IterationsResults.mat"))
        res = MAT.matread(joinpath(initguessdir(), "IterationsResults.mat"))
        AICc, i = findmin(res["AICc"])
        Float64[res["CA"][i], res["Rmajor"][i], res["MinorExpansion"][i]]
    end
    x0 = clamp.(x0, lb, ub)
    bounds = tuple.(lb, ub)

    #= toy problem
    local f(x) = (sleep(0.1); return sum(x.^4))
    x0 = randn(3)
    lb = fill(-2 * maximum(abs, x0), 3)
    ub = fill(+2 * maximum(abs, x0), 3)
    bounds = tuple.(lb, ub)
    =#

    call_f(x) = (cb_nlopt(); return f(x))

    function call_f_and_∇f!(x::Vector{Float64}, grad::Vector{Float64})
        δ = 0.02 .* (ub - lb)
        ê(i) = (e = zeros(length(x)); e[i] = 1.0; e)
        ∇dir(i) = (lb[i] + δ[i] < x[i] < ub[i] - δ[i]) ? rand((:FD, :BD)) : (x[i] < ub[i] - δ[i]) ? :FD : :BD # far from all boundaries --> randomly pick direction; else, choose direction away from nearby boundary
        xh(dir,he) = (dir === :FD) ? (x + he) : (x - he)
        ∇(dir,f0,f1,h) = (dir === :FD) ? (f1 - f0)/h : (f0 - f1)/h

        dirs = Symbol[∇dir(i) for i in 1:length(x)]
        xhs = Vector{Float64}[xh(dirs[i], δ[i] * ê(i)) for i in 1:length(x)]
        xs = Vector{Float64}[xhs; [copy(x)]]
        fs = pmap(call_f, xs) |> res -> convert(Vector{Float64}, res)
        grad .= Float64[∇(dirs[i], fs[end], fs[i], δ[i]) for i in 1:length(x)]

        """
        Loss = $(fs[end])
        Sample = $(x)
        Gradient = $(grad)
        """ |> println
        return fs[end]
    end

    f_nlopt(x::Vector{Float64}, grad::Vector{Float64}) = try
        length(grad) > 0 ? call_f_and_∇f!(x, grad) : call_f(x)
    catch e
        !(e isa NLopt.ForcedStop) && println(sprint(showerror, e, catch_backtrace()))
        rethrow(e)
    end

    opt = NLopt.Opt(:LD_LBFGS, length(bounds))
    opt.min_objective = f_nlopt
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.xtol_rel = 1e-3
    opt.ftol_rel = 1e-3
    opt.maxtime = maxtime()
    minf, minx, ret = NLopt.optimize(opt, x0)

    """
    Minimum:     AICc = $minf
    Minimizer:   [CA, Rmajor, MinorExpansion] = $minx
    Iterations:  $(opt.numevals)
    Return Code: $ret
    """ |> println

    # Save results
    MAT.matwrite(joinpath(jobdir(), "NLoptResults.mat"), Dict(
        "x" => copy(minx),
        "loss" => minf,
    ))

    BSON.bson(joinpath(jobdir(), "NLoptResults.bson"), Dict(
        "x" => deepcopy(minx),
        "loss" => deepcopy(minf),
        "ret" => deepcopy(ret),
        "lower_bounds" => deepcopy(opt.lower_bounds),
        "upper_bounds" => deepcopy(opt.upper_bounds),
        "stopval" => deepcopy(opt.stopval),
        "ftol_rel" => deepcopy(opt.ftol_rel),
        "ftol_abs" => deepcopy(opt.ftol_abs),
        "xtol_rel" => deepcopy(opt.xtol_rel),
        "xtol_abs" => deepcopy(opt.xtol_abs),
        "maxeval" => deepcopy(opt.maxeval),
        "maxtime" => deepcopy(opt.maxtime),
        "force_stop" => deepcopy(opt.force_stop),
        "algorithm" => deepcopy(opt.algorithm),
        "numevals" => deepcopy(opt.numevals),
        "errmsg" => deepcopy(opt.errmsg),
        # "population" => deepcopy(opt.population),
        # "vector_storage" => deepcopy(opt.vector_storage),
        # "results" => deepcopy(opt),
    ))

    return nothing
end

# Cleanup: save one large file instead of many small files
function cleanup()
    # Safest to read + re-save results within Matlab to ensure collect handling of class objects, function handles, etc.
    let
        files = readdir(glob"**/*.mat", jobdir()) .|> string
        outfile = joinpath(jobdir(), "AllIterationsResults.mat") |> string
        savetime = @elapsed mat"""
        AllIterationsResults = cell($(length(files)), 1);
        for ii = 1:numel(AllIterationsResults)
            AllIterationsResults{ii} = load($files{ii});
        end
        save($outfile, 'AllIterationsResults');
        """
        println("$(basename(jobdir())): collecting iteration results... ($(round(savetime, digits = 1)) s)")
    end

    # Zip all iteration results together
    let
        iterdirs = readdir(glob"*-worker-*", jobdir()) .|> string
        iterfiles = readdir(glob"*-worker-**/*", jobdir()) .|> string
        outfile = joinpath(jobdir(), "AllIterationsResults.zip") |> string
        try
            ziptime = @elapsed begin
                run(`zip -rq $outfile $iterdirs`; wait = true) # zip files

                zipinfo = read(`zipinfo -t $(jobdir())/AllIterationsResults.zip`, String) |> chomp
                numfiles = parse(Int, match(r"(\d+) files", zipinfo)[1])

                @assert length(iterfiles) + length(iterdirs) == numfiles # ensure all files accounted for
                run(`rm -rf $iterdirs`; wait = true)
            end
            println("$(basename(jobdir())): zipping iteration results... ($(round(ziptime, digits = 1)) s)")
        catch e
            println(sprint(showerror, e, catch_backtrace()))
        end
    end

    return nothing
end

logger(joinpath(jobdir(), "Diary")) do
    mxcall(:perforientation_bbopt_init, 0)
    # bbopt()
    nlopt()
    cleanup()
end

nothing
