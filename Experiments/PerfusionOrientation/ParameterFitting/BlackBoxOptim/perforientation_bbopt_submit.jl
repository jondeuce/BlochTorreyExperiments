using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Glob, ReadableRegex, Random, MAT

bash(str) = run(`bash -c $str`)

function submit(;
        JobNum::Int,
        MaxTime::Float64,
        Nmajor::Int,
        PVSvolume::Float64,
        Dtissue::Float64,
    )
    # Make output folder
    SweepDir = pwd()
    JobDir = joinpath(SweepDir, "Job-$(JobNum)_Dtissue-$(Dtissue)_Nmajor-$(Nmajor)_PVSvolume-$(PVSvolume)")
    mkpath(JobDir)

    # Copy scripts into output directory
    mkpath(joinpath(JobDir, "scripts"))
    map(readdir(@__DIR__; join = true)) do file
        cp(file, joinpath(JobDir, "scripts", basename(file)); force = true)
    end

    # Make sweep parameters file
    paramsfile = joinpath(JobDir, "SweepSettings.mat")
    params = Dict{String,Any}("Dtissue" => Dtissue, "PVSvolume" => PVSvolume, "Nmajor" => Nmajor)
    matwrite(paramsfile, params)

    # Submit job
    @info "Submitting job #$JobNum"
    """
    # Julia/project settings
    export JULIA_BINDIR="/home/jcd1994/julia-1.5.1/bin"
    # export JULIA_DEPOT_PATH="/scratch/st-arausch-1/jcd1994/.julia:" # https://github.com/JuliaLang/julia/issues/34918#issuecomment-593001758
    export JULIA_DEPOT_PATH="/scratch/st-arausch-1/jcd1994/.julia"
    export JULIA_PROJECT=$(@__DIR__)
    export JULIA_SCRIPT=$(joinpath(@__DIR__, "perforientation_bbopt.jl"))
    export JULIA_NUM_THREADS=1024

    # Job settings
    export SWEEP_DIR=$SweepDir
    export JOB_DIR=$JobDir
    export MAX_TIME_HOURS=$MaxTime

    # Make sweep settings
    export Dtissue=$Dtissue
    export PVSvolume=$PVSvolume
    export Nmajor=$Nmajor

    # Submit job
    qsub -N j-$JobNum -o $(joinpath(JobDir, "PBS-output.txt")) -o $(joinpath(JobDir, "PBS-error.txt")) -V $(joinpath(@__DIR__, "perforientation_bbopt.pbs"))
    """
    # |> println
    # |> bash
end

function submit_jobs(;
        ShuffleSubmit, # Submit jobs in random order
        MaxTime, # Simulation time
        Nmajor_all, # Number of major vessels
        PVSvolume_all, # PVS relative volume
        Dtissue_all, # Tissue diffusion
    )

    params_iter = enumerate(Iterators.product(Nmajor_all, PVSvolume_all, Dtissue_all)) |> collect
    if ShuffleSubmit
        params_iter = shuffle(MersenneTwister(0), params_iter)
    end

    job_params = map(params_iter) do (JobNum, (Nmajor, PVSvolume, Dtissue))
        submit(; JobNum, MaxTime, Nmajor, PVSvolume, Dtissue)
        return JobNum => (; MaxTime, Nmajor, PVSvolume, Dtissue)
    end

    return sort(Dict(vec(job_params)))
end

job_params = submit_jobs(;
    ShuffleSubmit = true,
    MaxTime = 47.0,
    Nmajor_all = 1:6,
    PVSvolume_all = [0.0, 0.5, 1.0, 2.0],
    Dtissue_all = [2.0, 3.0, 0.0],
)

resubmit(jobnum) = submit(; JobNum = jobnum, job_params[jobnum]...)

function capture_jobinfo()
    maybeint(m) = isnothing(m) ? m : parse(Int, m)
    maybestr(m) = isnothing(m) ? m : string(m)
    jobs = map(split(read(`qstat -u jcd1994`, String), '\n')) do l
        r = capture(one_or_more(DIGIT); as = :pid) * ".pbsha" *
            one_or_more(ANY) *
            "j-" * capture(one_or_more(DIGIT); as = :jobnum) *
            one_or_more(ANY) *
            "256gb 48:00 " * capture(WORD; as = :status) * " " *
            either(
                "--:--",
                capture(DIGIT * DIGIT; as = :hour) * ":" * capture(DIGIT * DIGIT; as = :min)
            )
        m = match(r, l)
        if !isnothing(m)
            return (pid = maybeint(m[:pid]), jobnum = maybeint(m[:jobnum]), status = maybestr(m[:status]), hour = maybeint(m[:hour]), min = maybeint(m[:min]))
        else
            return nothing
        end
    end
    filter(!isnothing, jobs)
end

function ensure_submit(todo = Set{Int}())
    while !isempty(todo)
        jobinfo = capture_jobinfo()
        for jobnum ∈ todo
            jobids = findall(i -> i.jobnum == jobnum, jobinfo)
            if isempty(jobids)
                if !isempty(readdir(Glob.GlobMatch("Job-$(jobnum)_*/AllIterationsResults.*"), pwd())) || !isempty(readdir(Glob.GlobMatch("Job-$(jobnum)_*/*-worker-*/*.mat"), pwd()))
                    @info "Skipping job #$jobnum: completed"
                else
                    @info "Submitting job #$jobnum: not in queue"
                    resubmit(jobnum)
                end
            else
                info = jobinfo[jobids]
                if all(i -> i.status == "E", info)
                    @info "Re-submitting job #$jobnum: error status"
                    resubmit(jobnum)
                elseif any(i -> i.status == "Q", info)
                    @info "Skipping job #$jobnum: queueing"
                elseif any(i -> i.status == "R", info)
                    running_info = info[findall(i -> i.status == "R", info)]
                    mins = map(i -> any(isnothing, (i.hour, i.min)) ? 0 : i.min + 60 * i.hour, running_info)
                    if any(>=(15), mins)
                        @info "Running job #$jobnum (success): run time >= 15 min"
                        delete!(todo, jobnum)
                    else
                        @info "Running job #$jobnum (waiting): run time < 15 min"
                    end
                else
                    @warn "Unknown state: $(info)"
                end
            end
        end
        @info "Remaining jobs: $todo"
        !isempty(todo) && sleep(60.0)
    end
end

ensure_submit(Set(collect(keys(job_params))))

nothing