using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Glob, ReadableRegex, Random, MAT

####
#### Submitting jobs
####

function submit(;
        JobNum::Int,
        JobTimeHours::Int,
        OptTime::Float64,
        Nmajor::Int,
        PVSvolume::Float64,
        Dtissue::Float64,
    )
    # Make output folder
    InitGuessSweepDir = "/project/st-arausch-1/jcd1994/GRE-DSC-PVS-Orientation2020Fall/2020-10-06-nlopt-v2"
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
    @info "Submitting job #$(JobNum)"
    pbsfile = joinpath(JobDir, "scripts", "jobfile.pbs")
    open(pbsfile; write = true) do io
        """
        #!/bin/bash
        #PBS -l walltime=$(JobTimeHours):00:00,select=4:ncpus=32:ompthreads=32:mem=64gb
        #PBS -N j-$(JobNum)
        #PBS -A st-arausch-1
        #PBS -m abe
        #PBS -j oe
        #PBS -o $(joinpath(JobDir, "PBS-output.txt"))
        #PBS -e $(joinpath(JobDir, "PBS-error.txt"))
        #PBS -V

        module load gcc/5.4.0 # for matlab
        module load intel-mkl/2019.3.199
        module load openmpi/3.1.4
        module load python/3.7.3
        module load matlab/R2019b

        # Julia/project settings
        export JULIA_BINDIR="/home/jcd1994/julia-1.5.1/bin"
        # export JULIA_DEPOT_PATH="/scratch/st-arausch-1/jcd1994/.julia:" # https://github.com/JuliaLang/julia/issues/34918#issuecomment-593001758
        export JULIA_DEPOT_PATH="/scratch/st-arausch-1/jcd1994/.julia"
        export JULIA_PROJECT=$(@__DIR__)
        export JULIA_SCRIPT=$(joinpath(@__DIR__, "perforientation_bbopt.jl"))
        export JULIA_NUM_THREADS=1024

        # Job settings
        export OPT_TIME_HOURS=$(OptTime)
        export SWEEP_DIR=$(SweepDir)
        export JOB_DIR=$(JobDir)
        export INIT_GUESS_DIR=$(joinpath(InitGuessSweepDir, basename(JobDir)))
        cd $(JobDir)

        # Sweep settings
        export Dtissue=$(Dtissue)
        export PVSvolume=$(PVSvolume)
        export Nmajor=$(Nmajor)
        export GeomLoadExisting=0
        """ |> s -> println(io, s)

        raw"""
        echo ------------------------------------------------------
        date; echo 'Job is running on node(s):'; cat ${PBS_NODEFILE}
        echo ------------------------------------------------------
        echo PBS: qsub is running on ${PBS_O_HOST}
        echo PBS: originating queue is ${PBS_O_QUEUE}
        echo PBS: executing queue is ${PBS_QUEUE}
        echo PBS: working directory is ${PBS_O_WORKDIR}
        echo PBS: execution mode is ${PBS_ENVIRONMENT}
        echo PBS: job identifier is ${PBS_JOBID}
        echo PBS: job name is ${PBS_JOBNAME}
        echo PBS: node file is ${PBS_NODEFILE}
        echo PBS: current home directory is ${PBS_O_HOME}
        echo PBS: PATH = ${PBS_O_PATH}
        echo ------------------------------------------------------
        time ${JULIA_BINDIR}/julia \
            --startup-file=no \
            --history-file=no \
            --machine-file=${PBS_NODEFILE} \
            ${JULIA_SCRIPT}
        echo ------------------------------------------------------
        date
        echo ------------------------------------------------------
        """ |> s -> println(io, s)
    end
    run(`qsub $pbsfile`)
end

function submit_jobs(;
        ShuffleSubmit, # Submit jobs in random order
        JobTimeHours, # Job time
        OptTime, # Simulation time
        Nmajor_all, # Number of major vessels
        PVSvolume_all, # PVS relative volume
        Dtissue_all, # Tissue diffusion
    )

    params_iter = enumerate(Iterators.product(Nmajor_all, PVSvolume_all, Dtissue_all)) |> collect
    if ShuffleSubmit
        params_iter = shuffle(MersenneTwister(0), params_iter)
    end

    # Optionally filter
    params_iter = filter(params_iter) do (JobNum, (Nmajor, PVSvolume, Dtissue))
        true
    end

    # Submit
    job_params = map(params_iter) do (JobNum, (Nmajor, PVSvolume, Dtissue))
        submit(; JobNum, JobTimeHours, OptTime, Nmajor, PVSvolume, Dtissue)
        return JobNum => (; JobTimeHours, OptTime, Nmajor, PVSvolume, Dtissue)
    end

    return sort(Dict(vec(job_params))), params_iter
end

job_params, params_iter = submit_jobs(;
    ShuffleSubmit = true,
    JobTimeHours = 48,
    OptTime = 42.0,
    Nmajor_all = 1:6,
    PVSvolume_all = [0.0, 0.5, 1.0, 2.0],
    Dtissue_all = [2.0, 3.0, 0.0],
)

####
#### Monitoring job progress + re-submitting failed jobs
####

resubmit(jobnum) = submit(; JobNum = jobnum, job_params[jobnum]...)

function capture_jobinfo()
    maybeint(m) = isnothing(m) ? m : parse(Int, m)
    maybestr(m) = isnothing(m) ? m : string(m)
    jobs = map(split(read(`qstat -u jcd1994`, String), '\n')) do l
        r = capture(one_or_more(DIGIT); as = :pid) * ".pbsha" *
            one_or_more(ANY) *
            "j-" * capture(one_or_more(DIGIT); as = :jobnum) *
            one_or_more(ANY) *
            " " * capture(WORD; as = :status) * " " *
            either(
                one_or_more(WHITESPACE) * "--" * one_or_more(WHITESPACE),
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

function monitor_jobinfo()
    jobinfo = sort(capture_jobinfo(); by = d -> d.jobnum)
    jobinfo_nums = Set(map(d -> d.jobnum, jobinfo))
    jobdirs = readdir(glob"Job-*_*", pwd())
    jobdir_num(jobdir) = parse(Int, match(r"Job-(\d+)_", jobdir)[1])
    jobdir_map = Dict{Int,String}(jobdir_num.(jobdirs) .=> jobdirs)
    jobs_to_check = intersect(
        setdiff(1:length(jobdirs), jobinfo_nums),
        filter(j -> !isfile(joinpath(jobdir_map[j], "stop.fired")), 1:length(jobdirs)),
    )

    @info "Total:    $(length(jobdirs))"
    @info "Status:   $(mapreduce(d -> d.status == "R", +, jobinfo)) R, $(mapreduce(d -> d.status == "Q", +, jobinfo)) Q, $(mapreduce(d -> d.status ∉ ("Q","R"), +, jobinfo)) other"
    @info "Finished: $(length(readdir(glob"**/stop.fired", pwd())))"
    @info "Check:    $(jobs_to_check)"
end
# while true; monitor_jobinfo(); sleep(60.0); end

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

# ensure_submit(Set(collect(keys(job_params))))

nothing
