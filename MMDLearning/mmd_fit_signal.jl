####
#### Code loading
####
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))

####
#### Load image data
####

const image   = DECAES.load_image(settings["prior"]["data"]["image"])
# const t2maps  = DECAES.MAT.matread(settings["prior"]["data"]["t2maps"])
# const t2dist  = DECAES.load_image(settings["prior"]["data"]["t2dist"])
# const t2parts = DECAES.MAT.matread(settings["prior"]["data"]["t2parts"])
const opts    = T2mapOptions(
    MatrixSize = size(image)[1:3],
    nTE        = size(image)[4],
    TE         = 8e-3,
    T2Range    = (15e-3, 2.0),
    nT2        = 40,
);
cp(
    settings["prior"]["data"]["settings"],
    joinpath(
        settings["data"]["out"],
        basename(settings["prior"]["data"]["settings"])
    )
)

####
#### Signal fitting
####

function test_signal_fit(signal::AbstractVector{T};
        TE::T                = T(opts.TE),
        T2Range::NTuple{2,T} = T.(opts.T2Range),
        nT2::Int             = opts.nT2,
        nTE::Int             = opts.nTE,
        ntheta::Int          = settings["data"]["ntheta"]::Int,
        theta_labels         = settings["data"]["theta_labels"]::Vector{String},
        losstype::Symbol     = Symbol(settings["prior"]["fitting"]["losstype"]::String),
        maxtime::T           = T(settings["prior"]["fitting"]["maxtime"]),
        plotfit::Bool        = false,
        kwargs...
    ) where {T}

    t2bins = T.(1000 .* DECAES.logrange(T2Range..., nT2)) # milliseconds
    t2times = T.(1000 .* TE .* (1:nTE)) # milliseconds
    xdata, ydata = t2times, signal./sum(signal)

    count = Ref(0)
    work = signal_model_work(T; nTE = nTE)
    ϵ = zeros(T,1)

    mlemodel = θ -> (count[] += 1; return signal_model!(work, uview(θ, 1:4), ϵ; TE = TE))
    mleloss = θ -> (@inbounds(work.buf .= .-logpdf.(Rician.(model(uview(θ, 1:4)), exp(θ[end])), ydata)); return sum(work.buf))

    l2model = θ -> (count[] += 1; return signal_model!(work, θ, ϵ; TE = TE))
    l2loss = θ -> (@inbounds(work.buf .= ydata .- model(θ)); return sum(abs2, work.buf))

    θbounds = theta_bounds(T)
    (losstype === :mle) && push!(θbounds, (-10.0, 0.0))
    model = losstype === :mle ? mlemodel : l2model
    loss = losstype === :mle ? mleloss : l2loss

    res = bboptimize(loss;
        SearchRange = θbounds,
        TraceMode = :silent,
        MaxTime = maxtime,
        MaxFuncEvals = typemax(Int),
        kwargs...
    )
    
    if plotfit
        θ = best_candidate(res)
        ℓ = best_fitness(res)
        @show count[]
        (losstype === :mle) && @show exp(θ[end])

        ymodel = model(θ)
        p = plot(xdata, [ydata ymodel];
            title = "fitness = $(round(ℓ; sigdigits = 3))",
            label = ["data" "model"],
            annotate = (t2times[3*(end÷4)], 0.5*maximum(ymodel), join(theta_labels .* " = " .* string.(round.(θ; sigdigits=4)), "\n")),
            xticks = t2times, xrotation = 70)
        display(p)
    end

    return make_save_df(res; losstype = losstype)
end

function make_save_df(res; losstype)
    # Results are BlackBoxOptim.OptimizationResults
    θ, ℓ = best_candidate(res), best_fitness(res)
    T = Float64
    return DataFrame(
        refcon   = T[180.0], # [deg]
        alpha    = T[θ[1]], # [deg]
        T2short  = T[θ[2]], # [ms]
        T2long   = T[θ[2] + θ[3]], # [ms]
        dT2      = T[θ[3]], # [ms]
        Ashort   = T[θ[4]], # [a.u.]
        Along    = T[1 - θ[4]], # [a.u.]
        T1short  = T[1000.0], # [ms]
        T1long   = T[1000.0], # [ms]
        logsigma = losstype === :mle ? T[θ[end]] : T[NaN],
        loss     = T[ℓ],
        # best_fitness   = T[best_fitness(res)],
        # elapsed_time   = T[res.elapsed_time],
        # f_calls        = Int[res.f_calls],
        # fit_scheme     = String[string(res.fit_scheme)],
        # iterations     = Int[res.iterations],
        # method         = String[string(res.method)],
        # start_time     = T[res.start_time],
        # stop_reason    = res.stop_reason,
        # archive_output = res.archive_output,
        # method_output  = res.method_output,
        # parameters     = res.parameters,
    )
end

function make_save_dict(res)
    # Results are BlackBoxOptim.OptimizationResults
    save_dict = Dict{String, Any}(
        "best_candidate"   => best_candidate(res),
        "best_fitness"     => best_fitness(res),
        "elapsed_time"     => res.elapsed_time,
        "f_calls"          => res.f_calls,
        "fit_scheme"       => string(res.fit_scheme),
        "iterations"       => res.iterations,
        "method"           => string(res.method),
        "start_time"       => res.start_time,
        "stop_reason"      => res.stop_reason,
        # "archive_output" => res.archive_output,
        # "method_output"  => res.method_output,
        # "parameters"     => res.parameters,
    )
    save_dict["settings"] = deepcopy(settings)
    save_dict["opts"] = Dict([string(f) => getfield(opts,f) for f in fieldnames(typeof(opts))])
    return save_dict
end

####
#### Signal fitting
####

const image_indices         = filter(I -> image[I,1] > 0, CartesianIndices(size(image)[1:3]))
const image_indices_batches = collect(Iterators.partition(image_indices, settings["prior"]["fitting"]["batchsize"]))
const image_indices_batch   = image_indices_batches[settings["prior"]["fitting"]["batchindex"]]

const batchindex = lpad(settings["prior"]["fitting"]["batchindex"], ndigits(length(image_indices_batches)), '0')
# const batchpath  = joinpath(settings["data"]["out"], "tmp-$batchindex") #TODO
# mkpath(batchpath) #TODO

df = mapreduce(vcat, image_indices_batch) do I
    # @show I #TODO
    df = test_signal_fit(image[I,:])
    df[!, :index] = NTuple{3,Int}[Tuple(I)]
    # BSON.bson(joinpath(batchpath, "tmp-$(join(lpad.(Tuple(I), 3, '0'), "_")).bson"), Dict{String,Any}("results" => deepcopy(df))) #TODO
    return df
end

DrWatson.@tagsave(
    joinpath(settings["data"]["out"], "results-$batchindex.bson"),
    Dict{String,Any}("results" => deepcopy(df));
    safe = true,
    gitpath = realpath(DrWatson.projectdir("..")),
)

#= Sample + fit random signals based on fit-to-noise ratio (FNR)
let I = rand(findall(I -> !isnan(t2maps["fnr"][I]) && 50 <= t2maps["fnr"][I] <= 100, CartesianIndices(size(image)[1:3])))
# let I = CartesianIndex(122, 112, 46)
    @show I
    test_signal_fit(image[I,:])
end
=#

#= Load indices of completed results
const resfiles = mapreduce(
        vcat,
        ["/project/st-arausch-1/jcd1994/simulations/nips2020/mlefit-v1"]
    ) do simdir
    mapreduce(vcat, readdir(simdir; join = true)) do dir
        resdir = joinpath(dir, "pbs-out", "tmp-" * lpad(basename(dir), 3, '0'))
        filter!(s -> endswith(s, ".bson"), readdir(resdir; join = true))
    end
end
const completed_indices = basename.(resfiles) .|> s -> CartesianIndex(map(x -> parse(Int, x), (s[5:7], s[9:11], s[13:15])))
const todo_indices = setdiff(image_indices, completed_indices)
const todo_indices_batches = collect(Iterators.partition(todo_indices, 1000))
=#

#= Load completed results
df = mapreduce(
        vcat,
        "/scratch/st-arausch-1/jcd1994/simulations/nips2020/mlefit-" .* ["v1"] #, "v2"]
    ) do simdir
    mapreduce(vcat, enumerate(readdir(simdir; join = true))) do (i, dir)
        resdir = joinpath(dir, "pbs-out", "tmp-" * lpad(basename(dir), 3, '0'))
        @info "$i / $(length(readdir(simdir))): $resdir"
        @time mapreduce(vcat, filter!(s -> endswith(s, ".bson"), readdir(resdir; join = true))) do file
            deepcopy(BSON.load(file)["results"])
        end
    end
end
DrWatson.@tagsave(
    "results-mlefit-v1.bson",
    Dict{String,Any}("results" => deepcopy(df));
    safe = true,
    gitpath = realpath(DrWatson.projectdir("..")),
)
=#

#= Shuffle + save mle fit results
df = mapreduce(
        vcat,
        "/scratch/st-arausch-1/jcd1994/simulations/nips2020/mlefit-" .* ["v1"]
    ) do simdir
    mapreduce(vcat, enumerate(readdir(simdir; join = true))) do (i, dir)
        resfile = joinpath(dir, "pbs-out", "results-" * lpad(basename(dir), 3, '0') * ".bson")
        @info "$i / $(length(readdir(simdir))): $(basename(resfile))"
        @time deepcopy(BSON.load(resfile)["results"])
    end
end
DrWatson.@tagsave(
    "/project/st-arausch-1/jcd1994/MMD-Learning/data/signal_fits/mlefits-shuffled.bson",
    Dict{String,Any}("results" => deepcopy(df[shuffle(MersenneTwister(0), 1:nrow(df)), :]));
    safe = true,
    gitpath = realpath(DrWatson.projectdir("..")),
)
=#

nothing
