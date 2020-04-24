####
#### Code loading
####
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
Random.seed!(0)

####
#### Load image data
####

const settings = load_settings()
const image    = DECAES.load_image(settings["prior"]["data"]["image"])
#const t2maps  = DECAES.MAT.matread(settings["prior"]["data"]["t2maps"])
#const t2dist  = DECAES.load_image(settings["prior"]["data"]["t2dist"])
#const t2parts = DECAES.MAT.matread(settings["prior"]["data"]["t2parts"])
const opts     = T2mapOptions(
    MatrixSize = size(image)[1:3],
    nTE        = size(image)[4],
    TE         = 8e-3,
    T2Range    = (15e-3, 2.0),
    nT2        = 40,
);
cp(settings["prior"]["data"]["settings"], joinpath(settings["data"]["out"], basename(settings["prior"]["data"]["settings"])))

####
#### Signal fitting
####

function signal_fit(
        signal::AbstractVector{T};
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
    ϵ = nothing

    mlemodel(θ) = (count[] += 1; return signal_model!(work, uview(θ, 1:4), ϵ; TE = TE))
    mleloss(θ) = (@inbounds(work.buf .= .-logpdf.(Rician.(mlemodel(uview(θ, 1:4)), exp(θ[end])), ydata)); return sum(work.buf))

    l2model(θ) = (count[] += 1; return signal_model!(work, θ, ϵ; TE = TE))
    l2loss(θ) = (@inbounds(work.buf .= ydata .- l2model(θ)); return sum(abs2, work.buf))

    θbounds = theta_bounds(T; ntheta = ntheta)
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
        @show BlackBoxOptim.stop_reason(res)
        (losstype === :mle) && @show exp(θ[end])

        ymodel = model(θ)
        p = plot(xdata, [ydata ymodel];
            title = "fitness = $(round(ℓ; sigdigits = 3))",
            label = ["data" "model"],
            annotate = (t2times[3*(end÷4)], 0.5*maximum(ymodel), join(["$lab = $(round(θ[i]; sigdigits=4))" for (i,lab) in enumerate(theta_labels)], "\n")),
            xticks = t2times, xrotation = 70)
        display(p)
    end

    return make_save_df(res; losstype = losstype)
end

function make_save_df(res; losstype)
    # Results are BlackBoxOptim.OptimizationResults
    θ = BlackBoxOptim.best_candidate(res)
    ℓ = BlackBoxOptim.best_fitness(res)
    fcalls = BlackBoxOptim.f_calls(res)
    opttime = BlackBoxOptim.elapsed_time(res)
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
        fcalls   = Int[fcalls],
        opttime  = T[opttime],
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
#### Batch signal fitting
####

#= Perform fitting of one "batch"
const image_indices         = shuffle(MersenneTwister(0), filter(I -> image[I,1] > 0, CartesianIndices(size(image)[1:3]))) #TODO
const image_indices_batches = collect(Iterators.partition(image_indices, settings["prior"]["fitting"]["batchsize"]))
const image_indices_batch   = image_indices_batches[settings["prior"]["fitting"]["batchindex"]]
const batchindex            = lpad(settings["prior"]["fitting"]["batchindex"], ndigits(length(image_indices_batches)), '0')

df = mapreduce(vcat, image_indices_batch) do I
    df = signal_fit(image[I,:])
    df[!, :index] = NTuple{3,Int}[Tuple(I)]
    return df
end

DrWatson.@tagsave(
    joinpath(settings["data"]["out"], "results-$batchindex.bson"),
    Dict{String,Any}("results" => deepcopy(df));
    safe = true,
    gitpath = realpath(DrWatson.projectdir("..")),
)
=#

#= Sample + fit random signals based on fit-to-noise ratio (FNR)
let I = rand(findall(I -> !isnan(t2maps["fnr"][I]) && 50 <= t2maps["fnr"][I] <= 100, CartesianIndices(size(image)[1:3])))
# let I = CartesianIndex(122, 112, 46)
    @show I
    signal_fit(image[I,:])
end
=#

#= Load indices of completed results
const resfiles = mapreduce(
        vcat,
        ["/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-v2"]
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
        "/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-" .* ["v2"] #, "v2"]
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
    "results-mlefit-v2.bson",
    Dict{String,Any}("results" => deepcopy(df));
    safe = true,
    gitpath = realpath(DrWatson.projectdir("..")),
)
=#

#= Shuffle + save mle fit results
df = mapreduce(
        vcat,
        "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/mlefit-" .* ["v2"]
    ) do simdir
    mapreduce(vcat, enumerate(readdir(simdir; join = true))) do (i, dir)
        resfile = joinpath(dir, "pbs-out", "results-" * lpad(basename(dir), 3, '0') * ".bson")
        if isfile(resfile)
            @info "$i / $(length(readdir(simdir))): $(basename(resfile))"
            @time deepcopy(BSON.load(resfile)["results"])
        else
            DataFrame()
        end
    end
end
DrWatson.@tagsave(
    "/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-v2/mlefits-shuffled.bson",
    Dict{String,Any}("results" => deepcopy(df[shuffle(MersenneTwister(0), 1:nrow(df)), :]));
    safe = true,
    gitpath = realpath(DrWatson.projectdir("..")),
)
=#

####
#### Investigating correcting fit results
####

#= =#
function make_mle_funs(
        signal::AbstractVector{T};
        TE::T                = T(opts.TE),
        T2Range::NTuple{2,T} = T.(opts.T2Range),
        nT2::Int             = opts.nT2,
        nTE::Int             = opts.nTE,
        ntheta::Int          = settings["data"]["ntheta"]::Int,
    ) where {T}

    t2bins = T.(1000 .* DECAES.logrange(T2Range..., nT2)) # milliseconds
    t2times = T.(1000 .* TE .* (1:nTE)) # milliseconds
    xdata, ydata = t2times, signal./sum(signal)

    work = signal_model_work(T; nTE = nTE)
    mlemodel(θ) = signal_model!(work, uview(θ, 1:4), nothing; TE = TE)
    mleloss(θ) = (@inbounds(work.buf .= .-logpdf.(Rician.(mlemodel(uview(θ, 1:4)), exp(θ[end])), ydata)); return sum(work.buf))

    return mlemodel, mleloss
end

function correct_results(df)
    loss0  = df[:, :loss]
    idx    = CartesianIndex.(df[:, :index])
    theta0 = permutedims(convert(Matrix, df[:, [:alpha, :T2short, :dT2, :Ashort, :logsigma]]))
    
    N = 1000
    loss = map(1:N) do i
        mlemodel, mleloss = make_mle_funs(image[idx[i],:])
        theta = theta0[:,i]
        theta[4] = 0.0
        # @show loss0[i]
        # @show mleloss(theta0[:,i])
        mleloss(theta)
    end

    # histogram(loss0) |> display
    histogram(loss0[1:N]) |> display
    histogram(loss[1:N]) |> display
    histogram(loss0[1:N] - loss[1:N]) |> display
    histogram(theta0[4,:]) |> display

    nothing
end

# correct_results(df)
empty!(Revise.queue_errors);
df = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-v2/mlefits-shuffled.bson")["results"])
let
    res = deepcopy(df)
    filter!(r -> !(999.99 <= r.dT2 && 999.99 <= r.T2short), res) # drop boundary failures
    # filter!(r -> r.dT2 <= 999.99, res) # drop boundary failures
    filter!(r -> r.dT2 <= 500, res) # drop long dT2 which can't be recovered
    filter!(r -> 0.01 <= r.Ashort <= 0.99, res) # drop degenerate (monoexponential) fits
    # filter!(r -> r.alpha <= 179.99, res)
    # filter!(r -> 0.1 <= r.Ashort <= 0.9, res) # window filter
    filter!(r -> r.T2short <= 100, res) # drop long T2short (very few points)
    filter!(r -> 8.01 <= r.T2short, res) # drop boundary failures
    filter!(r -> 8.01 <= r.dT2, res) # drop boundary failures
    filter!(r -> r.loss <= -250, res) # drop poor fits long T2short (very few points)

    @show nrow(res)/nrow(df)

    plot(map([:alpha, :T2short, :dT2, :Ashort, :loss]) do col
        @show col, quantile(res[!,col], 0.99)
        # lo, up = extrema(res[!,col])
        # trans = x -> -cos(pi * (x - lo)/(up - lo)) # scale [lo,up] -> [0,pi] -> [0,1]
        # trans = identity
        th = res[:,col]
        (col === :alpha) && (th .= cosd.(th))
        # (col === :Ashort) && (th .= sind.(90 .* th))
        histogram(th; lab = col)
    end...) |> display

    # filter!(r -> r.loss <= -350, res)
    filter!(r -> -275 <= r.loss <= -250, res)
    # filter!(r -> -250 <= r.loss <= -230, res)
    # filter!(r -> -225 <= r.loss <= -200, res)
    idx   = CartesianIndex.(res[:, :index])
    theta = permutedims(convert(Matrix, res[:, [:alpha, :T2short, :dT2, :Ashort, :logsigma]]))
    # map(sample(1:nrow(res), 10; replace = false)) do i
    #     y0 = image[idx[i],:]
    #     mlemodel, mleloss = make_mle_funs(y0)
    #     y = mlemodel(theta[:,i])
    #     plot([y0./sum(y0) y]; lab = ["true" "fit"], title = "loss = $(res.loss[i])") |> display
    # end

    nothing
end

#=
#TODO: for watching failed qsubs
while true
    for dir in readdir("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/mlefit-v2"; join = true)
        outpath = joinpath(dir, "pbs-out")
        if !isempty(filter(s -> startswith(s, "output"), readdir(outpath))) && isempty(filter(s -> startswith(s, "results"), readdir(outpath)))
            print("$(basename(dir)) ")
        end
    end
    println("\n")
    sleep(5.0)
end
=#

nothing
