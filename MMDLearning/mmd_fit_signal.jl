####
#### Code loading
####
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
Random.seed!(0)

####
#### Load image data
####

const mmd_settings = load_settings();
const image   = DECAES.load_image(mmd_settings["prior"]["data"]["image"]);
#const t2maps = DECAES.MAT.matread(mmd_settings["prior"]["data"]["t2maps"]);
#const t2dist = DECAES.load_image(mmd_settings["prior"]["data"]["t2dist"]);
#const t2parts= DECAES.MAT.matread(mmd_settings["prior"]["data"]["t2parts"]);
const opts    = T2mapOptions(
    MatrixSize = size(image)[1:3],
    nTE        = size(image)[4],
    TE         = 8e-3,
    T2Range    = (15e-3, 2.0),
    nT2        = 40,
);
# cp(mmd_settings["prior"]["data"]["settings"], joinpath(mmd_settings["data"]["out"], basename(mmd_settings["prior"]["data"]["settings"]))); #TODO

####
#### Signal fitting
####

function signal_fit(
        signal::AbstractVector{T};
        TE::T                = T(opts.TE),
        T2Range::NTuple{2,T} = T.(opts.T2Range),
        nT2::Int             = opts.nT2,
        nTE::Int             = opts.nTE,
        ntheta::Int          = mmd_settings["data"]["ntheta"]::Int,
        theta_labels         = mmd_settings["data"]["theta_labels"]::Vector{String},
        losstype::Symbol     = Symbol(mmd_settings["prior"]["fitting"]["losstype"]::String),
        maxtime::T           = T(mmd_settings["prior"]["fitting"]["maxtime"]),
        plotfit::Bool        = false,
        kwargs...
    ) where {T}

    t2bins = T.(1000 .* DECAES.logrange(T2Range..., nT2)) # milliseconds
    t2times = T.(1000 .* TE .* (1:nTE)) # milliseconds
    xdata, ydata = t2times, signal./sum(signal)

    count = Ref(0)
    work = signal_model_work(T; nTE = nTE)
    ϵ = nothing

    mlemodel(θ) = (count[] += 1; return signal_model!(work, uview(θ, 1:ntheta), ϵ; TE = TE))
    mleloss(θ) = (@inbounds(work.buf .= .-logpdf.(Rician.(mlemodel(uview(θ, 1:ntheta)), exp(θ[end])), ydata)); return sum(work.buf))

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
        #Along   = T[1 - θ[4]], # [a.u.] #TODO
        Along    = T[θ[5]], # [a.u.] #TODO
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
    save_dict["settings"] = deepcopy(mmd_settings)
    save_dict["opts"] = Dict([string(f) => getfield(opts,f) for f in fieldnames(typeof(opts))])
    return save_dict
end

####
#### Batch signal fitting
####

#= Perform fitting of one "batch"
const image_indices         = shuffle(MersenneTwister(0), filter(I -> image[I,1] > 0, CartesianIndices(size(image)[1:3]))) #TODO
const image_indices_batches = collect(Iterators.partition(image_indices, mmd_settings["prior"]["fitting"]["batchsize"]))
const image_indices_batch   = image_indices_batches[mmd_settings["prior"]["fitting"]["batchindex"]]
const batchindex            = lpad(mmd_settings["prior"]["fitting"]["batchindex"], ndigits(length(image_indices_batches)), '0')

df = mapreduce(vcat, image_indices_batch) do I
    df = signal_fit(image[I,:])
    df[!, :index] = NTuple{3,Int}[Tuple(I)]
    return df
end

DrWatson.@tagsave(
    joinpath(mmd_settings["data"]["out"], "results-$batchindex.bson"),
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
        ["/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-v3"]
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
        "/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-" .* ["v3"] #, "v3"]
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
    "results-mlefit-v3.bson",
    Dict{String,Any}("results" => deepcopy(df));
    safe = true,
    gitpath = realpath(DrWatson.projectdir("..")),
)
=#

#= Shuffle + save mle fit results
df = mapreduce(
        vcat,
        "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/mlefit-" .* ["v3"]
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
    "/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-v3/mlefits-shuffled.bson",
    Dict{String,Any}("results" => deepcopy(df[shuffle(MersenneTwister(0), 1:nrow(df)), :]));
    safe = true,
    gitpath = realpath(DrWatson.projectdir("..")),
)
=#

####
#### Investigating correcting fit results
####

function make_mle_funs(
        signal::AbstractVector{T};
        TE::T                = T(opts.TE),
        T2Range::NTuple{2,T} = T.(opts.T2Range),
        nT2::Int             = opts.nT2,
        nTE::Int             = opts.nTE,
        ntheta::Int          = mmd_settings["data"]["ntheta"]::Int,
    ) where {T}

    t2bins = T.(1000 .* DECAES.logrange(T2Range..., nT2)) # milliseconds
    t2times = T.(1000 .* TE .* (1:nTE)) # milliseconds
    xdata, ydata = t2times, signal./sum(signal)

    work = signal_model_work(T; nTE = nTE)
    mlemodel(θ) = signal_model!(work, uview(θ, 1:ntheta), nothing; TE = TE)
    mleloss(θ) = (@inbounds(work.buf .= .-logpdf.(Rician.(mlemodel(uview(θ, 1:ntheta)), exp(θ[end])), ydata)); return sum(work.buf))

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

#= plot distributions
=#
empty!(Revise.queue_errors);
df = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-v3/mlefits-shuffled-normalized.bson")["results"])
let
    res = deepcopy(df)
        filter!(r -> r.opttime <= 4.99, res) # drop non-converged fits
    filter!(r -> !(999.99 <= r.dT2 && 999.99 <= r.T2short), res) # drop boundary failures
    filter!(r -> r.dT2 <= 999.99, res) # drop boundary failures
    filter!(r -> r.dT2 <= 250, res) # drop long dT2 which can't be recovered
    filter!(r -> r.T2short <= 100, res) # drop long T2short (very few points)
    filter!(r -> 8.01 <= r.T2short, res) # drop boundary failures
    filter!(r -> 8.01 <= r.dT2, res) # drop boundary failures
    filter!(r -> 0.005 <= r.Ashort <= 0.15, res) # drop outlier fits (very few points)
    filter!(r -> 0.005 <= r.Along <= 0.15, res) # drop outlier fits (very few points)
    filter!(r -> r.loss <= -250, res) # drop poor fits (very few points)
    # filter!(r -> 0.01 <= r.Ashort <= 0.99, res) # drop degenerate (monoexponential) fits
    # filter!(r -> r.alpha <= 179.99, res)
    # filter!(r -> 0.1 <= r.Ashort <= 0.9, res) # window filter
    # filter!(r -> r.rmse <= 0.01, res) # drop poor fits (very few points)

    @show nrow(res)/nrow(df)

    plot(map([:alpha, :T2short, :dT2, :Ashort, :Along, :loss]) do col
        @show col, quantile(res[!,col], 0.99)
        th = res[:,col]
        lab = string(col)
        (col === :alpha) && (th .= cosd.(th); lab = "cosd(alpha)")
        # (col === :alpha) && (th .= sind.(th); lab = "sind(alpha)")
        histogram(th; lab = lab)
    end...) |> display

    # filter!(r -> r.loss <= -350, res)
    filter!(r -> -275 <= r.loss <= -250, res)
    # filter!(r -> -250 <= r.loss <= -225, res)
    # filter!(r -> -225 <= r.loss <= -200, res)
    idx   = CartesianIndex.(res[:, :index])
    theta = permutedims(convert(Matrix, res[:, [:alpha, :T2short, :dT2, :Ashort, :Along, :logsigma]]))
    # map(sample(1:nrow(res), 10; replace = false)) do i
    #     y0 = image[idx[i],:]
    #     mlemodel, mleloss = make_mle_funs(y0)
    #     y = mlemodel(theta[:,i])
    #     plot([y0./sum(y0) y]; lab = ["true" "fit"], title = "loss = $(res.loss[i]), rmse = $(res.rmse[i])") |> display
    # end

    nothing
end

#= add rmse column
let
    df.rmse = map(1:nrow(df)) do i
    # for i in 1:nrow(df)
        I = CartesianIndex(df[i, :index])
        theta = convert(Vector, df[i, [:alpha, :T2short, :dT2, :Ashort, :Along, :logsigma]])
        y0 = unitsum(image[I,:]; dims = 1)
        mlemodel, mleloss = make_mle_funs(y0)
        y = mlemodel(theta)
        return sqrt(sum(abs2, y .- y0))
        # df.rmse[i] = sqrt(sum(abs2, y .- y0))
        # if df.rmse[i] > 0.01
        #     plot([y y0]; title = "loss = $(df.loss[i]), rmse = $(df.rmse[i])") |> display
        # end
    end
end
=#

#= normalize Ashort, Along
empty!(Revise.queue_errors);
df = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/data/mlefit-v3/mlefits-shuffled-normalized.bson")["results"])
let
    nTE = 48
    TE = 8e-3
    sig_work = signal_model_work(Float64; nTE = nTE)
    for i in 1:10 #nrow(df)
        mod(i, 10000) == 0 && println("i = $i/$(nrow(df))")

        I = CartesianIndex(df[i, :index])
        theta = copy(convert(Vector, df[i, [:alpha, :T2short, :dT2, :Ashort, :Along, :logsigma]]))
        y0 = image[I,:]
        y0 ./= sum(y0)
        mlemodel, mleloss = make_mle_funs(y0)
        y = mlemodel(theta)

        # alpha, T2short, T2long, Ashort, Along
        theta[4:5] .*= 1+rand()
        signal = signal_model!(sig_work, theta[1:end-1], nothing; TE = TE, normalize = false)
        @show sum(signal)

        theta[4:5] ./= sum(signal)
        signal = signal_model!(sig_work, theta[1:end-1], nothing; TE = TE, normalize = false)
        @show sum(signal)

        @assert isapprox(signal, y)

        # df.Ashort[i] = theta[4]
        # df.Along[i] = theta[5]
    end
end
=#

#= padded mle samplers
=#
let
    sampleY, _, sampleθ = make_mle_data_samplers(mmd_settings["prior"]["data"]["image"]::String, mmd_settings["prior"]["data"]["thetas"]::String; ntheta = mmd_settings["data"]["ntheta"]::Int, plothist = false);
    Y = copy(sampleY(nothing; dataset = :train));
    θ = copy(sampleθ(nothing; dataset = :train));
    # @show all(isapprox.(sum(Y;dims=1), 1))
    # @show Y ≈ signal_model(θ, nothing; nTE = 48, TE = 8e-3, normalize = true)
    plot(
        histogram(cosd.(θ[1,:]); lab = "cosd(alpha)"),
        histogram(θ[2,:]; lab = "T2short"),
        histogram(θ[3,:]; lab = "dT2"),
        histogram(θ[4,:]; lab = "Ashort"),
        histogram(θ[5,:]; lab = "Along"),
        histogram(θ[5,θ[5,:].<0.01]; lab = "Along"),
    ) |> display
end

#=
#TODO: for watching failed qsubs
while true
    for dir in readdir("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/mlefit-v3"; join = true)
        outpath = joinpath(dir, "pbs-out")
        if !isempty(filter(s -> startswith(s, "output"), readdir(outpath))) && isempty(filter(s -> startswith(s, "results"), readdir(outpath)))
            print("$(basename(dir)) ")
        end
    end
    println("\n")
    sleep(5.0)
end
map(readdir("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/mlefit-v3"; join = true)) do dir
    isfile(joinpath(dir, "pbs-out", "settings.toml")) &&
    isfile(joinpath(dir, "pbs-out", "sweep_settings.toml")) &&
    isfile(joinpath(dir, "pbs-out", "masked-image-240x240x48x48.settings-240x240x48x48.txt"))
end |> sum
=#

####
#### Make learned maps
####

#=
m = BSON.load("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/cvae-test-v1/sweep/117/log/2020-04-24-T-17-31-25-591.acc=rmse_gamma=1_loss=l2_DenseLIGOCVAE_Dh=32_Nh=3_Xout=4_Zdim=10_act=relu_dropout=0.model-best.bson")[:model];
Is = filter(I -> image[I,1] > 0, CartesianIndices(size(image)[1:3]));
model_mu_std = m(unitsum(permutedims(image[Is,:]); dims = 1); nsamples = 100, stddev = true); #TODO
model_thetas, model_stds = model_mu_std[1:end÷2, ..], model_mu_std[end÷2+1:end, ..];

out = zeros(size(image)[1:3]..., size(model_thetas,1));
out[Is, :] .= permutedims(model_thetas);
CUTOFF = 50.0
BANDWIDTH = 2.0
mwf = zeros(size(image)[1:3]);
# mwf[Is] .= ((T2short, Ashort) -> ifelse(T2short < CUTOFF, Ashort, 0.0)).(out[Is, 2], out[Is, 4]);
mwf[Is] .= ((T2short, Ashort) -> Ashort * sigmoid(-(T2short - CUTOFF) / BANDWIDTH)).(out[Is, 2], out[Is, 4]);

Islice = (50:190, 210:-1:40, 24);
heatmap(permutedims(out[Islice...,1]) |> img -> (x -> acosd(clamp(x, -1.0, 1.0))).(img); aspect_ratio = :equal, clim = (120,180), title = "flip angle [deg]") |> p -> savefig(p, "flipangle.png");
heatmap(permutedims(out[Islice...,2]) |> img -> (x -> clamp(x < CUTOFF ? x : 0.0, 8.0, 100.0)).(img); aspect_ratio = :equal, clim = (0,CUTOFF), title = "T2short [ms]") |> p -> savefig(p, "T2short.png");
heatmap(permutedims(out[Islice...,3]) |> img -> (x -> clamp(x, 8.0, 500.0)).(img); aspect_ratio = :equal, clim = (0,500), title = "T2long [ms]") |> p -> savefig(p, "T2long.png");
heatmap(permutedims(out[Islice...,4]) |> img -> (x -> clamp(x, 0.0, 1.0)).(img); aspect_ratio = :equal, clim = (0,1), title = "Ashort [a.u.]") |> p -> savefig(p, "Ashort.png");

heatmap(permutedims(mwf[Islice...]) |> img -> (x -> clamp(x, 0.0, 1.0)).(img); aspect_ratio = :equal, clim = (0,1.0), title = "MWF [a.u.]") |> p -> savefig(p, "MWF.png");
heatmap(permutedims(t2parts["sfr"][Islice...]) |> img -> (x -> ifelse(isnan(x), 0.0, x)).(img); aspect_ratio = :equal, clim = (0,0.3), title = "sfr [a.u.]") |> p -> savefig(p, "sfr.png");
heatmap(permutedims(t2parts["sgm"][Islice...]) |> img -> (x -> ifelse(isnan(x), 0.0, 1000x)).(img); aspect_ratio = :equal, clim = (0,CUTOFF), title = "sgm [ms]") |> p -> savefig(p, "sgm.png");
heatmap(permutedims(t2maps["ggm"][Islice...]) |> img -> (x -> ifelse(isnan(x), 0.0, 1000x)).(img); aspect_ratio = :equal, clim = (0,500), title = "ggm [ms]") |> p -> savefig(p, "ggm.png");
=#

nothing
