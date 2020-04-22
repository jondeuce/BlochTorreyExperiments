using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

####
#### Plot sweep results
####

using MWFLearning
using GlobalUtils
using MAT, BSON, TOML, DataFrames
using Statistics, StatsPlots
# pyplot(size = (1200,800))
pyplot(size = (800,600))
# pyplot(size = (500,400))
empty!(Revise.queue_errors)

function flatten_dict!(dout::Dict{<:AbstractString, Any}, din::Dict{<:AbstractString, Any})
    for (k,v) in din
        if v isa Dict
            vnew = Dict{String, Any}([k * "." * kin => deepcopy(vin) for (kin,vin) in v])
            flatten_dict!(dout, vnew)
        else
            dout[k] = deepcopy(v)
        end
    end
    dout
end
flatten_dict(din::Dict{<:AbstractString, Any}) = flatten_dict!(Dict{String, Any}(), din)

findendswith(dir, suffix) = filter!(s -> endswith(s, suffix), readdir(dir)) |> x -> isempty(x) ? nothing : joinpath(dir, first(x))

function read_to_dataframe(sweep_dir)
    sweep = DataFrame()
    for (k,v) in flatten_dict(TOML.parsefile(joinpath(sweep_dir, "sweep_settings.toml")))
        sweep = hcat(sweep, DataFrame(Symbol(k) => typeof(v)[v]))
    end
    state = BSON.load(findendswith(joinpath(sweep_dir, "log"), ".errors.bson"))[:state]
    state = state[state.dataset .== :test, :]
    # dropmissing!(state) # drop rows with missings
    filter!(r -> all(x -> !((x isa Number && isnan(x)) || (x isa AbstractArray{<:Number} && any(isnan, x))), r), state) # drop rows with NaNs
    labelerr = skipmissing(state.labelerr)
    metrics = DataFrame(
        :loss     => minimum(skipmissing(state.loss)), # - sweep[1, Symbol("model.DenseLIGOCVAE.Zdim")]*(log(2π)-1)/2, # Correct for different Zdim's
        :acc      => maximum(skipmissing(state.acc)),
        :freq     => minimum((x->x[1]).(labelerr)),
        :phase    => minimum((x->x[2]).(labelerr)),
        :offset   => minimum((x->x[3]).(labelerr)),
        :amp      => minimum((x->x[4]).(labelerr)),
        :tconst   => minimum((x->x[5]).(labelerr)),
        :inf_med  => minimum(median.(labelerr)),
        :inf_mean => minimum(mean.(labelerr)),
        :inf_sup  => minimum(maximum.(labelerr)),
    )
    metrics.loss_epoch = Int[state.epoch[findfirst(state.loss .== metrics.loss[1])]]
    metrics.acc_epoch  = Int[state.epoch[findfirst(state.acc .== metrics.acc[1])]]

    return sweep, metrics
end

function check_read_path(sweep_dir, item)
    isdir(joinpath(sweep_dir, item)) &&
    isdir(joinpath(sweep_dir, item, "log")) &&
    isfile(joinpath(sweep_dir, item, "sweep_settings.toml")) &&
    !isnothing(findendswith(joinpath(sweep_dir, item, "log"), ".errors.bson"))
end

function dataframe_template(sweep_dir)
    for item in readdir(sweep_dir)
        if check_read_path(sweep_dir, item)
            templates = tryshow("Error reading directory: $item") do
                sweep, metrics = read_to_dataframe(joinpath(sweep_dir, item))
                return sweep[1:0,:]::DataFrame, metrics[1:0,:]::DataFrame
            end
            !isnothing(templates) && return templates
        end
    end
    error("No results found")
end

function read_results(sweep_dir)
    sweep_temp, metrics_temp = dataframe_template(sweep_dir)
    results = hcat(DataFrame(:folder => String[]), sweep_temp, metrics_temp)
    for item in readdir(sweep_dir)
        if check_read_path(sweep_dir, item)
            tryshow("Error reading directory: $item") do
                sweep, metrics = read_to_dataframe(joinpath(sweep_dir, item))
                results = append!(results, hcat(DataFrame(:folder => item), sweep, metrics))
            end
        end
    end
    return results, sweep_temp, metrics_temp
end

# Read results to DataFrame
results_dir = "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/toycvae-v1"
sweep_dir = joinpath(results_dir, "sweep");
df, sweep_temp, metrics_temp = read_results(sweep_dir);

# Show top results
show(stdout, first(sort(df, :acc; rev = true), 10); allrows = true, allcols = true); println("\n")
@show size(df);

function make_plots(df, sweepcols, metric; metadata = Dict(), showplot = false)
    dfp = dropmissing(df, metric)
    if isempty(dfp)
        @info "metric $metric has all missing values"
        return nothing
    end
    ps = []
    for sweepparam in filter(n -> length(unique(df[!,n])) > 1, sweepcols)
        s = x -> x isa Number ? x == round(x) ? string(round(Int, x)) : string(round(x; sigdigits = 3)) : string(x)
        p = plot(; xlabel = sweepparam, ylabel = metric, leg = :none, yscale = :identity)
        @df dfp  violin!(p, s.(cols(sweepparam)), cols(metric); marker = (0.2, :blue, stroke(0)))
        @df dfp boxplot!(p, s.(cols(sweepparam)), cols(metric); marker = (0.3, :orange, stroke(2)), alpha = 0.75)
        @df dfp dotplot!(p, s.(cols(sweepparam)), cols(metric); marker = (:black, stroke(0)))
        push!(ps, p)
    end
    p = plot(
        plot(title = DrWatson.savename(metadata; connector = ", "), grid = false, showaxis = false),
        plot(ps...);
        layout = @layout([a{0.01h}; b]),
    )
    showplot && display(p)
    return p
end
function recurse_make_plots(
        df,
        sweepcols = filter(n -> length(unique(df[!,n])) > 1, names(sweep_temp)),
        depth = 0;
        maxdepth = 2,
        metadata = Dict(),
    )
    foreach([:acc, :inf_mean]) do metric
        outpath = joinpath(results_dir, "summary/depth=$depth", DrWatson.savename(metadata))
        !isdir(outpath) && mkpath(outpath)

        p = make_plots(df, sweepcols, metric; metadata = metadata, showplot = depth == 0)
        savefig(p, joinpath(outpath, "by-$metric.png"))

        open(joinpath(outpath, "by-$metric.txt"); write = true) do io
            show(io, sort(dropmissing(df, metric), metric; rev = metric == :acc); allrows = true, allcols = true)
        end
    end
    if depth < maxdepth && length(sweepcols) > 1
        for (i,sweepcol) in enumerate(sweepcols)
            by(df, sweepcol) do g
                newsweepcols = sweepcols[[1:i-1; i+1:end]]
                meta = deepcopy(metadata)
                meta[sweepcol] = string(first(g[!, sweepcol]))
                recurse_make_plots(g, newsweepcols, depth + 1; maxdepth = maxdepth, metadata = meta)
            end
        end
    end
    nothing
end
recurse_make_plots(df)

nothing
