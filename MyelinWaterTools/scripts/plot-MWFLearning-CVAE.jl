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
    if state isa Dict
        function dict_to_df(dataset)
            nrows = length(state[:callbacks][dataset][:epoch])
            loopidx = dataset == :testing ? findall(in(state[:callbacks][dataset][:epoch]), intersect(state[:loop][:epoch])) : nothing
            # loopidx = findall(in(state[:callbacks][dataset][:epoch]), intersect(state[:loop][:epoch]))
            DataFrame(
                :epoch    => state[:callbacks][dataset][:epoch],
                :dataset  => fill(ifelse(dataset == :testing, :test, :train), nrows),
                :loss     => convert(Vector{Union{Float64, Missing}}, isnothing(loopidx) ? missings(nrows) : state[:loop][:loss][loopidx]),
                :acc      => convert(Vector{Union{Float64, Missing}}, state[:callbacks][dataset][:acc]),
                :ELBO     => convert(Vector{Union{Float64, Missing}}, isnothing(loopidx) ? missings(nrows) : state[:loop][:ELBO][loopidx]),
                :KL       => convert(Vector{Union{Float64, Missing}}, isnothing(loopidx) ? missings(nrows) : state[:loop][:KL][loopidx]),
                :labelerr => convert(Vector{Union{Vector{Float64}, Missing}}, state[:callbacks][dataset][:labelerr]),
            )
        end
        state = vcat(dict_to_df(:training), dict_to_df(:testing))
    end
    state = state[state.dataset .== :test, :]
    # dropmissing!(state) # drop rows with missings
    filter!(r -> all(x -> !((x isa Number && isnan(x)) || (x isa AbstractArray{<:Number} && any(isnan, x))), r), state) # drop rows with NaNs
    labelerr = skipmissing(state.labelerr)
    metrics = DataFrame(
        :loss         => minimum(skipmissing(state.loss)), # - sweep[1, Symbol("model.DenseLIGOCVAE.Zdim")]*(log(2π)-1)/2, # Correct for different Zdim's
        :acc          => maximum(skipmissing(state.acc)),
        :cosd_alpha   => minimum((x->x[1]).(labelerr)),
        :g_ratio      => minimum((x->x[2]).(labelerr)),
        :mwf          => minimum((x->x[3]).(labelerr)),
        :rel_T2mw     => minimum((x->x[4]).(labelerr)),
        :rel_T2iew    => minimum((x->x[5]).(labelerr)),
        :log_rel_K    => minimum((x->x[6]).(labelerr)),
        :inf_sup_iqr  => minimum((x->maximum(sort(x[2:end-1]))).(labelerr)),
        :inf_mean_iqr => minimum((x->mean(sort(x[2:end-1]))).(labelerr)),
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
# results_dir = "/project/st-arausch-1/jcd1994/simulations/ismrm2020/cvae-diff-med-2-v6";
# results_dir = "/project/st-arausch-1/jcd1994/ismrm2020/experiments/Spring-2020/permeability-training-1/cvae-diff-med-2/cvae-diff-med-2-100k-samples-v1"
results_dir = "/project/st-arausch-1/jcd1994/simulations/ismrm2020/cvae-diff-med-2-100k-samples-v2"
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
    foreach([:acc, :inf_mean_iqr]) do metric #names(metrics_temp)
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

#=
# Write sorted DataFrame to text file
foreach(names(metrics_temp)) do metric
    open(joinpath(results_dir, "results-by-$metric.txt"); write = true) do io
        show(io, sort(dropmissing(df, metric), metric; rev = metric == :acc); allrows = true, allcols = true)
    end
end

# Make plots
foreach(names(metrics_temp)) do metric
    p = make_plots(df, names(sweep_temp), metric)
    savefig(p, joinpath(results_dir, "results-by-$metric.png"))
end
=#

####
#### Visualize learned dense layers
####

#=
heatscale(x) = sign(x) * log(1+sqrt(abs(x)));
saveheatmap(c::Flux.Chain, name = "") = foreach(((i,l),) -> saveheatmap(l, name * "-$i"), enumerate(c.layers));
saveheatmap(l::Flux.Dense, name = "") = savefig(plot(heatmap(heatscale.(l.W[end:-1:1,:]); xticks = size(l.W,2)÷4 * (0:4), yticks = size(l.W,1)÷4 * (0:4)), heatmap(heatscale.(l.b[:,:]); xticks = 0:1); layout = @layout([a{0.9w} b{0.1w}])), @show(name));
saveheatmap(l, name = "") = nothing;

# model = BSON.load(joinpath(sweep_dir, "22", "best-model.bson"))["model"] |> deepcopy;
model = BSON.load(findendswith(joinpath(sweep_dir, "51", "log"), ".model-best.bson"))[:model] |> deepcopy;

rm.(filter!(s -> startswith(s,"dense-") && endswith(s,".png"), readdir(".")));
saveheatmap(model.E1, "dense-E1");
saveheatmap(model.E2, "dense-E2");
saveheatmap(model.D, "dense-D");
=#

#=
let results_dir = "/project/st-arausch-1/jcd1994/simulations/ismrm2020/cvae-diff-med-2-v4/sweep"

    function _check_read_path(results_dir, dir)
        isdir(joinpath(results_dir, dir)) &&
        isdir(joinpath(results_dir, dir, "log")) &&
        isfile(joinpath(results_dir, dir, "sweep_settings.toml")) &&
        !isnothing(findendswith(joinpath(results_dir, dir, "log"), ".errors.bson")) # &&
        # isnothing(findendswith(joinpath(results_dir, dir, "log"), ".errors.backup.bson"))
    end

    for dir in filter!(s -> parse(Int, s) >= 270, sort!(filter!(s->isdir(joinpath(results_dir,s)), readdir(results_dir)); by = s -> parse(Int, s)))
        if _check_read_path(results_dir, dir)
            @info dir
            tryshow("Error reading directory: $dir") do
                state_file             = findendswith(joinpath(results_dir, dir, "log"), "errors.bson")
                model_best_file        = findendswith(joinpath(results_dir, dir, "log"), "model-best.bson")
                model_checkpoint_file  = findendswith(joinpath(results_dir, dir, "log"), "model-checkpoint.bson")
                @time state            = BSON.load(state_file)
                @time model_best       = BSON.load(model_best_file)
                @time model_checkpoint = BSON.load(model_checkpoint_file)

                state[:state] = deepcopy(dropmissing(state[:state]))
                @time BSON.bson(state_file[1:end-5] * ".backup.bson", state)
            end
        end
    end
end
=#

nothing
