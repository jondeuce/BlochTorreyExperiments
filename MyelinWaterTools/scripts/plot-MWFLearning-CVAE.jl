using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

####
#### Plot sweep results
####

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

function read_to_dataframe(sweep_dir)
    sweep = DataFrame()
    for (k,v) in flatten_dict(TOML.parsefile(joinpath(sweep_dir, "sweep_settings.toml")))
        sweep = hcat(sweep, DataFrame(Symbol(k) => typeof(v)[v]))
    end
    state = BSON.load(joinpath(sweep_dir, "log", first(filter!(s->endswith(s, ".errors.bson"), readdir(joinpath(sweep_dir, "log"))))))[:state]
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
    labelerr = skipmissing(state.labelerr)
    metrics = DataFrame(
        :loss         => minimum(skipmissing(state.loss)) - sweep[1,Symbol("model.DenseLIGOCVAE.Zdim")]*(log(2π)-1)/2, # Correct for different Zdim's
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
    return sweep, metrics
end

function dataframe_template(results_dir)
    for item in readdir(results_dir)
        if isdir(joinpath(results_dir, item)) && isfile(joinpath(results_dir, item, "sweep_settings.toml")) && !isempty(filter!(s->endswith(s, ".errors.bson"), readdir(joinpath(results_dir, item, "log"))))
            sweep, metrics = read_to_dataframe(joinpath(results_dir, item))
            return sweep[1:0,:]::DataFrame, metrics[1:0,:]::DataFrame
        end
    end
    error("No results found")
end

function read_results(results_dir)
    sweep_temp, metrics_temp = dataframe_template(results_dir)
    results = hcat(DataFrame(:folder => String[]), sweep_temp, metrics_temp)
    for item in readdir(results_dir)
        if isdir(joinpath(results_dir, item)) && isfile(joinpath(results_dir, item, "sweep_settings.toml")) && !isempty(filter!(s->endswith(s, ".errors.bson"), readdir(joinpath(results_dir, item, "log"))))
            tryshow("Error reading directory: $item") do
                sweep, metrics = read_to_dataframe(joinpath(results_dir, item))
                results = append!(results, hcat(DataFrame(:folder => item), sweep, metrics))
            end
        end
    end
    return results, sweep_temp, metrics_temp
end

# Read results to DataFrame
results_dir = "/project/st-arausch-1/jcd1994/simulations/ismrm2020/cvae-diff-med-2-v3"
sweep_dir = joinpath(results_dir, "sweep")
df, sweep_temp, metrics_temp = read_results(sweep_dir);

# Write sorted DataFrame to text file
foreach(names(metrics_temp)) do metric
    open(joinpath(results_dir, "results-by-$metric.txt"); write = true) do io
        show(io, sort(dropmissing(df, metric), metric; rev = metric == :acc); allrows = true, allcols = true)
    end
end

# Show top results
show(stdout, first(sort(df, :acc; rev = true), 10); allrows = true, allcols = true); println("\n")
@show size(df);

function make_plots(df, sweepcols, metric, thresh = 1.0)
    dfp = dropmissing(df, metric)
    if isempty(dfp)
        @info "metric $metric has all missing values"
        return nothing
    end
    metricthresh = quantile(dfp[!, metric], thresh)
    filter!(r -> r[metric] < metricthresh, dfp)
    uniquesweepcols = sweepcols[map(n -> length(unique(df[!,n])), sweepcols) .> 1]
    ps = []
    for sweepparam in uniquesweepcols
        s = x -> x isa Number ? x == round(x) ? string(round(Int, x)) : string(round(x; sigdigits = 3)) : string(x)
        p = plot(; xlabel = sweepparam, ylabel = metric, leg = :none, yscale = :identity)
        @df dfp  violin!(p, s.(cols(sweepparam)), cols(metric); marker = (0.2, :blue, stroke(0)))
        @df dfp boxplot!(p, s.(cols(sweepparam)), cols(metric); marker = (0.3, :orange, stroke(2)), alpha = 0.75)
        @df dfp dotplot!(p, s.(cols(sweepparam)), cols(metric); marker = (:black, stroke(0)))
        push!(ps, p)
    end
    p = plot(ps...)
    display(p)
    return p
end
foreach(names(metrics_temp)) do metric
    p = make_plots(df, names(sweep_temp), metric)
    savefig(p, joinpath(results_dir, "results-by-$metric.png"))
end

nothing

####
#### Visualize learned dense layers
####

using MWFLearning

heatscale(x) = sign(x) * log(1+sqrt(abs(x)));
saveheatmap(c::Flux.Chain, name = "") = foreach(((i,l),) -> saveheatmap(l, name * "-$i"), enumerate(c.layers));
saveheatmap(l::Flux.Dense, name = "") = savefig(plot(heatmap(heatscale.(l.W[end:-1:1,:]); xticks = size(l.W,2)÷4 * (0:4), yticks = size(l.W,1)÷4 * (0:4)), heatmap(heatscale.(l.b[:,:]); xticks = 0:1); layout = @layout([a{0.9w} b{0.1w}])), @show(name));
saveheatmap(l, name = "") = nothing;

# model = BSON.load(joinpath(sweep_dir, "22", "best-model.bson"))["model"] |> deepcopy;
model_best_dir = joinpath(sweep_dir, "46", "log");
model_best_file = filter!(s -> endswith(s, ".model-best.bson"), readdir(model_best_dir))[1];
model = BSON.load(joinpath(model_best_dir, model_best_file))[:model] |> deepcopy;

rm.(filter!(s -> startswith(s,"dense-") && endswith(s,".png"), readdir(".")));
saveheatmap(model.E1, "dense-E1");
saveheatmap(model.E2, "dense-E2");
saveheatmap(model.D, "dense-D");
