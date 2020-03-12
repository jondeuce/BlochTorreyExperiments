using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MAT, BSON, TOML, DataFrames
using Statistics, StatsPlots
# pyplot(size = (1200,800))
pyplot(size = (800,600))
# pyplot(size = (500,400))

function flatten_dict!(dout::Dict{<:AbstractString, Any}, din::Dict{<:AbstractString, Any})
    for (k,v) in din
        if v isa Dict
            vnew = Dict{String, Any}(k .* "." .* keys(v) .=> deepcopy(values(v)))
            flatten_dict!(dout, vnew)
        else
            dout[k] = deepcopy(v)
        end
    end
    dout
end
flatten_dict(din::Dict{<:AbstractString, Any}) = flatten_dict!(Dict{String, Any}(), din)

function read_to_dataframe(sweep_dir)
    sweep = DataFrame(flatten_dict(TOML.parsefile(joinpath(sweep_dir, "sweep_settings.toml"))))
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
    metrics = DataFrame(
        :loss    => quantile(skipmissing(state.loss), 0.001),
        :acc     => quantile(skipmissing(state.acc), 0.999),
        :mwf     => quantile((x->x[3]).(skipmissing(state.labelerr)), 0.001),
        :infsup  => quantile((x->maximum(x[2:5])).(skipmissing(state.labelerr)), 0.001),
        :infmean => quantile((x->mean(x[2:5])).(skipmissing(state.labelerr)), 0.001),
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
            sweep, metrics = read_to_dataframe(joinpath(results_dir, item))
            results = append!(results, hcat(DataFrame(:folder => item), sweep, metrics))
        end
    end
    return results, sweep_temp, metrics_temp
end

# Read results to DataFrame
results_dir = "/project/st-arausch-1/jcd1994/simulations/ismrm2020/cvae-diff-med-2-v1"
df, sweep_temp, metrics_temp = read_results(results_dir);

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
    p = make_plots(df, names(sweep_temp)[1:end-5], metric)
    savefig(p, joinpath(results_dir, "results-by-$metric.png"))
end

nothing

# using Flux
# model = BSON.load(joinpath(sweep_dir, "22", "best-model.bson"))["model"] |> deepcopy;
# for (i,layer) in enumerate(model)
#     layer isa Flux.Dense && savefig(heatmap([layer.W[end:-1:1,:] repeat(layer.b, 1, 10)]), "dense$i.png")
# end
# @show sum(length, params(model));
