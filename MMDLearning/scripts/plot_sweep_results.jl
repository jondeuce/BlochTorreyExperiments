using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MAT, BSON, TOML, DataFrames
using Statistics, StatsPlots
# pyplot(size = (1200,800))
pyplot(size = (800,600))
# pyplot(size = (500,400))
empty!(Revise.queue_errors);

const SWEEP_FIELD = "gan" #TODO one of: "mmd", "gan", "vae"

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

# DataFrame has a constructor from Dict types already, but it broadcasts over vector values
dict_to_df(d::Dict)::DataFrame = push!(DataFrame(typeof.(values(d)), Symbol.(keys(d))), collect(values(d)))

function read_to_dataframe(file, field; max_retry::Int = 10)
    if endswith(file, ".toml")
        return dict_to_df(flatten_dict(TOML.parsefile(file)[field]))
    elseif endswith(file, ".bson")
        for i in 0:max_retry
            try
                return DataFrame(BSON.load(file)[field])
            catch e
                if e isa EOFError
                    @warn "Read end of file: $file\nRetry $i/$max_retry..."
                    sleep(5.0)
                else
                    rethrow(e)
                end
            end
        end
    else
        error("Unsupported file type")
    end
end

function dataframe_template(results_dir, file, field)
    for (root, dirs, files) in walkdir(results_dir)
        occursin("checkpoint", root) && continue
        if file ∈ files && "sweep_settings.toml" ∈ files && "settings.toml" ∈ files && "current-progress.bson" ∈ files #TODO: && "best-progress.bson" ∈ files
            return read_to_dataframe(joinpath(root, file), field)[1:0,:]::DataFrame
        end
    end
    error("$file not found")
end

function read_results(results_dir)
    sweep_temp = dataframe_template(results_dir, "sweep_settings.toml", SWEEP_FIELD)::DataFrame
    prog_temp  = dataframe_template(results_dir, "current-progress.bson", "progress")::DataFrame
    best_results = hcat(DataFrame(:folder => String[]), copy(sweep_temp), copy(prog_temp))
    curr_results = hcat(DataFrame(:folder => String[]), copy(sweep_temp), copy(prog_temp))
    for (root, dirs, files) in walkdir(results_dir)
        occursin("checkpoint", root) && continue
        if "sweep_settings.toml" ∈ files && "settings.toml" ∈ files && "current-progress.bson" ∈ files #TODO: && "best-progress.bson" ∈ files
            df_sweep = read_to_dataframe(joinpath(root, "sweep_settings.toml"), SWEEP_FIELD)
            df_curr = read_to_dataframe(joinpath(root, "current-progress.bson"), "progress")
            if isnothing(df_curr)
                @warn "Failed to load results: $root"
                continue
            end

            # ibest = findmin(df_curr.loss)[2]
            # ibest = findmin(df_curr.rmse)[2]
            ibest = !all(ismissing, df_curr.signal_fit_logL) ? findmin(skipmissing(df_curr.signal_fit_logL))[2] : nrow(df_curr)
            df_best_row = DataFrame(df_curr[ibest, :])
            df_curr_row = DataFrame(df_curr[end, :])

            df_best_row.time[1] = sum(df_curr.time[1:ibest])
            df_curr_row.time[1] = sum(df_curr.time)

            # mean over last WINDOW entries
            WINDOW = 250
            df_curr_row.signal_fit_logL[1] = mean(df_curr.signal_fit_logL[clamp(end-WINDOW+1,1,end) : end])
            df_curr_row.signal_fit_rmse[1] = mean(df_curr.signal_fit_rmse[clamp(end-WINDOW+1,1,end) : end])

            currfolder = basename(root)
            best_results = append!(best_results, hcat(DataFrame(:folder => currfolder), df_sweep, df_best_row))
            curr_results = append!(curr_results, hcat(DataFrame(:folder => currfolder), df_sweep, df_curr_row))
        end
    end
    return best_results, curr_results, sweep_temp, prog_temp
end

# Remove checkpointed models
#   ls -alhq1 **/sweep/**/checkpoint/checkpoint-model.epoch.*.bson | wc -l
#   rm **/sweep/**/checkpoint/checkpoint-model.epoch.*.bson

# Read results to DataFrame
# results_dir = "/project/st-arausch-1/jcd1994/MMD-Learning/toymmdopt_eps=1e-3/toymmdopt-v1"
# results_dir = "/project/st-arausch-1/jcd1994/MMD-Learning/toymmdopt_eps=1e-3/toymmdopt-v4"
# results_dir = "/project/st-arausch-1/jcd1994/MMD-Learning/toymmdopt_eps=1e-2/toymmdopt-v5"
# results_dir = "/project/st-arausch-1/jcd1994/MMD-Learning/toymmdopt_eps=1e-2/toymmdopt-v7"
# results_dir = "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7"
# results_dir = "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/ganopt-v2"
results_dir = "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/toyganopt-v3"
# results_dir = "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybridganopt-v1"
sweep_dir = joinpath(results_dir, "sweep");
df_best, df_curr, sweep_temp, prog_temp = read_results(sweep_dir);
# df = df_best
df = df_curr

# Write sorted DataFrame to text file
foreach([:signal_fit_rmse, :signal_fit_logL]) do metric #[:loss, :rmse, :mae, :linf]
    open(joinpath(results_dir, "results-by-$metric.txt"); write = true) do io
        savecols = Colon() #Not(:logsigma) #TODO
        dfsave = dropmissing(df, metric)[:, savecols]
        show(io, sort(dfsave, metric); allrows = true, allcols = true)
    end
end

# Show top results
let
    savecols = Colon() #Not([:logsigma]) #TODO
    show(stdout, first(sort(df[:,savecols], :signal_fit_logL), 10); allrows = true, allcols = true); println("\n")
    @show size(df);
end

function make_plots(df, sweepcols, metric, thresh = 0.5)
    dfp = dropmissing(df, metric)
    dfp = filter(r -> !any(x -> isa(x, Number) && isnan(x), r), dfp)
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
        p = plot(; xlabel = sweepparam, ylabel = metric, leg = :none, xrot = 30, yscale = :identity)
        @df dfp  violin!(p, s.(cols(sweepparam)), cols(metric); marker = (0.2, :blue, stroke(0)))
        @df dfp boxplot!(p, s.(cols(sweepparam)), cols(metric); marker = (0.3, :orange, stroke(2)), alpha = 0.75)
        @df dfp dotplot!(p, s.(cols(sweepparam)), cols(metric); marker = (:black, stroke(0)))
        push!(ps, p)
    end
    p = plot(ps...)
    savefig.(joinpath.(results_dir, "results-by-$metric" .* [".png", ".pdf"]))
    display(p)
    nothing
end
# make_plots.(Ref(df), Ref(names(sweep_temp)), [:loss, :rmse, :linf, :time], [0.5, 0.5, 0.5, 1.0]);
# make_plots.(Ref(df), Ref(names(sweep_temp)), [:signal_fit_rmse, :signal_fit_logL], 0.9);
make_plots.(Ref(filter(r -> r.signal_fit_logL < 0, dropmissing(df))), Ref(names(sweep_temp)), [:signal_fit_rmse, :signal_fit_logL], 1.0);

nothing

# using Flux
# model = BSON.load(joinpath(sweep_dir, "22", "best-model.bson"))["model"] |> deepcopy;
# for (i,layer) in enumerate(model)
#     layer isa Flux.Dense && savefig(heatmap([layer.W[end:-1:1,:] repeat(layer.b, 1, 10)]), "dense$i.png")
# end
# @show sum(length, params(model));
