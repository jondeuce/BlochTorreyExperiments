using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MAT, BSON, TOML, DataFrames, JuAFEM
using Statistics, StatsPlots
pyplot(size = (1200,800))
# pyplot(size = (800,600))
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

function read_to_dataframe(file)
    d = BSON.load(file)
    p = d[:sweepparams]
    p[:AxonPDensity] = d[:btparams_dict][:AxonPDensity]
    p[:g_ratio] = d[:btparams_dict][:g_ratio]
    p[:MVF] = d[:btparams_dict][:MVF]
    p[:MWF] = d[:btparams_dict][:MWF]
    p[:solve_time] = ifelse(d[:solve_time] > 1e9, d[:solve_time]/1e9, d[:solve_time]) # convert ns -> s
    return DataFrame(p)
end

function dataframe_template(results_dir)
    for (root, dirs, files) in walkdir(results_dir)
        !occursin("measurables", root) && continue
        for file in files
            if endswith(file, ".bson")
                return read_to_dataframe(joinpath(root, file))[1:0,:]::DataFrame
            end
        end
    end
    error("No .bson files found")
end

function read_results(results_dir)
    results = dataframe_template(results_dir)::DataFrame
    params_temp = results[:,Not(:solve_time)]
    for (root, dirs, files) in walkdir(results_dir)
        !occursin("measurables", root) && continue
        for file in files
            if endswith(file, ".bson")
                append!(results, read_to_dataframe(joinpath(root, file))::DataFrame)
            end
        end
    end
    return results, params_temp
end

# Read results to DataFrame
results_dir = "/project/st-arausch-1/jcd1994/simulations/ismrm2020/"
df, params_temp = read_results(results_dir);

# Write sorted DataFrame to text file
foreach([:solve_time]) do col
    open(joinpath(results_dir, "results-by-$col.txt"); write = true) do io
        show(io, sort(dropmissing(df, col), col); allrows = true, allcols = true)
    end
end
show(stdout, first(sort(df, :solve_time), 10); allrows = true, allcols = true); println("\n") # Show top results
@show size(df); # Show number of results

function make_plots(df, sweepcols, ycol)
    uniquesweepcols = sweepcols[map(n -> length(unique(df[!,n])), sweepcols) .> 1]
    ps = []
    for sweepparam in uniquesweepcols
        s = x -> x isa Number ? x == round(x) ? string(round(Int, x)) : string(round(x; sigdigits = 3)) : string(x)
        p = plot(; xlabel = sweepparam, xrot = 30, ylabel = ycol, leg = :none, yscale = :identity)
        @df df scatter!(p, cols(sweepparam), cols(ycol); marker = (2, :circle))
        push!(ps, p)
    end
    plot(ps...) |> display
    nothing
end
make_plots(df, names(df[!,Not(:solve_time)]), :solve_time);

nothing
