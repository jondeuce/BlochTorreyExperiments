using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MAT, BSON, TOML, DataFrames, JuAFEM
using Statistics, StatsPlots
pyplot(size = (1200,800))
# pyplot(size = (800,600))
# pyplot(size = (500,400))

function read_to_dataframe(file)
    d = BSON.load(file)
    
    # Bloch-Torrey params swept over
    p = d[:sweepparams]

    # Geometry params
    if haskey(d, :geomparams_dict)
        for (k,v) in d[:geomparams_dict]
            if k === :numfibres
                p[k] = v ÷ 9 # TODO FIXME periodic_unique cylinder saving
            else
                p[k] = v
            end
        end
    else
        p[:mwf] = d[:btparams_dict][:MWF] |> Float64
        p[:mvf] = d[:btparams_dict][:MVF] |> Float64
        p[:density] = d[:btparams_dict][:AxonPDensity] |> Float64
        p[:gratio] = d[:btparams_dict][:g_ratio] |> Float64
        p[:Npts] = missing
        p[:numfibres] = missing
        p[:Ntri] = missing
    end

    # Solve time
    if haskey(d, :solve_time)
        t = d[:solve_time]
        p[:solve_time] = ifelse(t > 1e9, t/1e9, t) |> Float64 # convert ns -> s
    else
        p[:solve_time] = missing
    end

    # Derived parameters
    df = DataFrame(p)
    relative_κ(K, D, TE = 10e-3) = D < 1000 ? missing : K * sqrt(TE/D) # TODO assuming D == 1000, TE = 10ms
    df[!, Symbol("K*√(TE/D)")] = relative_κ.(df[!, :K], df[!, :Dtiss])

    return df
end

function dataframe_template(results_dir)
    for (root, dirs, files) in walkdir(results_dir)
        !occursin("measurables", root) && continue
        for file in files
            if endswith(file, ".bson")
                df = read_to_dataframe(joinpath(root, file))::DataFrame
                any(ismissing, collect(df[1,:])) && continue
                return DataFrame(names(df) .=> missings.(eltype.(eachcol(df)), 0))
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

function make_plots(df, sweepcols, ycol; savefolder = joinpath(@__DIR__, "output"))
    uniquesweepcols = sweepcols[map(n -> length(unique(df[!,n])), sweepcols) .> 1]
    ps = []
    for sweepparam in uniquesweepcols
        s = x -> x isa Number ? x == round(x) ? string(round(Int, x)) : string(round(x; sigdigits = 3)) : string(x)
        p = plot(; xlabel = sweepparam, xrot = 30, xformatter = s, ylabel = ycol, leg = :none)
        @df df scatter!(p, cols(sweepparam), cols(ycol); marker = (2, :circle))
        push!(ps, p)
    end
    !isdir(savefolder) && mkpath(savefolder)
    savefig(plot(ps...), joinpath(savefolder, "MWF-generate-by-$ycol.png"))
    nothing
end
make_plots(df, names(df[!,Not(:solve_time)]), :solve_time);

nothing
