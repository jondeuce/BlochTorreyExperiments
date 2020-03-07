using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MAT, StatsPlots, DataFrames
# pyplot(size = (1200,800))
pyplot(size = (800,600))

function read_results(results_dir)
    results = DataFrame(method = String[], fitness = Float64[], sigma = Float64[], nterms = Int[])
    for (root, dirs, files) in walkdir(joinpath(@__DIR__, results_dir))
        for file in files
            if file == "bboptim_results.mat"
                res = MAT.matread(joinpath(root, file))
                push!(results, [res["method"], res["best_fitness"], res["settings"]["sigma"], res["settings"]["nterms"]])
            end
        end
    end
    return results
end

for ver in ["v2"]
results_dir = "results-$ver"
results = sort!(read_results(results_dir), [:method, :fitness])
p = let gdf = sort(collect(groupby(results, :sigma)), by = df -> df.sigma);  plot([plot(df.method, df.fitness; group = df.nterms, title = "sigma = $(df.sigma[1])",   m = (5,), l = (3,), xrot = 20, yscale = :log10) for df in gdf]...); end; display(p)
map(ext -> savefig(p, joinpath(results_dir, "fitness-group-sigma.$ext")), ["pdf", "png"])
p = let gdf = sort(collect(groupby(results, :nterms)), by = df -> df.nterms); plot([plot(df.method, df.fitness; group = df.sigma,  title = "nterms = $(df.nterms[1])", m = (5,), l = (3,), xrot = 20, yscale = :log10) for df in gdf]...); end; display(p)
map(ext -> savefig(p, joinpath(results_dir, "fitness-group-nterms.$ext")), ["pdf", "png"])
end

nothing
