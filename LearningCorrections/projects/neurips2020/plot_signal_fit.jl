using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DECAES, StatsPlots, DataFrames, GaussianMixtures
# pyplot(size = (1200,800))
# pyplot(size = (800,600))
pyplot(size = (500,400))

function read_results(results_dir)
    # results = DataFrame(refcon = Float64[], alpha = Float64[], Amw = Float64[], T2mw = Float64[], T1mw = Float64[], Aiew = Float64[], T2iew = Float64[], T1iew = Float64[])
    # results = DataFrame(refcon = Float64[], alpha = Float64[], T2short = Float64[], T2long = Float64[], Ashort = Float64[])
    results = DataFrame(refcon = Float64[], alpha = Float64[], T2short = Float64[], dT2 = Float64[], Ashort = Float64[])
    for (root, dirs, files) in walkdir(joinpath(@__DIR__, results_dir))
        for file in files
            if file == "bbsignalfit_results.mat"
                θ = DECAES.MAT.matread(joinpath(root, file))["thetas"]'
                df = similar(results, size(θ,1))
                df[!,:] .= θ
                append!(results, df)
            end
        end
    end
    results.T2long = results.T2short .+ results.dT2
    results.Along  = 1 .- results.Ashort
    return results
end

function histcorrplot(df, bounds = map(extrema, eachcol(df)))
    ps = Any[]
    for (j,nj) in enumerate(names(df)), (i,ni) in enumerate(names(df))
        p = if i > j
            histogram2d(df[!,i], df[!,j]; nbins = 50, cbar = :none, xlab = ni, xrot = 45, xlim = bounds[i], ylim = bounds[j])
        elseif i == j
            histogram(df[!,i]; nbins = 50, xlab = ni, xrot = 45, xlim = bounds[i], yticks = :none, leg = :none)
        else
            plot(grid = :none, ticks = :none)
        end
        push!(ps, p)
    end
    p = plot(ps...)
end

results_dir = joinpath(@__DIR__, "sigfit-v5")
results = read_results(results_dir)
p = histcorrplot(results); #display(p) #[!, [1,2,3,6,4,7]]
map(ext -> savefig(p, joinpath(results_dir, "corrplot.$ext")), ["pdf", "png"])

####
#### Fit gaussian mixture to (transformed) data
####

# Transform data by shifting data to γ * [-0.5, 0.5] and applying tanh.
# This makes data more smoothly centred around zero, with
#   γ = 2*tanh(3) ≈ 1.99
# sending boundary points to approx. +/- 3
f(x,a,b) = atanh((x - ((a+b)/2)) * (1.99 / (b-a)))
g(y,a,b) = ((a+b)/2) + tanh(y) * ((b-a) / 1.99)
f(x,t::NTuple{2}) = f(x,t...)
g(x,t::NTuple{2}) = g(x,t...)
trans!(fun, df, bounds) = (foreach(j -> df[!,j] .= fun.(df[!,j], Ref(bounds[j])), 1:ncol(df)); return df)

thetas = results[:, [:refcon, :alpha, :T2short, :T2long, :Ashort]] #results[:, [1,2,3,6,4,7]]
filter!(row -> !(row.Ashort ≈ 1) && 10 <= row.T2short <= 100 && row.T2long <= 500, thetas)
bounds = map(extrema, eachcol(thetas))
thetas_trans = trans!(f, copy(thetas), bounds)
bounds_trans = map(extrema, eachcol(thetas_trans))

let
    p = histcorrplot(thetas_trans, bounds_trans); savefig(p, "thetas-trans.png"); #display(p)
    p = histcorrplot(thetas, bounds); savefig(p, "thetas.png"); #display(p)
end

gmm_method = :kmeans
gmm = GMM(32, Matrix(thetas_trans); method = gmm_method, kind = :full, nInit = 1000, nIter = 100, nFinal = 100)
draws_trans = similar(thetas_trans)
draws_trans[!,:] .= rand(gmm, nrow(draws_trans))
filter!(draws_trans) do row
    for (j, val) in enumerate(row)
        bounds_trans[j][1] <= val <= bounds_trans[j][2] || return false
    end
    return true
end
draws = trans!(g, copy(draws_trans), bounds)

let
    p = histcorrplot(draws_trans, bounds_trans); savefig(p, "draws-trans-$gmm_method.png"); #display(p)
    p = histcorrplot(draws, bounds); savefig(p, "draws-$gmm_method.png"); #display(p)
end

nothing
