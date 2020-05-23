####
#### Code loading
####

include(joinpath(@__DIR__, "../src", "mmd_preamble.jl"))
using MWFLearning
Random.seed!(0)

####
#### Load image data
####

const mmd_settings = load_settings();
const save_folder  = mkpath(MWFLearning.getnow());

# Load data samplers
if !(@isdefined(sampleX) && @isdefined(sampleY) && @isdefined(sampleθ) && @isdefined(fits))
    global sampleX, sampleY, sampleθ, _, fits, _ = make_mle_data_samplers(
        mmd_settings["prior"]["data"]["image"]::String,
        mmd_settings["prior"]["data"]["thetas"]::String;
        ntheta = mmd_settings["data"]["ntheta"]::Int,
        padtrain = false,
        normalizesignals = false,
        filteroutliers = false,
        plothist = false, #TODO
    )
end
const θ = copy(sampleθ(nothing; dataset = :test));
const X = copy(sampleX(nothing; dataset = :test));
const Y = copy(sampleY(nothing; dataset = :test));
const Y_RES = minimum(filter(>(0), Y)); # resolution of signal values
const Y_EDGES = unique!(sort!(copy(vec(Y)))); # unique values
θ[1,:] .= cosd.(θ[1,:]); # transform flipangle -> cosd(flipangle)
θ[3,:] .= θ[2,:] .+ θ[3,:]; # transform dT2 -> T2long = T2short + dT2

####
#### Load models + make corrected data
####

const all_generators = Dict{String,Any}(
    # "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/224"), # MMD-kernel
    # "mmd" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/17"), # tstat-kernel
    # "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/225"), # tstat-kernel
    "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/117"), # tstat-kernel
    # "gan" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/ganopt-v3/sweep/96"),
    "gan" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/ganopt-v3/sweep/48"),
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/90"), # MMD-kernel
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/74"), # MMD-kernel
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/11"), # tstat-kernel
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/45"), # tstat-kernel
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/141"), # tstat-kernel
    "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/157"), # tstat-kernel
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/189"), # tstat-kernel
);
for (dataset, data) in all_generators
    data["models"] = deepcopy(BSON.load(joinpath(data["folder"], data["modelfile"])));
    data["settings"] = deepcopy(TOML.parsefile(joinpath(data["folder"], "settings.toml")));
end

# Save models
# BSON.bson(joinpath(save_folder, "plot-models.bson"), all_generators);

# Convenience functions
module Generator
    import Flux: Chain
    const G = Ref{Chain}()
    split_correction_and_noise(μlogσ) = μlogσ[1:end÷2, :], exp.(μlogσ[end÷2+1:end, :])
    noise_instance(X, ϵ) = ϵ .* randn(eltype(X), size(X)...)
    get_correction_and_noise(X) = split_correction_and_noise(G[](X))
    get_correction(X) = get_correction_and_noise(X)[1]
    get_noise(X) = get_correction_and_noise(X)[2]
    get_noise_instance(X) = noise_instance(X, get_noise(X))
    get_corrected_signal(X) = get_corrected_signal(X, get_correction_and_noise(X)...)
    get_corrected_signal(X, dX, ϵ) = get_corrected_signal(abs.(X .+ dX), ϵ)
    function get_corrected_signal(X, ϵ)
        ϵR, ϵI = noise_instance(X, ϵ), noise_instance(X, ϵ)
        Xϵ = @. sqrt((X + ϵR)^2 + ϵI^2)
        return Xϵ
    end
end

# Make corrected data
const GENERATOR_NAMES  = ["mmd", "gan", "hyb"];
const GENERATOR_MODELS = Chain[all_generators["mmd"]["models"]["mmd"], all_generators["gan"]["models"]["G"], all_generators["hyb"]["models"]["G"]];

let _models = BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-3/plot-models.bson")
    # GENERATOR_MODELS[1] = _models[:mmd_models]["mmd"]
    # GENERATOR_MODELS[2] = _models[:gan_models]["G"]
    GENERATOR_MODELS[3] = _models[:hyb_models]["G"]
end;
BSON.bson(joinpath(save_folder, "plot-models-with-hyb-plots-checkpoint-3.bson"), all_generators);

const Xs = Dict{String,Any}(
    "data" => Dict{String,Any}("Y" => Y),
    "mle" => Dict{String,Any}("X" => X, "Xeps0" => rand.(Rician.(X, exp.(fits.logsigma')))), # Xeps0: constant eps Rician noise from MLE fit
    [G => Dict{String,Any}() for G in GENERATOR_NAMES]...,
);
for (dataset, G) in zip(GENERATOR_NAMES, GENERATOR_MODELS)
    Generator.G[] = G
    g_delta, g_eps   = Generator.get_correction_and_noise(X);
    Xs[dataset]["Xhat"]    = Generator.get_corrected_signal(X, g_delta, g_eps); # add learned correction + learned noise
    Xs[dataset]["Xdelta"]  = abs.(X .+ g_delta); # add learned noise only
    Xs[dataset]["Xeps"]    = Generator.get_corrected_signal(X, g_eps); # add learned noise only
    Xs[dataset]["g_delta"] = g_delta;
    Xs[dataset]["g_eps"]   = g_eps;
end;

# Saving figs
_save_and_display(p, fname::String) = (map(ext -> savefig(p, joinpath(save_folder, fname * ext)), (".png", ".pdf")); nothing) #display(p)

# Histograms
using PyCall, Distances
const st = pyimport("scipy.stats")
_make_hist(x, edges) = fit(Histogram, x, UnitWeights{Float64}(length(x)), edges)
_dhist(h1::Histogram, h2::Histogram, d::Distances.PreMetric) = Distances.evaluate(d, h1.weights, h2.weights)
_dhist(h1::Histogram, h2::Histogram, f::Symbol) =
    f === :wasserstein_distance ? st.wasserstein_distance(h1.weights, h2.weights) :
    f === :chi_squared ? mapreduce((Pi,Qi) -> Pi + Qi == 0 ? 0 : (Pi - Qi)^2 / (2 * (Pi + Qi)), +, h1.weights, h2.weights) :
    error("Unknown histogram metric: $f")

####
#### Run NNLS on Y, Xhat, etc.
####

const t2mapopts = T2mapOptions(
    MatrixSize = (size(Y,2), 1, 1),
    nTE        = size(Y,1),
    TE         = 8e-3,
    nT2        = 40,
    T2Range    = (8e-3, 1.0), #TODO range changed from (15e-3, 2.0) to match CVAE outputs
    Threshold  = 0.0, #TODO important since signals are "normalized" down uniformly by 1e6
    Silent     = true,
);
const t2partopts = T2partOptions(
    MatrixSize = (size(Y,2), 1, 1),
    nT2        = t2mapopts.nT2,
    T2Range    = t2mapopts.T2Range,
    SPWin      = (prevfloat(t2mapopts.T2Range[1]), 40e-3), #TODO
    MPWin      = (nextfloat(40e-3), nextfloat(t2mapopts.T2Range[2])), #TODO
    Silent     = true,
);

#=
for (dataset, signals) in Xs, (name, _X) in signals
    # dataset == "hyb" || continue #TODO
    if name ∈ ["X", "Y", "Xeps0", "Xhat", "Xdelta", "Xeps"]
        print("dataset = $dataset, name = $name: ")
        @time begin
            Xs[dataset]["$(name)_t2maps"], Xs[dataset]["$(name)_t2dist"] = DECAES.T2mapSEcorr(reshape(permutedims(_X), (size(_X,2), 1, 1, size(_X,1))), t2mapopts)
            Xs[dataset]["$(name)_t2parts"] = DECAES.T2partSEcorr(Xs[dataset]["$(name)_t2dist"], t2partopts);
        end
    end
end;
=#

####
#### Noise level (epsilon) histograms
####

for G in GENERATOR_NAMES
    p = plot();
    common_args = Dict{Symbol,Any}(:normalized => true, :xlims => (-2.6, -1.2), :line => (2,), :tickfontsize => 10, :legendfontsize => 10)
    stephist!(p, log10.(exp.(fits.logsigma)); lab = L"Signalwise $\log_{10}(\epsilon_0)$ from MLE", common_args...);
    stephist!(p, log10.(vec(Xs[G]["g_eps"])); lab = L"Aggregated $\log_{10}(\epsilon)$ from $\ln(\epsilon) \sim g_{\epsilon}(X_i)$", common_args...);
    stephist!(p, log10.(vec(mean(Xs[G]["g_eps"]; dims = 1))); lab = L"Signalwise $\log_{10}$(Mean($\epsilon$))", common_args...);
    stephist!(p, log10.(vec(exp.(mean(log.(Xs[G]["g_eps"]); dims = 1)))); lab = L"Signalwise $\log_{10}$(GeoMean($\epsilon$))", common_args...);
    _save_and_display(p, "$(G)_log_noise_hist")

    p = plot();
    common_args = Dict{Symbol,Any}(:normalized => true, :xlims => (0, 0.03), :line => (2,), :tickfontsize => 10, :legendfontsize => 10)
    stephist!(p, exp.(fits.logsigma); lab = L"Signalwise $\epsilon_0$ from MLE", common_args...);
    stephist!(p, vec(Xs[G]["g_eps"]); lab = L"Aggregated $\epsilon$ from $\ln(\epsilon) \sim g_{\epsilon}(X_i)$", common_args...);
    stephist!(p, vec(mean(Xs[G]["g_eps"]; dims = 1)); lab = L"Signalwise Mean($\epsilon$)", common_args...);
    stephist!(p, vec(exp.(mean(log.(Xs[G]["g_eps"]); dims = 1))); lab = L"Signalwise GeoMean($\epsilon$)", common_args...);
    _save_and_display(p, "$(G)_noise_hist")
end

####
#### Signal intensity histograms
####

signal_hists = @time map(1:48) do j
    print("processing histogram $j/48:")
    @time begin
        binwidth = ceil(Int, mean(Xs["data"]["Y"][j,:]) / (50 * Y_RES))
        edges = Y_EDGES[1:binwidth:end]
        h = Dict()
        h["Y"] = _make_hist(Xs["data"]["Y"][j,:], edges)
        for Xmle in ["X", "Xeps0"]
            h[Xmle] = _make_hist(Xs["mle"][Xmle][j,:], edges)
        end
        for G in GENERATOR_NAMES, Xgen in ["Xhat"]
            h[G] = _make_hist(Xs[G][Xgen][j,:], edges)
        end
        h
    end
end;

for G in GENERATOR_NAMES
    common_args = Dict{Symbol,Any}(:seriestype => :steppost, :line => (1,), :tickfontsize => 10, :legendfontsize => 10)
    p = plot(
        map([1,16,24,32,40,48]) do j
            p = plot();
            hY, hX, hX̂ = signal_hists[j]["Y"], signal_hists[j]["X"], signal_hists[j][G]
            plot!(p, hY.edges[1], hY.weights; lab = "\$Y_{$j}\$", common_args...)
            plot!(p, hX.edges[1], hX.weights; lab = "\$X_{$j}\$", common_args...)
            plot!(p, hX̂.edges[1], hX̂.weights; lab = "\$\\hat{X}_{$j}\$", common_args...)
            xl = (0.0, hY.edges[1][1 + length(hY.weights) - findfirst(>=(250), hY.weights[end:-1:1])])
            xlims!(p, xl)
            p
        end...;
    );
    _save_and_display(p, "$(G)_signal_hist")
end

for G in GENERATOR_NAMES
    common_args = Dict{Symbol,Any}(:seriestype => :steppost, :line => (1,), :tickfontsize => 10, :legendfontsize => 10)
    p = plot(
        map([1,16,24,32,40,48]) do j
            p = plot();
            hY, hX, hX̂ = signal_hists[j]["Y"], signal_hists[j]["X"], signal_hists[j][G]
            plot!(p, hX.edges[1], (hX.weights .- hY.weights) ./ (maximum(hY.weights)); lab = "\$(X - Y)_{$j}\$", common_args...)
            plot!(p, hX̂.edges[1], (hX̂.weights .- hY.weights) ./ (maximum(hY.weights)); lab = "\$(\\hat{X} - Y)_{$j}\$", common_args...)
            i_last = 1 + length(hY.weights) - findfirst(>=(250), hY.weights[end:-1:1])
            xl = (0.0, hY.edges[1][i_last])
            yl = maximum(abs, (hX.weights[1:i_last] .- hY.weights[1:i_last]) ./ (maximum(hY.weights))) .* (-1,1)
            xlims!(p, xl)
            ylims!(p, yl)
            p
        end...;
    );
    _save_and_display(p, "$(G)_signal_hist_diff")
end

p = let
    common_args = Dict{Symbol,Any}(:seriestype => :line, :line => (2,), :tickfontsize => 10, :legendfontsize => 10)
    plot(
        map([Cityblock(), Euclidean(), ChiSqDist(), :wasserstein_distance]) do _dist_type
            ydata = map(product(signal_hists, ["X"; "Xeps0"; GENERATOR_NAMES])) do (h,key)
                # abs.(h[key].weights .- h["Y"].weights) ./ mean(h["Y"].weights)
                _dhist(h[key], h["Y"], _dist_type)
            end
            xdata = 1000 .* t2mapopts.TE .* (1:48)
            plot(xdata, ydata; # mean.(ydata);
                # ribbon = std.(ydata) ./ sqrt.(length.(ydata)),
                label = [L"$X - Y$" L"$X_{\epsilon_0} - Y$" L"$\hat{X}_{MMD} - Y$" L"$\hat{X}_{GAN} - Y$" L"$\hat{X}_{HYB} - Y$"],
                ylabel = "$_dist_type [a.u.]",
                xlabel = "Echo time [ms]",
                color = [:green :blue :red :orange :magenta],
                common_args...,
            )
        end...,
    )
end;
_save_and_display(p, "compare_genatr_signal_hist");

#=
for G in GENERATOR_NAMES
    common_args = Dict{Symbol,Any}(:line => (1,), :ylim => (0.0, 0.02), :tickfontsize => 8, :legendfontsize => 8, :legend => :topright, :xticks => :none)
    p = plot(
        map([1,8,16,32]) do j
            p = plot();
            Yj_sorted = sort!(Xs["data"]["Y"][j,:])
            plot!(p, abs.(sort!(Xs["mle"]["X"][j,:]) .- Yj_sorted); lab = L"$X - Y$" * " (echo $j)", seriestype = :steppre, common_args...);
            plot!(p, abs.(sort!(Xs["mmd"]["Xhat"][j,:]) - Yj_sorted); lab = L"$\hat{X} - Y$" * " (echo $j)", seriestype = :steppre, common_args...);
            p
        end...;
    );
    _save_and_display(p, "$(G)_signal_qqdiff")
end
=#

#=
for G in GENERATOR_NAMES
    p = plot();
    plot!(p, sort(Xs["data"]["Y"][1,:]); lab = L"$Y$", seriestype = :steppre);
    plot!(p, sort(Xs["mle"]["X"][1,:]); lab = L"$X$", seriestype = :steppre);
    plot!(p, sort(Xs[G]["Xhat"][1,:]); lab = L"$\hat{X}$", seriestype = :steppre);
    display(p)
end
=#

####
#### Compare T2 distributions/NNLS results
####

#=
let
    common_args = Dict{Symbol,Any}(
        :tickfontsize => 10, :legendfontsize => 10, :xscale => :log10, :xrot => 30.0,
        :xlabel => L"$T_2$ time [ms]",
        :xticks => Xs["data"]["Y_t2maps"]["t2times"][1:3:end],
        :xformatter => x -> string(round(1000x; sigdigits = 3)),
    )
    xdata = Xs["data"]["Y_t2maps"]["t2times"]
    ydata = hcat(
        vec(mean(Xs["data"]["Y_t2dist"]; dims = 1)),
        # vec(mean(Xs["mle"]["X_t2dist"]; dims = 1)),
        vec(mean(Xs["mle"]["Xeps0_t2dist"]; dims = 1)),
        [vec(mean(Xs[G]["Xhat_t2dist"]; dims = 1)) for G in GENERATOR_NAMES]...,
    )
    p = plot(xdata, ydata;
        label = [L"A_{\ell,Y}" L"A_{\ell,X_{\epsilon_0}}" L"$A_{\ell,\hat{X}_{MMD}}$" L"$A_{\ell,\hat{X}_{GAN}}$" L"$A_{\ell,\hat{X}_{HYB}}$"],
        ylabel = L"$T_2$ amplitude [a.u.]",
        marker = (5, [:circle :utriangle :square :diamond :dtriangle], [:blue :green :red :orange :magenta]),
        line = ([2 1 1 1], [:solid :dash :dash :dash :dash], [:blue :green :red :orange :magenta]),
        common_args...
    );
    _save_and_display(p, "compare_genatr_t2_distbn")

    xdata = Xs["data"]["Y_t2maps"]["t2times"]
    ydata = abs.(hcat(
        # vec(mean(Xs["mle"]["X_t2dist"]; dims = 1)),
        vec(mean(Xs["mle"]["Xeps0_t2dist"]; dims = 1)),
        [vec(mean(Xs[G]["Xhat_t2dist"]; dims = 1)) for G in GENERATOR_NAMES]...,
    ) .- vec(mean(Xs["data"]["Y_t2dist"]; dims = 1)));
    p = plot(xdata, ydata;
        label = [L"|A_{\ell,X_{\epsilon_0}} - A_{\ell,Y}|" L"$|A_{\ell,\hat{X}_{MMD}} - A_{\ell,Y}|$" L"$|A_{\ell,\hat{X}_{GAN}} - A_{\ell,Y}|$" L"$|A_{\ell,\hat{X}_{HYB}} - A_{\ell,Y}|$"] .* map(l2 -> ": \$\\ell^2\$ norm = $l2", round.(norm.(eachcol(ydata))'; sigdigits = 3)),
        ylabel = L"$T_2$ amplitude difference [a.u.]",
        marker = (5, [:utriangle :square :diamond :dtriangle], [:green :red :orange :magenta]),
        line = (3, [:solid :solid :solid :solid], [:green :red :orange :magenta]),
        common_args...
    );
    _save_and_display(p, "compare_genatr_t2_distbn_diff")
end

for G in GENERATOR_NAMES
    common_args = Dict{Symbol,Any}(
        :tickfontsize => 10, :legendfontsize => 10, :xscale => :log10, :xrot => 30.0,
        :xlabel => L"$T_2$ time [ms]",
        :xticks => Xs["data"]["Y_t2maps"]["t2times"][1:3:end],
        :xformatter => x -> string(round(1000x; sigdigits = 3)),
    )
    p = plot(
        Xs["data"]["Y_t2maps"]["t2times"],
        abs.(
            hcat(
                vec(mean(Xs["mle"]["X_t2dist"]; dims = 1)),
                vec(mean(Xs["mle"]["Xeps0_t2dist"]; dims = 1)),
                vec(mean(Xs[G]["Xeps_t2dist"]; dims = 1)),
                vec(mean(Xs[G]["Xdelta_t2dist"]; dims = 1)),
                vec(mean(Xs[G]["Xhat_t2dist"]; dims = 1)),
            ) .- vec(mean(Xs["data"]["Y_t2dist"]; dims = 1))
        );
        label = "\$|" .* ["A_{\\ell,X}" "A_{\\ell,X_{\\epsilon_0}}" "A_{\\ell,X_{\\epsilon}}" "A_{\\ell,X_{\\delta}}" "A_{\\ell,\\hat{X}}"] .* " - A_{\\ell,Y}|\$",
        ylabel = L"$T_2$ amplitude difference [a.u.]",
        marker = (5, [:utriangle :dtriangle :diamond :rtriangle :square], [:green :purple :orange :magenta :red]),
        line = (2, [:dash :dash :dash :dash :solid], [:green :purple :orange :magenta :red]),
        common_args...
    );
    _save_and_display(p, "$(G)_t2_distbn_diff")
end
=#

####
#### Plot learned correction vs. theta
####

for G in GENERATOR_NAMES
    function make_binned(x, y; binwidth)
        Is = partition(sortperm(x), binwidth)
        xbar = [mean(x[I]) for I in Is]
        ybar = [mean(vec(y[:,I])) for I in Is]
        ystd = [std(vec(y[:,I])) for I in Is]
        return xbar, ybar, ystd
    end
    common_args = Dict{Symbol,Any}(:line => (1,), :tickfontsize => 10, :legendfontsize => 10)
    theta_args = [
        Dict{Symbol,Any}(:label => L"$\cos(\alpha)$", :xlims => (-1.0, -0.5)),
        Dict{Symbol,Any}(:label => L"$T_{2,short}$",  :xlims => (8.0, 100.0)),
        Dict{Symbol,Any}(:label => L"$T_{2,long}$",   :xlims => (16.0, 1000.0)),
        Dict{Symbol,Any}(:label => L"$A_{2,short}$",  :xlims => (0.0, 3.0)),
        Dict{Symbol,Any}(:label => L"$A_{2,long}$",   :xlims => (0.0, 3.0)),
    ]
    p = plot(
        map(1:5) do j
            theta_means, g_delta_means, g_delta_stds = make_binned(θ[j,:], Xs[G]["g_delta"]; binwidth = 1000)
            plot(theta_means, g_delta_means; ribbon = g_delta_stds, common_args..., theta_args[j]...)
        end...
    );
    _save_and_display(p, "$(G)_delta_vs_theta")
end

nothing

####
#### Earth mover's distance (Wasserstein distance) utils
####

#= Requires heavy dependencies (https://discourse.julialang.org/t/computing-discrete-wasserstein-emd-distance-in-julia/9600/10)
    using LightGraphs, LightGraphsFlows
    import LightGraphs, LightGraphsFlows
    import JuMP, Ipopt # choose LP solver
    function earthmover_lg(a::Vector, b::Vector)
        @assert( sum(a) ≈ sum(b), "Input vectors are not stochastic...")
        @assert( length(a) == length(b), "Input vectors have different lengths...")
        len = length(a)
        g = LightGraphs.complete_digraph(len)
        cost,  capacity = zeros(len,len), ones(len,len)
        demand = b .- a
        for i in 1:len, j in 1:len
            if i != j
                cost[i,j] = abs(j - i)
            end
        end
        flow = LightGraphsFlows.mincost_flow(g, demand, capacity, cost, JuMP.with_optimizer(Ipopt.Optimizer))
        return sum( flow .* cost )
    end
    earthmover_lg([1,2,3], [3,2,1]) # should be 4
    earthmover_lg([3,2,1,4], [1,2,4,3]) # should be 5
=#

#= Earth mover's distance wrapping original C package (very buggy; tends to segfault)
    using EarthMoversDistance
    let d = Cityblock()
        @show earthmovers(Float64[1,2,3], Float64[3,2,1], d) # should be 4 (normalized by histogram sum?)
        @show earthmovers(Float64[3,2,1,4], Float64[1,2,4,3], d) # should be 5 (normalized by histogram sum?)
    end;
    let
        x1, x2 = sort(rand(8)), sort(rand(8))
        if @show(earthmovers(x1, x2, Cityblock())) == 0
            @show x1; @show x2
        end
    end;
=#
