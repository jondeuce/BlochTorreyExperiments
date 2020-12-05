####
#### Code loading
####

using MMDLearning
Random.seed!(0)
pyplot(size = (800,600))

####
#### Load image data
####

const mmd_settings = load_settings();
const save_folder  = mkpath(MMDLearning.getnow());

# Load data samplers
if !(@isdefined(sampleX) && @isdefined(sampleY) && @isdefined(sampleθ) && @isdefined(fits))
    global sampleX, sampleY, sampleθ, _, _, fits = make_mle_data_samplers(
        mmd_settings["prior"]["data"]["image"]::String,
        mmd_settings["prior"]["data"]["thetas"]::String;
        ntheta = mmd_settings["data"]["ntheta"]::Int,
        padtrain = false,
        normalizesignals = false,
        filteroutliers = false,
        plothist = false,
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
    #####
    ##### MMD models (t-statistic optimized kernels)
    #####
    # "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/117"), # tstat-kernel #rank: 1
    # "mmd" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/17"), # tstat-kernel #rank: 2
    # "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/225"), # tstat-kernel #rank: 3
    #####
    ##### Hybrid GAN + MMD models (t-statistic optimized kernels)
    #####
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/45"), # tstat-kernel #decent #rank: 4
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/11"), # tstat-kernel #decent #rank: 3
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/189"), # tstat-kernel #bad
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/169"), # tstat-kernel #bad
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/101"), # tstat-kernel #bad
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/141"), # tstat-kernel #decent #rank: 2
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/157"), # tstat-kernel #decent #rank: >2
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/143"), # tstat-kernel #decent #rank: 1
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/165"), # tstat-kernel #not great #rank: >1
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/133"), # tstat-kernel #bad #rank: >1
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/173"), # tstat-kernel #bad #rank: >1
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/109"), # tstat-kernel #good? #rank should be 1?
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/189"), # tstat-kernel #bad
    #####
    ##### MMD models (MMD optimized kernels)
    #####
    # "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/236"), # MMD-kernel (hdim = 256)
    # "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/224"), # MMD-kernel (hdim = 256)
    "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/126"), # MMD-kernel #best
    # "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/96"), # MMD-kernel
    # "mmd" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/63"), # MMD-kernel
    #####
    ##### Hybrid GAN + MMD models (MMD optimized kernels)
    #####
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/90"), # MMD-kernel #great
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/158"), # MMD-kernel #quite good
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/190"), # MMD-kernel #quite good
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/12"), # MMD-kernel #decent
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/74"), # MMD-kernel #bad
    # 
    # │ Row │ folder │ kernel.losstype │ hdim  │ signal_fit_logL │ modeltype │ Cityblock │ Euclidean │ ChiSqDist │ Wasserstein │
    # │     │ String │ String          │ Int64 │ Any             │ String    │ Int64     │ Float64   │ Float64   │ Float64     │
    # ├─────┼────────┼─────────────────┼───────┼─────────────────┼───────────┼───────────┼───────────┼───────────┼─────────────┤
    # │ 1   │ 89     │ tstatistic      │ 256   │                 │ current   │ 80606     │ 18078.8   │ 3668.42   │ 214.031     │
    # │ 2   │ 272    │ MMD             │ 256   │                 │ current   │ 94268     │ 22015.0   │ 3927.13   │ 240.334     │
    # │ 3   │ 67     │ MMD             │ 128   │                 │ current   │ 92351     │ 21371.8   │ 3868.36   │ 246.672     │
    # │ 4   │ 187    │ MMD             │ 128   │                 │ current   │ 93842     │ 18614.2   │ 1879.88   │ 246.774     │
    # │ 5   │ 243    │ MMD             │ 256   │                 │ current   │ 93924     │ 19876.1   │ 2826.87   │ 251.505     │
    # │ 6   │ 197    │ tstatistic      │ 128   │ -169.459        │ best      │ 91490     │ 19864.4   │ 3411.13   │ 255.975     │
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/272"),
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/67"),
    "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/187"),
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson",    "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/19"),
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/138"),
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson",    "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/275"),
    # "hyb" => Dict{String,Any}("modelfile" => "best-model.bson",    "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/187"),
    # "hyb" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/275"),
    #####
    ##### GAN models
    #####
    "gan" => Dict{String,Any}("modelfile" => "best-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/ganopt-v3/sweep/48"),
    # "gan" => Dict{String,Any}("modelfile" => "current-model.bson", "folder" => "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/ganopt-v3/sweep/96"),
);
for (dataset, data) in all_generators
    data["models"] = deepcopy(BSON.load(joinpath(data["folder"], data["modelfile"])));
    data["settings"] = deepcopy(TOML.parsefile(joinpath(data["folder"], "settings.toml")));
end
let prog = deepcopy(BSON.load(joinpath(all_generators["mmd"]["folder"], all_generators["mmd"]["modelfile"][1:end-10] * "progress.bson"))["progress"])
    all_generators["mmd"]["models"]["logsigma"] = copy(prog.logsigma[end])
end;

# Save models
BSON.bson(joinpath(save_folder, "plot-models.bson"), all_generators);

#= TODO
let _models = BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-3/plot-models.bson")
    # GENERATOR_MODELS[1] = _models[:mmd_models]["mmd"]
    # GENERATOR_MODELS[2] = _models[:gan_models]["G"]
    # GENERATOR_MODELS[3] = _models[:hyb_models]["G"]
    all_generators["hyb"]["models"]["G"] = deepcopy(_models[:hyb_models]["G"])
    all_generators["hyb"]["models"]["D"] = deepcopy(_models[:hyb_models]["D"])
    all_generators["hyb"]["models"]["logsigma"] = deepcopy(_models[:hyb_models]["logsigma"])
end;
BSON.bson(joinpath(save_folder, "plot-models-with-hyb-plots-checkpoint-3.bson"), all_generators);
=#

# Make corrected data
const GENERATOR_NAMES  = ["mmd", "gan", "hyb"];
const GENERATOR_MODELS = Chain[all_generators["mmd"]["models"]["mmd"], all_generators["gan"]["models"]["G"], all_generators["hyb"]["models"]["G"]];

const Xs = Dict{String,Any}(
    "data" => Dict{String,Any}("Y" => Y),
    "mle" => Dict{String,Any}("X" => X, "Xeps0" => rand.(Rician.(X, exp.(fits.logsigma')))), # Xeps0: constant eps Rician noise from MLE fit
    [G => Dict{String,Any}() for G in GENERATOR_NAMES]...,
);
for (dataset, G) in zip(GENERATOR_NAMES, GENERATOR_MODELS)
    # dataset == "hyb" || continue #TODO
    g_delta, g_eps = correction_and_noiselevel(G, X);
    Xs[dataset]["Xhat"]    = corrected_signal_instance(G, X, g_delta, g_eps); # add learned correction + learned noise
    Xs[dataset]["Xdelta"]  = abs.(X .+ g_delta); # add learned noise only
    Xs[dataset]["Xeps"]    = corrected_signal_instance(G, X, g_eps); # add learned noise only
    Xs[dataset]["g_delta"] = g_delta;
    Xs[dataset]["g_eps"]   = g_eps;
end;

const XLABELMAP = Dict{String,String}(
    [G => "\\hat{X}_{" * uppercase(G) * "}" for G in GENERATOR_NAMES]...,
    "Y"       => "Y",
    "X"       => "X",
    "Xeps0"   => raw"X_{\epsilon_0}",
    "Xeps"    => raw"X_{\epsilon}",
    "Xhat"    => raw"\hat{X}",
    "Xdelta"  => raw"X_{\delta}",
    "g_eps"   => raw"g_{\epsilon}",
    "g_delta" => raw"g_{\delta}",
)

#=
let m = 4096
    Is = sample(1:size(Y,2), m; replace = false)
    ptest_hyb = mmd_perm_test_power(all_generators["hyb"]["models"]["logsigma"], Xs["hyb"]["Xhat"][:,Is], Y[:,Is]; batchsize = m, nperms = 128, nsamples = 128)
    display(mmd_perm_test_power_plot(ptest_hyb))
    ptest_mmd = mmd_perm_test_power(all_generators["mmd"]["models"]["logsigma"], Xs["mmd"]["Xhat"][:,Is], Y[:,Is]; batchsize = m, nperms = 128, nsamples = 128)
    display(mmd_perm_test_power_plot(ptest_mmd))
end
=#

# Saving figs
_save_and_display(p, fname::String) = (map(ext -> savefig(p, joinpath(save_folder, fname * ext)), (".png", ".pdf")); nothing) #display(p)

# Histograms
using PyCall, Distances
const st = pyimport("scipy.stats")
_make_hist(x, edges) = fit(Histogram, x, UnitWeights{Float64}(length(x)), edges; closed = :left)
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

#= #TODO
=#
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

####
#### Noise level (epsilon) histograms
####

for G in GENERATOR_NAMES
    p = plot();
    common_args = Dict{Symbol,Any}(:normalized => true, :xlims => (-2.6, -1.2), :line => (2,), :tickfontsize => 10, :legendfontsize => 9)
    stephist!(p, log10.(exp.(fits.logsigma)); lab = L"Signalwise $\log_{10}(\epsilon_0)$ from MLE", common_args...);
    stephist!(p, log10.(vec(Xs[G]["g_eps"])); lab = L"Aggregated $\log_{10}(\epsilon)$ from $\ln(\epsilon) \sim g_{\epsilon}(X_i)$", common_args...);
    stephist!(p, log10.(vec(mean(Xs[G]["g_eps"]; dims = 1))); lab = L"Signalwise $\log_{10}$(Mean($\epsilon$))", common_args...);
    stephist!(p, log10.(vec(exp.(mean(log.(Xs[G]["g_eps"]); dims = 1)))); lab = L"Signalwise $\log_{10}$(GeoMean($\epsilon$))", common_args...);
    _save_and_display(p, "$(G)_log_noise_hist")

    p = plot();
    common_args = Dict{Symbol,Any}(:normalized => true, :xlims => (0, 0.03), :line => (2,), :tickfontsize => 10, :legendfontsize => 9)
    stephist!(p, exp.(fits.logsigma); lab = L"Signalwise $\epsilon_0$ from MLE", common_args...);
    stephist!(p, vec(Xs[G]["g_eps"]); lab = L"Aggregated $\epsilon$ from $\ln(\epsilon) \sim g_{\epsilon}(X_i)$", common_args...);
    stephist!(p, vec(mean(Xs[G]["g_eps"]; dims = 1)); lab = L"Signalwise Mean($\epsilon$)", common_args...);
    stephist!(p, vec(exp.(mean(log.(Xs[G]["g_eps"]); dims = 1))); lab = L"Signalwise GeoMean($\epsilon$)", common_args...);
    _save_and_display(p, "$(G)_noise_hist")
end

####
#### Signal intensity histograms
####

const whole_signal_hists = Dict{String,Any}();
const signal_hists_by_echo = [Dict{String,Any}() for _ in 1:t2mapopts.nTE];

print("processing whole signal histograms:")
@time let
    binwidth = ceil(Int, mean(vec(Xs["data"]["Y"])) / (50 * Y_RES))
    edges = Y_EDGES[1:binwidth:end]
    h = whole_signal_hists
    h["Y"] = fast_hist_1D(vec(Xs["data"]["Y"]), edges)
    for Xmle in ["X", "Xeps0"]
        h[Xmle] = fast_hist_1D(vec(Xs["mle"][Xmle]), edges)
    end
    for G in GENERATOR_NAMES, Xgen in ["Xhat"]
        h[G] = fast_hist_1D(vec(Xs[G][Xgen]), edges)
    end
end;

print("processing signal histograms by echo:")
@time for j in 1:t2mapopts.nTE
    binwidth = ceil(Int, mean(Xs["data"]["Y"][j,:]) / (50 * Y_RES))
    edges = Y_EDGES[1:binwidth:end]
    h = signal_hists_by_echo[j]
    h["Y"] = fast_hist_1D(Xs["data"]["Y"][j,:], edges)
    for Xmle in ["X", "Xeps0"]
        h[Xmle] = fast_hist_1D(Xs["mle"][Xmle][j,:], edges)
    end
    for G in GENERATOR_NAMES, Xgen in ["Xhat"]
        h[G] = fast_hist_1D(Xs[G][Xgen][j,:], edges)
    end
end;

for G in GENERATOR_NAMES
    common_args = Dict{Symbol,Any}(:seriestype => :steppost, :line => (1,), :tickfontsize => 10, :legendfontsize => 9)
    p = plot(
        map([1,16,24,32,40,48]) do j
            p = plot();
            hY, hX, hX̂ = signal_hists_by_echo[j]["Y"], signal_hists_by_echo[j]["X"], signal_hists_by_echo[j][G]
            plot!(p, hY.edges[1][1:end-1], hY.weights; lab = "\$Y_{$j}\$", common_args...)
            plot!(p, hX.edges[1][1:end-1], hX.weights; lab = "\$X_{$j}\$", common_args...)
            plot!(p, hX̂.edges[1][1:end-1], hX̂.weights; lab = "\$\\hat{X}_{$j}\$", common_args...)
            xl = (0.0, hY.edges[1][1:end-1][1 + length(hY.weights) - findfirst(>=(250), hY.weights[end:-1:1])])
            xlims!(p, xl)
            p
        end...;
    );
    _save_and_display(p, "$(G)_signal_hist")
end

for G in GENERATOR_NAMES
    common_args = Dict{Symbol,Any}(:seriestype => :steppost, :line => (1,), :tickfontsize => 10, :legendfontsize => 9)
    p = plot(
        map([1,16,24,32,40,48]) do j
            p = plot();
            hY, hX, hX̂ = signal_hists_by_echo[j]["Y"], signal_hists_by_echo[j]["X"], signal_hists_by_echo[j][G]
            plot!(p, hX.edges[1][1:end-1], (hX.weights .- hY.weights) ./ (maximum(hY.weights)); lab = "\$(X - Y)_{$j}\$", common_args...)
            plot!(p, hX̂.edges[1][1:end-1], (hX̂.weights .- hY.weights) ./ (maximum(hY.weights)); lab = "\$(\\hat{X} - Y)_{$j}\$", common_args...)
            i_last = 1 + length(hY.weights) - findfirst(>=(250), hY.weights[end:-1:1])
            xl = (0.0, hY.edges[1][1:end-1][i_last])
            yl = maximum(abs, (hX.weights[1:i_last] .- hY.weights[1:i_last]) ./ (maximum(hY.weights))) .* (-1,1)
            xlims!(p, xl)
            ylims!(p, yl)
            p
        end...;
    );
    _save_and_display(p, "$(G)_signal_hist_diff")
end

p = let
    common_args = Dict{Symbol,Any}(:seriestype => :line, :line => (2,), :yscale => :log10, :tickfontsize => 10, :legendfontsize => 9)
    plot(
        map([Cityblock(), Euclidean(), ChiSqDist(), :wasserstein_distance]) do _dhist_type
            Gplot = ["X"; "Xeps0"; GENERATOR_NAMES]
            xdata = 1000 .* t2mapopts.TE .* (1:t2mapopts.nTE)
            ydata = map(product(signal_hists_by_echo, Gplot)) do (h,G)
                _dhist(h[G], h["Y"], _dhist_type)
            end
            ylabeldata = map(product([whole_signal_hists], Gplot)) do (h,G)
                _dhist(h[G], h["Y"], _dhist_type)
            end
            yleg = map(enumerate(permutedims(Gplot))) do (i,G)
                "\$d($(XLABELMAP[G]), Y) = $(round(ylabeldata[i]; digits = 1))\$"
            end
            plot(xdata, ydata;
                label = yleg,
                ylabel = "d = $_dhist_type",
                xlabel = "Echo time [ms]",
                color = [:green :blue :red :orange :magenta],
                common_args...,
            )
        end...,
    )
end;
_save_and_display(p, "compare_genatr_signal_hist");

p = let
    common_args = Dict{Symbol,Any}(:seriestype => :steppost, :line => (1,), :tickfontsize => 10, :legendfontsize => 10)
    hY = whole_signal_hists["Y"]
    Gplot = GENERATOR_NAMES
    xdata = hY.edges[1][1:end-1]
    ydata = mapreduce(hcat, Gplot) do G
        hX̂ = whole_signal_hists[G]
        abs.(hX̂.weights .- hY.weights)
    end
    yleg = permutedims(["\$ |$(XLABELMAP[G]) - Y| \$" for G in Gplot])
    logyleg = permutedims(["\$ \\log_{10} (1 + |$(XLABELMAP[G]) - Y|) \$" for G in Gplot])
    p = plot(
        plot(xdata, ydata; ylabel = L"$|\Delta Histogram|$", xlabel = nothing, label = yleg),
        plot(xdata, log10.(1 .+ ydata); ylabel = L"$\log_{10}(1 + |\Delta Histogram|)$", xlabel = "Signal [a.u.]", label = logyleg);
        layout = @layout([a{0.5h}; b]),
        common_args...
    )
end;
_save_and_display(p, "compare_genatr_whole_signal_hist_diff");

####
#### Compare T2 distributions/NNLS results
####

#= #TODO
=#
let
    common_args = Dict{Symbol,Any}(
        :tickfontsize => 10, :legendfontsize => 9, :xscale => :log10, :xrot => 30.0,
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
    yleg = map(enumerate(permutedims(["Y", "Xeps0", "mmd", "gan", "hyb"]))) do (i,G)
        "\$ A_{\\ell, $(XLABELMAP[G])} \$"
    end
    p = plot(xdata, ydata;
        label = yleg,
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
    yleg = map(enumerate(permutedims(["Xeps0", "mmd", "gan", "hyb"]))) do (i,G)
        _l2norm = round(norm(ydata[:,i]); sigdigits = 3)
        "\$ |A_{\\ell, $(XLABELMAP[G])} - A_{\\ell,Y}| \$: \$ \\ell^2 \$ norm = \$ $_l2norm \$"
    end
    p = plot(xdata, ydata;
        label = yleg,
        ylabel = L"$T_2$ amplitude difference [a.u.]",
        marker = (5, [:utriangle :square :diamond :dtriangle], [:green :red :orange :magenta]),
        line = (3, [:solid :solid :solid :solid], [:green :red :orange :magenta]),
        common_args...
    );
    _save_and_display(p, "compare_genatr_t2_distbn_diff")
end

for G in GENERATOR_NAMES
    common_args = Dict{Symbol,Any}(
        :tickfontsize => 10, :legendfontsize => 9, :xscale => :log10, :xrot => 30.0,
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
    common_args = Dict{Symbol,Any}(:line => (1,), :tickfontsize => 10, :legendfontsize => 9)
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

#=
include("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/scripts/plot_sweep_results.jl")
function hist_distance_loop(
        df,
        X,
        Y;
        results_dir = "/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/"
    )
    global Y_RES, Y_EDGES
    binwidth = ceil(Int, mean(vec(Y)) / (50 * Y_RES))
    edges = Y_EDGES[1:binwidth:end]
    hY = fast_hist_1D(vec(Y), edges)

    df_print = DataFrame()
    for row in eachrow(sort(df, :signal_fit_logL))
        if row.signal_fit_logL > -169
            break
        end
        for mtype in ["current", "best"]
            G = deepcopy(BSON.load(joinpath(results_dir, row.folder, "$mtype-model.bson"))["G"])
            X̂ = corrected_signal_instance(G, X)
            hX̂ = fast_hist_1D(vec(X̂), edges)
            ds = (
                Cityblock = _dhist(hX̂, hY, Cityblock()),
                Euclidean = _dhist(hX̂, hY, Euclidean()),
                ChiSqDist = _dhist(hX̂, hY, ChiSqDist()),
                Wasserstein = _dhist(hX̂, hY, :wasserstein_distance),
            )
            df_print_row = hcat(
                DataFrame(row[[:folder, Symbol("kernel.losstype"), :hdim, :signal_fit_logL]]),
                DataFrame(; modeltype = mtype, ds...)
            )
            (mtype == "current") && (df_print_row[1,:signal_fit_logL] = nothing)
            append!(df_print, df_print_row)
        end
        show(sort(df_print, :Wasserstein); allrows = true, allcols = true); println("")
    end

    return df_print
end
df_dist = hist_distance_loop(df,X,Y)
=#

#=
│ Row │ folder │ kernel.losstype │ hdim  │ signal_fit_logL │ modeltype │ Cityblock │ Euclidean │ ChiSqDist │ Wasserstein │
│     │ String │ String          │ Int64 │ Any             │ String    │ Int64     │ Float64   │ Float64   │ Float64     │
├─────┼────────┼─────────────────┼───────┼─────────────────┼───────────┼───────────┼───────────┼───────────┼─────────────┤
│ 1   │ 272    │ MMD             │ 256   │                 │ current   │ 93329     │ 22296.8   │ 4047.58   │ 241.576     │
│ 2   │ 187    │ MMD             │ 128   │                 │ current   │ 92852     │ 18331.2   │ 1819.99   │ 247.628     │
│ 3   │ 19     │ MMD             │ 128   │ -170.731        │ best      │ 98404     │ 24584.0   │ 4765.81   │ 253.325     │
│ 4   │ 67     │ MMD             │ 128   │                 │ current   │ 94382     │ 21653.4   │ 4039.9    │ 253.573     │
=#

#=
let
    d = Dict{String,Any}()
    d["t2times"] = copy(Xs["data"]["Y_t2maps"]["t2times"])
    for (name, maps) in Xs
        d[name] = Dict{String,Any}()
        for (arrname, arr) in maps
            if endswith(arrname, "t2dist")
                d[name][arrname] = vec(mean(Xs[name][arrname]; dims = 1))
            end
        end
    end
    # for (name, t2maps) in d
    #     println(name); display(t2maps)
    # end
    MAT.matwrite("mri_t2dists.mat", deepcopy(d))
    d
end;
=#

####
#### Toy model data
####

#=
let
    local θ = toy_theta_sampler(10_000)
    local X = toy_signal_model(θ, nothing, 4)
    local Y = toy_signal_model(θ, 0.01, 2)
    local Ytrue = toy_signal_model(θ, nothing, 2)
    local toymodels = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-5/inference-2020-05-31-T-16-28-35-131/toy_inference.bson")["toy_models"])

    ####
    #### Load models + make corrected data
    ####

    # Make corrected data
    local GENERATOR_NAMES  = ["mmd", "gan", "hyb"];
    local GENERATOR_MODELS = Chain[toymodels["MMD"], toymodels["GAN"], toymodels["HYB"]];

    local Xs = Dict{String,Any}(
        "data" => Dict{String,Any}("Y" => copy(Y), "Ytrue" => copy(Ytrue)),
        "mle" => Dict{String,Any}("X" => copy(X), "Xeps0" => rand.(Rician.(X, 0.01))),
        [G => Dict{String,Any}() for G in GENERATOR_NAMES]...,
    );
    for (dataset, G) in zip(GENERATOR_NAMES, GENERATOR_MODELS)
        local g_delta, g_eps = correction_and_noiselevel(G, X);
        Xs[dataset]["Xhat"]    = corrected_signal_instance(G, X, g_delta, g_eps); # add learned correction + learned noise
        Xs[dataset]["Xdelta"]  = abs.(X .+ g_delta); # add learned noise only
        Xs[dataset]["Xeps"]    = corrected_signal_instance(G, X, g_eps); # add learned noise only
        Xs[dataset]["g_delta"] = copy(g_delta);
        Xs[dataset]["g_eps"]   = copy(g_eps);
    end;
    for (name, maps) in Xs
        println(name); display(Xs[name])
    end
    MAT.matwrite("toy_data.mat", deepcopy(Xs))
    Xs
end
=#

#=
let data = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-5/inference-2020-05-31-T-16-28-35-131/toy_inference.bson"))
    local toy_models = deepcopy(data["toy_models"])
    local toy_inf = deepcopy(data["toy_inference"])

    for (name, infdata) in toy_inf
        name == "CVAE" && continue
        println(name)
        local err = if name == "F2"
            local Y = toy_signal_model(toy_inf[name].θ, nothing, 2)
            local Xhat = toy_signal_model(toy_inf[name].θhat, nothing, 2)
            100 .* sqrt.(vec(mean(abs2, Y .- Xhat; dims = 1)))
        elseif name == "F4"
            local Y = toy_signal_model(toy_inf[name].θ, nothing, 2)
            local Xhat = toy_signal_model(toy_inf[name].θhat, nothing, 4)
            100 .* sqrt.(vec(mean(abs2, Y .- Xhat; dims = 1)))
        elseif name != "CVAE"
            local Y = toy_signal_model(toy_inf[name].θ, nothing, 2)
            local X = toy_signal_model(toy_inf[name].θhat, nothing, 4)
            local gdelta = toy_models[name](X)[1:div(end,2), :]
            local Xhat = abs.(X .+ gdelta)
            100 .* sqrt.(vec(mean(abs2, Y .- Xhat; dims = 1)))
        end
        local mu = mean(err)
        local s = std(err) / sqrt(length(err))
        @printf("rmse = %.3f +/- %.3f\n", mu, s)
    end
end
=#
