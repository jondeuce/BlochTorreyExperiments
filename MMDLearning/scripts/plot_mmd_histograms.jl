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

#const t2maps = DECAES.MAT.matread(mmd_settings["prior"]["data"]["t2maps"]);
#const t2dist = DECAES.load_image(mmd_settings["prior"]["data"]["t2dist"]);
#const t2parts= DECAES.MAT.matread(mmd_settings["prior"]["data"]["t2parts"]);

####
#### Load models + make corrected data
####

cvae_model = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/cvae-both-corr-v2/sweep/35/log/2020-05-01-T-12-05-36-268.acc=rmse_gamma=1_loss=l2_DenseLIGOCVAE_Dh=128_Nh=6_Xout=5_Zdim=6_act=relu_boundmean=true.model-best.bson")[:model]);
mmd_models = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/63/current-model.bson"))

# Convenience functions
split_correction_and_noise(μlogσ) = μlogσ[1:end÷2, :], exp.(μlogσ[end÷2+1:end, :])
noise_instance(X, ϵ) = ϵ .* randn(eltype(X), size(X)...)
get_correction_and_noise(X) = split_correction_and_noise(mmd_models["mmd"](mmd_models["vae.E"](X)))
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

# Make corrected data
const g_delta, g_eps = get_correction_and_noise(X);
const Xhat = get_corrected_signal(X, g_delta, g_eps); # add learned correction + learned noise
const Xdelta = abs.(X .+ g_delta); # add learned noise only
const Xeps = get_corrected_signal(X, g_eps); # add learned noise only
const Xeps0 = rand.(Rician.(X, exp.(fits.logsigma'))); # add "isotropic" noise from MLE fit

# Saving figs
_save_and_display(p, fname::String) = (map(ext -> savefig(p, fname * ext), (".png", ".pdf")); display(p))
_make_hist(x, edges) = fit(Histogram, x, UnitWeights{Float64}(length(x)), edges) #filter(<(maximum(x)), edges)

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
);
const t2partopts = T2partOptions(
    MatrixSize = (size(Y,2), 1, 1),
    nT2        = t2mapopts.nT2,
    T2Range    = t2mapopts.T2Range,
    SPWin      = (prevfloat(t2mapopts.T2Range[1]), 40e-3), #TODO
    MPWin      = (nextfloat(40e-3), nextfloat(t2mapopts.T2Range[2])), #TODO
);
if !(@isdefined(Y_t2maps) && @isdefined(Y_t2dist) && @isdefined(Y_t2parts))
    global Y_t2maps, Y_t2dist = DECAES.T2mapSEcorr(reshape(permutedims(Y), (size(Y,2), 1, 1, size(Y,1))), t2mapopts);
    global Y_t2parts = DECAES.T2partSEcorr(Y_t2dist, t2partopts);
end
if !(@isdefined(X_t2maps) && @isdefined(X_t2dist) && @isdefined(X_t2parts))
    global X_t2maps, X_t2dist = DECAES.T2mapSEcorr(reshape(permutedims(X), (size(X,2), 1, 1, size(X,1))), t2mapopts);
    global X_t2parts = DECAES.T2partSEcorr(X_t2dist, t2partopts);
end
if !(@isdefined(Xhat_t2maps) && @isdefined(Xhat_t2dist) && @isdefined(Xhat_t2parts))
    global Xhat_t2maps, Xhat_t2dist = DECAES.T2mapSEcorr(reshape(permutedims(Xhat), (size(Xhat,2), 1, 1, size(Xhat,1))), t2mapopts);
    global Xhat_t2parts = DECAES.T2partSEcorr(Xhat_t2dist, t2partopts);
end
if !(@isdefined(Xdelta_t2maps) && @isdefined(Xdelta_t2dist) && @isdefined(Xdelta_t2parts))
    global Xdelta_t2maps, Xdelta_t2dist = DECAES.T2mapSEcorr(reshape(permutedims(Xdelta), (size(Xdelta,2), 1, 1, size(Xdelta,1))), t2mapopts);
    global Xdelta_t2parts = DECAES.T2partSEcorr(Xdelta_t2dist, t2partopts);
end
if !(@isdefined(Xeps_t2maps) && @isdefined(Xeps_t2dist) && @isdefined(Xeps_t2parts))
    global Xeps_t2maps, Xeps_t2dist = DECAES.T2mapSEcorr(reshape(permutedims(Xeps), (size(Xeps,2), 1, 1, size(Xeps,1))), t2mapopts);
    global Xeps_t2parts = DECAES.T2partSEcorr(Xeps_t2dist, t2partopts);
end
if !(@isdefined(Xeps0_t2maps) && @isdefined(Xeps0_t2dist) && @isdefined(Xeps0_t2parts))
    global Xeps0_t2maps, Xeps0_t2dist = DECAES.T2mapSEcorr(reshape(permutedims(Xeps0), (size(Xeps0,2), 1, 1, size(Xeps0,1))), t2mapopts);
    global Xeps0_t2parts = DECAES.T2partSEcorr(Xeps0_t2dist, t2partopts);
end

####
#### Noise level (epsilon) histograms
####

p = plot();
common_args = Dict{Symbol,Any}(:normalized => true, :xlims => (-2.6, -1.2), :line => (2,), :tickfontsize => 10, :legendfontsize => 10)
stephist!(p, log10.(exp.(fits.logsigma)); lab = L"Signalwise $\log_{10}(\epsilon_0)$ from MLE", common_args...);
stephist!(p, log10.(vec(g_eps)); lab = L"Aggregated $\log_{10}(\epsilon)$ from $\ln(\epsilon) \sim g_{\epsilon}(X_i)$", common_args...);
stephist!(p, log10.(vec(mean(g_eps; dims = 1))); lab = L"Signalwise $\log_{10}$(Mean($\epsilon$))", common_args...);
stephist!(p, log10.(vec(exp.(mean(log.(g_eps); dims = 1)))); lab = L"Signalwise $\log_{10}$(GeoMean($\epsilon$))", common_args...);
_save_and_display(p, "log_noise_hist")

p = plot();
common_args = Dict{Symbol,Any}(:normalized => true, :xlims => (0, 0.03), :line => (2,), :tickfontsize => 10, :legendfontsize => 10)
stephist!(p, exp.(fits.logsigma); lab = L"Signalwise $\epsilon_0$ from MLE", common_args...);
stephist!(p, vec(g_eps); lab = L"Aggregated $\epsilon$ from $\ln(\epsilon) \sim g_{\epsilon}(X_i)$", common_args...);
stephist!(p, vec(mean(g_eps; dims = 1)); lab = L"Signalwise Mean($\epsilon$)", common_args...);
stephist!(p, vec(exp.(mean(log.(g_eps); dims = 1))); lab = L"Signalwise GeoMean($\epsilon$)", common_args...);
_save_and_display(p, "noise_hist")

####
#### Signal intensity histograms
####

common_args = Dict{Symbol,Any}(:seriestype => :steppost, :line => (1,), :tickfontsize => 10, :legendfontsize => 10)
p = plot(
    map([1,16,24,32,40,48]) do j
        p = plot();
        Yj = Y[j,:]
        binwidth = 50 * Y_RES
        edges = Y_EDGES[1 : ceil(Int, mean(Yj) / binwidth) : end]
        hY, hX, hX̂ = _make_hist(Yj, edges), _make_hist(X[j,:], edges), _make_hist(Xhat[j,:], edges)
        plot!(p, hY.edges[1], hY.weights; lab = "\$Y_{$j}\$", common_args...)
        plot!(p, hX.edges[1], hX.weights; lab = "\$X_{$j}\$", common_args...)
        plot!(p, hX̂.edges[1], hX̂.weights; lab = "\$\\hat{X}_{$j}\$", common_args...)
        xl = (0.0, hY.edges[1][1 + length(hY.weights) - findfirst(>=(250), hY.weights[end:-1:1])])
        xlims!(p, xl)
        p
    end...;
);
_save_and_display(p, "signal_hist")

common_args = Dict{Symbol,Any}(:seriestype => :steppost, :line => (1,), :tickfontsize => 10, :legendfontsize => 10)
p = plot(
    map([1,16,24,32,40,48]) do j
        p = plot();
        Yj = Y[j,:]
        binwidth = 50 * Y_RES
        edges = Y_EDGES[1 : ceil(Int, mean(Yj) / binwidth) : end]
        hY, hX, hX̂ = _make_hist(Yj, edges), _make_hist(X[j,:], edges), _make_hist(Xhat[j,:], edges)
        plot!(p, hX.edges[1], (hX.weights .- hY.weights) ./ (maximum(hY.weights)); lab = "\$(X - Y)_{$j}\$", common_args...)
        plot!(p, hX̂.edges[1], (hX̂.weights .- hY.weights) ./ (maximum(hY.weights)); lab = "\$(\\hat{X} - Y)_{$j}\$", common_args...)
        xl = (0.0, hY.edges[1][1 + length(hY.weights) - findfirst(>=(250), hY.weights[end:-1:1])])
        xlims!(p, xl)
        p
    end...;
);
_save_and_display(p, "signal_hist_diff")

#=
common_args = Dict{Symbol,Any}(:line => (1,), :ylim => (0.0, 0.02), :tickfontsize => 8, :legendfontsize => 8, :legend => :topright, :xticks => :none)
p = plot(
    map([1,8,16,32]) do j
        p = plot();
        Yj_sorted = sort!(Y[j,:])
        plot!(p, abs.(sort!(X[j,:]) .- Yj_sorted); lab = L"$X - Y$" * " (echo $j)", seriestype = :steppre, common_args...);
        plot!(p, abs.(sort!(Xhat[j,:]) - Yj_sorted); lab = L"$\hat{X} - Y$" * " (echo $j)", seriestype = :steppre, common_args...);
        p
    end...;
);
_save_and_display(p, "signal_qqdiff")
=#

#=
let
    p = plot();
    plot!(p, sort(Y[1,:]); lab = L"$Y$", seriestype = :steppre);
    plot!(p, sort(X[1,:]); lab = L"$X$", seriestype = :steppre);
    plot!(p, sort(Xhat[1,:]); lab = L"$\hat{X}$", seriestype = :steppre);
    display(p)
end
=#

####
#### Compare T2 distributions/NNLS results
####

p = plot(
    plot(
        Y_t2maps["t2times"],
        hcat(
            vec(mean(Y_t2dist; dims = 1)),
            vec(mean(X_t2dist; dims = 1)),
            vec(mean(Xhat_t2dist; dims = 1)),
            # vec(mean(Xeps0_t2dist; dims = 1)),
            # vec(mean(Xeps_t2dist; dims = 1))
        );
        label = L"Mean $T_2$ Distribution: " .* ["Y" "X" L"$\hat{X}$" L"$X_{\epsilon_0}$" L"$X_{\epsilon}$"],
        marker = (5, [:circle :utriangle :square], [:blue :green :red]),
        line = (2, [:solid :dash :dash], [:blue :green :red]),
        xticks = Y_t2maps["t2times"][1:3:end],
        xformatter = x -> "",
    ),
    plot(
        Y_t2maps["t2times"],
        abs.(
            hcat(
                vec(mean(X_t2dist; dims = 1)),
                vec(mean(Xeps0_t2dist; dims = 1)),
                vec(mean(Xeps_t2dist; dims = 1)),
                vec(mean(Xdelta_t2dist; dims = 1)),
                vec(mean(Xhat_t2dist; dims = 1)),
            ) .- vec(mean(Y_t2dist; dims = 1))
        );
        label = L"Mean $T_2$ Distribution: $|" .* ["X" "X_{\\epsilon_0}" "X_{\\epsilon}" "X_{\\delta}" "\\hat{X}"] .* L" - Y|$",
        marker = (5, [:utriangle :dtriangle :diamond :rtriangle :square], [:green :purple :orange :magenta :red]),
        line = (2, [:dash :dash :dash :dash :solid], [:green :purple :orange :magenta :red]),
        xticks = Y_t2maps["t2times"][1:3:end], xrot = 30, xformatter = x -> string(round(1000x; sigdigits = 3)) * " ms",
    );
    layout = @layout([a{0.4h}; b]), tickfontsize = 10, legendfontsize = 10, xscale = :log10,
);
_save_and_display(p, "t2_distbn")

####
#### Compare T2 distributions/NNLS results
####

p = let
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
    plot(
        map(1:5) do j
            Is = Colon()
            binwidth = 1000
            theta_means, g_delta_means, g_delta_stds = make_binned(θ[j,:], g_delta[:,:]; binwidth = 1000)
            plot(theta_means, g_delta_means; ribbon = g_delta_stds, common_args..., theta_args[j]...)
        end...
    )
end
_save_and_display(p, "delta_vs_theta")

nothing
