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

if !(@isdefined(image))
    global image = DECAES.load_image(mmd_settings["prior"]["data"]["image"]);
end

if !(@isdefined(t2maps) && @isdefined(t2dist))
    global t2mapopts = T2mapOptions(
        MatrixSize = size(image)[1:3],
        nTE        = size(image)[4],
        TE         = 8e-3,
        nT2        = 40,
        T2Range    = (8e-3, 1.0), #TODO range changed from (15e-3, 2.0) to match CVAE outputs
    );
    global t2maps, t2dist = DECAES.T2mapSEcorr(image, t2mapopts);
end

if !(@isdefined(t2parts))
    global t2partopts = T2partOptions(
        MatrixSize = size(image)[1:3],
        nT2        = t2mapopts.nT2,
        T2Range    = t2mapopts.T2Range,
        SPWin      = (prevfloat(t2mapopts.T2Range[1]), 40e-3),
        MPWin      = (nextfloat(40e-3), nextfloat(t2mapopts.T2Range[2])),
    );
    global t2parts = DECAES.T2partSEcorr(t2dist, t2partopts);
end

#const t2maps = DECAES.MAT.matread(mmd_settings["prior"]["data"]["t2maps"]);
#const t2dist = DECAES.load_image(mmd_settings["prior"]["data"]["t2dist"]);
#const t2parts= DECAES.MAT.matread(mmd_settings["prior"]["data"]["t2parts"]);

####
#### Make learned maps
####

cvae_model = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/cvae-both-corr-v2/sweep/35/log/2020-05-01-T-12-05-36-268.acc=rmse_gamma=1_loss=l2_DenseLIGOCVAE_Dh=128_Nh=6_Xout=5_Zdim=6_act=relu_boundmean=true.model-best.bson")[:model]);
Is = filter(I -> image[I,1] > 0, CartesianIndices(size(image)[1:3]));

@time model_mu_std = cvae_model(permutedims(image[Is,:] ./ 1e6); nsamples = 100, stddev = true); #TODO
model_thetas, model_stds = model_mu_std[1:end÷2, ..], model_mu_std[end÷2+1:end, ..];
model_thetas_image = fill(NaN, size(image)[1:3]..., size(model_thetas,1));
model_thetas_image[Is, :] .= permutedims(model_thetas);
# @time model_thetas = image[Is,:] ./ 1e6 |> permutedims |> copy |> x -> [cvae_model(x; nsamples = 1, stddev = false) for _ in 1:25] |> x -> cat(x...; dims = 3)
# model_thetas_image = fill(NaN, size(image)[1:3]..., size(model_thetas,1));
# model_thetas_image[Is, :] .= permutedims(mean(model_thetas; dims = 3)[:,:,1]);

CUTOFF = 30.0 #35.0 #1000 * t2partopts.SPWin[2] # cutoff in seconds
BANDWIDTH = 10.0

mwf = fill(NaN, size(image)[1:3]...);
# mwf[Is] .= ((Ashort, Along) -> Ashort / (Ashort + Along)).(model_thetas_image[Is, 4], model_thetas_image[Is, 5]); # no cutoff
# mwf[Is] .= ((T2short, Ashort, Along) -> ifelse(T2short < CUTOFF, Ashort / (Ashort + Along), 0.0)).(model_thetas_image[Is, 2], model_thetas_image[Is, 4], model_thetas_image[Is, 5]); # hard cutoff
mwf[Is] .= ((T2short, Ashort, Along) -> (Ashort / (Ashort + Along)) * sigmoid(-(T2short - CUTOFF) / BANDWIDTH)).(model_thetas_image[Is, 2], model_thetas_image[Is, 4], model_thetas_image[Is, 5]); # soft cutoff
# mwf[Is] .= ((T2short, Ashort, Along) -> (Ashort / (Ashort + Along)) * cdf(Normal(), -(T2short - CUTOFF) / BANDWIDTH)).(model_thetas_image[Is, 2], model_thetas_image[Is, 4], model_thetas_image[Is, 5]); # soft cutoff
# mwf[Is] .= vec(mean(((Ashort, Along) -> Ashort / (Ashort + Along)).(model_thetas[4:4,:,:], model_thetas[5:5,:,:]); dims = 3)); # no cutoff
# mwf[Is] .= vec(mean(((T2short, Ashort, Along) -> ifelse(T2short < CUTOFF, Ashort / (Ashort + Along), 0.0)).(model_thetas[2:2,:,:], model_thetas[4:4,:,:], model_thetas[5:5,:,:]); dims = 3)); # hard cutoff
# mwf[Is] .= vec(mean(((T2short, Ashort, Along) -> (Ashort / (Ashort + Along)) * sigmoid(-(T2short - CUTOFF) / BANDWIDTH)).(model_thetas[2:2,:,:], model_thetas[4:4,:,:], model_thetas[5:5,:,:]); dims = 3)); # soft cutoff

ggm = fill(NaN, size(image)[1:3]...);
ggm[Is] .= ((T2short, T2long, Ashort, Along) -> exp((Ashort * log(T2short) + Along * log(T2long)) / (Ashort + Along))).(model_thetas_image[Is, 2], model_thetas_image[Is, 3], model_thetas_image[Is, 4], model_thetas_image[Is, 5]); # general geometric mean

Islice = let Zslice = 24
    Is_Zslice = filter(I -> I[3] == Zslice, Is)
    XRange, YRange = extrema((x->x[1]).(Is_Zslice)), extrema((x->x[2]).(Is_Zslice))
    (XRange[1]:XRange[2], YRange[2]:-1:YRange[1], Zslice)
end

pyplot(size = (700,800))
_saveheatmap(fname::String) = p -> map(ext -> savefig(p, fname * ext), (".png", ".pdf")) # unreasonably large files: ".eps", ".svg"
common_args = Dict{Symbol,Any}(:aspect_ratio => :equal, :grid => :none, :ticks => :none, :axis => :none, :border => :none, :titlefontsize => 18, :tickfontsize => 14)

heatmap(permutedims(mwf[Islice...]) |> img -> (x -> clamp(x, 0.0, Inf)).(img); clim = (0,0.3), title = "CVAE: MWF [a.u.]", common_args...) |> _saveheatmap("cvae-mwf");
heatmap(permutedims(t2parts["sfr"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, x)).(img); clim = (0,0.3), title = "DECAES: MWF [a.u.]", common_args...) |> _saveheatmap("decaes-mwf");

heatmap(permutedims(ggm[Islice...]) |> img -> (x -> clamp(x, 0.0, Inf)).(img); clim = (0,250), title = "CVAE: Geometric Mean T2 [ms]", common_args...) |> _saveheatmap("cvae-ggm");
heatmap(permutedims(t2maps["ggm"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, 1000x)).(img); clim = (0,250), title = "DECAES: Geometric Mean T2 [ms]", common_args...) |> _saveheatmap("decaes-ggm");

heatmap(permutedims(log10.(ggm[Islice...])); clim = (1,3), title = "CVAE: Log10 Geometric Mean T2 [ms]", common_args...) |> _saveheatmap("cvae-log10ggm");
heatmap(permutedims(log10.(1000 .* t2maps["ggm"][Islice...])); clim = (1,3), title = "DECAES: Log10 Geometric Mean T2 [ms]", common_args...) |> _saveheatmap("decaes-log10ggm");

heatmap(permutedims(t2maps["alpha"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, x)).(img); clim = (120,180), title = "DECAES: Flip Angle [degrees]", common_args...) |> _saveheatmap("decaes-flipangle");
heatmap(permutedims(t2parts["sgm"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, 1000x)).(img); clim = 1000 .* t2partopts.SPWin, title = "DECAES: SGM [ms]", common_args...) |> _saveheatmap("decaes-sgm");
heatmap(permutedims(t2parts["mgm"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, 1000x)).(img); clim = 1000 .* t2partopts.MPWin, title = "DECAES: MGM [ms]", common_args...) |> _saveheatmap("decaes-mgm");

heatmap(permutedims(model_thetas_image[Islice...,1]) |> img -> (x -> acosd(clamp(x, -1.0, 1.0))).(img); clim = (120,180), title = "CVAE: Flip Angle [deg]", common_args...) |> _saveheatmap("cvae-flipangle");
heatmap(permutedims(model_thetas_image[Islice...,2]) |> img -> (x -> clamp(x, 8.0, 1000.0)).(img); clim = (0,100), title = "CVAE: T2short [ms]", common_args...) |> _saveheatmap("cvae-T2short");
heatmap(permutedims(model_thetas_image[Islice...,3]) |> img -> (x -> clamp(x, 8.0, 1000.0)).(img); clim = (0,1000), title = "CVAE: T2long [ms]", common_args...) |> _saveheatmap("cvae-T2long");
heatmap(permutedims(model_thetas_image[Islice...,4]) |> img -> (x -> clamp(x, 0.0, Inf)).(img); title = "CVAE: Ashort [a.u.]", common_args...) |> _saveheatmap("cvae-Ashort");
heatmap(permutedims(model_thetas_image[Islice...,5]) |> img -> (x -> clamp(x, 0.0, Inf)).(img); title = "CVAE: Along [a.u.]", common_args...) |> _saveheatmap("cvae-Along");

#=
r2(x,y) = 1 - sum(abs2, y .- x) / sum(abs2, y .- mean(y))
out = let
    _Is = findall(0.1 .< t2parts["sfr"] .< 0.3)
    map(Iterators.product(30.0 : 1.0 : 40.0, 1.0 : 0.5 : 10.0)) do (CUTOFF, BANDWIDTH)
        _r2 = r2(mwf[_Is], t2parts["sfr"][_Is])
        @show CUTOFF, BANDWIDTH, _r2
        return (CUTOFF, BANDWIDTH, _r2)
    end
end
=#

nothing
