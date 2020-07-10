####
#### Code loading
####

using MMDLearning
Random.seed!(0)

####
#### PyPlot
####

import PyPlot
const plt = PyPlot
const rcParams = plt.PyDict(plt.matplotlib."rcParams")
rcParams["font.size"] = 10
rcParams["text.usetex"] = false

function pyheatmap(data; formatter = nothing, filename = nothing, clim = nothing, cticks = nothing, title = nothing, savetypes = [".png", ".pdf"])
    plt.figure(figsize = (8.0, 8.0), dpi = 150.0)
    plt.set_cmap("plasma")
    fig, ax = plt.subplots()
    ax.set_axis_off()
    img = ax.imshow(data, aspect = "equal", interpolation = "nearest")
    plt.title(title)

    if formatter isa Function
        formatter = plt.matplotlib.ticker.FuncFormatter(formatter)
    end
    cbar = fig.colorbar(img, ticks = cticks, format = formatter, aspect = 40)
    cbar.ax.tick_params(labelsize = 12)

    if !isnothing(clim)
        img.set_clim(clim...)
    end

    if !isnothing(filename)
        foreach(ext -> plt.savefig(filename * ext, bbox_inches = "tight", dpi = 150.0), savetypes)
    end
    plt.close(fig)

    return nothing
end
# pyheatmap(randn(6,3), filename = "val", clim = (0,1))

####
#### Load image data
####

const mwi_settings = load_settings();
#const t2maps = DECAES.MAT.matread(mwi_settings["prior"]["data"]["t2maps"]);
#const t2dist = DECAES.load_image(mwi_settings["prior"]["data"]["t2dist"]);
#const t2parts= DECAES.MAT.matread(mwi_settings["prior"]["data"]["t2parts"]);

if !(@isdefined(image))
    global image = DECAES.load_image(mwi_settings["prior"]["data"]["image"]);
end;

if !(@isdefined(t2maps) && @isdefined(t2dist))
    global t2mapopts = T2mapOptions(
        MatrixSize = size(image)[1:3],
        nTE        = size(image)[4],
        TE         = 8e-3,
        nT2        = 40,
        T2Range    = (8e-3, 1.0), #TODO range changed from (15e-3, 2.0) to match CVAE outputs
    );
    global t2maps, t2dist = DECAES.T2mapSEcorr(image, t2mapopts);
end;

if !(@isdefined(t2parts))
    global t2partopts = T2partOptions(
        MatrixSize = size(image)[1:3],
        nT2        = t2mapopts.nT2,
        T2Range    = t2mapopts.T2Range,
        SPWin      = (prevfloat(t2mapopts.T2Range[1]), 40e-3),
        MPWin      = (nextfloat(40e-3), nextfloat(t2mapopts.T2Range[2])),
    );
    global t2parts = DECAES.T2partSEcorr(t2dist, t2partopts);
end;

cvae_model = deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/cvae-both-corr-v2/sweep/35/log/2020-05-01-T-12-05-36-268.acc=rmse_gamma=1_loss=l2_DenseLIGOCVAE_Dh=128_Nh=6_Xout=5_Zdim=6_act=relu_boundmean=true.model-best.bson")[:model]); # Trained on MMD-corrected signal + MMD-learned noise
Is = filter(I -> image[I,1] > 0, CartesianIndices(size(image)[1:3]));

@time model_mu_std = cvae_model(permutedims(image[Is,:] ./ 1e6); nsamples = 100, stddev = true); #TODO
model_thetas, model_stds = model_mu_std[1:end÷2, ..], model_mu_std[end÷2+1:end, ..];
model_thetas_image = fill(NaN, size(image)[1:3]..., size(model_thetas,1));
model_thetas_image[Is, :] .= permutedims(model_thetas);
# @time model_thetas = image[Is,:] ./ 1e6 |> permutedims |> copy |> x -> [cvae_model(x; nsamples = 1, stddev = false) for _ in 1:25] |> x -> cat(x...; dims = 3)
# model_thetas_image = fill(NaN, size(image)[1:3]..., size(model_thetas,1));
# model_thetas_image[Is, :] .= permutedims(mean(model_thetas; dims = 3)[:,:,1]);

function plot_mmd_mwi_maps(;zslice::Int = 25, savefolder = string(zslice))
    # Make maps
    CUTOFF = 30.0 #1000 * t2partopts.SPWin[2] # cutoff in seconds
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

    # Save maps
    FNAME(filename) = joinpath(mkpath(savefolder), filename)
    MS  = (x,pos) -> "$x ms"
    DEG = (x,pos) -> "$x" * L"\degree"

    Is_zslice = filter(I -> I[3] == zslice, Is)
    XRange, YRange = extrema((x->x[1]).(Is_zslice)), extrema((x->x[2]).(Is_zslice))
    Islice = (XRange[1]:XRange[2], YRange[1]:YRange[2], zslice)

    pyheatmap(permutedims(mwf[Islice...]) |> img -> (x -> clamp(x, 0.0, Inf)).(img);
        clim = (0,0.3), filename = FNAME("cvae-mwf"))# title = "MWF [a.u.]")
    pyheatmap(permutedims(t2parts["sfr"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, x)).(img);
        clim = (0,0.3), filename = FNAME("decaes-mwf"))# title = "MWF [a.u.]")

    pyheatmap(permutedims(ggm[Islice...]) |> img -> (x -> clamp(x, 0.0, Inf)).(img);
        clim = (0,250), filename = FNAME("cvae-ggm"), formatter = MS)# title = "Geometric Mean T2 [ms]")
    pyheatmap(permutedims(t2maps["ggm"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, 1000x)).(img);
        clim = (0,250), filename = FNAME("decaes-ggm"), formatter = MS)# title = "Geometric Mean T2 [ms]")

    pyheatmap(permutedims(log10.(ggm[Islice...]));
        clim = (1,3), filename = FNAME("cvae-log10ggm"), formatter = MS)# title = "Log10 Geometric Mean T2 [ms]")
    pyheatmap(permutedims(log10.(1000 .* t2maps["ggm"][Islice...]));
        clim = (1,3), filename = FNAME("decaes-log10ggm"), formatter = MS)# title = "Log10 Geometric Mean T2 [ms]")

    pyheatmap(permutedims(t2maps["alpha"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, x)).(img);
        clim = (120,180), filename = FNAME("decaes-flipangle"), formatter = DEG)# title = L"\alpha")
    pyheatmap(permutedims(t2parts["sgm"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, 1000x)).(img);
        clim = 1000 .* t2partopts.SPWin, filename = FNAME("decaes-sgm"), formatter = MS)# title = "SGM [ms]")
    pyheatmap(permutedims(t2parts["mgm"][Islice...]) |> img -> (x -> ifelse(isnan(x), NaN, 1000x)).(img);
        clim = 1000 .* t2partopts.MPWin, filename = FNAME("decaes-mgm"), formatter = MS)# title = "MGM [ms]")

    pyheatmap(permutedims(model_thetas_image[Islice...,1]) |> img -> (x -> acosd(clamp(x, -1.0, 1.0))).(img);
        clim = (120,180), filename = FNAME("cvae-flipangle"), formatter = DEG)# title = L"\alpha")
    pyheatmap(permutedims(model_thetas_image[Islice...,2]) |> img -> (x -> clamp(x, 8.0, 1000.0)).(img);
        clim = (0,100), filename = FNAME("cvae-T2short"), formatter = MS)# title = L"T_{2,short}")
    pyheatmap(permutedims(model_thetas_image[Islice...,3]) |> img -> (x -> clamp(x, 8.0, 1000.0)).(img);
        clim = (0,1000), filename = FNAME("cvae-T2long"), formatter = MS)# title = L"T_{2,long}")
    pyheatmap(permutedims(model_thetas_image[Islice...,4]) |> img -> (x -> clamp(x, 0.0, Inf)).(img);
        filename = FNAME("cvae-Ashort"))# title = L"A_{short}")
    pyheatmap(permutedims(model_thetas_image[Islice...,5]) |> img -> (x -> clamp(x, 0.0, Inf)).(img);
        filename = FNAME("cvae-Along"))# title = L"A_{long}")

    nothing
end
foreach(z -> plot_mmd_mwi_maps(zslice = z), [21,25])

####
#### CVAE T2 histograms
####

function make_binned(X, Y; binsize::Int)
    _unzip(y) = ((x->x[1]).(y), (x->x[2]).(y))
    X_sorted, Y_sorted = sort(collect(zip(X, Y)); by = first) |> _unzip
    X_binned, Y_binned = map(partition(1:length(X), binsize)) do Is
        mean(X_sorted[Is]), mean(Y_sorted[Is])
    end |> _unzip
    return X_binned, Y_binned
end

function cvae_t2_dist()
    pyplot(size = (800,600))
    local T2 = [model_thetas[2,:]; model_thetas[3,:]]
    local A  = [model_thetas[4,:]; model_thetas[5,:]]
    local common_args = Dict{Symbol,Any}(:ticks => :minor, :titlefontsize => 18, :tickfontsize => 14, :legendfontsize => 14)
    local p = plot(make_binned(T2, A; binsize = 500);
        label = "T2 Distribution", ylabel = "T2 Amplitude [a.u.]", xlabel = "T2 [ms]",
        xscale = :log10, xlim = (8, 1000), xticks = 10 .^ (1:0.5:3), xformatter = x -> string(round(x; digits = 1)),
    );
    map((".png", ".pdf")) do ext # unreasonably large files: ".eps", ".svg"
        savefig(p, "cvae_t2_dist" * ext)
    end
    nothing
end
cvae_t2_dist()

# (x,y) -> "\$$(x)^\\degree\$")

nothing
