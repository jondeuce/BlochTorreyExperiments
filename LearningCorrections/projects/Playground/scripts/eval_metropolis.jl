using DrWatson: @quickactivate
@quickactivate "Playground"
using Playground
lib.initenv()

#### Load physics model/images

if !@isdefined(phys)
    phys = lib.load_epgmodel_physics()

    lib.initialize!(phys; seed = 0, image_folders = [
        "2019-10-28_48echo_8msTE_CPMG",
        "2019-09-22_56echo_7msTE_CPMG",
        "2021-05-07_NeurIPS2021_64echo_10msTE_MockBiexpEPG_CPMG",
    ])

    # # Estimate mle labels using maximum likelihood estimation from pretrained cvae
    # lib.compute_mle_labels!(phys, derived["cvae"]; force_recompute = true)
    # lib.verify_mle_labels(phys)

    # # Estimate mle labels using random initial guess form the prior
    # lib.compute_mle_labels!(phys; force_recompute = true)
    # lib.verify_mle_labels(phys)

    # Load true labels
    lib.load_true_labels!(phys; force_reload = true)
    lib.verify_true_labels(phys)

    # Load mcmc labels
    lib.load_mcmc_labels!(phys; force_reload = true)
    lib.verify_mcmc_labels(phys)
end

#### Load physics model/images

function load_cvae(phys, checkpoint_folder)
    lib.set_checkpointdir!(checkpoint_folder)
    models = lib.load_checkpoint("current-models.jld2")
    cvae = lib.derived_cvae(phys, models["enc1"], models["enc2"], models["dec"]; nlatent = 0, zdim = 12, posterior = "TruncatedGaussian")
    return cvae
end

if !@isdefined(cvae_dict)
    # Load CVAEs, keyed by the data they were trained on:
    #   0: simulated data generated from the prior on the fly
    #   1: real dataset #1 (48 echo CPMG, 8 ms TE)
    #  -1: simulated data generated on the fly from CVAE posterior samples over dataset #1
    #   2: real dataset #2 (56 echo CPMG, 7 ms TE)
    #  -2: simulated data generated on the fly from CVAE posterior samples over dataset #2
    #   3: fixed precomputed simulated data generated from the prior
    cvae_dict = OrderedDict{Any,Any}(
        "0"     => load_cvae(phys, only(readdir(Glob.glob"*1913r95s/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [0]
        "03"    => load_cvae(phys, only(readdir(Glob.glob"*2wx65cke/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [0,3]
        "012"   => load_cvae(phys, only(readdir(Glob.glob"*g5wfhjnk/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [0,1,2]
        "2112"  => load_cvae(phys, only(readdir(Glob.glob"*3ao70r4e/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [-2,-1,1,2]
        "21012" => load_cvae(phys, only(readdir(Glob.glob"*1wi9jz3m/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [-2,-1,0,1,2]
    )
end

#### Eval CVAE samples

function sample_cvae(phys, cvae, Y; num_samples::Int, gpu_batch_size::Int)
    J_ranges   = collect(Iterators.partition(1:size(Y, 2), gpu_batch_size))
    θ′_samples = zeros(Float32, lib.ntheta(phys), size(Y,2), num_samples)
    for (j, (Y_gpu,)) in enumerate(CUDA.CuIterator((Y[:, J_range],) for J_range in J_ranges))
        θ′_sampler = lib.θZposterior_sampler(cvae, Y_gpu)
        for i in 1:size(θ′_samples, 3)
            θ′, _ = θ′_sampler()
            θ′_samples[:, J_ranges[j], i] .= lib.cpu32(θ′)
        end
    end
    return θ′_samples
end

function populate_cvae_samples(phys, cvae_dict; dataset::Symbol)
    θ_dicts = OrderedDict{Any,Any}()

    for (img_idx, img) in enumerate(phys.images)
        @info "Image $(img_idx)"
        θ_dicts[img_idx] = OrderedDict{Any,Any}()

        @info "Dataset $(dataset)"
        θ_dicts[img_idx][dataset] = OrderedDict{Any,Any}()

        θ_dicts[img_idx][dataset][100] = OrderedDict{Any,Any}(
            "True"  => lib.recursive_try_get(img.meta, [:true_labels, dataset, :theta]),
            "MCMC"  => lib.recursive_try_get(img.meta, [:mcmc_labels_100, dataset, :theta]),
        )

        mcmc_3000_burn_in = 500
        mcmc_3000_columns = lib.recursive_try_get(img.meta, [:mcmc_labels_3000, dataset, :columns])
        θ_dicts[img_idx][dataset][3000] = OrderedDict{Any,Any}(
            "True" => (mcmc_3000_columns === missing || θ_dicts[img_idx][dataset][100]["True"] === missing) ? missing : θ_dicts[img_idx][dataset][100]["True"][:, mcmc_3000_columns],
            "MCMC" => lib.recursive_try_get(img.meta, [:mcmc_labels_3000, dataset, :theta]) |> θ -> θ === missing ? missing : θ[:, :, mcmc_3000_burn_in + 1 : end],
        )

        for (cvae_name, cvae) in cvae_dict
            print("Populate CVAE: $(cvae_name) (100 samples) ")
            Y_100 = img.partitions[dataset]
            @time θ_dicts[img_idx][dataset][100]["CVAE-$(cvae_name)"] = sample_cvae(phys, cvae, Y_100; num_samples = 100, gpu_batch_size = 2^15)
            if mcmc_3000_columns !== missing
                print("Populate CVAE: $(cvae_name) (3000 samples) ")
                Y_3000 = Y_100[:, mcmc_3000_columns]
                @time θ_dicts[img_idx][dataset][3000]["CVAE-$(cvae_name)"] = sample_cvae(phys, cvae, Y_3000; num_samples = 3000 - mcmc_3000_burn_in, gpu_batch_size = 2^15)
            end
        end
    end

    return θ_dicts
end

if !@isdefined(θ_dicts)
    θ_dicts = populate_cvae_samples(phys, cvae_dict; dataset = :val)
end

#### Compute metrics

function mcmc_cdf_distances(θ₁, θ₂)
    dists    = lib.cdf_distance((1, 2), θ₁, θ₂) # array of tuples of ℓ₁ and ℓ₂ distances between the empirical cdf's
    ℓ₁       = (x->x[1]).(dists)
    ℓ₂       = (x->x[2]).(dists)
    return (; ℓ₁, ℓ₂)
end

function point_estimates(phys, θ_dicts; dataset::Symbol)

    df = DataFrame()
    μσ = x -> mean_and_std(vec(x))

    for (img_idx, img) in enumerate(phys.images)
        @info "Image $(img_idx)"

        print("Compute metrics ")
        @time for (θ_nsamples, θ_dict) in θ_dicts[img_idx][dataset], (label_set, θ) in θ_dict
            θ === missing && continue
            label_set == "True" && continue
            θ_widths = lib.θupper(phys) .- lib.θlower(phys)

            # Hoist outside loop to compute all at once
            if label_set !== "MCMC"
                @unpack ℓ₁, ℓ₂ = mcmc_cdf_distances(θ, θ_dict["MCMC"])
                ℓ₁ = ℓ₁ ./ θ_widths
                ℓ₂ = ℓ₂ ./ sqrt.(θ_widths)
            end

            # Push metrics to df
            for (i, lab) in enumerate(lib.θlabels(phys))
                metrics = OrderedDict{Any,Any}()
                metrics["Dataset"]    = img_idx
                metrics["Label set"]  = label_set
                metrics["Samples"]    = θ_nsamples
                metrics["Param"]      = lab
                if θ_dict["True"] !== missing
                    metrics["Mean"]   = μσ(100 .* abs.(mean(θ[i:i, :, :]; dims = 3) .- θ_dict["True"][i:i, :]) ./ θ_widths[i])
                    metrics["Median"] = μσ(100 .* abs.(lib.fast_median3(θ[i:i, :, :]) .- θ_dict["True"][i:i, :]) ./ θ_widths[i])
                end
                if label_set !== "MCMC"
                    @assert 0 <= minimum(ℓ₁[i:i, :]) <= maximum(ℓ₁[i:i, :]) <= 1
                    @assert 0 <= minimum(ℓ₂[i:i, :]) <= maximum(ℓ₂[i:i, :]) <= 1
                    metrics["Wasserstein"] = μσ(ℓ₁[i:i, :]) # scale to range [0,1]
                    metrics["Energy"]      = μσ(ℓ₂[i:i, :]) # scale to range [0,1]
                end
                push!(df, metrics; cols = :union)
            end
        end
    end

    for (group_key, group) in pairs(DataFrames.groupby(df, ["Dataset", "Param"]))
        @info group_key
        show(group; allrows = true)
        println("")
    end

    return df
end

#### Make plots

function default_plot_settings()
    
    # global rcParams = PyDict(PyPlot.matplotlib."rcParams")
    # rcParams["text.usetex"] = false # use matplotlib internal tex rendering
    # rcParams["mathtext.fontset"] = "cm" # "stix"
    # rcParams["font.family"] = "cmu serif" # "STIXGeneral"
    # rcParams["font.size"] = 12
    # rcParams["axes.titlesize"] = "medium"
    # rcParams["axes.labelsize"] = "medium"
    # rcParams["xtick.labelsize"] = "small"
    # rcParams["ytick.labelsize"] = "small"
    # rcParams["legend.fontsize"] = "small"
    
    Plots.pyplot()
    Plots.default()
    Plots.default(
        fontfamily = "serif",
        size = (1600, 900),
        titlefonthalign = :left, # for figure numbering
        titlefont = 20,
        guidefont = 20,
        tickfont = 14,
        legendfontsize = 12,
        grid = false,
        yminorgrid = false,
        xminorgrid = false,
    )
end

function error_histograms(phys, θ_dicts; dataset::Symbol)
    # MCMC data using only 100 samples
    θ_100_true = θ_dicts[3][dataset][100]["True"]
    θ_100_mcmc = θ_dicts[3][dataset][100]["MCMC"]
    θ_100_cvae = θ_dicts[3][dataset][100]["CVAE-03"]

    # MCMC data using 3000 samples (following 500 burn-in iterations)
    θ_3000_true = θ_dicts[3][dataset][3000]["True"]
    θ_3000_mcmc = θ_dicts[3][dataset][3000]["MCMC"][:, :, 501:end]
    θ_3000_cvae = θ_dicts[3][dataset][3000]["CVAE-03"][:, :, 501:end]

    ps = Any[]
    θ_widths = lib.θupper(phys) .- lib.θlower(phys)
    for (i, lab) in enumerate(lib.θlabels(phys))
        p = plot(; legend = :bottomleft)
        stephist!(p, (vec(θ_100_true[i,:] .- mean(θ_100_mcmc[i,:,:]; dims = 3))) ./ θ_widths[i], yscale = :log10, lab = "MCMC-100", line = (:solid, 1.5, :red))
        stephist!(p, (vec(θ_3000_true[i,:] .- mean(θ_3000_mcmc[i,:,:]; dims = 3))) ./ θ_widths[i], yscale = :log10, lab = "MCMC-3000", line = (:solid, 1.5, :green))
        stephist!(p, (vec(θ_100_true[i,:] .- mean(θ_100_cvae[i,:,:]; dims = 3))) ./ θ_widths[i], yscale = :log10, lab = "Metropolis-CVAE", line = (:solid, 1.5, :blue))
        stripped_lab = lab.s[2:end-1]
        xlabel!(p, L"\widehat{%$(stripped_lab)} - %$(stripped_lab)")
        push!(ps, p)
    end

    return ps
end

function pp_plot(phys, θ_dicts; dataset::Symbol)
    # MCMC data using only 100 samples
    θ_100_true = θ_dicts[3][dataset][100]["True"]
    θ_100_mcmc = θ_dicts[3][dataset][100]["MCMC"]
    θ_100_cvae = θ_dicts[3][dataset][100]["CVAE-03"]

    # MCMC data using 3000 samples (following 500 burn-in iterations)
    θ_3000_true = θ_dicts[3][dataset][3000]["True"]
    θ_3000_mcmc = θ_dicts[3][dataset][3000]["MCMC"][:, :, 501:end]
    θ_3000_cvae = θ_dicts[3][dataset][3000]["CVAE-03"][:, :, 501:end]

    empirical_cdf = x -> x |> sort |> cumsum |> y -> y ./ y[end]
    p_mcmc_100 = dropdims(mean(θ_100_mcmc .> θ_100_true; dims = 3); dims = 3) # p-statistic: "how often is θ > true θ?"
    p_mcmc_3000 = dropdims(mean(θ_3000_mcmc .> θ_3000_true; dims = 3); dims = 3) # p-statistic: "how often is θ > true θ?"
    p_cvae = dropdims(mean(θ_3000_cvae .> θ_3000_true; dims = 3); dims = 3) # p-statistic: "how often is θ > true θ?"
    p = plot()
    for (i, lab) in enumerate(lib.θlabels(phys))
        plot!(p, sort(p_mcmc_100[i,:]), empirical_cdf(p_mcmc_100[i,:]); seriestype = :steppost, line = (:solid, 1.5, :red), label = :none)
        plot!(p, sort(p_mcmc_3000[i,:]), empirical_cdf(p_mcmc_3000[i,:]); seriestype = :steppost, line = (:solid, 1.5, :green), label = :none)
        plot!(p, sort(p_cvae[i,:]), empirical_cdf(p_cvae[i,:]); seriestype = :steppost, line = (:solid, 1.5, :blue), label = :none)
    end
    plot!(p, [0, 1], [0, 1]; l = (3, :black, :dash), label = "Ideal", legend = :topleft)
    xlabel!(p, L"$p$-$p$ plot")

    return p
end

function wasserstein_histograms(phys, θ_dicts; dataset::Symbol)
    ps = Any[plot(; legend = :topright) for _ in 1:length(lib.θlabels(phys))]

    for img_idx = 1:2
        # MCMC data using 3000 samples (following 500 burn-in iterations)
        θ_3000_mcmc   = θ_dicts[img_idx][dataset][3000]["MCMC"][:, :, 501:end]
        θ_3000_super  = θ_dicts[img_idx][dataset][3000]["CVAE-0"][:, :, 501:end]
        θ_3000_hybrid = θ_dicts[img_idx][dataset][3000]["CVAE-21012"][:, :, 501:end]

        # MCMC data using only 100 samples
        θ_100_mcmc    = θ_dicts[img_idx][dataset][100]["MCMC"][:, sample(1:end, size(θ_3000_mcmc, 2); replace = false), :]
        θ_100_super   = θ_dicts[img_idx][dataset][100]["CVAE-0"][:, sample(1:end, size(θ_3000_hybrid, 2); replace = false), :]
        θ_100_hybrid  = θ_dicts[img_idx][dataset][100]["CVAE-21012"][:, sample(1:end, size(θ_3000_hybrid, 2); replace = false), :]

        θ_labels = lib.θlabels(phys)
        θ_widths = lib.θupper(phys) .- lib.θlower(phys)

        @unpack ℓ₁, ℓ₂ = mcmc_cdf_distances(θ_100_mcmc, θ_100_super)
        ℓ₁_100_super = ℓ₁ ./ θ_widths

        @unpack ℓ₁, ℓ₂ = mcmc_cdf_distances(θ_100_mcmc, θ_100_hybrid)
        ℓ₁_100_hybrid = ℓ₁ ./ θ_widths

        @unpack ℓ₁, ℓ₂ = mcmc_cdf_distances(θ_3000_mcmc, θ_3000_super)
        ℓ₁_3000_super = ℓ₁ ./ θ_widths

        @unpack ℓ₁, ℓ₂ = mcmc_cdf_distances(θ_3000_mcmc, θ_3000_hybrid)
        ℓ₁_3000_hybrid = ℓ₁ ./ θ_widths

        xlims = Dict(1 => (0, 1), 2 => (0, 1), 3 => (0, 1), 4 => (0, 0.6), 5 => (0, 1), 6 => (0, 0.4), 7 => (0, 0.03))
        for (i, lab) in enumerate(lib.θlabels(phys))
            p = ps[i]
            stephist!(p, vec(ℓ₁_100_super[i,:]), yscale = :log10, lab = ifelse(img_idx == 1, "CVAE (MCMC-100)", :none), line = (:dash, 1.5, :red), xlims = xlims[i])
            stephist!(p, vec(ℓ₁_3000_super[i,:]), yscale = :log10, lab = ifelse(img_idx == 1, "CVAE (MCMC-3000)", :none), line = (:dash, 1.5, :green), xlims = xlims[i])
            stephist!(p, vec(ℓ₁_100_hybrid[i,:]), yscale = :log10, lab = ifelse(img_idx == 1, "Metropolis-CVAE (MCMC-100)", :none), line = (:solid, 1.5, :red), xlims = xlims[i])
            stephist!(p, vec(ℓ₁_3000_hybrid[i,:]), yscale = :log10, lab = ifelse(img_idx == 1,  "Metropolis-CVAE (MCMC-3000)", :none), line = (:solid, 1.5, :green), xlims = xlims[i])
            stripped_lab = lab.s[2:end-1]
            xlabel!(p, L"W_1(\tilde{p}(\widehat{%$(stripped_lab)}), \, \tilde{p}(%$(stripped_lab)))")
        end
    end

    return ps
end

function qq_plot(phys, θ_dicts; dataset::Symbol)
    img_idx = 1 #TODO

    # MCMC data using 3000 samples (following 500 burn-in iterations)
    θ_3000_mcmc = θ_dicts[img_idx][dataset][3000]["MCMC"][:, :, 501:end]
    θ_3000_cvae = θ_dicts[img_idx][dataset][3000]["CVAE-21012"][:, :, 501:end]

    # MCMC data using only 100 samples
    θ_100_mcmc = θ_dicts[img_idx][dataset][100]["MCMC"][:, sample(1:end, size(θ_3000_mcmc, 2); replace = false), :]
    θ_100_cvae = θ_dicts[img_idx][dataset][100]["CVAE-21012"][:, sample(1:end, size(θ_3000_cvae, 2); replace = false), :]

    θ_labels = lib.θlabels(phys)
    θ_lower  = lib.θlower(phys)
    θ_widths = lib.θupper(phys) .- lib.θlower(phys)

    quantile_vec = x -> quantile(vec(x), 0.01:0.01:0.99)
    p = plot()
    for (i, lab) in enumerate(lib.θlabels(phys))
        # nrm = x -> (x .- θ_lower[i]) ./ θ_widths[i]
        nrm = x -> (x .- minimum(x)) ./ (maximum(x) .- minimum(x))
        q_100_mcmc  = nrm(quantile_vec(θ_100_mcmc[i,:,:]))
        q_3000_mcmc = nrm(quantile_vec(θ_3000_mcmc[i,:,:]))
        q_3000_cvae = nrm(quantile_vec(θ_3000_cvae[i,:,:]))
        plot!(p, q_3000_cvae, q_100_mcmc; seriestype = :steppost, line = (:solid, 1.5, :red), label = :none)
        plot!(p, q_3000_cvae, q_3000_mcmc; seriestype = :steppost, line = (:solid, 1.5, :green), label = :none)
    end
    plot!(p, [0, 1], [0, 1]; l = (3, :black, :dash), label = "Ideal", legend = :topleft)
    xlabel!(p, L"$q$-$q$ plot")

    return p
end

function figure1(phys, θ_dicts; dataset::Symbol)
    default_plot_settings()
    p1_to_p7 = error_histograms(phys, θ_dicts; dataset)
    p8 = pp_plot(phys, θ_dicts; dataset)
    ps = [p1_to_p7..., p8]
    for (i,p) in enumerate(ps)
        title!(p, string.('A':'Z')[i])
    end
    p = plot(ps...; layout = (2,4))
    display(p)
    savefig(p, "figure1.eps")
    savefig(p, "figure1.svg")
    savefig(p, "figure1.pdf")
end

function figure2(phys, θ_dicts; dataset::Symbol)
    default_plot_settings()
    p1_to_p7 = wasserstein_histograms(phys, θ_dicts; dataset)
    p8 = qq_plot(phys, θ_dicts; dataset)
    ps = [p1_to_p7..., p8]
    for (i,p) in enumerate(ps)
        title!(p, string.('A':'Z')[i])
    end
    p = plot(ps...; layout = (2,4))
    display(p)
    savefig(p, "figure2.eps")
    savefig(p, "figure2.svg")
    savefig(p, "figure2.pdf")
end

#### Temporary

function copy_mat_files(; dataset::Symbol)
    in_dir = "/home/jdoucette/Documents/projects/2021-05-24_NeurIPS2021_Initial_Submission/mcmc_3000_samples/image-3"
    out_dir = "/home/jdoucette/Documents/code/BlochTorreyExperiments-shared/LearningCorrections/projects/Playground/data/images/2019-09-22_56echo_7msTE_CPMG/julia-mcmc-biexpepg"
    mat_files = readdir(Glob.GlobMatch("*_dataset-$(dataset)_*.mat"), in_dir)
    mat_out = DECAES.MAT.matread(mat_files[1])
    for i in 2:length(mat_files)
        mat = DECAES.MAT.matread(mat_files[i])
        for k in keys(mat_out)
            append!(mat_out[k], mat[k])
        end
    end

    all_inds = CartesianIndex.(mat_out["image_x"], mat_out["image_y"], mat_out["image_z"])
    seen_inds = Set{eltype(all_inds)}()
    is_unique_cols = zeros(Bool, length(all_inds))
    for i in 1:3000:length(all_inds)
        I = all_inds[i]
        is_unique_cols[i .+ (0:2999)] .= I ∉ seen_inds
        push!(seen_inds, I)
    end

    for k in keys(mat_out)
        mat_out[k] = mat_out[k][is_unique_cols][1:5000*3000]
    end
    @show length(unique(CartesianIndex.(mat_out["image_x"], mat_out["image_y"], mat_out["image_z"])))

    # DECAES.MAT.matwrite(joinpath(out_dir, "mcmc_theta_samples_3000.mat"), mat_out)
    mat_out
end

function copy_all_mat_files()
    mat_test = copy_mat_files(dataset = :test)
    mat_val = copy_mat_files(dataset = :val)
    mat_out = DECAES.MAT.matread("/home/jdoucette/Documents/projects/2021-05-24_NeurIPS2021_Initial_Submission/mcmc_3000_samples/image-3_dataset-train_0000001-to-0005000.mat")
    for k in keys(mat_out)
        append!(mat_out[k], mat_val[k])
        append!(mat_out[k], mat_test[k])
    end
    @show length(unique(CartesianIndex.(mat_out["image_x"], mat_out["image_y"], mat_out["image_z"])))

    out_dir = "/home/jdoucette/Documents/code/BlochTorreyExperiments-shared/LearningCorrections/projects/Playground/data/images/2021-05-07_NeurIPS2021_64echo_10msTE_MockBiexpEPG_CPMG/julia-mcmc-biexpepg"
    DECAES.MAT.matwrite(joinpath(out_dir, "mcmc_theta_samples_3000.mat"), mat_out)
    mat_out
end

function test_annot()
    ps = map(1:8) do i
        p = plot()
        plot!(p, 100 .* rand(10,10))
        annotate!(p, [(0, 1, "A")])
        i==1 && for (k,v) in Plots.getattr(p)
            display(k => v)
        end
        p
    end
    plot(ps...)
end


#=

@time sample_cvae(phys, cvae_dict["2112"], phys.images[1].partitions[:val]; num_samples = 1, gpu_batch_size = 2^15)

=#
