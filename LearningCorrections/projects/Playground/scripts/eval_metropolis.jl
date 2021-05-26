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
    cvae_dict = OrderedDict{String, Any}(
        "0"     => load_cvae(phys, only(readdir(Glob.glob"*1913r95s/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [0]
        "03"    => load_cvae(phys, only(readdir(Glob.glob"*2wx65cke/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [0,3]
        "012"   => load_cvae(phys, only(readdir(Glob.glob"*g5wfhjnk/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [0,1,2]
        "2112"  => load_cvae(phys, only(readdir(Glob.glob"*3ao70r4e/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [-2,-1,1,2]
        "21012" => load_cvae(phys, only(readdir(Glob.glob"*1wi9jz3m/files", lib.projectdir("wandb")))) |> lib.gpu, # train_indices = [-2,-1,0,1,2]
    )
end

#### Compute metrics

function mcmc_cdf_distances(θ₁, θ₂)
    dists    = lib.cdf_distance((1, 2), θ₁, θ₂) # array of tuples of ℓ₁ and ℓ₂ distances between the empirical cdf's
    ℓ₁       = (x->x[1]).(dists)
    ℓ₂       = (x->x[2]).(dists)
    return (; ℓ₁, ℓ₂)
end

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

function point_estimates(phys, cvae_pairs; dataset::Symbol = :val)

    df = DataFrame()
    μσ = x -> mean_and_std(vec(x))

    for (img_idx, img) in enumerate(phys.images)
        @info "Image $(img_idx)"

        θ_100_dict = OrderedDict{String,Any}(
            "True"  => lib.recursive_try_get(img.meta, [:true_labels, dataset, :theta]),
            "MCMC"  => lib.recursive_try_get(img.meta, [:mcmc_labels_100, dataset, :theta]),
        )

        mcmc_3000_burn_in = 500
        mcmc_3000_columns = lib.recursive_try_get(img.meta, [:mcmc_labels_3000, dataset, :columns])
        θ_3000_dict = OrderedDict{String,Any}(
            "True" => (mcmc_3000_columns === missing || θ_100_dict["True"] === missing) ? missing : θ_100_dict["True"][:, mcmc_3000_columns],
            "MCMC" => lib.recursive_try_get(img.meta, [:mcmc_labels_3000, dataset, :theta]) |> θ -> θ === missing ? missing : θ[:, :, mcmc_3000_burn_in + 1 : end],
        )

        for (cvae_name, cvae) in cvae_pairs
            print("Populate CVAE: $(cvae_name) (100 samples) ")
            Y_100 = img.partitions[dataset]
            @time θ_100_dict["CVAE-$(cvae_name)"] = sample_cvae(phys, cvae, Y_100; num_samples = 100, gpu_batch_size = 2^15)
            if mcmc_3000_columns !== missing
                print("Populate CVAE: $(cvae_name) (3000 samples) ")
                Y_3000 = Y_100[:, mcmc_3000_columns]
                @time θ_3000_dict["CVAE-$(cvae_name)"] = sample_cvae(phys, cvae, Y_3000; num_samples = 3000 - mcmc_3000_burn_in, gpu_batch_size = 2^15)
            end
        end

        print("Compute metrics ")
        @time for (θ_nsamples, θ_dict) in [100 => θ_100_dict, 3000 => θ_3000_dict], (label_set, θ) in θ_dict
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
                metrics = OrderedDict{AbstractString, Any}()
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

#### Temporary

function copy_mat_files()
    in_dir = "/home/jdoucette/Documents/projects/2021-05-24_NeurIPS2021_Initial_Submission/mcmc_3000_samples"
    out_dir = "/home/jdoucette/Documents/code/BlochTorreyExperiments-shared/LearningCorrections/projects/Playground/data/images/2019-09-22_56echo_7msTE_CPMG/julia-mcmc-biexpepg"
    mat_files = readdir(Glob.glob"image-2_*.mat", in_dir)
    mat_out = DECAES.MAT.matread(mat_files[1])
    for i in 2:length(mat_files)
        mat = DECAES.MAT.matread(mat_files[i])
        for k in keys(mat_out)
            append!(mat_out[k], mat[k])
        end
    end
    DECAES.MAT.matwrite(joinpath(out_dir, "mcmc_theta_samples_3000.mat"), mat_out)
    mat_out
end

function corr_coeff(x,y)
    x̄ = mean(x)
    ȳ = mean(y)
    sum(@. (x - x̄) * (y - ȳ))^2  / (sum(@. (x - x̄)^2) * sum(@. (y - ȳ)^2)), Statistics.cor(x, y)^2
end

function test_mcmc_3000(cvae, img, dataset)
    local θ_cols_ = img.meta[:mcmc_labels_3000][dataset][:columns]
    local θ_mcmc_ = img.meta[:mcmc_labels_3000][dataset][:theta][:, :, end-2499:end]
    local Y_mcmc_ = img.partitions[dataset][:, θ_cols_] |> lib.gpu
    local θ′_sampler_ = lib.θZposterior_sampler(cvae, Y_mcmc_)
    local θ′_samples_ = zeros(Float32, size(θ_mcmc_)...)
    metrics = Dict{Any,Any}()
    for i in 1:size(θ′_samples_, 3)
        local θ′, _ = θ′_sampler_()
        θ′_samples_[:,:,i] .= lib.cpu32(θ′)
    end
    local θ_dists_ = mcmc_cdf_distances(θ_mcmc_, θ′_samples_)
    for (i, lab) in enumerate(lib.θasciilabels(phys)), (ℓ_name, ℓ) in pairs(θ_dists_)
        metrics[Symbol("$(lab)_$(ℓ_name)_CVAE")] = ℓ[i]
    end
    return metrics
end

#=

@time sample_cvae(phys, cvae_dict["2112"], phys.images[1].partitions[:val]; num_samples = 1, gpu_batch_size = 2^15)

=#
