using Working
pyplot(size=(800,600))

ismrm_dir = "/home/jdoucette/Documents/code/MWI-MMD-CVAE/ISMRM-2021/"
image_infos = [
    (TE =  8e-3, refcon = 180.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Example_48echo_8msTE/data-in/ORIENTATION_B0_08_WIP_MWF_CPMG_CS_AXIAL_5_1.masked-image.nii.gz"),
    (TE =  7e-3, refcon = 180.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Example_56echo_7msTE_CPMG/data-in/MW_TRAINING_001_WIP_CPMG56_CS_half_2_1.masked-image.mat"),
    (TE = 10e-3, refcon = 150.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Siemens_t2_mapping_se_mc_32echo_10msTE_120degRefCon/0008_t2_mapping_se_mc_ms_32TE_single_slice_1E_masked_image_rotated.mat"),
    (TE = 10e-3, refcon = 180.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Siemens_t2_mapping_se_mc_32echo_10msTE_120degRefCon/0008_t2_mapping_se_mc_ms_32TE_single_slice_1E_masked_image_rotated.mat"), # same image with deliberately wrong refcon
]

#### MLE inference

for model_path in readdir(joinpath(ismrm_dir, "models"); join = true)
    @info basename(model_path)
    JL_CHECKPOINT_FOLDER[] = model_path
    global settings = make_settings()
    global phys = make_physics(settings; image_infos)
    global models, derived = make_models!(phys, settings, load_checkpoint())

    for (i,img) in enumerate(phys.images)
        for data_source in (:image, :simulated), data_subset in (:mask,)
            @info "image $i / $(length(phys.images)) -- data source: $data_source -- data_subset: $data_subset"
            @time global init, res = mle_biexp_epg(
                phys, models, derived, img;
                data_source, data_subset,
                verbose = true, checkpoint = false, dryrun = false, batch_size = 12*1024,
                savefolder = joinpath(ismrm_dir, "analysis", basename(model_path), "mle_inference_image_$i"),
            )
        end
    end
end

for model_path in readdir(Glob.glob"ignite-cvae*", joinpath(ismrm_dir, "analysis"))
    @info basename(model_path)
    for results_file in readdir(Glob.glob"mle_inference_image_*/*-results-final-*.mat", model_path)
        @info basename(results_file)
        res = DECAES.MAT.matread(results_file)
        @show mean(res["loss"])
    end
end

#### Heat maps etc.

for model_path in readdir(joinpath(ismrm_dir, "models"); join = true)
    @info basename(model_path)
    JL_CHECKPOINT_FOLDER[] = model_path
    global settings = make_settings()
    global phys = make_physics(settings; image_infos)
    global models, derived = make_models!(phys, settings, load_checkpoint())

    slices = [24:24, 162:162, 1:1, 1:1]
    slicedim = [3, 2, 3, 3]
    savetypes = [".png", ".pdf"]

    for (i,img) in collect(enumerate(phys.images))
        @time global peak_settings, peak_cvae, peak_decaes = peak_separation(
            phys, models, derived, img;
            savefolder = joinpath(ismrm_dir, "analysis", basename(model_path), "eval_image_$i"),
            savetypes = savetypes,
        )
        @time eval_mri_model(
            phys, models, derived, img;
            slices = slices[i],
            slicedim = slicedim[i],
            naverage = 10,
            savefolder = joinpath(ismrm_dir, "analysis", basename(model_path), "eval_image_$i"),
            savetypes = savetypes,
            mle_image_path = joinpath(ismrm_dir, "analysis", basename(model_path), "mle_inference_image_$i"),
            mle_sim_path = joinpath(ismrm_dir, "analysis", basename(model_path), "mle_inference_image_$i"),
            batch_size = 12*1024,
            force_decaes = false,
            force_histograms = false,
            posterior_mode = :maxlikelihood,
            quiet = false,
            dataset = :val, # :val or (for final model comparison) :test
        )
    end
end
