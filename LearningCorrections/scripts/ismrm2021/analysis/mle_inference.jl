Revise.includet(joinpath(@__DIR__, "..", "setup.jl"))

ismrm_dir = "/home/jdoucette/Documents/code/MWI-MMD-CVAE/ISMRM-2021/"
image_infos = [
    (TE =  8e-3, refcon = 180.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Example_48echo_8msTE/data-in/ORIENTATION_B0_08_WIP_MWF_CPMG_CS_AXIAL_5_1.masked-image.nii.gz"),
    (TE =  7e-3, refcon = 180.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Example_56echo_7msTE_CPMG/data-in/MW_TRAINING_001_WIP_CPMG56_CS_half_2_1.masked-image.mat"),
    (TE = 10e-3, refcon = 150.0, path = "/home/jdoucette/Documents/code/MWI-Data-Catalog/Siemens_t2_mapping_se_mc_32echo_10msTE_120degRefCon/0008_t2_mapping_se_mc_ms_32TE_single_slice_1E_masked_image_rotated.mat"),
]

for model_path in readdir(joinpath(ismrm_dir, "models"); join = true)
    @info basename(model_path)
    ENV["JL_CHECKPOINT_FOLDER"] = model_path
    local settings = make_settings()
    local phys = make_physics(settings; image_infos)
    local models, derived = make_models!(phys, settings, load_checkpoint())

    for (i,img) in enumerate(phys.images)
        local init, res = mle_biexp_epg(
            phys, models, derived, img;
            verbose = true, checkpoint = false, dryrun = false, batch_size = 12*1024,
            savefolder = joinpath(ismrm_dir, "analysis", basename(model_path), "mle_inference_image_$i"),
        )
    end
end

#=
let
    for model_path in readdir(Glob.glob"ignite-cvae*", joinpath(ismrm_dir, "analysis"))
        @info basename(model_path)
        for results_dir in readdir(Glob.glob"mle_inference_image_*/*-results-final-*.mat", model_path)
            @info basename(results_dir)
            res = DECAES.MAT.matread(results_dir)
            @show mean(res["loss"])
        end
    end
end
=#
