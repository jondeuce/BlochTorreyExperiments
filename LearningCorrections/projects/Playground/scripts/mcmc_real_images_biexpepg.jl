using DrWatson: @quickactivate
@quickactivate "Playground"
using Playground
lib.initenv()

# Save library code for future reference
lib.save_project_code(joinpath(pwd(), "project"))

# Load images from: DrWatson.projectdir()/data/images/{image_folders}
phys = lib.load_epgmodel_physics()
lib.initialize!(
    phys;
    seed = 0,
    image_folders = [
        "2019-10-28_48echo_8msTE_CPMG",
        "2019-09-22_56echo_7msTE_CPMG",
        "2021-05-07_NeurIPS2021_64echo_10msTE_MockBiexpEPG_CPMG",
    ]
)

for img_idx in 1:length(phys.images), dataset in [:train, :val, :test]
    lib.mcmc_biexp_epg(
        phys;
        img_idx         = img_idx,
        num_samples     = 3000,
        dataset         = dataset,
        total_chains    = 5000,
        seed            = 0,
        shuffle         = true,
        save            = true,
        checkpoint      = true,
        checkpoint_freq = 100,  # checkpoint every `checkpoint_freq` iterations
        progress_freq   = 15.0, # update progress bar every `progress_freq` seconds
    )
end
