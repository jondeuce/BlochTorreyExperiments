####
#### Perform MCMC
####
####    Resulting chains are stored in /mcmc/simulated/{timestamp}.
####    Checkpoint files can be deleted after finishing.
####    Move image*.mat files to the respective /data/images/{image} folder when finished,
####    and update the `image_info.toml` file for use with the rest of the pipeline.
####

using DrWatson: @quickactivate
@quickactivate "NeurIPS2021"
using NeurIPS2021
lib.initenv()

function main(;
        num_samples::Int,     # number of samples per chain to draw from NUTS sampler
        total_chains::Int,    # number of signals to process; pass Colon() to process all signals
        checkpoint_freq::Int, # checkpoint every `checkpoint_freq` iterations
        dataset::Symbol,
    )

    # Save output data in /data/mcmc/{timestamp}
    savefolder = mkpath(DrWatson.datadir("mcmc", lib.getnow()))

    # Save library code for future reference
    lib.save_project_code(joinpath(savefolder, "project"))

    # Load images from: joinpath.(DrWatson.projectdir(), "data/images", image_folders)
    phys = lib.load_epgmodel_physics()
    lib.initialize!(
        phys;
        seed = 0,
        image_folders = [
            "Simulated_BiexpEPG_CPMG_64echo_10msTE",
        ]
    )

    # Perform MCMC, saving results
    for img_idx in 1:length(phys.images)
        lib.mcmc_biexp_epg(
            phys;
            img_idx         = img_idx,
            num_samples     = num_samples,
            dataset         = dataset,
            total_chains    = total_chains,
            seed            = 0,
            shuffle         = true, # shuffle to balance thread work loads; unrelated to train/val/test split
            save            = true,
            savefolder      = savefolder,
            checkpoint      = true,
            checkpoint_freq = checkpoint_freq,
            progress_freq   = 15.0, # update progress bar every `progress_freq` seconds
        )
    end
end

# Draw 100 MCMC samples across whole dataset
main(;
    num_samples     = 100,
    total_chains    = Colon(),
    checkpoint_freq = 10_000,
    dataset         = :mask,
)

# Draw 3000 MCMC samples for 5000 samples from each of train, val, test sets
for dataset = [:train, :val, :test]
    main(;
        num_samples     = 3_000,
        total_chains    = 5_000,
        checkpoint_freq = 500,
        dataset         = dataset,
    )
end
