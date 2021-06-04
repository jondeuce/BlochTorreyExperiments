# Metropolis-CVAEs

## Preface

Due to subject confidentiality, the MRI data used in this study cannot be provided.
Therefore, the code and data provided here reproduces the results of the two experiments:

1. Traditional CVAE trained on simulated data with known labels, and
2. Metropolis-CVAE trained on simulated data with known labels as well as simulated data initialized with random pseudo labels drawn from the respective prior distributions.

The data used for training and evaluation consists of approximately 5 GB of 200 000 MRI signals simulated using the biexponential EPG model, as well as their corresponding MCMC posterior samples (100 draws for all 200 000 signals, and 3000 draws for 5000 signals from each of the train, validation, and testing splits).

This data is too large to submit as supplementary material (100 MB limit).
The supplementary material including this `data/` folder can, however, be found at the following (anonymized) Google Drive link:

    https://drive.google.com/drive/folders/1TSFb7B8nAvB8bvB9ZftYzxy3K6ravsf7?usp=sharing

## Installation

The required Julia environment is described by the `Project.toml` and `Manifest.toml` files.
To install the relevant packages, using Julia v1.6+, run the `make.jl` script as follows:

    julia make.jl

The included Julia scripts make use of external Python libraries.
The required libraries will be automatically installed in an anaconda environment specific to Julia (default location `~/.julia/conda/3/`) using the above `make.jl` file.
However, for completeness, the `requirements.txt` file detailing the conda environment is provided.
Additionally, the platform-specific `conda.txt` file detailing the explicit conda environment is provided.
This file can be used to recreate the conda environment via e.g. `~/.julia/conda/3/bin/conda create --name <env> --file conda.txt`, but this should not be necessary.

## Training

To perform training using traditional CVAEs with only simulated data + labels, run the following:

    # Set the --threads=XX flag as appropriate for your machine to enable cpu multithreading
    julia --threads=32 scripts/train_metropolis.jl --data.labels.train_indices 0 --data.labels.eval_indices 0 --data.labels.train_fractions 1.0

To perform training using Metropolis-CVAEs with both simulated data + labels and unlabeled simulated data initialized with random pseudo labels drawn from the prior, run the following:

    # Set the --threads=XX flag as appropriate for your machine to enable cpu multithreading
    julia --threads=32 scripts/train_metropolis.jl --data.labels.train_indices 0 1 --data.labels.eval_indices 0 1 --data.labels.train_fractions 0.5 0.5

Files produced during training (log files, model checkpoints, loss plots, etc.) will be stored in the `log/`.

Note: the training script takes approximately 12-24 hours to run, depending on your CPU and GPU.
Time estimate derived from running Julia v1.6.0 with 32 threads on a AMD Ryzen 9 3950X 16-Core CPU with 64 GB of RAM and a single Nvidia GeForce RTX 3080 GPU with 10 GB of VRAM.

## Generating Simulated Data

To generate the simulated data used in this study, run the following:

    # Set the --threads=XX flag as appropriate for your machine to enable cpu multithreading
    julia --threads=32 scripts/synthetic_data_biexpepg.jl

Generated simulated data will be stored in the `/data/simulated` folder.

## Generating MCMC Samples

To perform MCMC using the NUTS sampler on the generated simulated data, run the following:

    # Set the --threads=XX flag as appropriate for your machine to enable cpu multithreading
    julia --threads=32 scripts/mcmc_biexpepg.jl

Generated MCMC samples will be stored in the `/data/mcmc` folder.

Note: this script will take approximately 72 hours to run, depending on your CPU.
Time estimate derived from running Julia v1.6.0 with 32 threads on a AMD Ryzen 9 3950X 16-Core CPU with 64 GB of RAM.

## Generating Figures

Figures from the paper can be reproduced as follows:

    # Set the --threads=XX flag as appropriate for your machine to enable cpu multithreading
    julia --threads=32 scripts/eval_metropolis.jl

Note that figure 3 differs from the paper due to the fact that simulated data is used instead of the real MRI data, which cannot be provided due to the aforementioned subject confidentiality.
