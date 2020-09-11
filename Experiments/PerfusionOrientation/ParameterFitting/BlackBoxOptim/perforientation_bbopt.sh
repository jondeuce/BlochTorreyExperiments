#!/bin/bash

# Julia/project settings
export JULIA_PATH=/home/jcd1994/julia-1.5.1/bin/julia
export JULIA_PROJECT=/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/Experiments/PerfusionOrientation/ParameterFitting/BlackBoxOptim #TODO
export JULIA_SCRIPT=${JULIA_PROJECT}/perforientation_bbopt.jl
export SWEEP_DIR=$PWD

JOBCOUNTER=0
for Dtissue in 2.0 3.0 0.0 # Tissue diffusion
do
for PVSvolume in 1.0 0.0 3.0 # PVS relative volume
do
for Nmajor in {1..9} # Number of major vessels
do
JOBCOUNTER=$(expr $JOBCOUNTER + 1)
echo "Job #${JOBCOUNTER}"

# Job settings
export JOB_DIR=${SWEEP_DIR}/Job-${JOBCOUNTER}_Dtissue-${Dtissue}_Nmajor-${Nmajor}_PVSvolume-${PVSvolume}
mkdir -p $JOB_DIR

# Copy this script to output directory
cp ${JULIA_PROJECT}/perforientation_bbopt.sh ${JOB_DIR}/
cp ${JULIA_PROJECT}/perforientation_bbopt.pbs ${JOB_DIR}/

# Make sweep settings
export Dtissue=${Dtissue}
export PVSvolume=${PVSvolume}
export Nmajor=${Nmajor}
${JULIA_PATH} --startup-file=no --project=${JULIA_PROJECT} -e 'using MAT; matwrite(joinpath(ENV["JOB_DIR"], "SweepSettings.mat"), Dict("Dtissue" => parse(Float64, ENV["Dtissue"]), "PVSvolume" => parse(Float64, ENV["PVSvolume"]), "Nmajor" => parse(Float64, ENV["Nmajor"])))'

# Submit job
qsub -N job-${JOBCOUNTER} -o ${JOB_DIR}/PBS-output.txt -o ${JOB_DIR}/PBS-error.txt -V ${JULIA_PROJECT}/perforientation_bbopt.pbs

done
done
done
