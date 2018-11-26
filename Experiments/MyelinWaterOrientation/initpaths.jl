# ============================================================================ #
# Module path loading (do this before using Revise to ensure Revise sees them)
# ============================================================================ #
let
    global BTHOME, BTMASTERPATH, MWOPATH, MWOFOLDERS
    BTHOME = realpath(joinpath(@__DIR__, "../../..")) # assume we are in MyelinWaterOrientation folder
    BTMASTERPATH = joinpath(BTHOME, "BlochTorreyExperiments-master/")
    MWOPATH = joinpath(BTMASTERPATH, "Experiments/MyelinWaterOrientation/")
    push!(LOAD_PATH, MWOPATH)

    # Add paths to modules defined in folders
    MWOFOLDERS = ["BlochTorrey", "CirclePacking", "DistMesh", "ExpmvHigham", "Geometry", "Utils", "MWFUtils"]
    MWOFOLDERS = joinpath.(MWOPATH, MWOFOLDERS)
    append!(LOAD_PATH, MWOFOLDERS)

    # Might need to define this environment variable
    # ENV["MATLAB_HOME"] = "/usr/local/MATLAB/R2017b"#/bin/matlab" # curie (don't need it)
    # ENV["MATLAB_HOME"] = "C:\\Users\\Jonathan\\Downloads\\Mathworks Matlab R2016a\\R2016a"
end
