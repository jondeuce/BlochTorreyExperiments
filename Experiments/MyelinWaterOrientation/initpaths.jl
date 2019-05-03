# ============================================================================ #
# Module path loading (do this before using Revise to ensure Revise sees them)
# ============================================================================ #
BTHOME = joinpath(@__DIR__, "../../..") |> realpath # assume we are in MyelinWaterOrientation folder
BTBRANCHPATH = joinpath(@__DIR__, "../..") |> realpath
MWOPATH = joinpath(BTBRANCHPATH, "Experiments/MyelinWaterOrientation/") |> realpath

# Add module paths to global LOAD_PATH
for dir in filter(isdir, realpath.(joinpath.(MWOPATH, readdir(MWOPATH))))
    (dir ∉ LOAD_PATH) && push!(LOAD_PATH, dir)
end

# # Add module paths to global LOAD_PATH recursively
# (MWOPATH ∉ LOAD_PATH) && push!(LOAD_PATH, MWOPATH)
# for (root, dirs, files) in walkdir(MWOPATH)
#     realroot = root |> realpath;
#     for dir in dirs
#         realdir = joinpath(realroot, dir) |> realpath
#         (realdir ∉ LOAD_PATH) && push!(LOAD_PATH, realdir)
#     end
# end

# Might need to define this environment variable
# ENV["MATLAB_HOME"] = "/usr/local/MATLAB/R2017b"#/bin/matlab" # curie (don't need it)
# ENV["MATLAB_HOME"] = "C:\\Program Files\\MATLAB\\R2018b\\"
