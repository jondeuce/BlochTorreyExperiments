# ============================================================================ #
# Module path loading (do this before using Revise to ensure Revise sees them)
# ============================================================================ #
BTHOME = joinpath(@__DIR__, "../..") |> realpath # assume we are in MyelinWaterTools folder
BTBRANCHPATH = joinpath(@__DIR__, "..") |> realpath
MWTPATH = joinpath(BTBRANCHPATH, "MyelinWaterTools/") |> realpath

# Add module paths to global LOAD_PATH
for dir in filter(isdir, realpath.(joinpath.(MWTPATH, readdir(MWTPATH))))
    srcdir = realpath(joinpath(dir, "src"))
    (dir ∉ LOAD_PATH) && push!(LOAD_PATH, dir)
    isdir(srcdir) && (srcdir ∉ LOAD_PATH) && push!(LOAD_PATH, srcdir)
end

# # Add module paths to global LOAD_PATH recursively
# (MWTPATH ∉ LOAD_PATH) && push!(LOAD_PATH, MWTPATH)
# for (root, dirs, files) in walkdir(MWTPATH)
#     realroot = root |> realpath;
#     for dir in dirs
#         realdir = joinpath(realroot, dir) |> realpath
#         (realdir ∉ LOAD_PATH) && push!(LOAD_PATH, realdir)
#     end
# end

# Might need to define this environment variable
# ENV["MATLAB_HOME"] = "/usr/local/MATLAB/R2017b"#/bin/matlab" # curie (don't need it)
# ENV["MATLAB_HOME"] = "C:\\Program Files\\MATLAB\\R2018b\\"
