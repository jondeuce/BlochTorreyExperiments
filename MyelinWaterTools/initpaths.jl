# ============================================================================ #
# Module path loading (do this before using Revise to ensure Revise sees them)
# ============================================================================ #
BTHOME = joinpath(@__DIR__, "../..") |> realpath # assume we are in MyelinWaterTools folder
BTBRANCHPATH = joinpath(@__DIR__, "..") |> realpath
MWTPATH = joinpath(BTBRANCHPATH, "MyelinWaterTools/") |> realpath

# Add folders, as well as "src" and "test" subfolders, to global LOAD_PATH
for dir in joinpath.(MWTPATH, readdir(MWTPATH))
    srcdir = joinpath(dir, "src")
    testdir = joinpath(dir, "test")
    map((dir, srcdir, testdir)) do subdir
        if isdir(subdir)
            realsubdir = realpath(subdir)
            (realsubdir ∉ LOAD_PATH) && push!(LOAD_PATH, realsubdir)
        end
    end
end

# Might need to define this environment variable
# ENV["MATLAB_HOME"] = "/usr/local/MATLAB/R2017b"#/bin/matlab" # curie (don't need it)
# ENV["MATLAB_HOME"] = "C:\\Program Files\\MATLAB\\R2018b\\"
