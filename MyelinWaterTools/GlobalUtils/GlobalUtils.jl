module GlobalUtils

import LibGit2
export make_reproduce

function make_reproduce(
        appending_lines = "";
        fname = "reproduce.jl",
        force = false
    )
    repo = LibGit2.GitRepo(joinpath(@__DIR__, "../.."))
    hash = LibGit2.GitHash(repo, "HEAD")

    exists = isfile(fname)
    if force || !exists
        if force && exists
            @info "Overwriting existing reproduce.jl file"
        else
            @info "Creating reproduce.jl file"
        end
        open(fname, "w") do io
            str =
                """
                import Pkg, LibGit2
                let
                    repo = LibGit2.clone(
                        "https://github.com/jondeuce/BlochTorreyExperiments/",
                        "BlochTorreyExperiments")
                    LibGit2.checkout!(repo, "$(string(hash))")
                    Pkg.activate("BlochTorreyExperiments/MyelinWaterTools/")
                    Pkg.instantiate()
                end
                """
            str = reduce(*, appending_lines; init = str)
            write(io, str)
        end
    else
        @info "File $fname exists and will not be overwritten"
    end

    return nothing
end

end # module GlobalUtils