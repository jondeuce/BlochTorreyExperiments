import LibGit2
let
    repo = LibGit2.GitRepo(joinpath(@__DIR__, "../.."))
    hash = LibGit2.GitHash(repo, "HEAD")

    open("reproduce.jl", "w") do io
        str =
            """
            import LibGit2
            repo = LibGit2.clone(
                "https://github.com/jondeuce/BlochTorreyExperiments/",
                "BlochTorreyExperiments")
            LibGit2.checkout!(repo, "$(string(hash))")
            cd("BlochTorreyExperiments")
            """
        write(io, str)
    end
end