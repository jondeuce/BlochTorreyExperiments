if VERSION < v"1.6"
    @info "Julia v1.6+ is required; version $VERSION found"
    @info "Exiting..."
    exit(0)
end

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
