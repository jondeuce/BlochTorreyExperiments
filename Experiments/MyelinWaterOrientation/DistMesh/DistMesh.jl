# ============================================================================ #
# DistMesh
# ============================================================================ #

module DistMesh

# using LinearAlgebra
using StatsBase
# using Random
using Tensors # for gradients of tensor functions, and Vec type
using MATLAB # for plotting
using VoronoiDelaunay # for Delaunay triangulation

export distmesh2d, delaunay2, delaunay2!
export huniform, fixmesh, boundedges, simpplot
export dblock, drectangle, drectangle0, dsphere, dcircle, dshell
export ddiff, dintersect, dunion

include("src/utils.jl")
include("src/distances.jl")
include("src/distmesh2d.jl")

end # module DistMesh

nothing
