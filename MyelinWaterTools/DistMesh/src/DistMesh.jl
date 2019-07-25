# ============================================================================ #
# DistMesh
# ============================================================================ #

module DistMesh

using LinearAlgebra, Statistics, StatsBase
using Random
using Tensors # for gradients of tensor functions, and Vec type
using VoronoiDelaunay # for Delaunay triangulation

using RecipesBase # for plotting
import RecipesBase: plot, plot!
export simpplot, simpplot!

export kmg2d, distmesh2d, hgeom, delaunay2, delaunay2!
export huniform, fixmesh, boundedges, edges, edges!
export dblock, drectangle, drectangle0, dsphere, dcircle, dshell
export ddiff, dintersect, dunion

include("utils.jl")
include("delaunay.jl")
include("distances.jl")
include("distmesh2d.jl")
include("kmg2d.jl")

end # module DistMesh