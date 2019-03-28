module MeshUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #
using GeometryUtils
using DistMesh
using VoronoiDelaunay

using JuAFEM
using JuAFEM: vertices, faces, edges
using SparseArrays, Statistics

using RecipesBase
# using MATLAB # Only need for MAT_* methods, which are no longer used

export getfaces, simpplot, disjoint_rect_mesh_with_tori
export nodevector, nodematrix, cellvector, cellmatrix, nodecellmatrices
export mxbbox, mxaxis

include("src/utils.jl")

end # module MeshUtils

nothing