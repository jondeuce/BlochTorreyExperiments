# ============================================================================ #
# Generic geometry utilities for use within the JuAFEM.jl/Tensors.jl framework
# ============================================================================ #

module GeometryUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using Reexport
@reexport using LinearAlgebra, SparseArrays, Statistics, StatsBase
@reexport using JuAFEM, Tensors
@reexport using DistMesh
@reexport using Rotations
using Tensors: Vec
using JuAFEM: vertices, faces, edges
export vertices, faces, edges

using StaticArrays: SVector
using PolynomialRoots
using Optim
using LineSearches
using RecipesBase
# using MATLAB # Only need for MAT_* methods, which are no longer used

export Vec1d, Vec1f, Vec2d, Vec2f, Vec3d, Vec3f,
       fielddim, norm2, rotmat, pulsemat2, pulsemat3, transverse, longitudinal,
       hadamardproduct, ⊙, skewprod, ⊠
export Ellipse, Circle, Rectangle, VecOfCircles, VecOfEllipses, VecOfRectangles,
       origin, radius, radii, widths, corners, geta, getb, getc, getF1, getF2,
       dimension, floattype, xmin, ymin, xmax, ymax, area, volume,
       scale_shape, translate_shape, inscribed_square,
       signed_edge_distance, minimum_signed_edge_distance,
       bounding_box, bounding_circle, crude_bounding_circle, opt_bounding_ellipse, opt_bounding_circle, intersect_area, intersection_points, tile_rectangle,
       is_inside, is_overlapping, is_any_overlapping, is_on_circle, is_on_any_circle, is_in_circle, is_in_any_circle, is_inside, is_outside, is_on_boundary
export getfaces, nodevector, nodematrix, cellvector, cellmatrix, nodecellmatrices,
       mxbbox, mxaxis, dcircles, dexterior, hcircles
export disjoint_rect_mesh_with_tori

include("types.jl")
include("shapeutils.jl")
include("gridutils.jl")
include("meshutils.jl")

end # module GeometryUtils
