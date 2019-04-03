# ============================================================================ #
# Generic geometry utilities for use within the JuAFEM.jl/Tensors.jl framework
# ============================================================================ #

module GeometryUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using Statistics
using Tensors
using StaticArrays: SVector
using LinearAlgebra
using PolynomialRoots
using Optim
using LineSearches
using RecipesBase

export Vec1d, Vec1f, Vec2d, Vec2f, Vec3d, Vec3f
export fielddim, norm2, rotmat, hadamardproduct, ⊙, skewprod, ⊠
export Ellipse, Circle, Rectangle, VecOfCircles, VecOfEllipses, VecOfRectangles
export origin, radius, radii, widths, corners, geta, getb, getc, getF1, getF2
export dimension, floattype, xmin, ymin, xmax, ymax, area, volume
export scale_shape, translate_shape, inscribed_square
export signed_edge_distance, minimum_signed_edge_distance
export bounding_box, bounding_circle, crude_bounding_circle, opt_bounding_ellipse, opt_bounding_circle, intersect_area, intersection_points, tile_rectangle
export is_inside, is_overlapping, is_any_overlapping, is_on_circle, is_on_any_circle, is_in_circle, is_in_any_circle, is_inside, is_outside, is_on_boundary

include("src/types.jl")
include("src/utils.jl")

end # module GeometryUtils

nothing
