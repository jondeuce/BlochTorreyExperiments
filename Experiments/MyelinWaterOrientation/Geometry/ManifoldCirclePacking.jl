# ============================================================================ #
# Manifold circle packing algorithm
# ============================================================================ #

module ManifoldCirclePacking

export pack

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using GeometryUtils
using Tensors
# using VoronoiDelaunay
# using Statistics
# using StaticArrays: SVector
# using LinearAlgebra
# using PolynomialRoots
using Optim
# using LineSearches

# @inline d(c1::Circle, c2::Circle) = norm(origin(c1) - origin(c2)) - radius(c1) - radius(c2)
@inline d(o1::Vec, o2::Vec, r1, r2) = norm(o1-o2)-r1-r2
@inline ∇d(o1::Vec, o2::Vec, r1, r2) = (o1-o2)/norm(o1-o2)


end # module ManifoldCirclePacking

nothing
