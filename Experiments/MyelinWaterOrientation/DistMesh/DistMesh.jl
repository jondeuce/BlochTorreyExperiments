# ============================================================================ #
# Tools for circle packing
# ============================================================================ #

module DistMesh

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using GeometryUtils
using LinearAlgebra
# using Statistics
using Random
using DiffBase, ForwardDiff
# using Optim, LineSearches, Roots
using JuAFEM # for gradients of tensor functions, and Vec type
using MATLAB

using VoronoiDelaunay, GeometricalPredicates

export delaunay2, distmesh2d, scaleto, scaleto!

include("distmesh2d.jl")

# Simple function for scaling vector of Vec's to range [a,b]
function scaleto!(p::AbstractVector{Vec{2,Float64}}, a, b)
    N = length(p)
    P = reinterpret(Float64, p) # must be Float64
    Pmin, Pmax = minimum(P), maximum(P)
    P .= ((b - a)/(Pmax - Pmin)) .* (P .- Pmin) .+ a
    clamp!(P, a, b) # to be safe
    return p, Pmin, Pmax
end
scaleto(p::AbstractVector{Vec{2,Float64}}, a, b) = scaleto!(copy(p), a, b)

function scaleto(p::AbstractVector{Vec{2,T}}, a, b) where {T}
    x, Xmin, Xmax = scaleto(Vector{Vec{2,Float64}}(p), Float64(a), Float64(b))
    return Vector{Vec{2,T}}(x), T(Xmin), T(Xmax)
end

end # module DistMesh

nothing
