# ---------------------------------------------------------------------------- #
# Types
# ---------------------------------------------------------------------------- #
struct Ellipse{dim,T}
    F1::Vec{dim,T} # focus #1
    F2::Vec{dim,T} # focus #2
    b::T # semi-minor axis
end

struct Circle{dim,T}
    center::Vec{dim,T}
    r::T
end

struct Rectangle{dim,T}
    mins::Vec{dim,T}
    maxs::Vec{dim,T}
end

const VecOfCircles{dim} = Vector{Circle{dim,T}} where T
const VecOfEllipses{dim} = Vector{Ellipse{dim,T}} where T
const VecOfRectangles{dim} = Vector{Rectangle{dim,T}} where T

# ---------------------------------------------------------------------------- #
# Plot Recipes
# ---------------------------------------------------------------------------- #
vertices_for_plotting(c::Circle{2}) = [Tuple(origin(c)) .+ radius(c) .* (cos(θ), sin(θ)) for θ in range(0, 2π, length=100)]
vertices_for_plotting(r::Rectangle{2}) = [Tuple.(corners(r))..., Tuple(corners(r)[1])]
@recipe f(::Type{C}, c::C) where C <: Circle{2} = vertices_for_plotting(c)
@recipe f(::Type{R}, r::R) where R <: Rectangle{2} = vertices_for_plotting(r)