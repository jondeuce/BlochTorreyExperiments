# ---------------------------------------------------------------------------- #
# Types
# ---------------------------------------------------------------------------- #
struct Ellipse{dim,T}
    F1::Vec{dim,T} # focus #1
    F2::Vec{dim,T} # focus #2
    b::T # semi-minor axis
end
Base.show(io::IO, ::MIME"text/plain", c::Ellipse) = print(io, "$(typeof(c)) with foci $(c.F1) and $(c.F2), and semi-minor axis $(c.b)")
Base.show(io::IO, c::Ellipse) = print(io, "F1 = $(c.F1), F2 = $(c.F2), b = $(c.b)")

struct Circle{dim,T}
    center::Vec{dim,T}
    r::T
end
Base.show(io::IO, ::MIME"text/plain", c::Circle) = print(io, "$(typeof(c)) with origin = $(c.center) and radius = $(c.r)")
Base.show(io::IO, c::Circle) = print(io, "O = $(c.center), R = $(c.r)")

struct Rectangle{dim,T}
    mins::Vec{dim,T}
    maxs::Vec{dim,T}
end
_bounds_string(r::Rectangle{dim}) where {dim} = reduce(*, [" × [$(r.mins[d]), $(r.maxs[d])]" for d in 2:dim]; init = "[$(r.mins[1]), $(r.maxs[1])]")
Base.show(io::IO, ::MIME"text/plain", r::Rectangle) = print(io, "$(typeof(r)) with bounds = " * _bounds_string(r))
Base.show(io::IO, r::Rectangle) = print(io, _bounds_string(r))

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