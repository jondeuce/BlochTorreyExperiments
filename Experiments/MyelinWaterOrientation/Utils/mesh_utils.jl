# ---------------------------------------------------------------------------- #
# Tools for adding circles to meshes within JuAFEM.jl/Tensors.jl framework
# ---------------------------------------------------------------------------- #

struct Circle{dim,T}
    center::Vec{dim,T}
    r::T
end

function is_on_circle(x::Vec{dim,T}, grid::Grid, c::Circle{dim,T}, thresh::T) where {dim,T}
    dx = x - c.center
    return abs(dx⋅dx - c.r^2) <= thresh
end

function add_circle!(grid::Grid, c::Circle{dim,T}, thresh::T) where {dim,T}
    #r², thresh² = c.r^2, thresh^2
    for i in 1:length(grid.nodes)
        x = getcoordinates(grid.nodes[i])
        dx = x - c.center
        if abs(norm(dx) - c.r) <= thresh
            α = c.r/norm(dx)
            x = c.center + α * dx
            grid.nodes[i] = Node(x)
        end
    end
end
function add_circle(grid::Grid, c::Circle{dim,T}, thresh::T) where {dim,T}
    g = deepcopy(grid)
    add_circle!(g, c, thresh)
    return g
end

function add_circles!(grid::Grid, cs::Vector{Circle{dim,T}}, thresh::T) where {dim,T}
    for c in cs
        add_circle!(grid, c, thresh)
    end
end
function add_circles(grid::Grid, cs::Vector{Circle{dim,T}}, thresh::T) where {dim,T}
    g = deepcopy(grid)
    add_circles!(g, cs, thresh)
    return g
end

nothing
