# ============================================================================ #
# Tools for working with circles within JuAFEM.jl Grid's
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# square_mesh_with_circles
# ---------------------------------------------------------------------------- #
function square_mesh_with_circles(r::Rectangle{2,T},
                                  cs::Vector{Circle{2,T}},
                                  h0::T,
                                  eta::T;
                                  isunion::Bool = true) where T
    # TODO: add minimum angle threshold
    const dim = 2
    const nfaces = 3 # per triangle
    const nnodes = 3 # per triangle

    bbox = [xmin(r) ymin(r);
            xmax(r) ymax(r)]
    centers = reinterpret(T, origin.(cs), (dim,length(cs)))'
    radii = radius.(cs)

    const nargout = 2
    p, t = mxcall(:squaremeshwithcircles, nargout, bbox, centers, radii, h0, eta, isunion);

    cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)];
    nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)];
    fullgrid = Grid(cells, nodes);

    # Manually add boundary sets for the four square edges and circle boundaries
    addfaceset!(fullgrid, "left",   x -> x[1] ≈ xmin(r), all=true)
    addfaceset!(fullgrid, "right",  x -> x[1] ≈ xmax(r), all=true)
    addfaceset!(fullgrid, "top",    x -> x[2] ≈ ymax(r), all=true)
    addfaceset!(fullgrid, "bottom", x -> x[2] ≈ ymin(r), all=true)
    addfaceset!(fullgrid, "circles", x -> is_on_any_circle(x, cs), all=true)

    # Boundary matrix and boundary face set
    all_boundaries = union(values.(getfacesets(fullgrid))...)
    fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
    addfaceset!(fullgrid, "boundary", all_boundaries)

    # Generate face sets
    addcellset!(fullgrid, "exterior", x -> !is_in_any_circle(x, cs); all=false)
    addcellset!(fullgrid, "circles",  x ->  is_in_any_circle(x, cs); all=true)
    addnodeset!(fullgrid, "exterior", x -> !is_in_any_circle(x, cs) || is_on_any_circle(x, cs))
    addnodeset!(fullgrid, "circles",  x ->  is_in_any_circle(x, cs) || is_on_any_circle(x, cs))

    # Generate exterior grid
    subgrids = Grid[]
    cellset = getcellset(fullgrid, "exterior")
    nodeset = getnodeset(fullgrid, "exterior")
    push!(subgrids, form_subgrid(fullgrid, cellset, nodeset, all_boundaries))

    # Generate circle grids
    for c in cs
        nodeset = filter(node->is_in_circle(getcoordinates(getnodes(fullgrid, node)), c), getnodeset(fullgrid, "circles"))
        cellset = filter(cell->all(node -> node ∈ nodeset, getcells(fullgrid, cell).nodes), getcellset(fullgrid, "circles"))
        push!(subgrids, form_subgrid(fullgrid, cellset, nodeset, all_boundaries))
    end

    return fullgrid, subgrids
end

function form_subgrid(parent_grid::Grid{dim,N,T,M},
                      cellset::Set{Int},
                      nodeset::Set{Int},
                      boundaryset::Set{Tuple{Int,Int}}) where {dim,N,T,M}
    cells = Triangle[]
    nodes = Node{dim,T}[]
    sizehint!(cells, length(cellset))
    sizehint!(nodes, length(nodeset))

    nodemap = spzeros(T,Int,getnnodes(parent_grid))
    for (i,node) in zip(1:length(nodeset), nodeset)
        push!(nodes, getnodes(parent_grid, node))
        nodemap[node] = i
    end

    cellmap = spzeros(T,Int,getncells(parent_grid))
    for (i,cell) in zip(1:length(cellset), cellset)
        parentcellnodes = getcells(parent_grid, cell).nodes
        push!(cells, Triangle(map(n->nodemap[n], parentcellnodes)))
        cellmap[cell] = i
    end

    boundary = Set{Tuple{Int,Int}}()
    for (cell, face) in boundaryset
        newcell = cellmap[cell]
        if newcell ≠ 0
            push!(boundary, (newcell, face))
        end
    end
    boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, boundary))

    grid = Grid(cells, nodes, boundary_matrix=boundary_matrix)
    addfaceset!(grid, "boundary", boundary)

    return grid
end

# ---------------------------------------------------------------------------- #
# is_on_circle/is_on_any_circle
# ---------------------------------------------------------------------------- #
@inline function is_on_circle(x::Vec{dim,T},
                              c::Circle{dim,T},
                              thresh::T=sqrt(eps(T))) where {dim,T}
    dx = x - c.center
    return abs(dx⋅dx - c.r^2) <= thresh
end
function is_on_any_circle(x::Vec{dim,T},
                          cs::Vector{Circle{dim,T}},
                          thresh::T=sqrt(eps(T))) where {dim,T}
    return any(c->is_on_circle(x, c, thresh), cs)
end

# ---------------------------------------------------------------------------- #
# is_in_circle/is_in_any_circle
# ---------------------------------------------------------------------------- #
@inline function is_in_circle(x::Vec{dim,T},
                              c::Circle{dim,T},
                              thresh::T=sqrt(eps(T))) where {dim,T}
    dx = x - c.center
    return dx⋅dx <= (c.r + thresh)^2
end
function is_in_any_circle(x::Vec{dim,T},
                          cs::Vector{Circle{dim,T}},
                          thresh::T=sqrt(eps(T))) where {dim,T}
    return any(c->is_in_circle(x, c, thresh), cs)
end

# ---------------------------------------------------------------------------- #
# project_circle
# ---------------------------------------------------------------------------- #
function project_circle!(grid::Grid, c::Circle{dim,T}, thresh::T) where {dim,T}
    for i in eachindex(grid.nodes)
        x = getcoordinates(grid.nodes[i])
        dx = x - c.center
        normdx = norm(dx)
        if abs(normdx - c.r) <= thresh
            x = c.center + (c.r/normdx) * dx
            grid.nodes[i] = Node(x)
        end
    end
end
function project_circle(grid::Grid, c::Circle, thresh)
    g = deepcopy(grid)
    project_circle!(g, c, thresh)
    return g
end

function project_circles!(grid::Grid, cs::Vector{Circle}, thresh)
    for c in cs
        project_circle!(grid, c, thresh)
    end
end
function project_circles(grid::Grid, cs::Vector{Circle}, thresh)
    g = deepcopy(grid)
    project_circles!(g, cs, thresh)
    return g
end

nothing
