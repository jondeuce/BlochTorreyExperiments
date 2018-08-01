# ============================================================================ #
# Tools for working with circles within JuAFEM.jl Grid's
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# square_mesh_with_circles
# ---------------------------------------------------------------------------- #
function square_mesh_with_circles(rect::Rectangle{2,T},
                                  circles::Vector{Circle{2,T}},
                                  h0::T,
                                  eta::T;
                                  isunion::Bool = true) where T
    # TODO: add minimum angle threshold
    const dim = 2
    const nfaces = 3 # per triangle
    const nnodes = 3 # per triangle

    bbox = [xmin(rect) ymin(rect);
            xmax(rect) ymax(rect)]
    centers = reinterpret(T, origin.(circles), (dim,length(circles)))'
    radii = radius.(circles)

    const nargout = 2
    p, t = mxcall(:squaremeshwithcircles, nargout, bbox, centers, radii, h0, eta, isunion)

    cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
    nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
    fullgrid = Grid(cells, nodes)

    # Manually add boundary sets for the four square edges and circle boundaries
    addfaceset!(fullgrid, "left",   x -> x[1] ≈ xmin(rect), all=true)
    addfaceset!(fullgrid, "right",  x -> x[1] ≈ xmax(rect), all=true)
    addfaceset!(fullgrid, "top",    x -> x[2] ≈ ymax(rect), all=true)
    addfaceset!(fullgrid, "bottom", x -> x[2] ≈ ymin(rect), all=true)
    addfaceset!(fullgrid, "circles", x -> is_on_any_circle(x, circles), all=true)

    # Boundary matrix and boundary face set
    all_boundaries = union(values.(getfacesets(fullgrid))...)
    fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
    addfaceset!(fullgrid, "boundary", all_boundaries)

    # Generate face sets
    addcellset!(fullgrid, "exterior", x -> !is_in_any_circle(x, circles); all=false)
    addcellset!(fullgrid, "circles",  x ->  is_in_any_circle(x, circles); all=true)
    addnodeset!(fullgrid, "exterior", x -> !is_in_any_circle(x, circles) || is_on_any_circle(x, circles))
    addnodeset!(fullgrid, "circles",  x ->  is_in_any_circle(x, circles) || is_on_any_circle(x, circles))

    # Generate exterior grid
    subgrids = Grid[]
    cellset = getcellset(fullgrid, "exterior")
    nodeset = getnodeset(fullgrid, "exterior")
    push!(subgrids, form_subgrid(fullgrid, cellset, nodeset, all_boundaries))

    # Generate circle grids
    nodefilter = (nodenum, circle)  -> is_in_circle(getcoordinates(getnodes(fullgrid, nodenum)), circle)
    cellfilter = (cellnum, nodeset) -> all(nodenum -> nodenum ∈ nodeset, vertices(getcells(fullgrid, cellnum)))
    for circle in circles
        nodeset = filter(nodenum -> nodefilter(nodenum, circle), getnodeset(fullgrid, "circles"))
        cellset = filter(cellnum -> cellfilter(cellnum, nodeset), getcellset(fullgrid, "circles"))
        push!(subgrids, form_subgrid(fullgrid, cellset, nodeset, all_boundaries))
    end

    return fullgrid, subgrids
end

# ---------------------------------------------------------------------------- #
# square_mesh_with_tori
# ---------------------------------------------------------------------------- #
function square_mesh_with_tori(rect::Rectangle{2,T},
                               inner_circles::Vector{Circle{2,T}},
                               outer_circles::Vector{Circle{2,T}},
                               h0::T,
                               eta::T;
                               isunion::Bool = true) where T
    # Ensure that outer circles strictly contain inner circles
    @assert all(zip(inner_circles, outer_circles)) do c
        c_in, c_out = c
        is_inside(c_in, c_out, <)
    end

    # Ensure that outer circles are strictly non-overlapping
    @assert !is_any_overlapping(outer_circles, <)

    # TODO: add minimum angle threshold
    const dim = 2
    const nfaces = 3 # per triangle
    const nnodes = 3 # per triangle

    bbox = [xmin(rect) ymin(rect);
            xmax(rect) ymax(rect)]
    all_circles = vcat(outer_circles, inner_circles)
    all_centers = reinterpret(T, origin.(all_circles), (dim, length(all_circles)))'
    all_radii = radius.(all_circles)

    const nargout = 2
    p, t = mxcall(:squaremeshwithcircles, nargout, bbox, all_centers, all_radii, h0, eta, isunion)

    cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
    nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
    fullgrid = Grid(cells, nodes)

    # Helper functions
    is_on_rectangle     = x -> x[1] ≈ xmin(rect) || x[1] ≈ xmax(rect) || x[2] ≈ ymax(rect) || x[2] ≈ ymin(rect)
    is_in_outer_circles = x -> is_in_any_circle(x, outer_circles)
    is_in_inner_circles = x -> is_in_any_circle(x, inner_circles)
    is_on_outer_circles = x -> is_on_any_circle(x, outer_circles)
    is_on_inner_circles = x -> is_on_any_circle(x, inner_circles)

    # Manually add boundary sets for the four square edges and circle boundaries
    addfaceset!(fullgrid, "left",   x -> x[1] ≈ xmin(rect), all=true)
    addfaceset!(fullgrid, "right",  x -> x[1] ≈ xmax(rect), all=true)
    addfaceset!(fullgrid, "top",    x -> x[2] ≈ ymax(rect), all=true)
    addfaceset!(fullgrid, "bottom", x -> x[2] ≈ ymin(rect), all=true)
    addfaceset!(fullgrid, "outer_circles", x -> is_on_outer_circles(x), all=true)
    addfaceset!(fullgrid, "inner_circles", x -> is_on_inner_circles(x), all=true)

    # Boundary matrix (including inner boundaries) and boundary face set
    all_boundaries = union(values.(getfacesets(fullgrid))...)
    fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
    addfaceset!(fullgrid, "boundary", all_boundaries)

    is_in_exterior  = x -> !is_in_outer_circles(x) || is_on_outer_circles(x)
    is_in_tori      = x ->  is_in_outer_circles(x) && (!is_in_inner_circles(x) || is_on_inner_circles(x))
    is_in_interior  = x ->  is_in_inner_circles(x)
    is_on_exterior  = x -> (is_on_outer_circles(x) || is_on_rectangle(x)) && (!is_in_outer_circles(x) || is_on_outer_circles(x))
    is_on_tori      = x ->  is_on_outer_circles(x) || is_on_inner_circles(x)
    is_on_interior  = x ->  is_on_inner_circles(x)

    # Generate face sets and node sets
    addcellset!(fullgrid, "exterior", is_in_exterior; all=true)
    addnodeset!(fullgrid, "exterior", is_in_exterior)
    addcellset!(fullgrid, "tori", is_in_tori; all=true)
    addnodeset!(fullgrid, "tori", is_in_tori)
    addcellset!(fullgrid, "interior", is_in_interior; all=true)
    addnodeset!(fullgrid, "interior", is_in_interior)

    # Generate exterior grid
    cellset = getcellset(fullgrid, "exterior")
    nodeset = getnodeset(fullgrid, "exterior")
    exteriorgrid = form_subgrid(fullgrid, cellset, nodeset, all_boundaries)

    # Generate tori and interior grids
    get_x = (nodenum) -> getcoordinates(getnodes(fullgrid, nodenum))
    get_c = (cellnum) -> getcells(fullgrid, cellnum)
    cellfilter = (cellnum, nodeset) -> all(nodenum -> nodenum ∈ nodeset, vertices(get_c(cellnum)))
    nodefilter = (nodenum, circle) -> is_inside(get_x(nodenum), circle)

    # Create individual tori grids by filtering on the entire "tori" set
    torigrids = Grid[]
    for circle in outer_circles
        nodeset = filter(nodenum -> nodefilter(nodenum, circle), getnodeset(fullgrid, "tori"))
        cellset = filter(cellnum -> cellfilter(cellnum, nodeset), getcellset(fullgrid, "tori"))
        push!(torigrids, form_subgrid(fullgrid, cellset, nodeset, all_boundaries))
    end

    # Create individual interior grids by filtering on the entire "interior" set
    interiorgrids = Grid[]
    for circle in inner_circles
        nodeset = filter(nodenum -> nodefilter(nodenum, circle), getnodeset(fullgrid, "interior"))
        cellset = filter(cellnum -> cellfilter(cellnum, nodeset), getcellset(fullgrid, "interior"))
        push!(interiorgrids, form_subgrid(fullgrid, cellset, nodeset, all_boundaries))
    end

    return fullgrid, exteriorgrid, torigrids, interiorgrids
end

# ---------------------------------------------------------------------------- #
# form_subgrid
# ---------------------------------------------------------------------------- #
function form_subgrid(parent_grid::Grid{dim,N,T,M},
                      cellset::Set{Int},
                      nodeset::Set{Int},
                      boundaryset::Set{Tuple{Int,Int}}) where {dim,N,T,M}
    cells = Triangle[]
    nodes = Node{dim,T}[]
    sizehint!(cells, length(cellset))
    sizehint!(nodes, length(nodeset))

    nodemap = spzeros(Int, Int, getnnodes(parent_grid))
    for (i,nodenum) in zip(1:length(nodeset), nodeset)
        push!(nodes, getnodes(parent_grid, nodenum))
        nodemap[nodenum] = i
    end

    cellmap = spzeros(Int, Int, getncells(parent_grid))
    for (i,cellnum) in zip(1:length(cellset), cellset)
        parent_cellnodeset = vertices(getcells(parent_grid, cellnum))
        push!(cells, Triangle(map(n->nodemap[n], parent_cellnodeset)))
        cellmap[cellnum] = i
    end

    boundary = Set{Tuple{Int,Int}}()
    for (cellnum, face) in boundaryset
        newcell = cellmap[cellnum]
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
# mxplot: call the MATLAB function `simpplot` to plot the grid
# ---------------------------------------------------------------------------- #
function mxplot(g::Grid)
    p = zeros(Float64, getnnodes(g), 2)
    @inbounds for i in 1:getnnodes(g)
        p[i,1], p[i,2] = getcoordinates(getnodes(g)[i])
    end

    t = zeros(Float64, getncells(g), 3)
    for i in 1:size(t,1)
        t[i,1], t[i,2], t[i,3] = vertices(getcells(g)[i])
    end

    # Call `simpplot(p,t)`
    mxcall(:simpplot, 0, p, t)
end

# ---------------------------------------------------------------------------- #
# is_on_circle/is_on_any_circle
# ---------------------------------------------------------------------------- #
@inline function is_on_circle(x::Vec{dim,T},
                              circle::Circle{dim,T},
                              thresh::T=sqrt(eps(T))) where {dim,T}
    dx = x - origin(circle)
    return abs(dx⋅dx - radius(circle)^2) <= thresh
end
function is_on_any_circle(x::Vec{dim,T},
                          circles::Vector{Circle{dim,T}},
                          thresh::T=sqrt(eps(T))) where {dim,T}
    return any(circle->is_on_circle(x, circle, thresh), circles)
end

# ---------------------------------------------------------------------------- #
# is_in_circle/is_in_any_circle
# ---------------------------------------------------------------------------- #
@inline function is_in_circle(x::Vec{dim,T},
                              circle::Circle{dim,T},
                              thresh::T=sqrt(eps(T))) where {dim,T}
    dx = x - origin(circle)
    return dx⋅dx <= (radius(circle) + thresh)^2
end
function is_in_any_circle(x::Vec{dim,T},
                          circles::Vector{Circle{dim,T}},
                          thresh::T=sqrt(eps(T))) where {dim,T}
    return any(circle->is_in_circle(x, circle, thresh), circles)
end

# ---------------------------------------------------------------------------- #
# project_circle
# ---------------------------------------------------------------------------- #
function project_circle!(grid::Grid, circle::Circle{dim,T}, thresh::T) where {dim,T}
    for i in eachindex(grid.nodes)
        x = getcoordinates(grid.nodes[i])
        dx = x - origin(circle)
        normdx = norm(dx)
        if abs(normdx - radius(circle)) <= thresh
            x = origin(circle) + (radius(circle)/normdx) * dx
            grid.nodes[i] = Node(x)
        end
    end
end
function project_circle(grid::Grid, circle::Circle, thresh)
    g = deepcopy(grid)
    project_circle!(g, circle, thresh)
    return g
end

function project_circles!(grid::Grid, circles::Vector{Circle}, thresh)
    for circle in circles
        project_circle!(grid, circle, thresh)
    end
end
function project_circles(grid::Grid, circles::Vector{Circle}, thresh)
    g = deepcopy(grid)
    project_circles!(g, circles, thresh)
    return g
end

nothing
