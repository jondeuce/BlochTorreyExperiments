# ============================================================================ #
# Tools for working with circles within JuAFEM.jl Grid's
# ============================================================================ #

module MeshUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #
using GeometryUtils
using JuAFEM
using JuAFEM: vertices, faces, edges
using MATLAB, SparseArrays, Statistics

export getfaces, mxplot,
       circle_mesh_with_tori,
       rect_mesh_with_circles, disjoint_rect_mesh_with_tori, rect_mesh_with_tori,
       form_tori_subgrids, form_disjoint_grid, form_subgrid

# ---------------------------------------------------------------------------- #
# Misc grid utils
# ---------------------------------------------------------------------------- #
function getfaces(grid::Grid{2})
    # faces will be approx. double counted, and Euler characteristic says #faces (edges) ~ #nodes + #cells in 2D
    faceset = Set{Tuple{Int,Int}}()
    sizehint!(faceset, 2*(getnnodes(grid) + getncells(grid)))
    for (cell_idx, cell) in enumerate(getcells(grid))
        for (face_idx, face) in enumerate(faces(cell))
            push!(faceset, (cell_idx, face_idx))
        end
    end
    JuAFEM._warn_emptyset(faceset)
    return faceset
end

# ---------------------------------------------------------------------------- #
# rect_mesh_with_circles
# ---------------------------------------------------------------------------- #
function rect_mesh_with_circles(rect_bdry::Rectangle{2,T},
                                circles::Vector{Circle{2,T}},
                                h0::T,
                                eta::T;
                                isunion::Bool = true) where T
    # TODO: add minimum angle threshold
    dim = 2
    nfaces = 3 # per triangle
    nnodes = 3 # per triangle

    bbox = [xmin(rect_bdry) ymin(rect_bdry);
            xmax(rect_bdry) ymax(rect_bdry)]
    centers = reinterpret(T, origin.(circles), (dim,length(circles)))'
    radii = radius.(circles)

    nargout = 2
    p, t = mxcall(:squaremeshwithcircles, nargout, bbox, centers, radii, h0, eta, isunion)

    cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
    nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
    fullgrid = Grid(cells, nodes)

    # Manually add boundary sets for the four square edges and circle boundaries
    addfaceset!(fullgrid, "left",   x -> x[1] ≈ xmin(rect_bdry), all=true)
    addfaceset!(fullgrid, "right",  x -> x[1] ≈ xmax(rect_bdry), all=true)
    addfaceset!(fullgrid, "top",    x -> x[2] ≈ ymax(rect_bdry), all=true)
    addfaceset!(fullgrid, "bottom", x -> x[2] ≈ ymin(rect_bdry), all=true)
    addfaceset!(fullgrid, "circles", x -> is_on_any_circle(x, circles), all=true)

    # Boundary matrix and boundary face set
    all_boundaries = union(values.(getfacesets(fullgrid))...)
    fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
    addfaceset!(fullgrid, "boundary", all_boundaries)

    if isunion
        # Generate cell and node sets
        addcellset!(fullgrid, "exterior", x -> !is_in_any_circle(x, circles); all=false)
        addnodeset!(fullgrid, "exterior", x -> !is_in_any_circle(x, circles) || is_on_any_circle(x, circles))
        addcellset!(fullgrid, "circles",  x ->  is_in_any_circle(x, circles); all=true)
        addnodeset!(fullgrid, "circles",  x ->  is_in_any_circle(x, circles) || is_on_any_circle(x, circles))

        # Generate exterior grid
        subgrids = typeof(fullgrid)[]
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
    else
        subgrids = typeof(fullgrid)[]
    end

    return fullgrid, subgrids
end

# ---------------------------------------------------------------------------- #
# rect_mesh_with_tori
# ---------------------------------------------------------------------------- #
function disjoint_rect_mesh_with_tori(rect_bdry::Rectangle{2,T},
                                      inner_circles::Vector{Circle{2,T}},
                                      outer_circles::Vector{Circle{2,T}},
                                      h0::T,
                                      eta::T) where {T}
    # Ensure that outer circles strictly contain inner circles, and that outer
    # circles are strictly non-overlapping
    @assert length(inner_circles) == length(outer_circles)
    @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
    @assert !is_any_overlapping(outer_circles, <)

    println("0/$(length(outer_circles)): Exterior")
    exteriorgrid = form_disjoint_grid(rect_bdry, inner_circles, outer_circles, h0, eta, :exterior)
    # exteriorgrid = Grid[]

    # interiorgrids = Grid[]
    # torigrids = Grid[]
    interiorgrids = typeof(exteriorgrid)[]
    torigrids = typeof(exteriorgrid)[]
    @inbounds for i = 1:length(outer_circles)
        println("$i/$(length(outer_circles)): Interior")
        new_bdry = intersect(rect_bdry, bounding_box(inner_circles[i]))
        if !(area(new_bdry) ≈ zero(T))
            push!(interiorgrids, form_disjoint_grid(new_bdry, [inner_circles[i]], [outer_circles[i]], h0, eta, :interior))
        else
            push!(interiorgrids, Grid(Triangle[], Node{2,T}[]))
        end

        println("$i/$(length(outer_circles)): Annular")
        new_bdry = intersect(rect_bdry, bounding_box(outer_circles[i]))
        if !(area(new_bdry) ≈ zero(T))
            push!(torigrids, form_disjoint_grid(new_bdry, [inner_circles[i]], [outer_circles[i]], h0, eta, :tori))
        else
            push!(torigrids, Grid(Triangle[], Node{2,T}[]))
        end
    end

    return exteriorgrid, torigrids, interiorgrids
end

function form_disjoint_grid(rect_bdry::Rectangle{2,T},
                            inner_circles::Vector{Circle{2,T}},
                            outer_circles::Vector{Circle{2,T}},
                            h0::T,
                            eta::T,
                            regiontype::Symbol) where {T}

    dim = 2
    nargout = 2
    isunion = false
    outer_centers = reinterpret(T, origin.(outer_circles), (dim, length(outer_circles)))'
    inner_centers = reinterpret(T, origin.(inner_circles), (dim, length(inner_circles)))'
    outer_radii   = radius.(outer_circles)
    inner_radii   = radius.(inner_circles)

    bbox = [xmin(rect_bdry) ymin(rect_bdry);
            xmax(rect_bdry) ymax(rect_bdry)]
    if regiontype == :exterior
        regnumber = 1.0
        p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber)
    elseif regiontype == :tori
        regnumber = 2.0
        p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber, inner_centers, inner_radii)
    elseif regiontype == :interior
        regnumber = 3.0
        p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber, inner_centers, inner_radii)
    else
        error("Invalid regiontype == $regiontype.")
    end

    cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
    nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
    grid = Grid(cells, nodes)

    # Ensure points near circles are exactly on circles
    project_circles!(grid, inner_circles, 1e-6*h0)
    project_circles!(grid, outer_circles, 1e-6*h0)

    # Manually add boundary sets for the four square edges and circle boundaries
    is_on_outer_circles = x -> is_on_any_circle(x, outer_circles)
    is_on_inner_circles = x -> is_on_any_circle(x, inner_circles)
    is_on_rectangle     = x -> x[1] ≈ xmin(rect_bdry) || x[1] ≈ xmax(rect_bdry) ||
                               x[2] ≈ ymax(rect_bdry) || x[2] ≈ ymin(rect_bdry)
    is_boundary = x -> is_on_outer_circles(x) || is_on_inner_circles(x) || is_on_rectangle(x)

    # Boundary matrix (including inner boundaries) and boundary face set
    addfaceset!(grid, "boundary", is_boundary, all=true)
    grid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, getfaceset(grid, "boundary")))

    return grid
end

function rect_mesh_with_tori(rect_bdry::Rectangle{2,T},
                             inner_circles::Vector{Circle{2,T}},
                             outer_circles::Vector{Circle{2,T}},
                             h0::T,
                             eta::T) where {T}
    # Ensure that outer circles strictly contain inner circles, and that outer
    # circles are strictly non-overlapping
    @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
    @assert !is_any_overlapping(outer_circles, <)

    dim = 2
    nfaces = 3 # per triangle
    nnodes = 3 # per triangle

    # TODO: add minimum angle threshold?
    nargout = 2
    all_circles = vcat(outer_circles, inner_circles)
    all_centers = reinterpret(T, origin.(all_circles), (dim, length(all_circles)))'
    all_radii   = radius.(all_circles)
    bbox = [xmin(rect_bdry) ymin(rect_bdry);
            xmax(rect_bdry) ymax(rect_bdry)]
    p, t = mxcall(:squaremeshwithcircles, nargout, bbox, all_centers, all_radii, h0, eta, isunion)

    cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
    nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
    fullgrid = Grid(cells, nodes)

    # Ensure points near circles are exactly on circles
    project_circles!(fullgrid, inner_circles, 1e-6*h0)
    project_circles!(fullgrid, outer_circles, 1e-6*h0)

    # Manually add boundary sets for the four square edges and circle boundaries
    addfaceset!(fullgrid, "left",   x -> x[1] ≈ xmin(rect_bdry), all=true)
    addfaceset!(fullgrid, "right",  x -> x[1] ≈ xmax(rect_bdry), all=true)
    addfaceset!(fullgrid, "top",    x -> x[2] ≈ ymax(rect_bdry), all=true)
    addfaceset!(fullgrid, "bottom", x -> x[2] ≈ ymin(rect_bdry), all=true)
    addfaceset!(fullgrid, "inner_circles", x -> is_on_any_circle(x, inner_circles), all=true)
    addfaceset!(fullgrid, "outer_circles", x -> is_on_any_circle(x, outer_circles), all=true)

    # Boundary matrix (including inner boundaries) and boundary face set
    all_boundaries = union(values.(getfacesets(fullgrid))...)
    fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
    addfaceset!(fullgrid, "boundary", all_boundaries)

    return fullgrid
end

function form_tori_subgrids(fullgrid::Grid{dim,N,T,M},
                            rect_bdry::Rectangle{2,T},
                            inner_circles::Vector{Circle{2,T}},
                            outer_circles::Vector{Circle{2,T}}) where {dim,N,T,M}
    # Helper functions
    is_in_outer_circles = x -> is_in_any_circle(x, outer_circles)
    is_in_inner_circles = x -> is_in_any_circle(x, inner_circles)
    is_on_outer_circles = x -> is_on_any_circle(x, outer_circles)
    is_on_inner_circles = x -> is_on_any_circle(x, inner_circles)
    is_on_rectangle     = x -> x[1] ≈ xmin(rect_bdry) || x[1] ≈ xmax(rect_bdry) ||
                               x[2] ≈ ymax(rect_bdry) || x[2] ≈ ymin(rect_bdry)

    is_in_exterior  = x -> !is_in_outer_circles(x) || is_on_outer_circles(x)
    is_in_tori      = x ->  is_in_outer_circles(x) && (!is_in_inner_circles(x) || is_on_inner_circles(x))
    is_in_interior  = x ->  is_in_inner_circles(x)
    is_on_exterior  = x -> (is_on_outer_circles(x) || is_on_rectangle(x)) && (!is_in_outer_circles(x) || is_on_outer_circles(x))
    is_on_tori      = x ->  is_on_outer_circles(x) || is_on_inner_circles(x)
    is_on_interior  = x ->  is_on_inner_circles(x)

    # Generate face sets and node sets
    delete!(fullgrid.cellsets, "exterior"); addcellcenterset!(fullgrid, "exterior", x -> !is_in_outer_circles(x))#; all=false)
    delete!(fullgrid.cellsets, "tori");     addcellcenterset!(fullgrid, "tori",     x -> !is_in_inner_circles(x) && is_in_outer_circles(x))#; all=false)
    delete!(fullgrid.cellsets, "interior"); addcellcenterset!(fullgrid, "interior", x ->  is_in_inner_circles(x))#; all=false)
    delete!(fullgrid.nodesets, "exterior"); addnodeset!(fullgrid, "exterior", cellset_to_nodeset(fullgrid, "exterior"))
    delete!(fullgrid.nodesets, "tori");     addnodeset!(fullgrid, "tori",     cellset_to_nodeset(fullgrid, "tori"))
    delete!(fullgrid.nodesets, "interior"); addnodeset!(fullgrid, "interior", cellset_to_nodeset(fullgrid, "interior"))

    # Generate exterior grid
    cellset = getcellset(fullgrid, "exterior")
    nodeset = getnodeset(fullgrid, "exterior")
    exteriorgrid = form_subgrid(fullgrid, cellset, nodeset, getfaceset(fullgrid, "boundary"))

    # Generate tori and interior grids
    get_x = (nodenum) -> getcoordinates(getnodes(fullgrid, nodenum))
    get_c = (cellnum) -> getcells(fullgrid, cellnum)
    cellfilter = (cellnum, circle) -> is_inside(cellcenter(fullgrid, cellnum), circle)

    # Create individual tori grids by filtering on the entire "tori" set
    torigrids = Grid{dim,N,T,M}[]
    for circle in outer_circles
        cellset = filter(cellnum -> cellfilter(cellnum, circle), getcellset(fullgrid, "tori"))
        nodeset = cellset_to_nodeset(fullgrid, cellset)
        push!(torigrids, form_subgrid(fullgrid, cellset, nodeset, getfaceset(fullgrid, "boundary")))
    end

    # Create individual interior grids by filtering on the entire "interior" set
    interiorgrids = Grid{dim,N,T,M}[]
    for circle in inner_circles
        cellset = filter(cellnum -> cellfilter(cellnum, circle), getcellset(fullgrid, "interior"))
        nodeset = cellset_to_nodeset(fullgrid, cellset)
        push!(interiorgrids, form_subgrid(fullgrid, cellset, nodeset, getfaceset(fullgrid, "boundary")))
    end

    return exteriorgrid, torigrids, interiorgrids
end

# ---------------------------------------------------------------------------- #
# circle_mesh_with_tori
# ---------------------------------------------------------------------------- #
function circle_mesh_with_tori( circle_bdry::Circle{2,T},
                                inner_circles::Vector{Circle{2,T}},
                                outer_circles::Vector{Circle{2,T}},
                                h0::T,
                                eta::T ) where {T}
    # Ensure that outer circles strictly contain inner circles, and that outer
    # circles are strictly non-overlapping
    @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
    @assert !is_any_overlapping(outer_circles, <)

    dim = 2 # grid dimension
    nfaces = 3 # per triangle
    nnodes = 3 # per triangle

    # TODO: add minimum angle threshold?
    nargout = 2
    isunion = true
    regiontype = 0 # union type
    bcircle = [origin(circle_bdry)..., radius(circle_bdry)]
    outer_centers = copy(transpose(reshape(reinterpret(T, origin.(outer_circles)), (dim, length(outer_circles)))))
    inner_centers = copy(transpose(reshape(reinterpret(T, origin.(inner_circles)), (dim, length(inner_circles)))))
    p, t = mxcall(:circularmeshwithtori, nargout,
        bcircle, outer_centers, radius.(outer_circles), inner_centers, radius.(inner_circles),
        h0, eta, isunion, regiontype )

    cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
    nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
    fullgrid = Grid(cells, nodes)

    # Ensure points near circles are exactly on circles
    project_circle!(fullgrid, circle_bdry, 1e-6*h0)
    project_circles!(fullgrid, inner_circles, 1e-6*h0)
    project_circles!(fullgrid, outer_circles, 1e-6*h0)

    # Manually add boundary sets for the four square edges and circle boundaries
    addfaceset!(fullgrid, "boundary_circle", x -> is_on_circle(x, circle_bdry), all=true)
    addfaceset!(fullgrid, "inner_circles",   x -> is_on_any_circle(x, inner_circles), all=true)
    addfaceset!(fullgrid, "outer_circles",   x -> is_on_any_circle(x, outer_circles), all=true)

    # Boundary matrix (including inner boundaries) and boundary face set
    all_boundaries = union(values.(getfacesets(fullgrid))...)
    fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
    addfaceset!(fullgrid, "boundary", all_boundaries)

    return fullgrid
end

function form_tori_subgrids(fullgrid::Grid{dim,N,T,M},
                            circle_bdry::Circle{2,T},
                            inner_circles::Vector{Circle{2,T}},
                            outer_circles::Vector{Circle{2,T}}) where {dim,N,T,M}
    # Helper functions
    is_in_outer_circles = x -> is_in_any_circle(x, outer_circles)
    is_in_inner_circles = x -> is_in_any_circle(x, inner_circles)
    is_on_outer_circles = x -> is_on_any_circle(x, outer_circles)
    is_on_inner_circles = x -> is_on_any_circle(x, inner_circles)
    is_on_boundary      = x -> is_on_circle(x, circle_bdry)

    is_in_exterior  = x -> !is_in_outer_circles(x) || is_on_outer_circles(x)
    is_in_tori      = x ->  is_in_outer_circles(x) && (!is_in_inner_circles(x) || is_on_inner_circles(x))
    is_in_interior  = x ->  is_in_inner_circles(x)
    is_on_exterior  = x -> (is_on_outer_circles(x) || is_on_boundary(x)) && (!is_in_outer_circles(x) || is_on_outer_circles(x))
    is_on_tori      = x ->  is_on_outer_circles(x) || is_on_inner_circles(x)
    is_on_interior  = x ->  is_on_inner_circles(x)

    # Generate face sets and node sets
    delete!(fullgrid.cellsets, "exterior"); addcellcenterset!(fullgrid, "exterior", x -> !is_in_outer_circles(x))#; all=false)
    delete!(fullgrid.cellsets, "tori");     addcellcenterset!(fullgrid, "tori",     x -> !is_in_inner_circles(x) && is_in_outer_circles(x))#; all=false)
    delete!(fullgrid.cellsets, "interior"); addcellcenterset!(fullgrid, "interior", x ->  is_in_inner_circles(x))#; all=false)
    delete!(fullgrid.nodesets, "exterior"); addnodeset!(fullgrid, "exterior", cellset_to_nodeset(fullgrid, "exterior"))
    delete!(fullgrid.nodesets, "tori");     addnodeset!(fullgrid, "tori",     cellset_to_nodeset(fullgrid, "tori"))
    delete!(fullgrid.nodesets, "interior"); addnodeset!(fullgrid, "interior", cellset_to_nodeset(fullgrid, "interior"))

    # Generate exterior grid
    cellset = getcellset(fullgrid, "exterior")
    nodeset = getnodeset(fullgrid, "exterior")
    exteriorgrid = form_subgrid(fullgrid, cellset, nodeset, getfaceset(fullgrid, "boundary"))

    # Generate tori and interior grids
    get_x = (nodenum) -> getcoordinates(getnodes(fullgrid, nodenum))
    get_c = (cellnum) -> getcells(fullgrid, cellnum)
    cellfilter = (cellnum, circle) -> is_inside(cellcenter(fullgrid, cellnum), circle)

    # Create individual tori grids by filtering on the entire "tori" set
    torigrids = Grid{dim,N,T,M}[]
    for circle in outer_circles
        cellset = filter(cellnum -> cellfilter(cellnum, circle), getcellset(fullgrid, "tori"))
        nodeset = cellset_to_nodeset(fullgrid, cellset)
        push!(torigrids, form_subgrid(fullgrid, cellset, nodeset, getfaceset(fullgrid, "boundary")))
    end

    # Create individual interior grids by filtering on the entire "interior" set
    interiorgrids = Grid{dim,N,T,M}[]
    for circle in inner_circles
        cellset = filter(cellnum -> cellfilter(cellnum, circle), getcellset(fullgrid, "interior"))
        nodeset = cellset_to_nodeset(fullgrid, cellset)
        push!(interiorgrids, form_subgrid(fullgrid, cellset, nodeset, getfaceset(fullgrid, "boundary")))
    end

    return exteriorgrid, torigrids, interiorgrids
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
# creating cellsets
# ---------------------------------------------------------------------------- #
function cellset_to_nodeset(grid::Grid, cellset::Union{Set{Int},Vector{Int}})
    nodeset = Set{Int}()
    for c in cellset
        cell = getcells(grid, c)
        push!(nodeset, vertices(cell)...)
    end
    return nodeset
end
cellset_to_nodeset(grid::Grid, name::String) = cellset_to_nodeset(grid, getcellset(grid, name))

@inline cellcenter(grid::Grid, cell::Cell) = mean(getcoordinates.(getindex.((getnodes(grid),), cell.nodes)))
@inline cellcenter(grid::Grid, cellnum::Int) = cellcenter(grid, getcells(grid, cellnum))
function cellcenterset(grid::Grid, f::Function)
    cells = Set{Int}()
    for (i, cell) in enumerate(getcells(grid))
        x = cellcenter(grid, cell)
        f(x) && push!(cells, i)
    end
    JuAFEM._warn_emptyset(cells)
    return cells
end
function addcellcenterset!(grid::Grid, name::String, f::Function)
    JuAFEM._check_setname(grid.cellsets, name)
    grid.cellsets[name] = cellcenterset(grid, f)
    return grid
end

# ---------------------------------------------------------------------------- #
# mxplot: call the MATLAB function `simpplot` to plot the grid
# ---------------------------------------------------------------------------- #
function mxplot(g::Grid)
    # Node positions matrix
    p = zeros(Float64, getnnodes(g), 2)
    @inbounds for i in 1:getnnodes(g)
        p[i,1], p[i,2] = getcoordinates(getnodes(g)[i])
    end

    # Triangle indices matrix
    t = zeros(Float64, getncells(g), 3)
    @inbounds for i in 1:getncells(g)
        t[i,1], t[i,2], t[i,3] = vertices(getcells(g)[i])
    end

    # Call `simpplot(p,t)`
    mxcall(:simpplot, 0, p, t)
    return nothing
end

# ---------------------------------------------------------------------------- #
# project_circle
# ---------------------------------------------------------------------------- #
function project_circle!(grid::Grid, circle::Circle{dim,T}, thresh::T) where {dim,T}
    for i in eachindex(grid.nodes)
        x = getcoordinates(getnodes(grid)[i])
        dx = x - origin(circle)
        normdx = norm(dx)
        if abs(normdx - radius(circle)) <= thresh
            x = origin(circle) + (radius(circle)/normdx) * dx
            grid.nodes[i] = Node(x)
        end
    end
    return grid
end
function project_circles!(grid::Grid, circles::Vector{C}, thresh) where {C <: Circle}
    for circle in circles
        project_circle!(grid, circle, thresh)
    end
    return grid
end

project_circle(grid::Grid, circle::Circle, thresh) = project_circle!(deepcopy(grid), circle, thresh)
project_circles(grid::Grid, circles::Vector{C}, thresh) where {C <: Circle} = project_circles!(deepcopy(grid), circles, thresh)

end # module MeshUtils

nothing
