# ---------------------------------------------------------------------------- #
# DistMesh helper functions and extensions
# ---------------------------------------------------------------------------- #

@inline mxbbox(r::Rectangle{2}) = [xmin(r) ymin(r); xmax(r) ymax(r)]
@inline mxaxis(r::Rectangle{2}) = [xmin(r) xmax(r) ymin(r) ymax(r)]

# Clamping functions: (parameters η, γ, α)
# -Provides a continuous transition from h = h0 to h = η*h0 when d goes from
#  abs(d) = h0 to abs(d) = γ*h0, with h clamped at h0 for abs(d) < h0 and
#  η*h0 for abs(d) > γ*h0
# -sqrt/linear/quad clamps are special cases of the power clamp with exponent α
# -lower α gives faster increasing edge sizes away from boundary; higher α gives slower
@inline sqrtclamp(d, h0, η, γ) = clamp(η * sqrt(h0 * abs(d) / γ), h0, η * h0)
@inline linearclamp(d, h0, η, γ) = clamp(η * abs(d) / γ, h0, η * h0)
@inline quadclamp(d, h0, η, γ) = clamp((η / h0) * (abs(d) / γ)^2, h0, η * h0)
@inline powerclamp(d, h0, η, γ, α) = clamp(η * h0 * (abs(d) / (γ * h0))^α, h0, η * h0)

# Signed distance functions for rectangle, circle, and shell
@inline DistMesh.drectangle0(x::Vec, r::Rectangle) = drectangle0(x, xmin(r), xmax(r), ymin(r), ymax(r))
@inline DistMesh.dcircle(x::Vec, c::Circle) = dcircle(x, origin(c), radius(c))
@inline DistMesh.dshell(x::Vec, c_in::Circle, c_out::Circle) = dshell(x, origin(c_in), radius(c_in), radius(c_out))

# Signed distance function for many circles
function dcircles(x::Vec, cs::Vector{Circle{2,T}}) where {T}
    d = T(dunion()) # initial value s.t. dunion(d, x) == x for all x
    @inbounds for i in eachindex(cs)
        d = dunion(d, dcircle(x, cs[i])) # union of all circle distances
    end
    return d
end

# Signed distance function for region exterior to `r`, but inside of `cs`
@inline function dexterior(x::Vec, r::Rectangle{2}, cs::Vector{C}) where {C<:Circle{2}}
    return ddiff(drectangle0(x, r), dcircles(x, cs))
end

# Edge length functions
@inline function hcircles(x::Vec, h0, η, γ, α, cs::Vector{C}) where {C <: Circle}
    return powerclamp(dcircles(x, cs), h0, η, γ, α)
end
@inline function hcircle(x::Vec, h0, η, γ, α, c::Circle)
    return powerclamp(dcircle(x, c), h0, η, γ, α)
end
@inline function hshell(x::Vec, h0, η, γ, α, c_in::Circle, c_out::Circle)
    return powerclamp(dshell(x, c_in, c_out), h0, η, γ, α)
end

# Plot Grid or vector of Grids. Vector of Grids are combined into one large Grid
# before plotting for speed, so that simpplot need only be called once
function DistMesh.SimpPlotGrid(gs::Vector{G}) where {G <: Grid{2,3}}
    return DistMesh.SimpPlotGrid(nodecellmatrices(gs)...)
end
function DistMesh.SimpPlotGrid(g::G) where {G <: Grid{2,3}}
    return DistMesh.SimpPlotGrid(nodevector(g), cellvector(g))
end

# ---------------------------------------------------------------------------- #
# Misc grid utils
# ---------------------------------------------------------------------------- #

@inline floattype(g::Grid{dim,N,T,M}) where {dim,N,T,M} = T

# `JuAFEM.Grid` constructor given a vector of points and a vector of tuples of
# integers representing cell vertice indices.
#   `dim`: Spatial dimension of domain
#   `T`:   Float type
#   `N`:   Number of nodes per finite element
#   `M`:   Number of faces per finite element (default is number of nodes)
#   `Ne`:  Number of nodes per finite element face (2 in 2D, 3+ in 3D)
# NOTE: assumes cells are properly oriented
function JuAFEM.Grid(
        p::AbstractVector{Vec{dim,T}},
        t::AbstractVector{NTuple{N,Int}},
        M::Int = N # default guess is num faces = num nodes
    ) where {dim,N,T}

    # Check for empty grid
    if isempty(p) || isempty(t)
        return Grid(Cell{dim,N,M}[], Node{dim,T}[])
    end

    # Create initial grid
    grid = Grid(Cell{dim,N,M}.(t), Node.(Tuple.(p)))

    # Add boundary set and boundary matrix, including inner boundaries
    e = boundaryfaceset(grid)
    addfaceset!(grid, "boundary", e)
    grid.boundary_matrix = JuAFEM.boundaries_to_sparse(e)

    return grid
end

function JuAFEM.Grid(grids::AbstractVector{G}) where {G<:Grid}
    # For nodes, can simply concatenate them
    nodes = reduce(vcat, copy(getnodes(g)) for g in grids)

    # For cells, indices must be shifted to account for new node indices
    idxshifts = cumsum([getnnodes(g) for g in grids])
    cells = [copy(getcells(g)) for g in grids]
    for i in 2:length(cells)
        c = copy(reinterpret(Int, cells[i])) # shouldn't modify immutables in-place
        c .+= idxshifts[i-1] # shift all integer indices
        copyto!(cells[i], reinterpret(eltype(cells[i]), c))
    end
    cells = reduce(vcat, cells)

    # Create union of grids
    grid = Grid(cells, nodes)

    # Add boundary set and boundary matrix, including inner boundaries
    # NOTE: this should already be done by the Grid constructor?
    e = boundaryfaceset(grid)
    addfaceset!(grid, "boundary", e)
    grid.boundary_matrix = JuAFEM.boundaries_to_sparse(e)

    return grid
end

function boundaryfaceset(g::Grid{dim,N,T,M}) where {dim,N,T,M}
    nodes, cells = getnodes(g), getcells(g)
    edgeindices = boundedges(g) # Vector{NTuple{M,Int}}

    # Brute force search for edges
    # faceset_brute = Set{Tuple{Int,Int}}() # tuples of (cell_idx, face_idx)
    # sizehint!(faceset_brute, length(edgeindices))
    # for e in edgeindices
    #     for (cell_idx, cell) in enumerate(cells)
    #         face_idx = findfirst(f -> all(ei ∈ f for ei in e), faces(cell))
    #         !(face_idx == nothing) && push!(faceset_brute, (cell_idx, face_idx))
    #         # for (face_idx,f) in enumerate(faces(cell))
    #         #     v = all(ei -> ei ∈ f, e)
    #         #     v && push!(faceset_brute, (cell_idx, face_idx))
    #         # end
    #     end
    # end

    # Create an array of (cell_index, face_index) pairs, as well as an array of
    # nodetuples which stores the node indices for each face
    Np = length(faces(cells[1])[1]) # number of points per cell face
    cellfaces = NTuple{2,Int}[]; sizehint!(cellfaces, M*length(cells)) #[(c,f) for f in faces(c) for c in 1:length(cells)]
    nodetuples = NTuple{Np,Int}[]; sizehint!(cellfaces, M*length(cells))
    for (ci,c) in enumerate(cells)
        for (fi,f) in enumerate(faces(c))
            push!(cellfaces, (ci,fi)) # `fi` is the face index for the cell `ci`
            push!(nodetuples, f) # `f` is a pair of Int's (node indices)
        end
    end

    # Find edges by taking the intersect, and then scanning through to find the
    # corresponding indices
    facetuples = intersect(nodetuples, edgeindices) # order is preserved for arrays
    faceindices = Int[]
    sizehint!(faceindices, length(facetuples))
    ix = 1
    for f in facetuples
        # facetuples order is preserved, so can just linearly move through to find indices
        while (ix <= length(nodetuples)) && !(f == nodetuples[ix])
            ix += 1
        end
        if ix <= length(nodetuples)
            push!(faceindices, ix)
        else
            break
        end
    end

    # Push resulting (cell_index, face_index) pairs to a Set
    faceset = Set{Tuple{Int,Int}}() # tuples of (cell_idx, face_idx)
    sizehint!(faceset, length(nodetuples))
    for ix in faceindices
        push!(faceset, cellfaces[ix])
    end

    JuAFEM._warn_emptyset(faceset)
    return faceset
end

function DistMesh.boundedges(g::Grid{dim,N,T,M}) where {dim,N,T,M}
    return boundedges(
        copy(reinterpret(Vec{dim,T}, getnodes(g))),
        copy(reinterpret(NTuple{N,Int}, getcells(g))))
end

# Area of triangle on 2D grid
function GeometryUtils.area(g::Grid{2,3,T,3}, cell::Int) where {T}
    A, B, C = getcoordinates(g, cell)
    D = ((B - A) × (C - A))[3] # 3rd element of cross product, aka signed norm
    return abs(D)/2 # half of unsigned parallelpiped volume (area)
end
# Area of 2D grid
function GeometryUtils.area(g::Grid{2,3,T,3}) where {T}
    nc = getncells(g)
    return nc == 0 ? zero(T) : sum(c -> area(g,c), 1:nc)
end

# Get faceset for entire 2D grid
function getfaces(grid::Grid{2})
    # faces will be approx. double counted, and Euler characteristic says
    # num. faces (edges) ~ num. nodes + num. cells (2D)
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

# Creating cellsets
function cellset_to_nodeset(grid::Grid, cellset::Union{Set{Int},Vector{Int}})
    nodeset = Set{Int}()
    for c in cellset
        cell = getcells(grid, c)
        push!(nodeset, vertices(cell)...)
    end
    return nodeset
end
cellset_to_nodeset(grid::Grid, name::String) = cellset_to_nodeset(grid, getcellset(grid, name))

# Creating cellcentersets
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

# Project points nearly on circles to being exactly on them
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
project_circle(grid::Grid, circle::Circle, thresh) = project_circle!(deepcopy(grid), circle, thresh)

function project_circles!(grid::Grid, circles::Vector{C}, thresh) where {C <: Circle}
    for circle in circles
        project_circle!(grid, circle, thresh)
    end
    return grid
end
project_circles(grid::Grid, circles::Vector{C}, thresh) where {C <: Circle} = project_circles!(deepcopy(grid), circles, thresh)

# Form node positions matrix
function nodematrix(g::Grid{dim,N,T,M}) where {dim,N,T,M}
    p = zeros(T, getnnodes(g), dim) # dim is spatial dimension of grid
    @inbounds for i in 1:getnnodes(g)
        x = getcoordinates(getnodes(g)[i])
        for j in 1:dim
            p[i,j] = x[j]
        end
    end
    return p
end
nodevector(g::Grid) = getcoordinates.(getnodes(g))

# Form triangle indices matrix
function cellmatrix(g::Grid{dim,N,T,M}) where {dim,N,T,M}
    c = zeros(Int, getncells(g), N) # N is number of nodes per cell
    @inbounds for i in 1:getncells(g)
        x = vertices(getcells(g)[i])
        for j in 1:N
            c[i,j] = x[j]
        end
    end
    return c
end
cellvector(g::Grid) = vertices.(getcells(g))

# Return combined nodematrix and cellmatrix of a vector of grids,
# renumbering nodes accordingly
function nodecellmatrices(gs::Vector{G}) where {G <: Grid{2,3}}
    ps = nodematrix.(gs) # Vector of matrices of node positions
    ts = cellmatrix.(gs) # Vector of matrices of triangle indices
    idxshifts = cumsum(size.(ps,1))
    for i in 2:length(ts)
        ts[i] .+= idxshifts[i-1] # correct for indices shifts
    end
    p = reduce(vcat, ps) # Node positions matrix
    t = reduce(vcat, ts) # Triangle indices matrix
    return p, t
end

# Generic forming of a subgrid from a cellset + nodeset + boundaryset of a parent grid
function form_subgrid(
        parent_grid::Grid{dim,N,T,M},
        cellset::Set{Int},
        nodeset::Set{Int},
        boundaryset::Set{Tuple{Int,Int}}
    ) where {dim,N,T,M}

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

# Form interior, tori, and exterior subgrids from a parent grid given the
# rectangular exterior boundary and inner/outer circle boundaries
function form_tori_subgrids(
        fullgrid::Grid{dim,N,T,M},
        rect_bdry::Rectangle{2,T},
        inner_circles::Vector{Circle{2,T}},
        outer_circles::Vector{Circle{2,T}}
    ) where {dim,N,T,M}

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

# Form interior, tori, and exterior subgrids from a parent grid given the
# circular exterior boundary and inner/outer circle boundaries
function form_tori_subgrids(
        fullgrid::Grid{dim,N,T,M},
        circle_bdry::Circle{2,T},
        inner_circles::Vector{Circle{2,T}},
        outer_circles::Vector{Circle{2,T}}
    ) where {dim,N,T,M}

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
