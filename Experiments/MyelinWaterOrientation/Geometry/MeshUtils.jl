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
using DistMesh

export getfaces, simpplot, disjoint_rect_mesh_with_tori
export mxbbox, mxaxis

# ---------------------------------------------------------------------------- #
# Misc grid utils
# ---------------------------------------------------------------------------- #

# `JuAFEM.Grid` constructor given a vector of points and a vector of tuples of
# triangle vertices.
#NOTE: assumes triangles are properly oriented
function JuAFEM.Grid(
        p::AbstractVector{Vec{2,T}},
        t::AbstractVector{NTuple{3,Int}},
        e::AbstractVector{NTuple{2,Int}} = boundedges(p, t)
    ) where {T}

    # Create initial grid
    grid = Grid(Triangle.(t), Node.(Tuple.(p)))

    # Add boundary set and boundary matrix, including inner boundaries
    addfaceset!(grid, "boundary", Set{NTuple{2,Int}}(e))
    grid.boundary_matrix = JuAFEM.boundaries_to_sparse(e)

    return grid
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

# Form a subgrid from a cellset + nodeset + boundaryset of a parent grid
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

# ---------------------------------------------------------------------------- #
# simpplot: call the DistMesh function `simpplot` to plot the grid
# ---------------------------------------------------------------------------- #

# Form node positions matrix
function nodematrix(g::Grid{2,3,T}) where {T}
    p = zeros(T, getnnodes(g), 2)
    @inbounds for i in 1:getnnodes(g)
        p[i,1], p[i,2] = getcoordinates(getnodes(g)[i])
    end
    return p
end

# Form triangle indices matrix
function trimatrix(g::Grid{2,3,T}) where {T}
    t = zeros(Int, getncells(g), 3)
    @inbounds for i in 1:getncells(g)
        t[i,1], t[i,2], t[i,3] = vertices(getcells(g)[i])
    end
    return t
end

# Plot grid or vector of grids
function DistMesh.simpplot(gs::Vector{G}; kwargs...) where {G <: Grid{2,3}}
    ps = nodematrix.(gs) # Vector of matrices of node positions
    ts = trimatrix.(gs) # Vector of matrices of triangle indices
    idxshifts = cumsum(size.(ps,1))
    for i in 2:length(ts)
        ts[i] .+= idxshifts[i-1] # correct for indices shifts
    end
    p = reduce(vcat, ps) # Node positions matrix
    t = reduce(vcat, ts) # Triangle indices matrix
    simpplot(p, t; kwargs...) # Plot grid
    return nothing
end
DistMesh.simpplot(g::Grid{2,3}; kwargs...) = simpplot(nodematrix(g), trimatrix(g); kwargs...)

# ---------------------------------------------------------------------------- #
# Helper functions for DistMesh, etc.
# ---------------------------------------------------------------------------- #

@inline mxbbox(r::Rectangle{2}) = [xmin(r) ymin(r); xmax(r) ymax(r)]
@inline mxaxis(r::Rectangle{2}) = [xmin(r) xmax(r) ymin(r) ymax(r)]

# Clamping functions: η γ α
# -Provides a continuous transition from h = h0 to h = η*h0 when d goes from
#  abs(d) = h0 to abs(d) = γ*h0, with h clamped at h0 for abs(d) < h0 and
#  η*h0 for abs(d) > γ*h0
# -sqrt/linear/quad clamps are special cases of the power clamp with exponent α
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

function tile_rectangle(r::Rectangle{2}, tiling = (1,1))
    (tiling == (1,1)) && return [r] # Trivial case

    m, n = tiling
    xs = range(xmin(r), stop = xmax(r), length = m+1)
    ys = range(ymin(r), stop = ymax(r), length = n+1)

    R = typeof(r)
    rs = R[]
    for i in 1:m, j in 1:n
        push!(rs, R(Vec{2}((xs[i], ys[j])), Vec{2}((xs[i+1], ys[j+1]))))
    end

    return rs
end

# ---------------------------------------------------------------------------- #
# disjoint_rect_mesh_with_tori
# ---------------------------------------------------------------------------- #
function disjoint_rect_mesh_with_tori(
        rect_bdry::Rectangle{2,T},
        inner_circles::Vector{Circle{2,T}},
        outer_circles::Vector{Circle{2,T}},
        h_min::T, # minimum edge length
        h_max::T = h_min, # maximum edge length (default to uniform)
        h_range::T = T(10*h_min), # distance over which h increases from h_min to h_max
        h_rate::T = T(0.7); # rate of increase of h from circle boundaries (power law)
        exterior_tiling = (1,1), # tile exterior grid into (m,n) subgrids
        maxstalliters = 500, # default to no limit
        plotgrids = false, # plot resulting grids
        plotgridprogress = false # plot grids as they are created
    ) where {T}

    # Useful defines
    V, G = Vec{2,T}, Grid{2,3,T,3}
    h0 = h_min # typical h-value
    eta = T(h_max/h0) # approx ratio between largest/smallest edges, i.e. max ≈ eta * h0
    gamma = T(h_range/h0) # max edge length of `eta * h0` occurs approx. `gamma * h0` from circle edges
    alpha = h_rate # power law for edge length

    # Ensure that:
    # -there are the same number of outer/inner circles, and at least 1 of each
    # -outer circles strictly contain inner circles
    # -outer/inner circles have the same origins
    # -outer circles are strictly non-overlapping
    @assert length(inner_circles) == length(outer_circles) >= 1
    @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
    @assert all(c -> origin(c[1]) ≈ origin(c[2]), zip(inner_circles, outer_circles))
    @assert !is_any_overlapping(outer_circles, <)

    function circle_bdry_points(p, e, c, h0, thresh = h0/100)
        e_unique = unique!(sort!(copy(reinterpret(Int, e)))) # unique indices of boundary points
        p_bdry = filter(x -> is_on_circle(x, c, thresh), p[e_unique]) # keep points which are on `c`
        return p_bdry
    end

    # Project points onto circles/boundaries if they are within a distance `thresh`
    function fix_gridpoints!(p, thresh = h0/100)
        @inbounds for i in eachindex(p)
            for c in Iterators.flatten((inner_circles, outer_circles))
                dx = p[i] - origin(c)
                normdx = norm(dx)
                if abs(normdx - radius(c)) <= thresh
                    p[i] = origin(c) + (radius(c)/normdx) * dx
                end
            end
            x = xmin(rect_bdry); (x-thresh <= p[i][1] <= x+thresh) && (p[i] = V((x, p[i][2])))
            x = xmax(rect_bdry); (x-thresh <= p[i][1] <= x+thresh) && (p[i] = V((x, p[i][2])))
            y = ymin(rect_bdry); (y-thresh <= p[i][2] <= y+thresh) && (p[i] = V((p[i][1], y)))
            y = ymax(rect_bdry); (y-thresh <= p[i][2] <= y+thresh) && (p[i] = V((p[i][1], y)))
        end
        return p
    end

    # Initialize Grids
    exteriorgrids, interiorgrids, torigrids = G[], G[], G[]
    parent_circle_indices = Int[]

    # Fixed points for exteriorgrid
    pfix_ext = V[]

    @inbounds for i = 1:length(outer_circles)
        # Fixed points for inner/outer circles, as well as boundary points
        c_in, c_out = inner_circles[i], outer_circles[i]
        pfix_int, pfix_out, pfix_int_bdry, pfix_out_bdry = V[], V[], V[], V[]
        push!(parent_circle_indices, i)

        println("$i/$(length(outer_circles)): Interior")
        int_bdry = intersect(rect_bdry, bounding_box(c_in))
        if !(√area(int_bdry) ≈ zero(T))
            fd = x -> dintersect(drectangle0(x, int_bdry), dcircle(x, c_in))
            fh = x -> hcircle(x, h0, eta, gamma, alpha, c_in)
            pfix_int = vcat(pfix_int, intersection_points(rect_bdry, c_in)) # only fix w.r.t rect_bdry to avoid tangent points being fixed

            p, t = distmesh2d(
                fd, fh, h0, mxbbox(int_bdry), pfix_int;
                PLOT = plotgridprogress, MAXSTALLITERS = maxstalliters
            )
            fix_gridpoints!(p)

            e = boundedges(p, t)
            pfix_int_bdry = vcat(pfix_int_bdry, circle_bdry_points(p, e, c_in, h0))

            plotgrids && simpplot(p, t; newfigure = true)
            push!(interiorgrids, Grid(p, t, e))
        else
            push!(interiorgrids, Grid(Triangle[], Node{2,T}[]))
        end

        println("$i/$(length(outer_circles)): Annular")
        out_bdry = intersect(rect_bdry, bounding_box(c_out))
        if !(√area(out_bdry) ≈ zero(T))
            fd = x -> dintersect(drectangle0(x, out_bdry), dshell(x, c_in, c_out))
            fh = x -> hshell(x, h0, eta, gamma, alpha, c_in, c_out)
            pfix_out = vcat(pfix_out, pfix_int_bdry,
            intersection_points(rect_bdry, c_in),
            intersection_points(rect_bdry, c_out)) # only fix w.r.t rect_bdry to avoid tangent points being fixed

            p, t = distmesh2d(
                fd, fh, h0, mxbbox(out_bdry), pfix_out;
                PLOT = plotgridprogress, MAXSTALLITERS = maxstalliters
            )
            fix_gridpoints!(p)

            e = boundedges(p, t)
            pfix_out_bdry = vcat(pfix_out_bdry, circle_bdry_points(p, e, c_out, h0))
            pfix_ext = vcat(pfix_ext, pfix_out_bdry)

            plotgrids && simpplot(p, t; newfigure = true)
            push!(torigrids, Grid(p, t, e))
        else
            push!(torigrids, Grid(Triangle[], Node{2,T}[]))
        end
    end

    for (k, ext_bdry) in enumerate(tile_rectangle(rect_bdry, exterior_tiling))
        # Add intersection points of circles with sub-exterior
        pfix_sub_ext = copy(pfix_ext)
        for c in Iterators.flatten((inner_circles, outer_circles))
            pfix_sub_ext = vcat(pfix_sub_ext, intersection_points(ext_bdry, c))
        end

        # Keep unique points
        pfix_sub_ext = filter!(pfix_sub_ext) do p
            xmin(ext_bdry) <= p[1] <= xmax(ext_bdry) && ymin(ext_bdry) <= p[2] <= ymax(ext_bdry)
        end
        !isempty(pfix_sub_ext) && unique!(sort!(pfix_sub_ext; by = first))

        # Form exterior grid
        println("$k/$(prod(exterior_tiling)): Exterior")
        fd = x -> dexterior(x, ext_bdry, outer_circles)
        fh = x -> hcircles(x, h0, eta, gamma, alpha, outer_circles)

        p, t = distmesh2d(
            fd, fh, h0, mxbbox(ext_bdry), pfix_sub_ext;
            PLOT = plotgridprogress, MAXSTALLITERS = maxstalliters
        )
        fix_gridpoints!(p)

        push!(exteriorgrids, Grid(p, t))
        plotgrids && simpplot(p, t; newfigure = true)
    end

    return exteriorgrids, torigrids, interiorgrids, parent_circle_indices
end

# ---------------------------------------------------------------------------- #
# Form tori subgrids with rectangular boundary
# ---------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------- #
# Form tori subgrids with circular boundary
# ---------------------------------------------------------------------------- #
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


# ============================================================================ #
# ============================================================================ #
#
# MATLAB-based DistMesh grid generation; much slower and less tested
#
# ============================================================================ #
# ============================================================================ #


# ---------------------------------------------------------------------------- #
# MAT_rect_mesh_with_circles
# ---------------------------------------------------------------------------- #
function MAT_rect_mesh_with_circles(
        rect_bdry::Rectangle{2,T},
        circles::Vector{Circle{2,T}},
        h0::T,
        eta::T;
        isunion::Bool = true
    ) where {T}

    # TODO: add minimum angle threshold?
    dim = 2
    nfaces = 3 # per triangle
    nnodes = 3 # per triangle

    bbox = mxbbox(rect_bdry)
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
# MAT_disjoint_rect_mesh_with_tori
# ---------------------------------------------------------------------------- #
function MAT_disjoint_rect_mesh_with_tori(
        rect_bdry::Rectangle{2,T},
        inner_circles::Vector{Circle{2,T}},
        outer_circles::Vector{Circle{2,T}},
        h0::T,
        eta::T;
        fixcorners::Bool = true,
        fixcirclepoints::Bool = true
    ) where {T}

    # Ensure that outer circles strictly contain inner circles, and that outer
    # circles are strictly non-overlapping
    @assert length(inner_circles) == length(outer_circles)
    @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
    @assert !is_any_overlapping(outer_circles, <)

    println("0/$(length(outer_circles)): Exterior")
    exteriorgrid = form_disjoint_grid(rect_bdry, inner_circles, outer_circles, h0, eta, :exterior, fixcorners, fixcirclepoints)
    # exteriorgrid = Grid[]

    # interiorgrids = Grid[]
    # torigrids = Grid[]
    interiorgrids = typeof(exteriorgrid)[]
    torigrids = typeof(exteriorgrid)[]
    @inbounds for i = 1:length(outer_circles)
        println("$i/$(length(outer_circles)): Interior")
        new_bdry = intersect(rect_bdry, bounding_box(inner_circles[i]))
        if !(area(new_bdry) ≈ zero(T))
            push!(interiorgrids, form_disjoint_grid(new_bdry, [inner_circles[i]], [outer_circles[i]], h0, eta, :interior, fixcorners, fixcirclepoints))
        else
            push!(interiorgrids, Grid(Triangle[], Node{2,T}[]))
        end

        println("$i/$(length(outer_circles)): Annular")
        new_bdry = intersect(rect_bdry, bounding_box(outer_circles[i]))
        if !(area(new_bdry) ≈ zero(T))
            push!(torigrids, form_disjoint_grid(new_bdry, [inner_circles[i]], [outer_circles[i]], h0, eta, :tori, fixcorners, fixcirclepoints))
        else
            push!(torigrids, Grid(Triangle[], Node{2,T}[]))
        end
    end

    return exteriorgrid, torigrids, interiorgrids
end

# ---------------------------------------------------------------------------- #
# MAT_form_disjoint_grid
# ---------------------------------------------------------------------------- #
function MAT_form_disjoint_grid(
        rect_bdry::Rectangle{2,T},
        inner_circles::Vector{Circle{2,T}},
        outer_circles::Vector{Circle{2,T}},
        h0::T,
        eta::T,
        regiontype::Symbol,
        fixcorners::Bool = true,
        fixcirclepoints::Bool = true
    ) where {T}

    dim = 2
    nargout = 2
    isunion = false
    to_array(cs) = reinterpret(T, origin.(cs), (dim, length(cs))) |> transpose |> Matrix
    outer_centers = to_array(outer_circles)
    inner_centers = to_array(inner_circles)
    outer_radii   = radius.(outer_circles)
    inner_radii   = radius.(inner_circles)

    bbox = mxbbox(rect_bdry)
    if regiontype == :exterior
        regnumber = 1.0
        p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber)
    elseif regiontype == :tori
        regnumber = 2.0
        p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber, inner_centers, inner_radii, fixcorners, fixcirclepoints)
    elseif regiontype == :interior
        regnumber = 3.0
        p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber, inner_centers, inner_radii, fixcorners, fixcirclepoints)
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

# ---------------------------------------------------------------------------- #
# MAT_rect_mesh_with_tori
# ---------------------------------------------------------------------------- #
function MAT_rect_mesh_with_tori(
        rect_bdry::Rectangle{2,T},
        inner_circles::Vector{Circle{2,T}},
        outer_circles::Vector{Circle{2,T}},
        h0::T,
        eta::T
    ) where {T}

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
    bbox = mxbbox(rect_bdry)
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

# ---------------------------------------------------------------------------- #
# MAT_circle_mesh_with_tori
# ---------------------------------------------------------------------------- #
function MAT_circle_mesh_with_tori(
        circle_bdry::Circle{2,T},
        inner_circles::Vector{Circle{2,T}},
        outer_circles::Vector{Circle{2,T}},
        h0::T,
        eta::T
    ) where {T}

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

end # module MeshUtils

nothing
