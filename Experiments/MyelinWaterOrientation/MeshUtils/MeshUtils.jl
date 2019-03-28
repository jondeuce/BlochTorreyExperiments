# ============================================================================ #
# Tools for working with circles within JuAFEM.jl Grid's
# ============================================================================ #

module MeshUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #
using GeometryUtils
using DistMesh
using VoronoiDelaunay

using JuAFEM
using JuAFEM: vertices, faces, edges
using SparseArrays, Statistics

using RecipesBase
# using MATLAB # Only need for MAT_* methods, which are no longer used

export getfaces, simpplot, disjoint_rect_mesh_with_tori
export nodevector, nodematrix, cellvector, cellmatrix, nodecellmatrices
export mxbbox, mxaxis

# DEBUG
export dcircles
# export dexterior, hcircles, hcircle, hshell
# export sqrtclamp, linearclamp, quadclamp, powerclamp

# ---------------------------------------------------------------------------- #
# Misc grid utils
# ---------------------------------------------------------------------------- #

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
        # CellType = eltype(cells[i]) # cell type of i'th grid
        # shift = idxshifts[i-1] # shift all integer indices
        # map!(c -> CellType(c.nodes .+ shift), cells[i], cells[i])
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
        copy(reinterpret(NTuple{N,Int}, getcells(g)))
    )
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

# Plot Grid or vector of Grids. Vector of Grids are combined into one large Grid
# before plotting for speed, so that simpplot need only be called once
function DistMesh.SimpPlotGrid(gs::Vector{G}) where {G <: Grid{2,3}}
    return DistMesh.SimpPlotGrid(nodecellmatrices(gs)...)
end
function DistMesh.SimpPlotGrid(g::G) where {G <: Grid{2,3}}
    return DistMesh.SimpPlotGrid(nodevector(g), cellvector(g))
end

# ---------------------------------------------------------------------------- #
# Helper functions for DistMesh, etc.
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
        plotgrids = false, # plot resulting grids
        plotgridprogress = false, # plot grids as they are created
        CIRCLESTALLITERS = 1000, # stall iterations for interior/tori grids
        EXTERIORSTALLITERS = 500, # stall iterations for exterior grids
    ) where {T}

    # Useful defines
    V, G = Vec{2,T}, Grid{2,3,T,3}
    D_BDRY = h_min*sqrt(eps(T)) # points within D_BDRY from boundary edges are deemed on the boundary
    D_CIRC = h_min # circles with at least D_CIRC within the bounding box are included

    # Ensure that:
    # -there are the same number of outer/inner circles, and at least 1 of each
    # -outer circles strictly contain inner circles
    # -outer/inner circles have the same origins
    # -outer circles are strictly non-overlapping
    @assert length(inner_circles) == length(outer_circles) >= 1
    @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
    @assert all(c -> origin(c[1]) ≈ origin(c[2]), zip(inner_circles, outer_circles))
    @assert !is_any_overlapping(outer_circles, <)

    function check_include_circle(c)
        # Allow circle if the circle is sufficiently inside the boundary rectangle;
        # here, if intersect(c,rect_bdry) has size above a threshold in all dimensions
        c_bounding_box = intersect(rect_bdry, bounding_box(c))
        return min(widths(c_bounding_box)...) ≥ D_CIRC
    end

    function circle_bdry_points(p::AbstractVector{V}, e, c, thresh = D_BDRY) where {V <: Vec{2}}
        e_unique = unique!(sort!(copy(reinterpret(Int, e)))) # unique indices of boundary points
        p_bdry = filter(x -> is_on_circle(x, c, thresh), p[e_unique]) # keep points which are on `c`
        return p_bdry
    end
    function circle_bdry_points(g::Grid{dim,N,T,M}, c, thresh = D_BDRY) where {dim,N,T,M}
        e_unique = unique!(sort!(copy(reinterpret(Int, boundedges(g)))))
        p_bdry_all = Vec{dim,T}[getcoordinates(getnodes(g)[i]) for i in e_unique]
        p_bdry = filter(x -> is_on_circle(x, c, thresh), p_bdry_all)
        return p_bdry
    end

    # Project points onto circles/boundaries if they are within a distance `thresh`
    function fix_gridpoints!(p, t,
            rect_bdry = rect_bdry,
            inner_circles = inner_circles,
            outer_circles = outer_circles
        )

        e = boundedges(p, t) # vectors of NTuple{2,Int}'s of boundary edge points
        bdry_indices = unique(reinterpret(Int, e)) # boundary point indices
        for i in bdry_indices
            pᵢ = p[i]
            d_in, idx_in = findmin([abs(dcircle(pᵢ, c)) for c in inner_circles])
            d_out, idx_out = findmin([abs(dcircle(pᵢ, c)) for c in outer_circles])
            d_rect = abs(drectangle0(pᵢ, rect_bdry))

            if d_rect ≤ min(d_in, d_out)
                dxL, dxR = abs(pᵢ[1] - xmin(rect_bdry)), abs(pᵢ[1] - xmax(rect_bdry))
                dyL, dyR = abs(pᵢ[2] - ymin(rect_bdry)), abs(pᵢ[2] - ymax(rect_bdry))
                _, idx = findmin((dxL, dxR, dyL, dyR))

                # Project onto nearest rectangle wall
                p[i] = idx == 1 ? V((xmin(rect_bdry), pᵢ[2])) :
                       idx == 2 ? V((xmax(rect_bdry), pᵢ[2])) :
                       idx == 3 ? V((pᵢ[1], ymin(rect_bdry))) :
                                  V((pᵢ[1], ymax(rect_bdry)))
            else
                # Project onto nearest circle
                c = d_in < d_out ? inner_circles[idx_in] : outer_circles[idx_out]
                dx = pᵢ - origin(c)
                p[i] = origin(c) + (radius(c)/norm(dx)) * dx
            end
        end

        return p
    end

    function remove_extra_boundary_points(p, t, p_allowed, p_input, fd, thresh = D_BDRY)
        p_to_remove = filter(x -> !any(norm(x-y) < thresh for y in p_allowed), p_input)
        if !isempty(p_to_remove)
            p = filter(x -> !any(norm(x-y) < thresh for y in p_to_remove), p)
            t = delaunay2(p) # new Delaunay triangulation
            pmid = V[(p[t[1]] + p[t[2]] + p[t[3]])/3 for t in t] # Compute centroids
            t = t[fd.(pmid) .< thresh] # Keep interior triangles
            p, t, _ = fixmesh(p, t)
        end
        return p, t
    end

    # Initialize interior/tori grids
    interiorgrids, torigrids = G[], G[]
    parent_circle_indices = collect(1:length(inner_circles))

    # Initialize exterior grid
    exteriorgrids = Matrix{G}(undef, exterior_tiling)
    tiled_ext_bdry = tile_rectangle(rect_bdry, exterior_tiling)

    # Initialize plot, if any
    local fighandle
    # plotgrids && (fighandle = plot(;seriestype = :simpplot))
    plotgrids && (fighandle = simpplot())

    @inbounds for i = 1:length(inner_circles)
        # Fixed points for inner/outer circles, as well as boundary points
        c_in = inner_circles[i]
        int_bdry = intersect(rect_bdry, bounding_box(c_in))

        println("$i/$(length(inner_circles)): Interior")
        if !check_include_circle(c_in)
            push!(interiorgrids, Grid(Triangle[], Node{2,T}[]))
            continue
        end

        halfwidth = min(radius(c_in)/5, 0.5 * min(widths(int_bdry)...))
        h0 = min(h_min, halfwidth)
        h1 = min((h_max/h_min) * h0, halfwidth) # preserve ratio eta, or clamp
        h2 = h_range # don't want to shrink h_range

        eta = h1/h0
        gamma = h2/h0
        alpha = h_rate

        fd = x -> dintersect(drectangle0(x, int_bdry), dcircle(x, c_in))
        fh = x -> hcircle(x, h0, eta, gamma, alpha, c_in)

        pfix = vcat(
            filter(x -> is_in_circle(x, c_in), [corners(rect_bdry)...]), # rect_bdry corners
            intersection_points(rect_bdry, c_in)
        )
        !isempty(pfix) && unique!(sort!(pfix; by = first))

        p, t = distmesh2d(
            fd, fh, h0, mxbbox(int_bdry), pfix;
            PLOT = plotgridprogress, MAXSTALLITERS = CIRCLESTALLITERS
        )
        fix_gridpoints!(p, t, rect_bdry)

        # @show length(p)
        # println("...DEBUG\n")

        # Push interior grid, and plot if requested
        push!(interiorgrids, Grid(p, t))
        if plotgrids
            simpplot!(fighandle, interiorgrids[end]; color = :yellow)
            # plot!(fighandle, Circle{2,T}(mean(p), 2*h_min); color = :white, seriestype = :shape, annotations = (mean(p)..., string(i)))
            plot!(fighandle; annotations = (mean(p)..., string(i)))
            display(fighandle)
        end
    end

    @inbounds for i = 1:length(outer_circles)
        # Fixed points for inner/outer circles, as well as boundary points
        c_in, c_out = inner_circles[i], outer_circles[i]
        out_bdry = intersect(rect_bdry, bounding_box(c_out))

        println("$i/$(length(outer_circles)): Annular")
        if !check_include_circle(c_out)
            push!(torigrids, Grid(Triangle[], Node{2,T}[]))
            continue
        end
        
        halfwidth = min(0.5 * (radius(c_out) - radius(c_in)), 0.5 * min(widths(out_bdry)...))
        h0 = min(h_min, halfwidth)
        # h1 = min((h_max/h_min) * h0, halfwidth) # preserve ratio eta, or clamp
        h1 = h0 # sheaths should just be high resolution layers. they only take a small portion of the domain, anyways
        h2 = h_range # don't want to shrink h_range

        eta = h1/h0
        gamma = h2/h0
        alpha = h_rate

        fd = x -> dintersect(drectangle0(x, out_bdry), dshell(x, c_in, c_out))
        fh = x -> hshell(x, h0, eta, gamma, alpha, c_in, c_out)

        pfix_prev = circle_bdry_points(interiorgrids[i], c_in)
        pfix = vcat(
            filter(x -> is_in_circle(x, c_out) && !is_in_circle(x, c_in), [corners(rect_bdry)...]), # rect_bdry corners
            intersection_points(rect_bdry, c_out), # only fix w.r.t rect_bdry to avoid tangent points being fixed
            pfix_prev # interior circle boundary points from inner grid
        )
        !isempty(pfix) && unique!(sort!(pfix; by = first))

        p, t = distmesh2d(
            fd, fh, h0, mxbbox(out_bdry), pfix;
            PLOT = plotgridprogress, MAXSTALLITERS = CIRCLESTALLITERS
        )
        fix_gridpoints!(p, t, rect_bdry)

        pfix_now = circle_bdry_points(p, boundedges(p,t), c_in)
        p, t = remove_extra_boundary_points(p, t, pfix_prev, pfix_now, fd)

        # @show length(p)
        # println("...DEBUG\n")

        # Push tori grid, and plot if requested
        push!(torigrids, Grid(p, t)) #push!(torigrids, Grid(p, t, e))
        plotgrids && display(simpplot!(fighandle, torigrids[end]; color = :blue))
    end

    for i in 1:exterior_tiling[1]
        for j in 1:exterior_tiling[2]
            # Get current exterior boundary
            ext_bdry = tiled_ext_bdry[i,j]

            # Fixed boundary points are those to the left/below current grid
            boundary_points_below_and_left = (p) -> begin
                p_bdry = V[]
                (j > 1) && (p_bdry = vcat(p_bdry, filter(x -> x[1] ≈ xmax(tiled_ext_bdry[i,j-1]), p)))
                (i > 1) && (p_bdry = vcat(p_bdry, filter(x -> x[2] ≈ ymax(tiled_ext_bdry[i-1,j]), p)))
                !isempty(p_bdry) && unique!(sort!(p_bdry; by = first))
                return p_bdry
            end

            fixed_boundary_points_below_and_left = () -> begin
                p_bdry = V[]
                (j > 1) && (p_bdry = vcat(p_bdry, boundary_points_below_and_left(getcoordinates.(getnodes(exteriorgrids[i,j-1])))))
                (i > 1) && (p_bdry = vcat(p_bdry, boundary_points_below_and_left(getcoordinates.(getnodes(exteriorgrids[i-1,j])))))
                !isempty(p_bdry) && unique!(sort!(p_bdry; by = first))
                return p_bdry
            end

            # Create Delaunay tessellation
            println("$i/$(exterior_tiling[1]), $j/$(exterior_tiling[2]): Exterior")
            h0 = h_min
            eta = h_max/h_min
            gamma = h_range/h_min
            alpha = h_rate
            
            fd = x -> dexterior(x, ext_bdry, outer_circles)
            fh = x -> hcircles(x, h0, eta, gamma, alpha, outer_circles)

            pfix_prev = vcat(
                fixed_boundary_points_below_and_left(), # fixed points from previous domain
                reduce(vcat, circle_bdry_points(torigrids[i], c_out) for (i, c_out) in enumerate(outer_circles)) # boundary points from outer circles
            )
            pfix = vcat(
                filter(x -> !is_in_any_circle(x, outer_circles), [corners(ext_bdry)...]), # ext_bdry corners, if not in an outer_circle
                pfix_prev
            )
            !isempty(pfix) && unique!(sort!(filter!(p->is_inside(p, ext_bdry), pfix); by = first))

            p, t = distmesh2d(
                fd, fh, h0, mxbbox(ext_bdry), pfix;
                PLOT = plotgridprogress, MAXSTALLITERS = EXTERIORSTALLITERS
            )
            fix_gridpoints!(p, t, ext_bdry)

            e = boundedges(p,t)
            pfix_now = reduce(vcat, circle_bdry_points(p, e, c_out) for c_out in outer_circles)
            p, t = remove_extra_boundary_points(p, t, pfix_prev, pfix_now, fd)

            # Form exterior grid, and plot if requested
            exteriorgrids[i,j] = Grid(p, t)
            plotgrids && display(simpplot!(fighandle, exteriorgrids[i,j]; color = :cyan))
        end
    end

    # Plot resulting grid
    plotgrids && display(fighandle)

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
# function MAT_rect_mesh_with_circles(
#         rect_bdry::Rectangle{2,T},
#         circles::Vector{Circle{2,T}},
#         h0::T,
#         eta::T;
#         isunion::Bool = true
#     ) where {T}
# 
#     # TODO: add minimum angle threshold?
#     dim = 2
#     nfaces = 3 # per triangle
#     nnodes = 3 # per triangle
# 
#     bbox = mxbbox(rect_bdry)
#     centers = reinterpret(T, origin.(circles), (dim,length(circles)))'
#     radii = radius.(circles)
# 
#     nargout = 2
#     p, t = mxcall(:squaremeshwithcircles, nargout, bbox, centers, radii, h0, eta, isunion)
# 
#     cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
#     nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
#     fullgrid = Grid(cells, nodes)
# 
#     # Manually add boundary sets for the four square edges and circle boundaries
#     addfaceset!(fullgrid, "left",   x -> x[1] ≈ xmin(rect_bdry), all=true)
#     addfaceset!(fullgrid, "right",  x -> x[1] ≈ xmax(rect_bdry), all=true)
#     addfaceset!(fullgrid, "top",    x -> x[2] ≈ ymax(rect_bdry), all=true)
#     addfaceset!(fullgrid, "bottom", x -> x[2] ≈ ymin(rect_bdry), all=true)
#     addfaceset!(fullgrid, "circles", x -> is_on_any_circle(x, circles), all=true)
# 
#     # Boundary matrix and boundary face set
#     all_boundaries = union(values.(getfacesets(fullgrid))...)
#     fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
#     addfaceset!(fullgrid, "boundary", all_boundaries)
# 
#     if isunion
#         # Generate cell and node sets
#         addcellset!(fullgrid, "exterior", x -> !is_in_any_circle(x, circles); all=false)
#         addnodeset!(fullgrid, "exterior", x -> !is_in_any_circle(x, circles) || is_on_any_circle(x, circles))
#         addcellset!(fullgrid, "circles",  x ->  is_in_any_circle(x, circles); all=true)
#         addnodeset!(fullgrid, "circles",  x ->  is_in_any_circle(x, circles) || is_on_any_circle(x, circles))
# 
#         # Generate exterior grid
#         subgrids = typeof(fullgrid)[]
#         cellset = getcellset(fullgrid, "exterior")
#         nodeset = getnodeset(fullgrid, "exterior")
#         push!(subgrids, form_subgrid(fullgrid, cellset, nodeset, all_boundaries))
# 
#         # Generate circle grids
#         nodefilter = (nodenum, circle)  -> is_in_circle(getcoordinates(getnodes(fullgrid, nodenum)), circle)
#         cellfilter = (cellnum, nodeset) -> all(nodenum -> nodenum ∈ nodeset, vertices(getcells(fullgrid, cellnum)))
#         for circle in circles
#             nodeset = filter(nodenum -> nodefilter(nodenum, circle), getnodeset(fullgrid, "circles"))
#             cellset = filter(cellnum -> cellfilter(cellnum, nodeset), getcellset(fullgrid, "circles"))
#             push!(subgrids, form_subgrid(fullgrid, cellset, nodeset, all_boundaries))
#         end
#     else
#         subgrids = typeof(fullgrid)[]
#     end
# 
#     return fullgrid, subgrids
# end

# ---------------------------------------------------------------------------- #
# MAT_disjoint_rect_mesh_with_tori
# ---------------------------------------------------------------------------- #
# function MAT_disjoint_rect_mesh_with_tori(
#         rect_bdry::Rectangle{2,T},
#         inner_circles::Vector{Circle{2,T}},
#         outer_circles::Vector{Circle{2,T}},
#         h0::T,
#         eta::T;
#         fixcorners::Bool = true,
#         fixcirclepoints::Bool = true
#     ) where {T}
# 
#     # Ensure that outer circles strictly contain inner circles, and that outer
#     # circles are strictly non-overlapping
#     @assert length(inner_circles) == length(outer_circles)
#     @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
#     @assert !is_any_overlapping(outer_circles, <)
# 
#     println("0/$(length(outer_circles)): Exterior")
#     exteriorgrid = form_disjoint_grid(rect_bdry, inner_circles, outer_circles, h0, eta, :exterior, fixcorners, fixcirclepoints)
#     # exteriorgrid = Grid[]
# 
#     # interiorgrids = Grid[]
#     # torigrids = Grid[]
#     interiorgrids = typeof(exteriorgrid)[]
#     torigrids = typeof(exteriorgrid)[]
#     @inbounds for i = 1:length(outer_circles)
#         println("$i/$(length(outer_circles)): Interior")
#         new_bdry = intersect(rect_bdry, bounding_box(inner_circles[i]))
#         if !(area(new_bdry) ≈ zero(T))
#             push!(interiorgrids, form_disjoint_grid(new_bdry, [inner_circles[i]], [outer_circles[i]], h0, eta, :interior, fixcorners, fixcirclepoints))
#         else
#             push!(interiorgrids, Grid(Triangle[], Node{2,T}[]))
#         end
# 
#         println("$i/$(length(outer_circles)): Annular")
#         new_bdry = intersect(rect_bdry, bounding_box(outer_circles[i]))
#         if !(area(new_bdry) ≈ zero(T))
#             push!(torigrids, form_disjoint_grid(new_bdry, [inner_circles[i]], [outer_circles[i]], h0, eta, :tori, fixcorners, fixcirclepoints))
#         else
#             push!(torigrids, Grid(Triangle[], Node{2,T}[]))
#         end
#     end
# 
#     return exteriorgrid, torigrids, interiorgrids
# end

# ---------------------------------------------------------------------------- #
# MAT_form_disjoint_grid
# ---------------------------------------------------------------------------- #
# function MAT_form_disjoint_grid(
#         rect_bdry::Rectangle{2,T},
#         inner_circles::Vector{Circle{2,T}},
#         outer_circles::Vector{Circle{2,T}},
#         h0::T,
#         eta::T,
#         regiontype::Symbol,
#         fixcorners::Bool = true,
#         fixcirclepoints::Bool = true
#     ) where {T}
# 
#     dim = 2
#     nargout = 2
#     isunion = false
#     to_array(cs) = reinterpret(T, origin.(cs), (dim, length(cs))) |> transpose |> Matrix
#     outer_centers = to_array(outer_circles)
#     inner_centers = to_array(inner_circles)
#     outer_radii   = radius.(outer_circles)
#     inner_radii   = radius.(inner_circles)
# 
#     bbox = mxbbox(rect_bdry)
#     if regiontype == :exterior
#         regnumber = 1.0
#         p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber)
#     elseif regiontype == :tori
#         regnumber = 2.0
#         p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber, inner_centers, inner_radii, fixcorners, fixcirclepoints)
#     elseif regiontype == :interior
#         regnumber = 3.0
#         p, t = mxcall(:squaremeshwithcircles, nargout, bbox, outer_centers, outer_radii, h0, eta, isunion, regnumber, inner_centers, inner_radii, fixcorners, fixcirclepoints)
#     else
#         error("Invalid regiontype == $regiontype.")
#     end
# 
#     cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
#     nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
#     grid = Grid(cells, nodes)
# 
#     # Ensure points near circles are exactly on circles
#     project_circles!(grid, inner_circles, 1e-6*h0)
#     project_circles!(grid, outer_circles, 1e-6*h0)
# 
#     # Manually add boundary sets for the four square edges and circle boundaries
#     is_on_outer_circles = x -> is_on_any_circle(x, outer_circles)
#     is_on_inner_circles = x -> is_on_any_circle(x, inner_circles)
#     is_on_rectangle     = x -> x[1] ≈ xmin(rect_bdry) || x[1] ≈ xmax(rect_bdry) ||
#                                x[2] ≈ ymax(rect_bdry) || x[2] ≈ ymin(rect_bdry)
#     is_boundary = x -> is_on_outer_circles(x) || is_on_inner_circles(x) || is_on_rectangle(x)
# 
#     # Boundary matrix (including inner boundaries) and boundary face set
#     addfaceset!(grid, "boundary", is_boundary, all=true)
#     grid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, getfaceset(grid, "boundary")))
# 
#     return grid
# end

# ---------------------------------------------------------------------------- #
# MAT_rect_mesh_with_tori
# ---------------------------------------------------------------------------- #
# function MAT_rect_mesh_with_tori(
#         rect_bdry::Rectangle{2,T},
#         inner_circles::Vector{Circle{2,T}},
#         outer_circles::Vector{Circle{2,T}},
#         h0::T,
#         eta::T
#     ) where {T}
# 
#     # Ensure that outer circles strictly contain inner circles, and that outer
#     # circles are strictly non-overlapping
#     @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
#     @assert !is_any_overlapping(outer_circles, <)
# 
#     dim = 2
#     nfaces = 3 # per triangle
#     nnodes = 3 # per triangle
# 
#     # TODO: add minimum angle threshold?
#     nargout = 2
#     all_circles = vcat(outer_circles, inner_circles)
#     all_centers = reinterpret(T, origin.(all_circles), (dim, length(all_circles)))'
#     all_radii   = radius.(all_circles)
#     bbox = mxbbox(rect_bdry)
#     p, t = mxcall(:squaremeshwithcircles, nargout, bbox, all_centers, all_radii, h0, eta, isunion)
# 
#     cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
#     nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
#     fullgrid = Grid(cells, nodes)
# 
#     # Ensure points near circles are exactly on circles
#     project_circles!(fullgrid, inner_circles, 1e-6*h0)
#     project_circles!(fullgrid, outer_circles, 1e-6*h0)
# 
#     # Manually add boundary sets for the four square edges and circle boundaries
#     addfaceset!(fullgrid, "left",   x -> x[1] ≈ xmin(rect_bdry), all=true)
#     addfaceset!(fullgrid, "right",  x -> x[1] ≈ xmax(rect_bdry), all=true)
#     addfaceset!(fullgrid, "top",    x -> x[2] ≈ ymax(rect_bdry), all=true)
#     addfaceset!(fullgrid, "bottom", x -> x[2] ≈ ymin(rect_bdry), all=true)
#     addfaceset!(fullgrid, "inner_circles", x -> is_on_any_circle(x, inner_circles), all=true)
#     addfaceset!(fullgrid, "outer_circles", x -> is_on_any_circle(x, outer_circles), all=true)
# 
#     # Boundary matrix (including inner boundaries) and boundary face set
#     all_boundaries = union(values.(getfacesets(fullgrid))...)
#     fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
#     addfaceset!(fullgrid, "boundary", all_boundaries)
# 
#     return fullgrid
# end

# ---------------------------------------------------------------------------- #
# MAT_circle_mesh_with_tori
# ---------------------------------------------------------------------------- #
# function MAT_circle_mesh_with_tori(
#         circle_bdry::Circle{2,T},
#         inner_circles::Vector{Circle{2,T}},
#         outer_circles::Vector{Circle{2,T}},
#         h0::T,
#         eta::T
#     ) where {T}
# 
#     # Ensure that outer circles strictly contain inner circles, and that outer
#     # circles are strictly non-overlapping
#     @assert all(c -> is_inside(c[1], c[2], <), zip(inner_circles, outer_circles))
#     @assert !is_any_overlapping(outer_circles, <)
# 
#     dim = 2 # grid dimension
#     nfaces = 3 # per triangle
#     nnodes = 3 # per triangle
# 
#     # TODO: add minimum angle threshold?
#     nargout = 2
#     isunion = true
#     regiontype = 0 # union type
#     bcircle = [origin(circle_bdry)..., radius(circle_bdry)]
#     outer_centers = copy(transpose(reshape(reinterpret(T, origin.(outer_circles)), (dim, length(outer_circles)))))
#     inner_centers = copy(transpose(reshape(reinterpret(T, origin.(inner_circles)), (dim, length(inner_circles)))))
#     p, t = mxcall(:circularmeshwithtori, nargout,
#         bcircle, outer_centers, radius.(outer_circles), inner_centers, radius.(inner_circles),
#         h0, eta, isunion, regiontype )
# 
#     cells = [Triangle((t[i,1], t[i,2], t[i,3])) for i in 1:size(t,1)]
#     nodes = [Node((p[i,1], p[i,2])) for i in 1:size(p,1)]
#     fullgrid = Grid(cells, nodes)
# 
#     # Ensure points near circles are exactly on circles
#     project_circle!(fullgrid, circle_bdry, 1e-6*h0)
#     project_circles!(fullgrid, inner_circles, 1e-6*h0)
#     project_circles!(fullgrid, outer_circles, 1e-6*h0)
# 
#     # Manually add boundary sets for the four square edges and circle boundaries
#     addfaceset!(fullgrid, "boundary_circle", x -> is_on_circle(x, circle_bdry), all=true)
#     addfaceset!(fullgrid, "inner_circles",   x -> is_on_any_circle(x, inner_circles), all=true)
#     addfaceset!(fullgrid, "outer_circles",   x -> is_on_any_circle(x, outer_circles), all=true)
# 
#     # Boundary matrix (including inner boundaries) and boundary face set
#     all_boundaries = union(values.(getfacesets(fullgrid))...)
#     fullgrid.boundary_matrix = JuAFEM.boundaries_to_sparse(collect(Tuple{Int,Int}, all_boundaries))
#     addfaceset!(fullgrid, "boundary", all_boundaries)
# 
#     return fullgrid
# end

end # module MeshUtils

nothing
