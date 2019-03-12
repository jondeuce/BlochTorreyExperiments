# ---------------------------------------------------------------------------- #
# DistMesh helper functions
# ---------------------------------------------------------------------------- #

@inline huniform(x::Vec{dim,T}) where {dim,T} = one(T)#eltype(eltype(x)))
@inline norm2(x::Vec) = x⋅x
@inline Base.Tuple(v::Vec) = Tensors.get_data(v)

@inline function triangle_area(p1::Vec{2}, p2::Vec{2}, p3::Vec{2})
    d12, d13 = p2 - p1, p3 - p1
    return (d12[1] * d13[2] - d12[2] * d13[1])/2
end
@inline triangle_area(t::NTuple{3,Int}, p::AbstractArray{V}) where {V<:Vec{2}} = triangle_area(p[t[1]], p[t[2]], p[t[3]])

@inline triangle_quality(a, b, c) = (b+c-a)*(c+a-b)*(a+b-c)/(a*b*c)
@inline triangle_quality(p1::Vec{2}, p2::Vec{2}, p3::Vec{2}) = triangle_quality(norm(p2-p1), norm(p3-p2), norm(p1-p3))
@inline triangle_quality(t::NTuple{3,Int}, p::AbstractArray{V}) where {V<:Vec{2}} = triangle_quality(p[t[1]], p[t[2]], p[t[3]])

function mesh_rho(p::AbstractVector{V}, t::AbstractVector{NTuple{3,Int}}) where {V <: Vec{2}}
    edge_min, edge_max = extrema([norm(p[t[1]] - p[t[2]]) for t in t])
    return edge_max/edge_min
end
mesh_quality(p::AbstractVector{V}, t::AbstractVector{NTuple{3,Int}}) where {V <: Vec{2}} = minimum(triangle_quality(t,p) for t in t)

function to_vec(p::AbstractMatrix{T}) where {T}
    N = size(p, 2)
    P = reinterpret(Vec{N,T}, transpose(p)) |> vec |> copy
    return P
end

function to_tuple(p::AbstractMatrix{T}) where {T}
    N = size(p, 2)
    P = reinterpret(NTuple{N,T}, transpose(p)) |> vec |> copy
    return P
end

function to_mat(P::AbstractVector{NTuple{N,T}}) where {N,T}
    M = length(P)
    p = reshape(reinterpret(T, P), (N,M)) |> transpose |> copy
    return p
end
to_mat(P::AbstractVector{Vec{N,T}}) where {N,T} = to_mat(reinterpret(NTuple{N,T}, P))

function init_points(bbox::Matrix{T}, h0::T) where {T}
    # xrange, yrange = bbox[1,1]:h0:bbox[2,1], bbox[1,2]:h0*sqrt(T(3))/2:bbox[2,2]
    Nx = ceil(Int, (bbox[2,1]-bbox[1,1])/h0)
    Ny = ceil(Int, (bbox[2,2]-bbox[1,2])/(h0*sqrt(T(3))/2))
    xrange, yrange = range(bbox[1,1], bbox[2,1], length=Nx), range(bbox[1,2], bbox[2,2], length=Ny)
    p = zeros(Vec{2,T}, Ny, Nx)
    @inbounds for (i,y) in enumerate(yrange), (j,x) in enumerate(xrange)
        iseven(i) && (x += h0/2)            # Shift even rows
        p[i,j] = Vec{2,T}((x,y))            # Add to list of node coordinates
    end
    return vec(p)
end

#HGEOM - Mesh size function based on medial axis distance
#--------------------------------------------------------------------
# Inputs:
#    p     : points coordinates
#    fd    : signed distance function
#    pmax  : approximate medial axis points
#
# output:
#    fh    : mesh size function of points p
#--------------------------------------------------------------------
#  (c) 2011, Koko J., ISIMA, koko@isima.fr
#--------------------------------------------------------------------
function hgeom(
        fd, # distance function
        h0::T, # nominal edge length
        bbox::Matrix{T}, # bounding box (2x2 matrix [xmin ymin; xmax ymax])
        pfix::AbstractVector{Vec{2,T}} = Vec{2,T}[]; # fixed points
        alpha::T = T(1.0), # Relative edge lengths constant; alpha = 0.25 -> rel. length = 5X; 0.5 -> 3X; 1.0 -> 2X
        rtol::T = sqrt(eps(T)) # relative tolerance for medial axis search
    ) where {T}

    # Make and return hgeom
    pmedial, pgrid = medial_axis_search(fd, h0/4, bbox)
    pmedial = unique!(sort!([pfix; pmedial])) # Prepend (unique) fixed points
    max_fh1 = maximum(x->abs(fd(x)), pgrid)
    max_fh2 = sqrt(maximum(x->minimum(y->norm2(x-y), pmedial), pgrid))

    function hgeom(x)
        fh1 = fd(x)/max_fh1 # Normalized signed distance fuction
        fh2 = sqrt(minimum(y->norm2(x-y), pmedial))/max_fh2 # Normalized medial axis distance
        fh = alpha + min(abs(fh1), one(T)) + min(fh2, one(T)) # Size function (capped at 1 since max_fh1/max_fh2 are approximate)
        return fh
    end

    return hgeom
end

function medial_axis(
        fd, # distance function
        h0::T, # nominal edge length
        bbox::Matrix{T} # bounding box (2x2 matrix [xmin ymin; xmax ymax])
    ) where {T}

    # Numerical gradient (NOTE: this is for finding discontinuities in the gradient; can't use auto-diff!)
    V = Vec{2,T}
    e1, e2 = basevec(V)
    @inline dfd_dx(x,h) = (fd(x + h*e1) - fd(x - h*e1))/(2h)
    @inline dfd_dy(x,h) = (fd(x + h*e2) - fd(x - h*e2))/(2h)
    @inline    ∇fd(x,h) = V((dfd_dx(x,h), dfd_dy(x,h)))

    # Compute the approximate medial axis
    pcandidate = V[]
    local pgrid
    while isempty(pmedial) # shouldn't ever take more than one iteration
        pgrid = cartesian_grid_generator(bbox, h0)
        pmedial = collect(Iterators.filter(x -> fd(x) <= 0 && norm(∇fd(x,h0/8)) < 0.99, pgrid))
        isempty(pmedial) && (h0 /= 2)
    end

    return pmedial, pgrid
end

function medial_axis_search(
        fd, # distance function
        h0::T, # nominal edge length
        bbox::Matrix{T}, # bounding box (2x2 matrix [xmin ymin; xmax ymax])
        ∇fd = x -> Tensors.gradient(fd, x); # gradient function
        rtol::T = sqrt(eps(T))
    ) where {T}

    V = Vec{2,T}
    e1, e2 = basevec(V)
    
    @inline isinterior(x) = fd(x) <= 0
    @inline iscontinuous(x,h) = (dx = h*∇fd(x); ∇fd(x+dx)⋅∇fd(x-dx) > 1 - (h/h0)^2) # scale-invariant adaptive thresholding
    @inline ismedial(x,h) = isinterior(x) && !iscontinuous(x,h)  

    # Search for candidate approximate medial axis points
    pcandidate = V[]
    local pgrid
    while isempty(pcandidate) # shouldn't ever take more than a couple iterations
        pgrid = cartesian_grid_generator(bbox, h0)
        pcandidate = collect(Iterators.filter(x->ismedial(x,h0/2), pgrid))
        isempty(pcandidate) && (h0 /= 2)
    end
    
    # Binary search for medial axis point on the interval [x0-h0/2*∇x0, x0+h0/2*∇x0].
    # Returns exactly one found point, or nothing (does not search for multiple points in the interval)
    function binarysearch(x0,h0,e=∇fd(x0))::Union{V,Nothing}
        x, h = x0, h0/2 # Candidate x satisfies ismedial(x,h/2)
        while (h > rtol * h0) && !(x == nothing)
            s = (h/2)*e
            x = ismedial(x-s,h/2) ? x-s :
                ismedial(x+s,h/2) ? x+s :
                nothing
            h /= 2
        end
        return x
    end

    pmedial = V[]
    sizehint!(pmedial, length(pcandidate))
    for p0 in pcandidate
        p = binarysearch(p0,h0)
        !(p == nothing) && push!(pmedial, p::V)
    end
    resize!(pmedial, length(pmedial))
    pmedial = threshunique(pmedial, norm2; rtol = zero(T), atol = h0^2)
    
    @show length(pcandidate)
    @show length(pmedial)

    return pmedial, pgrid
end

function cartesian_grid_generator(bbox::Matrix{T}, h0::T; vec=false) where {T}
    @assert size(bbox) == (2,2)
    Nx = round(Int, (bbox[2,1]-bbox[1,1])/h0) + 1
    Ny = round(Int, (bbox[2,2]-bbox[1,2])/h0) + 1
    xrange = range(bbox[1,1], bbox[2,1], length=Nx)
    yrange = range(bbox[1,2], bbox[2,2], length=Ny)
    g = vec ? (Vec{2,T}((x,y)) for x in xrange for y in yrange) : # generates a (Nx*Ny)-length vector
              (Vec{2,T}(x) for x in Iterators.product(xrange, yrange)) # generates a Nx x Ny 2d grid
    return g
end

# ---------------------------------------------------------------------------- #
# Unique, sorting, etc.
# ---------------------------------------------------------------------------- #

# NOTE: Slow, and likely suboptimal
function threshunique(
        x::AbstractVector{T},
        norm = LinearAlgebra.norm;
        rtol = √eps(T),
        atol = eps(T),
    ) where {T}

    uniqueset = Vector{T}()
    sizehint!(uniqueset, length(x))
    ex = eachindex(x)
    # idxs = Vector{eltype(ex)}()
    for i in ex
        xi = x[i]
        norm_xi = norm(xi)
        isunique = true
        for xj in uniqueset
            norm_xj = norm(xj)
            if norm(xi - xj) < max(rtol*max(norm_xi, norm_xj), atol)
                isunique = false
                break
            end
        end
        isunique && push!(uniqueset, xi)
        # push!(idxs, i)
    end

    return uniqueset
end

# NOTE: Much faster than threshunique
# function gridunique(
#         x::AbstractVector{Vec{2,T}};
#         rtol = √eps(T),
#         atol = eps(T)
#     ) where {T}
# 
#     exts = [extrema(x[i] for x in x) for i in 1:2]
#     mins = ntuple(i -> exts[i][1], 2)
#     widths = ntuple(i -> exts[i][2] - exts[i][1], 2)
#     h0 = max.(atol, rtol.*widths)
#     Ns = ceil.(widths./h0)
#     hs = widths./Ns
# 
#     # to_tup(x, mins, hs) = round.((Tuple(x) .- mins) ./ hs)
#     # to_vec(t, mins, hs) = Vec(mins .+ hs .* t)
#     to_tup(x, hs) = round.(Tuple(x) ./ hs) #DEBUG
#     to_vec(t, hs) = Vec(hs .* t)
#     
#     #DEBUG
#     if any(isnan, first.(x)) #isnan(rtol) || isnan(ptol)
#         println("ERROR: NaN within `gridunique` before `to_tup`")
#     end
# 
#     # idx = to_tup.(x, Ref(mins), Ref(hs)) |> sort! |> unique!
#     idx = to_tup.(x, Ref(hs)) |> sort! |> unique! #DEBUG
# 
#     #DEBUG
#     if any(isnan, first.(idx)) #isnan(rtol) || isnan(ptol)
#         @show exts
#         @show mins
#         @show widths
#         @show h0
#         @show Ns
#         @show hs
#         println("ERROR: NaN within `gridunique` after `to_tup`")
#     end
# 
#     return to_vec.(idx, Ref(mins), Ref(hs))
# end

# NOTE: Much faster than threshunique
gridunique(x::AbstractVector{Vec{2,T}}, h0::T) where {T} = map(x->Vec{2,T}(h0 .* round.(Tuple(x) ./ h0)), x)

function findunique(A::AbstractArray{T}) where T
    ex = eachindex(A); Ti = eltype(ex)
    C = unique(A) # Must preserve order
    Cset = Set{T}(); sizehint!(Cset, length(C))
    Cmap = Dict(zip(C, Ti(1):Ti(length(C))))
    iA = Ti[]; sizehint!(iA, length(A))
    iC = Ti[]; sizehint!(iC, length(C))
    @inbounds for i in ex
        ai = A[i]
        push!(iC, Cmap[ai])
        if !(ai in Cset)
            push!(iA, i)
            push!(Cset, ai)
        end
    end
    C, iA, iC
end

# Extremely slow, conceptually simple version
function findunique_slow(A)
    C = unique(A)
    iA = findfirst.(isequal.(C), (A,))
    iC = findfirst.(isequal.(A), (C,))
    return C, iA, iC
end

# Simple sorting of 2-tuples (optimal sorting network from https://github.com/JeffreySarnoff/SortingNetworks.jl)
@inline function sorttuple(a, b)
    a, b = minmax(a, b)
    return (a, b)
end
@inline sorttuple(t::NTuple{2}) = sorttuple(t[1], t[2])

# Simple sorting of 3-tuples (optimal sorting network from https://github.com/JeffreySarnoff/SortingNetworks.jl)
function sorttuple(a, b, c)
    b, c = minmax(b, c)
    a, c = minmax(a, c)
    a, b = minmax(a, b)
    return (a, b, c)
end
@inline sorttuple(t::NTuple{3}) = sorttuple(t[1], t[2], t[3])

# edges of triangle vector
function getedges!(e::AbstractVector{NTuple{2,Int}}, t::AbstractVector{NTuple{3,Int}}; sorted = false)
    @assert length(e) == 3*length(t)
    if !sorted
        # Standard "counter-clockwise" order: (a,b), (b,c), (c,a)
        @inbounds for (i,tt) in enumerate(t)
            a, b, c = tt
            e[3i-2] = (a, b)
            e[3i-1] = (b, c)
            e[3i  ] = (c, a)
        end
    else
        # Sort edge indices first, and return sorted tuples: (a,b), (b,c), (a,c)
        @inbounds for (i,tt) in enumerate(t)
            a, b, c = sorttuple(tt)
            e[3i-2] = (a, b)
            e[3i-1] = (b, c)
            e[3i  ] = (a, c) # NOTE: not (c,a), as a <= c
        end
    end
    return e
end
getedges(t::AbstractVector{NTuple{3,Int}}; kwargs...) = getedges!(Vector{NTuple{2,Int}}(undef, 3*length(t)), t; kwargs...)

# ---------------------------------------------------------------------------- #
# FIXMESH  Remove duplicated/unused nodes and fix element orientation.
#   P, T, PIX = FIXMESH(P,T)
# ---------------------------------------------------------------------------- #

function fixmesh(
        p::AbstractVector{Vec{2,T}},
        t::AbstractVector{NTuple{3,Int}} = Vector{NTuple{3,Int}}(),
        ptol = 1024*eps(T)
    ) where {T}

    if isempty(p) || isempty(t)
        return p, t, collect(1:length(p))
    end

    p_matrix(p) = reshape(reinterpret(T, p), (2, length(p))) |> transpose |> copy
    p_vector(p) = vec(reinterpret(Vec{2,T}, transpose(p))) |> copy
    t_matrix(t) = reshape(reinterpret(Int, t), (3, length(t))) |> transpose |> copy
    t_vector(t) = vec(reinterpret(NTuple{3,Int}, transpose(t))) |> copy

    p = p_matrix(p)
    snap = maximum(maximum(p, dims=1) - minimum(p, dims=1)) * ptol
    _, ix, jx = findunique(p_vector(round.(p./snap).*snap))
    p = p_vector(p)[ix]

    pix = Int[]
    if !isempty(t)
        t = t_matrix(t)
        t = reshape(jx[t], size(t))

        pix, ix1, jx1 = findunique(vec(t))
        t = reshape(jx1, size(t))
        p = p[pix]
        pix = ix[pix]

        t = t_vector(t)
        @inbounds for (i,tt) in enumerate(t)
            A = triangle_area(p[tt[1]], p[tt[2]], p[tt[3]])
            A < 0 && (t[i] = (tt[2], tt[1], tt[3])) # flip if volume is negative
        end
    end

    return p, t, pix
end

# ---------------------------------------------------------------------------- #
# BOUNDEDGES Find boundary edges from triangular mesh
#   E = BOUNDEDGES(P,T)
# ---------------------------------------------------------------------------- #

function boundedges(
        p::AbstractVector{Vec{2,T}},
        t::AbstractVector{NTuple{3,Int}}
    ) where {T}

    # Check for empty inputs
    (isempty(p) || isempty(t)) && return NTuple{2,Int}[]

    # Form all edges, non-duplicates are boundary edges
    t = to_mat(t)
    edges = [t[:,[1,2]]; t[:,[2,3]]; t[:,[3,1]]] #NOTE changed from above, but shouldn't matter
    node3 = [t[:,3]; t[:,1]; t[:,2]]
    edges = sort(edges; dims = 2) # for finding unique edges, make sure they're ordered the same
    _, ix, jx = findunique(to_tuple(edges))
    
    # Histogram edges are 1:max(jx)+1, closed on the left i.e. [a,b). First bin is [1,2), n'th bin is [n,n+1)
    h = fit(Histogram, jx, weights(ones(Int, size(jx))), 1:length(ix)+1; closed = :left)
    ix_unique = ix[h.weights .== 1]
    edges = edges[ix_unique, :]
    node3 = node3[ix_unique]
    
    # Orientation
    b_flip = triangle_area.(p[edges[:,1]], p[edges[:,2]], p[node3]) .< 0 #NOTE this is different in DistMesh; they check for > 0 due to clockwise ordering
    edges[b_flip, [1,2]] = edges[b_flip, [2,1]]
    edges = sort!(to_tuple(edges))

    return edges
end

# ---------------------------------------------------------------------------- #
# simpplot
# ---------------------------------------------------------------------------- #

# Light grid wrapper structure for simpplot. Non-AbstractPlot arguments to simpplot
# are forwarded to the SimpPlotGrid constructor.
struct SimpPlotGrid{T}
    p::Vector{Vec{2,T}}
    t::Vector{NTuple{3,Int}}
    SimpPlotGrid() = SimpPlotGrid(Vec{2,Float64}[], NTuple{3,Int}[])
    SimpPlotGrid(g::SimpPlotGrid) = deepcopy(g)
    SimpPlotGrid(p::AbstractVector{Vec{2,T}}, t::AbstractVector{NTuple{3,Int}}) where {T} = new{T}(p,t)
    function SimpPlotGrid(p::AbstractMatrix{T}, t::AbstractMatrix{Int}) where {T}
        size(p,2) == 2 || throw(DimensionMismatch("p must be a matrix of two columns, i.e. representing 2-vectors"))
        size(t,2) == 3 || throw(DimensionMismatch("t must be a matrix of three columns, i.e. representing triangles"))
        new{T}(to_vec(p), to_tuple(t))
    end
end

@userplot SimpPlot

@recipe function f(h::SimpPlot)
    grid = parse_simpplot_args(h.args...)
    if !(typeof(grid) <: SimpPlotGrid)
        error("Arguments not properly parsed; expected two AbstractArray's, got: $unwrapped")
    end
    p, t = grid.p, grid.t
    V = eltype(p)
    T = eltype(V)

    # Default and common attributes
    seriestype   --> :shape
    aspect_ratio :=  :equal
    legend       :=  false
    hover        :=  nothing

    if plotattributes[:seriestype] == :shape
        # Plot all shapes as a single path, using NaN's as line segment breaks
        x = sizehint!(T[], 5*length(t))
        y = sizehint!(T[], 5*length(t))
        for tt in t
            # Push the first point twice so that the `Shape` path is a closed loop
            p1, p2, p3 = p[tt[1]], p[tt[2]], p[tt[3]]
            push!(x, p1[1]); push!(x, p2[1]); push!(x, p3[1]); push!(x, p1[1]); push!(x, T(NaN))
            push!(y, p1[2]); push!(y, p2[2]); push!(y, p3[2]); push!(y, p1[2]); push!(y, T(NaN))
        end
        x, y
    else
        # Plot all points as a single path, using NaN's as line segment breaks
        edges = getedges(t; sorted = true) |> sort! |> unique!
        x = sizehint!(T[], 3*length(t))
        y = sizehint!(T[], 3*length(t))
        for e in edges
            p1, p2 = p[e[1]], p[e[2]]
            push!(x, p1[1]); push!(x, p2[1]); push!(x, T(NaN))
            push!(y, p1[2]); push!(y, p2[2]); push!(y, T(NaN))
        end
        seriestype := :path
        x, y
    end
end

parse_simpplot_args(g::SimpPlotGrid) = g
parse_simpplot_args(args...) = SimpPlotGrid(args...) # construct SimpPlotGrid
parse_simpplot_args(plts::P, args...) where {P <: Union{<:AbstractPlot, <:AbstractArray{<:AbstractPlot}}} = parse_simpplot_args(args...) # skip plot objects
