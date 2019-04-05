# ---------------------------------------------------------------------------- #
# Grid methods
# ---------------------------------------------------------------------------- #

# JuAFEM doesn't define show methods without ::MIME"text/plain", and this gives verbose outputs for e.g. arrays of Grid's
Base.show(io::IO, grid::JuAFEM.Grid) = print(io, "$(typeof(grid)) with $(JuAFEM.getncells(grid)) $(JuAFEM.celltypes[eltype(grid.cells)]) cells and $(JuAFEM.getnnodes(grid)) nodes")

# ---------------------------------------------------------------------------- #
# AbstractDomain methods
# ---------------------------------------------------------------------------- #

# Interpolation is done by simply creating a `Dirichlet` constraint on every
# face of the domain and applying it to the vector `u`. This is really quite
# slow and wasteful, and there is almost definitely a better way to implement
# this, but it just isn't a bottleneck and this is easy.
function interpolate!(
        u::AbstractVector{Tu},
        f::Function,
        domain::AbstractDomain{Tu,uType}
    ) where {Tu, uType<:FieldType{Tu}}
    
    ch = ConstraintHandler(getdofhandler(domain))
    ∂Ω = getfaces(getgrid(domain))
    uDim = fielddim(uType)
    dbc = JuAFEM.Dirichlet(:u, ∂Ω, (x,t) -> f(x), [1:uDim;])
    add!(ch, dbc)
    close!(ch)
    update!(ch, zero(Tu)) # time zero
    apply!(u, ch)

    return u
end
function interpolate(f::Function, domain::AbstractDomain{Tu,uType}) where {Tu, uType<:FieldType{Tu}}
    return interpolate!(zeros(Tu, ndofs(getdofhandler(domain))), f, domain)
end

# Optimization for when we can guarantee that the degrees of freedom `u` are
# purely nodal and we just want to assign a constant vector `u0` to each node
function interpolate!(u::AbstractVector{Tu}, u0::uType, domain::AbstractDomain{Tu,uType}) where {Tu, uType<:FieldType{Tu}}
    # Check that `u` has the correct length
    @assert length(u) == ndofs(getdofhandler(domain))
    if length(u) == fielddim(uType) * getnnodes(getgrid(domain))
        # degrees of freedom are nodal; can efficiently assign directly
        _u = reinterpret(uType, u) # rename is important - allows to return u below with original type
        _u .= Ref(u0)
    else
        # degrees of freedom are not nodal; call general projection
        interpolate!(u, x->u0, domain)
    end
    return u
end
function interpolate(u0::uType, domain::AbstractDomain{Tu,uType}) where {Tu, uType<:FieldType{Tu}}
    return interpolate!(zeros(Tu, ndofs(getdofhandler(domain))), u0, domain)
end

function integrate(u::AbstractVector{Tu}, domain::AbstractDomain{Tu,uType}) where {Tu, uType<:FieldType{Tu}}
    @assert length(u) == ndofs(getdofhandler(domain))
    u = reinterpret(uType, u)
    w = reinterpret(uType, getquadweights(domain))
    # Integrate. ⊙ == hadamardproduct is the Hadamard product of the Vec's.
    S = u[1] ⊙ w[1]
    @inbounds for i in 2:length(u)
        S += u[i] ⊙ w[i]
    end
    return S
end

# "Vectorized" versions of functions for convenience
integrate(U::VectorOfVectors, domains::VectorOfDomains) = sum(map((u,d) -> integrate(u,d), U, domains))

interpolate!(U::VectorOfVectors, f::Function, domains::VectorOfDomains) = map!((u,d) -> interpolate!(u, f, d), U, U, domains)
interpolate(f::Function, domains::VectorOfDomains{Tu}) where {Tu} = interpolate!(VectorOfVectors{Tu}[zeros(Tu, ndofs(d)) for d in domains], f, domains)

interpolate!(U::VectorOfVectors, u0::uType, domains::VectorOfDomains{Tu,uType}) where {Tu, uType<:FieldType{Tu}} = map!((u,d) -> interpolate!(u, u0, d), U, U, domains)
interpolate(u0::uType, domains::VectorOfDomains{Tu,uType}) where {Tu, uType<:FieldType{Tu}} = interpolate!(VectorOfVectors{Tu}[zeros(Tu, ndofs(d)) for d in domains], u0, domains)

# ---------------------------------------------------------------------------- #
# ParabolicDomain methods
# ---------------------------------------------------------------------------- #

@inline getgrid(d::ParabolicDomain) = d.grid
@inline getdofhandler(d::ParabolicDomain) = d.dh
@inline getcellvalues(d::ParabolicDomain) = d.cellvalues
@inline getfacevalues(d::ParabolicDomain) = d.facevalues
@inline getrefshape(d::ParabolicDomain) = d.refshape
@inline getquadorder(d::ParabolicDomain) = d.quadorder
@inline getfuncinterporder(d::ParabolicDomain) = d.funcinterporder
@inline getgeominterporder(d::ParabolicDomain) = d.geominterporder
@inline getmass(d::ParabolicDomain) = d.M
@inline getmassfact(d::ParabolicDomain) = d.Mfact
@inline getstiffness(d::ParabolicDomain) = d.K
@inline getquadweights(d::ParabolicDomain) = d.w
@inline JuAFEM.ndofs(d::ParabolicDomain) = JuAFEM.ndofs(getdofhandler(d))

factorize!(d::ParabolicDomain) = (d.Mfact = cholesky(getmass(d)); return d)
# function factorize!(d::ParabolicDomain)
#     M = getmass(d)
#     m = M[1:2:end, 1:2:end]
#     @assert m ≈ M[2:2:end, 2:2:end]
#     d.Mfact = cholesky(m)
#     return d
# end
LinearAlgebra.norm(u, domain::ParabolicDomain) = √dot(u, getmass(domain) * u)

# GeometryUtils.area(d::ParabolicDomain{uDim}) where {uDim} = sum(@views getquadweights(d)[1:uDim:end]) # summing the quad weights for one component gives area
GeometryUtils.area(d::ParabolicDomain{Tu,uType,2}) where {Tu,uType} = area(getgrid(d)) # calculate area of 2D grid directly

# Show methods
function _compact_show_sparse(io, S::SparseMatrixCSC)
    print(io, S.m, "×", S.n, " ", typeof(S), " with ", nnz(S), " stored ", nnz(S) == 1 ? "entry" : "entries")
end
function _compact_show_sparse(io, A::Symmetric{T,<:SparseMatrixCSC{T}}) where {T}
    S = A.data; xnnz = nnz(S)
    print(io, S.m, "×", S.n, " ", typeof(A), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
end
function _compact_show_factorization(io, F::Union{Nothing, <:Factorization})
    F == nothing && (show(io, F); return)
    m, n = size(F)
    print(io, m, "×", n, " ", typeof(F), " with ", nnz(F), " stored ", nnz(F) == 1 ? "entry" : "entries")
end
function Base.show(io::IO, d::ParabolicDomain)
    compact = get(io, :compact, false)
    if compact
        print(io, "$(typeof(d)) with $(ndofs(d)) degrees of freedom")
    else
        print(io, "$(typeof(d)) with:")
        print(io, "\n  grid: "); show(io, getgrid(d))
        print(io, "\n     M: "); _compact_show_sparse(io, getmass(d))
        print(io, "\n Mfact: "); _compact_show_factorization(io, getmassfact(d))
        print(io, "\n     K: "); _compact_show_sparse(io, getstiffness(d))
        print(io, "\n     w: ", length(getquadweights(d)), "-element ", typeof(getquadweights(d)))
    end
end

# ---------------------------------------------------------------------------- #
# MyelinDomain methods
# ---------------------------------------------------------------------------- #

@inline getregion(m::MyelinDomain) = m.region
@inline getdomain(m::MyelinDomain) = m.domain
@inline getoutercircles(m::MyelinDomain) = m.outercircles
@inline getinnercircles(m::MyelinDomain) = m.innercircles
@inline getoutercircle(m::MyelinDomain, i::Int) = m.outercircles[i]
@inline getinnercircle(m::MyelinDomain, i::Int) = m.innercircles[i]
@inline getouterradius(m::MyelinDomain, i::Int) = radius(getoutercircle(m,i))
@inline getinnerradius(m::MyelinDomain, i::Int) = radius(getinnercircle(m,i))
@inline numfibres(m::MyelinDomain) = length(getoutercircles(m))

Lazy.@forward MyelinDomain.domain (getgrid, getdofhandler, getcellvalues, getfacevalues, getrefshape, getquadorder, getfuncinterporder, getgeominterporder, getmass, getmassfact, getstiffness, getquadweights)
Lazy.@forward MyelinDomain.domain (factorize!, addquadweights!)
Lazy.@forward MyelinDomain.domain (JuAFEM.ndofs, LinearAlgebra.norm, GeometryUtils.area)

function Base.show(io::IO, m::MyelinDomain)
    compact = get(io, :compact, false)
    if compact
        print(io, "$(typeof(m)) with $(ndofs(m)) degrees of freedom and $(numfibres(m)) fibres")
    else
        print(io, "$(typeof(m)) with $(numfibres(m)) fibres and:")
        print(io, "\n  grid: "); show(io, getgrid(m))
        print(io, "\n     M: "); _compact_show_sparse(io, getmass(m))
        print(io, "\n Mfact: "); _compact_show_factorization(io, getmassfact(m))
        print(io, "\n     K: "); _compact_show_sparse(io, getstiffness(m))
        print(io, "\n     w: ", length(getquadweights(m)), "-element ", typeof(getquadweights(m)))
    end
end

function createmyelindomains(
        tissuegrids::AbstractVector{Grid{gDim,Nd,T,Nf}},
        myelingrids::AbstractVector{Grid{gDim,Nd,T,Nf}},
        axongrids::AbstractVector{Grid{gDim,Nd,T,Nf}},
        outercircles::AbstractVector{Circle{2,T}},
        innercircles::AbstractVector{Circle{2,T}},
        ferritins::AbstractVector{Vec{3,T}} = Vec{3,T}[],
        ::Type{uType} = Vec{2,T}; #Default to same float type as grid
        kwargs...
    ) where {gDim,T,Nd,Nf,Tu,uType<:FieldType{Tu}}

    @assert length(outercircles) == length(innercircles) == length(myelingrids) == length(axongrids)

    isgridempty(g::Grid) = (getnnodes(g) == 0 || getncells(g) == 0)

    Mtype = MyelinDomain{R,Tu,uType,gDim,T,Nd,Nf} where {R}
    ms = Vector{Mtype}()

    for (i, a) in enumerate(axongrids)
        isgridempty(a) && continue
        push!(ms, MyelinDomain(AxonRegion(i), a, outercircles, innercircles, ferritins, uType; kwargs...))
    end

    for (i, m) in enumerate(myelingrids)
        isgridempty(m) && continue
        push!(ms, MyelinDomain(MyelinRegion(i), m, outercircles, innercircles, ferritins, uType; kwargs...))
    end

    for t in tissuegrids
        isgridempty(t) && continue
        push!(ms, MyelinDomain(TissueRegion(), t, outercircles, innercircles, ferritins, uType; kwargs...))
    end

    return ms
end

# Create interface domain from vector of MyelinDomain's which are all assumed
# to have the same outercircles, innercircles, and ferritins
function MyelinDomain(
        region::PermeableInterfaceRegion,
        prob::MyelinProblem,
        ms::AbstractVector{<:TriangularMyelinDomain{R,Tu,uType,T} where R}
    ) where {Tu,uType,T}
    domain = ParabolicDomain(region, prob, ms)
    myelindomain = TriangularMyelinDomain{PermeableInterfaceRegion,Tu,uType,T,typeof(domain)}(
        PermeableInterfaceRegion(),
        domain,
        ms[1].outercircles, # assume these are the same for all domains
        ms[1].innercircles, # assume these are the same for all domains
        ms[1].ferritins # assume these are the same for all domains
    )
    return myelindomain
end

function ParabolicDomain(
        region::PermeableInterfaceRegion,
        prob::MyelinProblem,
        ms::AbstractVector{<:TriangularMyelinDomain{R,Tu,uType,T} where {R}}
    ) where {Tu,uType,T}

    # Construct one large ParabolicDomain containing all grids
    gDim, Nd, Nf = 2, 3, 3 # Triangular 2D domain
    grid = Grid(getgrid.(ms)) # combine grids into single large grid
    domain = ParabolicDomain(grid, uType;
        refshape = getrefshape(ms[1]), # assume these are the same for all domains
        quadorder = getquadorder(ms[1]), # assume these are the same for all domains
        funcinterporder = getfuncinterporder(ms[1]), # assume these are the same for all domains
        geominterporder = getgeominterporder(ms[1]) # assume these are the same for all domains
    )
    domain.M = blockdiag(getmass.(ms)...)
    domain.K = blockdiag(getstiffness.(ms)...)
    domain.w = reduce(vcat, getquadweights.(ms))

    # Find interface pairs
    cells, nodes = getcells(getgrid(domain)), getnodes(getgrid(domain))
    boundaryfaceset = getfaceset(getgrid(domain), "boundary") # set of 2-tuples of (cellid, faceid)
    nodepairs = NTuple{2,Int}[JuAFEM.faces(cells[f[1]])[f[2]] for f in boundaryfaceset] # pairs of node indices
    nodecoordpairs = NTuple{2,Vec{gDim,T}}[(getcoordinates(nodes[n[1]]), getcoordinates(nodes[n[2]])) for n in nodepairs] # pairs of node coordinates

    # Sort pairs by midpoints and read off pairs
    bymidpoint = (np) -> (mid = (np[1] + np[2])/2; return norm2(mid), angle(mid))
    nodecoordindices = sortperm(nodecoordpairs; by = bymidpoint) # sort pairs by midpoint location
    interfaceindices = Vector{NTuple{4,Int}}()
    sizehint!(interfaceindices, length(nodecoordpairs)÷2)
    @inbounds for i in 1:length(nodecoordindices)-1
        i1, i2 = nodecoordindices[i], nodecoordindices[i+1]
        np1, np2 = nodecoordpairs[i1], nodecoordpairs[i2]
        if norm2(np1[1] - np2[2]) < eps(T) && norm2(np1[2] - np2[1]) < eps(T)
            push!(interfaceindices, (nodepairs[i1]..., nodepairs[i2]...))
        end
    end

    # # Brute force search for pairs
    # interfaceindices_brute = Vector{NTuple{4,Int}}()
    # sizehint!(interfaceindices_brute, length(nodecoordpairs)÷2)
    # @inbounds for i1 in 1:length(nodecoordpairs)
    #     np1 = nodecoordpairs[i1] # pair of Vec's
    #     for i2 in 1:i1-1
    #         np2 = nodecoordpairs[i2] # pair of Vec's
    #         # For properly oriented triangles, the edge nodes will be stored in
    #         # opposite order coincident edges which share a triangle face, i.e.
    #         # np1[1] should equal np2[2], and vice-versa
    #         if norm2(np1[1] - np2[2]) < eps(T) && norm2(np1[2] - np2[1]) < eps(T)
    #             # The nodes are stored as e.g. np1 = (A,B) and np2 = (B,A).
    #             # We want to keep them in this order, as is expected by the
    #             # local stiffness matrix `Se` below
    #             ip1, ip2 = nodepairs[i1], nodepairs[i2] # pairs of node indices
    #             push!(interfaceindices_brute, (ip1..., ip2...))
    #         end
    #     end
    # end

    # Local permeability interaction matrix, unscaled by length
    κ = prob.params.K_perm
    Se = Tu(-κ/6) .* Tu[ 2  1 -1 -2   # Minus sign in front since we build the negative stiffness matrix
                         1  2 -2 -1   # `Se` represents the local stiffness matrix of a zero volume (line segment) interface element
                        -1 -2  2  1   # The segment interfaces between the pairs of nodes (A1,B1) and (A2,B2), where A nodes and B nodes have the same coordinates
                        -2 -1  1  2 ] # `Se` is ordered such that it acts on [A1,B1,B2,A2]
    _Se = similar(Se) # temp matrix for storing ck * Se

    # S matrix global indices
    Is, Js, Ss = Vector{Int}(), Vector{Int}(), Vector{Tu}()
    uDim = fielddim(uType)
    sizehint!(Is, length(Se) * uDim * length(interfaceindices))
    sizehint!(Js, length(Se) * uDim * length(interfaceindices))
    sizehint!(Ss, length(Se) * uDim * length(interfaceindices))

    for idx in interfaceindices
        ck = Tu(norm(getcoordinates(nodes[idx[1]]) - getcoordinates(nodes[idx[2]]))) # length of edge segment
        _Se .= ck .* Se
        dof = uDim .* idx .- (uDim-1) # node indices --> first dof indices (i.e. 1st component of u)
        for (j,dof_j) in enumerate(dof)
            for (i,dof_i) in enumerate(dof)
                for d in 0:uDim-1
                    push!(Is, dof_i + d) # first dof indices --> d'th dof indices (i.e. d'th component of u)
                    push!(Js, dof_j + d)
                    push!(Ss, _Se[i,j])
                end
            end
        end
    end

    # Form final stiffness matrix
    I, J, V = findnz(domain.K)
    domain.K = sparse([I; Is], [J; Js], [V; Ss])

    return domain
end

# ---------------------------------------------------------------------------- #
# Stiffness matrix and mass matrix assembly:
#   Assemble matrices for a BlochTorreyProblem on a ParabolicDomain
# ---------------------------------------------------------------------------- #

# Assemble the `BlochTorreyProblem` system $M u_t = K u$ on the domain `domain`.
function doassemble!(
        domain::ParabolicDomain{Tu,uType},
        prob::BlochTorreyProblem{Tu}
    ) where {Tu,uType}

    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    # First, we create assemblers for the stiffness matrix `K` and the mass
    # matrix `M`. The assemblers are just thin wrappers around `K` and `M`
    # and some extra storage to make the assembling faster.
    assembler_K = start_assemble(getstiffness(domain), getquadweights(domain))
    assembler_M = start_assemble(getmass(domain))

    # Next, we allocate the element stiffness matrix and element mass matrix
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    Ke = zeros(Tu, n_basefuncs, n_basefuncs)
    Me = zeros(Tu, n_basefuncs, n_basefuncs)
    we = zeros(Tu, n_basefuncs)

    # It is now time to loop over all the cells in our grid. We do this by iterating
    # over a `CellIterator`. The iterator caches some useful things for us, for example
    # the nodal coordinates for the cell, and the local degrees of freedom.
    @inbounds for cell in CellIterator(getdofhandler(domain))
        # Always remember to reset the element stiffness matrix and
        # element mass matrix since we reuse them for all elements.
        fill!(Ke, zero(Tu))
        fill!(Me, zero(Tu))
        fill!(we, zero(Tu))

        # Get the coordinates of the cell
        coords = getcoordinates(cell)

        # For each cell we also need to reinitialize the cached values in `cellvalues`.
        JuAFEM.reinit!(getcellvalues(domain), cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `Me`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`, and the quadrature
        # coordinate can be queried from `cellvalues` by `spatial_coordinate`
        for q_point in 1:getnquadpoints(getcellvalues(domain))
            dΩ = getdetJdV(getcellvalues(domain), q_point)
            coords_qp = spatial_coordinate(getcellvalues(domain), q_point, coords)

            # calculate the heat conductivity and heat source at point `coords_qp`
            D = prob.Dcoeff(coords_qp)
            R = prob.Rdecay(coords_qp)
            ω = prob.Omega(coords_qp)

            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            for i in 1:n_basefuncs
                v  = shape_value(getcellvalues(domain), q_point, i)
                ∇v = shape_gradient(getcellvalues(domain), q_point, i)
                we[i] += sum(v) * dΩ
                for j in 1:n_basefuncs
                    u = shape_value(getcellvalues(domain), q_point, j)
                    ∇u = shape_gradient(getcellvalues(domain), q_point, j)
                    Ke[i, j] -= (D * (∇v ⊡ ∇u) + R * (v ⋅ u) - ω * (v ⊠ u)) * dΩ
                    Me[i, j] += (v ⋅ u) * dΩ
                end
            end
        end

        # The last step in the element loop is to assemble `Ke` and `Me`
        # into the global `K` and `M` with `assemble!`.
        assemble!(assembler_K, celldofs(cell), Ke, we)
        assemble!(assembler_M, celldofs(cell), Me)
    end

    return domain
end

# ---------------------------------------------------------------------------- #
# Assembly on a MyelinDomain of a MyelinProblem
# ---------------------------------------------------------------------------- #

function doassemble!(myelindomain::MyelinDomain, prob::MyelinProblem)
    doassemble!(getdomain(myelindomain), BlochTorreyProblem(prob, myelindomain))
    return myelindomain
end

# ---------------------------------------------------------------------------- #
# Assembly on a ParabolicDomain
# ---------------------------------------------------------------------------- #

# Assembly quadrature weights for the parabolic domain `domain`.
function addquadweights!(domain::ParabolicDomain{Tu,uType}) where {Tu,uType}
    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    we = zeros(Tu, n_basefuncs)
    fill!(getquadweights(domain), zero(Tu))

    @inbounds for cell in CellIterator(getdofhandler(domain))
        # Reset element residual and reinit cellvalues
        fill!(we, zero(Tu))
        JuAFEM.reinit!(getcellvalues(domain), cell)
        # Integrate all components of shape function `v` and add to weights vector
        for q_point in 1:getnquadpoints(getcellvalues(domain))
            dΩ = getdetJdV(getcellvalues(domain), q_point)
            for i in 1:n_basefuncs
                v  = shape_value(getcellvalues(domain), q_point, i)
                we[i] += sum(v) * dΩ # sum(v) is short for adding v[1] ... v[vdim] contributions
            end
        end
        # Assemble the element residual `we` into the global residual vector `w`
        assemble!(getquadweights(domain), celldofs(cell), we)
    end

    return domain
end

# Assemble the standard mass and stiffness matrices on the ParabolicDomain
# `domain`. The resulting system is $M u_t = K u$ and is equivalent to the weak
# form of the heat equation $u_t = k Δu$ with k = 1. `M` is positive definite,
# and `K` is negative definite.
function doassemble!(domain::ParabolicDomain{Tu,uType}) where {Tu,uType}
    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    Ke = zeros(Tu, n_basefuncs, n_basefuncs)
    Me = zeros(Tu, n_basefuncs, n_basefuncs)
    we = zeros(Tu, n_basefuncs)

    # Next we create assemblers for the stiffness matrix `K` and the mass
    # matrix `M`. The assemblers are just thin wrappers around `K` and `M`
    # and some extra storage to make the assembling faster.
    assembler_K = start_assemble(getstiffness(domain), getquadweights(domain))
    assembler_M = start_assemble(getmass(domain))

    # It is now time to loop over all the cells in our grid. We do this by iterating
    # over a `CellIterator`. The iterator caches some useful things for us, for example
    # the nodal coordinates for the cell, and the local degrees of freedom.
    @inbounds for cell in CellIterator(getdofhandler(domain))
        # Always remember to reset the element stiffness matrix and
        # element mass matrix since we reuse them for all elements.
        fill!(Ke, zero(Tu))
        fill!(Me, zero(Tu))
        fill!(we, zero(Tu))

        # For each cell we also need to reinitialize the cached values in `cellvalues`.
        JuAFEM.reinit!(getcellvalues(domain), cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `Me`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`, and the quadrature
        # coordinate can be queried from `cellvalues` by `spatial_coordinate`
        for q_point in 1:getnquadpoints(getcellvalues(domain))
            dΩ = getdetJdV(getcellvalues(domain), q_point)

            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            for i in 1:n_basefuncs
                v  = shape_value(getcellvalues(domain), q_point, i)
                ∇v = shape_gradient(getcellvalues(domain), q_point, i)
                we[i] += (ones(v) ⋅ v) * dΩ # v[1] and v[2] are never non-zero together
                for j in 1:n_basefuncs
                    u = shape_value(getcellvalues(domain), q_point, j)
                    ∇u = shape_gradient(getcellvalues(domain), q_point, j)
                    Ke[i,j] -= (∇v ⊡ ∇u) * dΩ
                    Me[i,j] += (v ⋅ u) * dΩ
                end
            end
        end

        # The last step in the element loop is to assemble `Ke` and `Me`
        # into the global `K` and `M` with `assemble!`.
        assemble!(assembler_K, celldofs(cell), Ke, we)
        assemble!(assembler_M, celldofs(cell), Me)
    end

    return domain
end

# ---------------------------------------------------------------------------- #
# Example loop for evaluating surface/boundary integrals
# ---------------------------------------------------------------------------- #

# Loop over the edges of a cell to add interface contributions to `Ke`
function add_interface!(Ke::AbstractMatrix{Tu}, facevalues::FaceVectorValues, cell) where {Tu}#, q_point, coords, func::Function)
    # TODO: make this a working function
    @warn "Function add_interface! is only a sketch of an implementation; returning Ke"
    return Ke

    # Allocate local interface element matrices.
    n_basefuncs = getnbasefunctions(getfacevalues(domain))
    Se = zeros(Tu, 2*n_basefuncs, 2*n_basefuncs)
    
    for face in 1:nfaces(cell)
        if onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "Interface")
            # Initialize face values
            JuAFEM.reinit!(getfacevalues(domain), cell, face)
    
            for q_point in 1:getnquadpoints(facevalues)
                dΓ = getdetJdV(facevalues, q_point)
                coords_qp = spatial_coordinate(facevalues, q_point, coords)
    
                # calculate the heat conductivity and heat source at point `coords_qp`
                f = func(coords_qp)
                fdΓ = f * dΓ
    
                for i in 1:getnbasefunctions(facevalues)
                    n = getnormal(facevalues, q_point)
                    v = shape_value(facevalues, q_point, i)
                    vfdΓ = v * fdΓ
                    for j in 1:n_basefuncs
                        ∇u = shape_gradient(facevalues, q_point, j)
                        Ke[i,j] += (∇u⋅n) * vfdΓ
                    end
                end
            end
        end
    end
end