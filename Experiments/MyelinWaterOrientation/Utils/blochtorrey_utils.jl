# ---------------------------------------------------------------------------- #
# Bloch-Torrey parameters type
# ---------------------------------------------------------------------------- #

# Struct of BlochTorreyParameters. T is the float type.
@with_kw struct BlochTorreyParameters{T}
    B0::T             =    T(3.0)           # External magnetic field [T]
    gamma::T          =    T(2.67515255e8)  # Gyromagnetic ratio [rad/(T*s)]
    theta::T          =    T(π/2)           # Main magnetic field angle w.r.t B0 [rad/(T*s)]
    g_ratio::T        =    T(0.8370)        # g-ratio (original 0.71), 0.84658 for healthy, 0.8595 for MS.
    R2_sp::T          =    T(1.0/15e-3)     # #TODO play with these? Relaxation rate of small pool [s^-1] (Myelin) (Xu et al. 2017) (15e-3s)
    R2_lp::T          =    T(1.0/63e-3)     # #TODO play with these? 1st attempt was 63E-3. 2nd attempt 76 ms
    R2_Tissue::T      =    T(14.5)          # Relaxation rate of tissue [s^-1]
    R2_water::T       =    T(1.0/2.2)       # Relaxation rate of pure water
    D_Tissue::T       =    T(2000.0)        # #TODO reference? Diffusion coefficient in tissue [um^2/s]
    D_Sheath::T       =    T(1000.0)        # #TODO reference? Diffusion coefficient in myelin sheath [um^2/s]
    D_Axon::T         =    T(2500.0)        # #TODO reference? Diffusion coefficient in axon interior [um^2/s]
    D_Blood::T        =    T(3037.0)        # Diffusion coefficient in blood [um^2/s]
    D_Water::T        =    T(3037.0)        # Diffusion coefficient in water [um^2/s]
    R_mu::T           =    T(0.46)          # Axon mean radius [um] ; this is taken to be outer radius.
    R_shape::T        =    T(5.7)           # Axon shape parameter for Gamma distribution (Xu et al. 2017)
    R_scale::T        =    T(0.46/5.7)      # Axon scale parameter for Gamma distribution (Xu et al. 2017)
    AxonPDensity::T   =    T(0.83)          # Axon packing density based region in white matter. (Xu et al. 2017) (originally 0.83)
    AxonPDActual::T   =    T(0.64)          # The actual axon packing density you're aiming for.
    PD_sp::T          =    T(0.5)           # Relative proton density (Myelin)
    PD_lp::T          =    T(1.0)           # Relative proton density (Intra Extra)
    PD_Fe::T          =    T(1.0)           # Relative proton density (Ferritin)
    ChiI::T           =    T(-60e-9)        # Isotropic susceptibility of myelin [ppb] (check how to get it) (Xu et al. 2017)
    ChiA::T           =    T(-120e-9)       # Anisotropic Susceptibility of myelin [ppb] (Xu et al. 2017)
    E::T              =    T(10e-9)         # Exchange component to resonance freqeuency [ppb] (Wharton and Bowtell 2012)
    R2_Fe::T          =    T(1.0/1e-6)      # Relaxation rate of iron in ferritin. Assumed to be really high.
    R2_WM::T          =    T(1.0/70e-3)     # Relaxation rate of frontal WM. This is empirical;taken from literature. (original 58.403e-3) (patient 58.4717281111171e-3)
    R_Ferritin::T     =    T(4.0e-3)        # Ferritin mean radius [um].
    R_conc::T         =    T(0.0)           # Conntration of iron in the frontal white matter. [mg/g] (0.0424 in frontal WM) (0.2130 in globus pallidus; deep grey matter)
    Rho_tissue::T     =    T(1.073)         # White matter tissue density [g/ml]
    ChiTissue::T      =    T(-9.05e-6)      # Isotropic susceptibility of tissue
    ChiFeUnit::T      =    T(1.4e-9)        # Susceptibility of iron per ppm/ (ug/g) weight fraction of iron.
    ChiFeFull::T      =    T(520.0e-6)      # Susceptibility of iron for ferritin particle FULLY loaded with 4500 iron atoms. (use volume of FULL spheres) (from Contributions to magnetic susceptibility)
    Rho_Iron::T       =    T(7.874)         # Iron density [g/cm^3]
end

radiidistribution(p::BlochTorreyParameters) = Distributions.Gamma(p.R_shape, p.R_mu/p.R_shape)

# ---------------------------------------------------------------------------- #
# BlochTorreyProblem type
# ---------------------------------------------------------------------------- #

abstract type AbstractParabolicProblem{T} end

struct MyelinProblem{T} <: AbstractParabolicProblem{T}
    params::BlochTorreyParameters{T}
end

struct BlochTorreyProblem{T,D,R,W} <: AbstractParabolicProblem{T}
    Dcoeff::D
    Rdecay::R
    Omega::W
    BlochTorreyProblem{T}(d::D,r::R,w::W) where {T,D,R,W} = new{T,D,R,W}(d,r,w)
end

# ---------------------------------------------------------------------------- #
# Myelin grid type
# ---------------------------------------------------------------------------- #

# Abstract domain type. The type parameters are:
#   `dim`:  Spatial dimension of domain
#   `Nd`:   Number of nodes per finite element
#   `T`:    Float type used
#   `Nf`:   Number of faces per finite element
abstract type AbstractDomain{dim,Nd,T,Nf} end

# Interpolation is done by simply creating a `Dirichlet` constraint on every
# face of the domain and applying it to the vector `u`. This is really quite
# slow and wasteful, and there is almost definitely a better way to implement
# this, but it just isn't a bottleneck and this is easy.
function interpolate!(u::Vector{T}, f::Function, domain::AbstractDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    ch = ConstraintHandler(getdofhandler(domain))
    ∂Ω = getfaces(getgrid(domain))
    dbc = JuAFEM.Dirichlet(:u, ∂Ω, (x,t) -> f(x), collect(1:dim))
    add!(ch, dbc)
    close!(ch)
    update!(ch, 0.0) # time zero
    apply!(u, ch)
    return u
end
function interpolate(f::Function, domain::AbstractDomain{dim}) where {dim}
    u = zeros(ndofs(getdofhandler(domain)))
    return interpolate!(u, f, domain)
end

function interpolate!(u::Vector{T}, u0::Vec{dim,T}, domain::AbstractDomain) where {dim,T}
    # Check that `u` has the correct length
    @assert length(u) == ndofs(getdofhandler(domain))

    if length(u) == dim*getnnodes(getgrid(domain))
        # degrees of freedom are nodal; can efficiently assign directly
        u = reinterpret(Vec{dim,T}, u)
        @inbounds for i in 1:length(u)
            u[i] = u0
        end
        u = reinterpret(T, u)
    else
        # degrees of freedom are not nodal; call general projection
        interpolate!(u, x->u0, domain)
    end
    return u
end
function interpolate!(u0::Vec{dim,T}, domain::AbstractDomain) where {dim,T}
    u = zeros(ndofs(getdofhandler(domain)))
    return interpolate!(u, f, domain)
end


function integrate(u::Vector{T}, domain::AbstractDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    u = reinterpret(Vec{dim,T}, u)
    w = reinterpret(Vec{dim,T}, getquadweights(domain))

    # Integrate. ⊙ == hadamardproduct is the Hadamard product of the Vec's.
    S = zero(Vec{dim,T})
    @inbounds for i in 1:length(u)
        S += u[i] ⊙ w[i]
    end

    return S
end

# ---------------------------------------------------------------------------- #
# Generic parabolic domain grid type
# ---------------------------------------------------------------------------- #

mutable struct ParabolicDomain{dim,Nd,T,Nf} <: AbstractDomain{dim,Nd,T,Nf}
    grid::Grid{dim,Nd,T,Nf}
    dh::DofHandler{dim,Nd,T,Nf}
    cellvalues::CellValues{dim,T}
    facevalues::FaceValues{dim,T}
    M::Symmetric{T,<:SparseMatrixCSC{T}}
    Mfact::Union{Factorization{T},Nothing}
    K::SparseMatrixCSC{T}
    w::Vector{T}
end

#TODO inner constructor?
function ParabolicDomain(grid::Grid{dim,Nd,T,Nf};
    udim = 2,
    refshape = RefTetrahedron,
    quadorder = 3,
    funcinterporder = 1,
    geominterporder = 1) where {dim,Nd,T,Nf}

    @assert udim == 2 #TODO: where is this assumption? likely, assume dim(u) == dim(grid) somewhere

    # Quadrature and interpolation rules and corresponding cellvalues/facevalues
    func_interp = Lagrange{dim, refshape, funcinterporder}()
    geom_interp = Lagrange{dim, refshape, geominterporder}()
    quadrule = QuadratureRule{dim, refshape}(quadorder)
    quadrule_face = QuadratureRule{dim-1, refshape}(quadorder)
    cellvalues = CellVectorValues(T, quadrule, func_interp, geom_interp)
    facevalues = FaceVectorValues(T, quadrule_face, func_interp, geom_interp)
    # cellvalues = CellScalarValues(T, quadrule, func_interp, geom_interp)
    # facevalues = FaceScalarValues(T, quadrule_face, func_interp, geom_interp)

    # Degree of freedom handler
    dh = DofHandler(grid)
    push!(dh, :u, udim, func_interp)
    close!(dh)

    # Mass matrix, inverse mass matrix, stiffness matrix, and weights vector
    M = create_symmetric_sparsity_pattern(dh)
    K = create_sparsity_pattern(dh)
    w = zeros(T, ndofs(dh))
    Mfact = nothing

    ParabolicDomain{dim,Nd,T,Nf}(grid, dh, cellvalues, facevalues, M, Mfact, K, w)
end

@inline numsubdomains(d::ParabolicDomain) = 1
@inline getsubdomain(d::ParabolicDomain, i::Int) = i == 1 ? d : error("i=$i, but ParabolicDomain has only 1 subdomain.")
@inline getgrid(d::ParabolicDomain) = d.grid
@inline getdofhandler(d::ParabolicDomain) = d.dh
@inline getcellvalues(d::ParabolicDomain) = d.cellvalues
@inline getfacevalues(d::ParabolicDomain) = d.facevalues
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

function _compact_show_sparse(io, S::SparseMatrixCSC)
    print(io, S.m, "×", S.n, " ", typeof(S), " with ", nnz(S), " stored ", nnz(S) == 1 ? "entry" : "entries")
end
function _compact_show_sparse(io, A::Symmetric{T,<:SparseMatrixCSC{T}}) where {T}
    S = A.data; xnnz = nnz(S)
    print(io, S.m, "×", S.n, " ", typeof(A), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
end
function _compact_show_factorization(io, F::Union{<:Factorization, Nothing})
    F == nothing && (show(io, F); return)
    m, n = size(F)
    print(io, m, "×", n, " ", typeof(F), " with ", nnz(F), " stored ", nnz(F) == 1 ? "entry" : "entries")
end
function Base.show(io::IO, d::ParabolicDomain)
    compact = get(io, :compact, false)
    if compact
        print(io, "$(typeof(d)) with $(ndofs(d.dh)) degrees of freedom")
    else
        print(io, "$(typeof(d)) with:")
        print(io, "\n  grid: "); show(io, d.grid)
        print(io, "\n     M: "); _compact_show_sparse(io, d.M)
        print(io, "\n Mfact: "); _compact_show_factorization(io, d.Mfact)
        print(io, "\n     K: "); _compact_show_sparse(io, d.K)
        print(io, "\n     w: ", length(d.w), "-element ", typeof(d.w))
    end
end

# ---------------------------------------------------------------------------- #
# Assembly on ParabolicDomain
# ---------------------------------------------------------------------------- #

# Assembly quadrature weights for the parabolic domain `domain`.
function addquadweights!(domain::ParabolicDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    we = zeros(n_basefuncs)
    fill!(getquadweights(domain), 0)

    @inbounds for cell in CellIterator(getdofhandler(domain))
        # Reset element residual and reinit cellvalues
        fill!(we, 0)
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

# # Assemble the standard mass and stiffness matrices on the ParabolicDomain
# # `domain`. The resulting system is $M u_t = K u$ and is equivalent to the weak
# # form of the heat equation $u_t = k Δu$ with k = 1. `M` is positive definite,
# # and `K` is negative definite.
# function doassemble!(domain::ParabolicDomain)
#     # This assembly function is only for CellScalarValues
#     @assert typeof(getcellvalues(domain)) <: CellScalarValues
#
#     # We allocate the element stiffness matrix and element force vector
#     # just once before looping over all the cells instead of allocating
#     # them every time in the loop.
#     n_basefuncs = getnbasefunctions(getcellvalues(domain))
#     Ke = zeros(n_basefuncs, n_basefuncs)
#     Me = zeros(n_basefuncs, n_basefuncs)
#     we = zeros(n_basefuncs)
#
#     # Next we create assemblers for the stiffness matrix `K` and the mass
#     # matrix `M`. The assemblers are just thin wrappers around `K` and `M`
#     # and some extra storage to make the assembling faster.
#     assembler_K = start_assemble(getstiffness(domain), getquadweights(domain))
#     assembler_M = start_assemble(getmass(domain))
#
#     # It is now time to loop over all the cells in our grid. We do this by iterating
#     # over a `CellIterator`. The iterator caches some useful things for us, for example
#     # the nodal coordinates for the cell, and the local degrees of freedom.
#     @inbounds for cell in CellIterator(getdofhandler(domain))
#         # Always remember to reset the element stiffness matrix and
#         # element mass matrix since we reuse them for all elements.
#         fill!(Ke, 0)
#         fill!(Me, 0)
#         fill!(we, 0)
#
#         # For each cell we also need to reinitialize the cached values in `cellvalues`.
#         JuAFEM.reinit!(getcellvalues(domain), cell)
#
#         # It is now time to loop over all the quadrature points in the cell and
#         # assemble the contribution to `Ke` and `Me`. The integration weight
#         # can be queried from `cellvalues` by `getdetJdV`, and the quadrature
#         # coordinate can be queried from `cellvalues` by `spatial_coordinate`
#         for q_point in 1:getnquadpoints(getcellvalues(domain))
#             dΩ = getdetJdV(getcellvalues(domain), q_point)
#
#             # For each quadrature point we loop over all the (local) shape functions.
#             # We need the value and gradient of the testfunction `v` and also the gradient
#             # of the trial function `u`. We get all of these from `cellvalues`.
#             for i in 1:n_basefuncs
#                 v  = shape_value(getcellvalues(domain), q_point, i)
#                 ∇v = shape_gradient(getcellvalues(domain), q_point, i)
#                 we[i] += sum(v) * dΩ # v[1] and v[2] are never non-zero together
#                 for j in 1:n_basefuncs
#                     u = shape_value(getcellvalues(domain), q_point, j)
#                     ∇u = shape_gradient(getcellvalues(domain), q_point, j)
#                     Ke[i, j] -= (∇v ⋅ ∇u) * dΩ
#                     Me[i, j] += (v * u) * dΩ
#                 end
#             end
#         end
#
#         # The last step in the element loop is to assemble `Ke` and `Me`
#         # into the global `K` and `M` with `assemble!`.
#         # assemble!(assembler_K, celldofs(cell), Ke, we)
#         # assemble!(assembler_M, celldofs(cell), Me)
#         for d in 1:2
#             assemble!(assembler_K, celldofs(cell)[d:2:end], Ke, we)
#             assemble!(assembler_M, celldofs(cell)[d:2:end], Me)
#         end
#     end
#
#     return domain
# end

# Assemble the standard mass and stiffness matrices on the ParabolicDomain
# `domain`. The resulting system is $M u_t = K u$ and is equivalent to the weak
# form of the heat equation $u_t = k Δu$ with k = 1. `M` is positive definite,
# and `K` is negative definite.
function doassemble!(domain::ParabolicDomain)
    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    Ke = zeros(n_basefuncs, n_basefuncs)
    Me = zeros(n_basefuncs, n_basefuncs)
    we = zeros(n_basefuncs)

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
        fill!(Ke, 0)
        fill!(Me, 0)
        fill!(we, 0)

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
# Myelin grid type
# ---------------------------------------------------------------------------- #

const MyelinBoundary{dim,T} = Union{<:Circle{dim,T}, <:Rectangle{dim,T}, <:Ellipse{dim,T}}

mutable struct MyelinDomain{dim,Nd,T,Nf} <: AbstractDomain{dim,Nd,T,Nf}
    fullgrid::Grid{dim,Nd,T,Nf}
    outercircles::Vector{Circle{dim,T}}
    innercircles::Vector{Circle{dim,T}}
    domainboundary::MyelinBoundary{dim,T}
    tissuedomain::ParabolicDomain{dim,Nd,T,Nf}
    myelindomains::Vector{ParabolicDomain{dim,Nd,T,Nf}}
    axondomains::Vector{ParabolicDomain{dim,Nd,T,Nf}}
end

function MyelinDomain(
        fullgrid::Grid{dim,Nd,T,Nf},
        outercircles::Vector{Circle{dim,T}},
        innercircles::Vector{Circle{dim,T}},
        domainboundary::MyelinBoundary{dim,T},
        tissuegrid::Grid{dim,Nd,T,Nf},
        myelingrids::Vector{Grid{dim,Nd,T,Nf}},
        axongrids::Vector{Grid{dim,Nd,T,Nf}};
        kwargs...) where {dim,Nd,T,Nf}
    return MyelinDomain(fullgrid, outercircles, innercircles, domainboundary,
        ParabolicDomain(tissuegrid; kwargs...),
        ParabolicDomain.(myelingrids; kwargs...),
        ParabolicDomain.(axongrids; kwargs...))
end

@inline numtissuedomains(m::MyelinDomain) = 1
@inline nummyelindomains(m::MyelinDomain) = length(m.myelindomains)
@inline numaxondomains(m::MyelinDomain) = length(m.axondomains)
@inline numsubdomains(m::MyelinDomain) = 1 + nummyelindomains(m) + numaxondomains(m)
@inline getsubdomain(m::MyelinDomain, i::Int) = (i == 1 ? m.tissuedomain :
    1 <= i-1 <= nummyelindomains(m) ? m.myelindomains[i-1] :
    1 <= i-1-nummyelindomains(m) <= numaxondomains(m) ? m.axondomains[i-1-nummyelindomains(m)] :
    error("i=$i, but this MyelinDomain has only $(numsubdomains(m)) subdomains."))
@inline getsubdomains(m::MyelinDomain) = vcat(m.tissuedomain, m.myelindomains, m.axondomains)

@inline getgrid(m::MyelinDomain) = m.fullgrid
@inline tissuegrid(m::MyelinDomain) = getgrid(m.tissuedomain)
@inline myelingrid(m::MyelinDomain, i::Int) = getgrid(m.myelindomains[i])
@inline axongrid(m::MyelinDomain, i::Int) = getgrid(m.axondomains[i])
@inline getoutercircles(m::MyelinDomain) = m.outercircles
@inline getinnercircles(m::MyelinDomain) = m.innercircles
@inline getboundary(m::MyelinDomain) = m.domainboundary

@inline getoutercircle(m::MyelinDomain, i::Int) = m.outercircles[i]
@inline getinnercircle(m::MyelinDomain, i::Int) = m.innercircles[i]
@inline getouterradius(m::MyelinDomain, i::Int) = radius(getoutercircle(m,i))
@inline getinnerradius(m::MyelinDomain, i::Int) = radius(getinnercircle(m,i))
@inline numfibres(m::MyelinDomain) = length(getoutercircles(m))

packingdensity(m::MyelinDomain) = estimate_density(getoutercircles(m))
factorize!(m::MyelinDomain) = (map(factorize!, getsubdomains(m)); return m)

function integrate(U::AbstractVector, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    @assert length(U) == numsubdomains(domain)
    Σ = zero(Vec{dim,T})
    @inbounds for i in 1:numsubdomains(domain)
        subdomain = getsubdomain(domain, i)
        @assert length(U[i]) == ndofs(getdofhandler(subdomain))
        Σ += integrate(U[i], subdomain)
    end
    return Σ
end

function interpolate!(U::Vector{Vector{T}},
                      f::Function,
                      domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    @assert length(U) == numsubdomains(domain)
    @inbounds for i in 1:length(U)
        subdomain = getsubdomain(domain, i)
        @assert length(U[i]) == ndofs(getdofhandler(subdomain))
        interpolate!(U[i], f, subdomain)
    end
    return U
end
function interpolate(f::Function, domain::MyelinDomain{dim}) where {dim}
    U = [zeros(ndofs(getsubdomain(domain, i))) for i in 1:numsubdomains(domain)]
    return interpolate!(U, f, domain)
end

function interpolate!(U::Vector{Vector{T}},
                      u0::Vec{dim,T},
                      domain::MyelinDomain) where {dim,T}
    @assert length(U) == numsubdomains(domain)
    @inbounds for i in 1:length(U)
        subdomain = getsubdomain(domain, i)
        @assert length(U[i]) == ndofs(getdofhandler(subdomain))
        interpolate!(U[i], u0, subdomain)
    end
    return U
end
function interpolate(u0::Vec{dim,T}, domain::MyelinDomain) where {dim,T}
    U = [zeros(ndofs(getsubdomain(domain, i))) for i in 1:numsubdomains(domain)]
    return interpolate!(U, u0, domain)
end

function LinearAlgebra.norm(U::Vector{Vector{T}}, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    @assert length(U) == numsubdomains(domain)
    return √sum(i->norm(U[i], getsubdomain(domain,i))^2, 1:numsubdomains(domain))
end

function Base.show(io::IO, m::MyelinDomain)
    compact = get(io, :compact, false)
    len = length(m.outercircles)
    subs = getsubdomains(m)
    ndof = sum(ndofs.(getdofhandler.(subs)))
    plural_s = len == 1 ? "" : "s"
    print(io, "$(typeof(m)) with $(len) fibre", plural_s, ", ",
              "$ndof total degrees of freedom, and $(length(subs)) subdomains")
    if !compact
        showdomain = (d) -> (print(io, "\n  "); show(IOContext(io, :compact => true), d))
        print(io, ":\n1 tissue domain:"); showdomain(m.tissuedomain)
        print(io, "\n$len myelin domain", plural_s, ":"); showdomain.(m.myelindomains)
        print(io, "\n$len axon domain", plural_s, ":"); showdomain.(m.axondomains)
    end
end

# ---------------------------------------------------------------------------- #
# Assmeble mass and stiffness matrices for MyelinProblem and BlochTorreyProblem
# ---------------------------------------------------------------------------- #

function doassemble!(prob::MyelinProblem{T},
                     domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    # Exterior region
    Rdecay = (x) -> prob.params.R2_lp
    Dcoeff = (x) -> prob.params.D_Tissue
    Omega = (x) -> omega_tissue(x, domain, prob.params)
    doassemble!(BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega), domain.tissuedomain)

    # Myelin sheath region
    Rdecay = (x) -> prob.params.R2_sp
    Dcoeff = (x) -> prob.params.D_Sheath
    for i in 1:numfibres(domain)
        Omega = (x) -> omega_myelin(x, domain, prob.params, i)
        doassemble!(BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega), domain.myelindomains[i])
    end

    # Axon region
    Rdecay = (x) -> prob.params.R2_lp
    Dcoeff = (x) -> prob.params.D_Axon
    for i in 1:numfibres(domain)
        Omega = (x) -> omega_axon(x, domain, prob.params, i)
        doassemble!(BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega), domain.axondomains[i])
    end

    return domain
end

# Assemble the `BlochTorreyProblem` system $M u_t = K u$ on the domain `domain`.
function doassemble!(prob::BlochTorreyProblem, domain::ParabolicDomain)
    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    Ke = zeros(n_basefuncs, n_basefuncs)
    Me = zeros(n_basefuncs, n_basefuncs)
    we = zeros(n_basefuncs)

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
        fill!(Ke, 0)
        fill!(Me, 0)
        fill!(we, 0)

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
            R = prob.Rdecay(coords_qp)
            D = prob.Dcoeff(coords_qp)
            ω = prob.Omega(coords_qp)

            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            for i in 1:n_basefuncs
                v  = shape_value(getcellvalues(domain), q_point, i)
                ∇v = shape_gradient(getcellvalues(domain), q_point, i)
                we[i] += sum(v) * dΩ # v[1] and v[2] are never non-zero together
                for j in 1:n_basefuncs
                    u = shape_value(getcellvalues(domain), q_point, j)
                    ∇u = shape_gradient(getcellvalues(domain), q_point, j)
                    Ke[i, j] -= (D * ∇v ⊡ ∇u + R * v ⋅ u - ω * v ⊠ u) * dΩ
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
# Example loop for evaluating surface/boundary integrals
# ---------------------------------------------------------------------------- #
# Loop over the edges of the cell for contributions to `Ke`. For example, if
# "Neumann Boundary" is a subset of boundary points, use:
#     `onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "Neumann Boundary")`
function surface_integral!(Ke, facevalues::FaceVectorValues{dim}, cell, q_point, coords, func::Function) where {dim}
   for face in 1:nfaces(cell)
       if !onboundary(cell, face)
           # Initialize face values
           reinit!(facevalues, cell, face)

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

# ---------------------------------------------------------------------------- #
# Creating LinearMap's for M*du/dt = K*u ODE systems
# ---------------------------------------------------------------------------- #

# Wrap the action of Mfact\K in a LinearMap
struct ParabolicLinearMap{T} <: LinearMap{T}
    M::AbstractMatrix{T}
    Mfact::Factorization{T}
    K::AbstractMatrix{T}
    function ParabolicLinearMap(M::AbstractMatrix{T}, Mfact::Factorization{T}, K::AbstractMatrix{T}) where {T}
        @assert (size(M) == size(K)) && (size(M,1) == size(M,2))
        @assert (size(M) == size(Mfact) || size(M) == 2 .* size(Mfact))
        new{T}(M, Mfact, K)
    end
end
ParabolicLinearMap(d::ParabolicDomain) = ParabolicLinearMap(getmass(d), getmassfact(d), getstiffness(d))

# Properties
Base.size(A::ParabolicLinearMap) = size(A.K)
LinearAlgebra.issymmetric(A::ParabolicLinearMap) = false
LinearAlgebra.ishermitian(A::ParabolicLinearMap) = false
LinearAlgebra.isposdef(A::ParabolicLinearMap) = false

@static if VERSION < v"0.7.0"
    # Multiplication action
    Minv_K_mul_u!(Y, X, K, Mfact) = (A_mul_B!(Y, K, X); copy!(Y, Mfact\Y); return Y)
    Kt_Minv_mul_u!(Y, X, K, Mfact) = (At_mul_B!(Y, K, Mfact\X); return Y)
    Kc_Minv_mul_u!(Y, X, K, Mfact) = (Ac_mul_B!(Y, K, Mfact\X); return Y)

    # Multiplication with Vector or Matrix
    Base.A_mul_B!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
    Base.At_mul_B!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = Kt_Minv_mul_u!(Y, X, A.K, A.Mfact)
    Base.Ac_mul_B!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = Kc_Minv_mul_u!(Y, X, A.K, A.Mfact)
    Base.A_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
    Base.At_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Kt_Minv_mul_u!(Y, X, A.K, A.Mfact)
    Base.Ac_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Kc_Minv_mul_u!(Y, X, A.K, A.Mfact)
    Base.:(*)(A::ParabolicLinearMap, X::AbstractVector) = A_mul_B!(similar(X, promote_type(eltype(A), eltype(X)), size(A, 1)), A, X)
    Base.:(*)(A::ParabolicLinearMap, X::AbstractMatrix) = A_mul_B!(similar(X, promote_type(eltype(A), eltype(X)), size(A, 1)), A, X)
else
    #TODO Check that this actually needs to be defined, and isn't a bug in LinearMaps
    LinearMaps.A_mul_B!( Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = mul!(Y, A, X)
    LinearMaps.At_mul_B!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = mul!(Y, transpose(A), X)
    LinearMaps.Ac_mul_B!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = mul!(Y, adjoint(A), X)
    LinearMaps.A_mul_B!( Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = mul!(Y, A, X)
    LinearMaps.At_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = mul!(Y, transpose(A), X)
    LinearMaps.Ac_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = mul!(Y, adjoint(A), X)
    LinearAlgebra.A_mul_B!( Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = mul!(Y, A, X)
    LinearAlgebra.At_mul_B!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = mul!(Y, transpose(A), X)
    LinearAlgebra.Ac_mul_B!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = mul!(Y, adjoint(A), X)
    LinearAlgebra.A_mul_B!( Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = mul!(Y, A, X)
    LinearAlgebra.At_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = mul!(Y, transpose(A), X)
    LinearAlgebra.Ac_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = mul!(Y, adjoint(A), X)

    # For taking powers of LinearMaps, e.g. A^2
    Base.to_power_type(A::Union{<:LinearMaps.AdjointMap, <:LinearMaps.TransposeMap, <:LinearMap}) = A

    function reassemble!(U,ux,uy)
        @assert size(ux,1) == size(uy,1)
        @assert size(U,1) == 2*size(ux,1)
        @inbounds for (i,iU) in enumerate(1:2:2*size(ux,1))
            @views U[iU  , :] = ux[i,:]
            @views U[iU+1, :] = uy[i,:]
        end
        return U
    end

    function Minv_u(Mfact, X)
        if size(Mfact,2) == size(X,1)
            Y = Mfact\X
            return Y
        elseif 2*size(Mfact,2) == size(X,1)
            Y = similar(X)
            x, y = X[1:2:end,:], X[2:2:end,:]
            reassemble!(Y, Mfact\x, Mfact\y)
            return Y
        else
            throw(DimensionMismatch("Minv_u"))
        end
    end

    # Multiplication action
    Minv_K_mul_u!(Y, X, K, Mfact) = (mul!(Y, K, X); copyto!(Y, Minv_u(Mfact, Y)); return Y)
    Kt_Minv_mul_u!(Y, X, K, Mfact) = (mul!(Y, transpose(K), Minv_u(Mfact, X)); return Y)
    Kc_Minv_mul_u!(Y, X, K, Mfact) = (mul!(Y, adjoint(K), Minv_u(Mfact, X)); return Y)

    # Multiplication with Vector or Matrix
    LinearAlgebra.mul!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
    LinearAlgebra.mul!(Y::AbstractVector, A::LinearMaps.TransposeMap{T, ParabolicLinearMap{T}}, X::AbstractVector) where {T} = Kt_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
    LinearAlgebra.mul!(Y::AbstractVector, A::LinearMaps.AdjointMap{T, ParabolicLinearMap{T}}, X::AbstractVector) where {T} = Kc_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
    LinearAlgebra.mul!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
    LinearAlgebra.mul!(Y::AbstractMatrix, A::LinearMaps.TransposeMap{T, ParabolicLinearMap{T}}, X::AbstractMatrix) where {T} = Kt_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
    LinearAlgebra.mul!(Y::AbstractMatrix, A::LinearMaps.AdjointMap{T, ParabolicLinearMap{T}}, X::AbstractMatrix) where {T} = Kc_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
end # if VERSION < v"0.7.0"

function LinearAlgebra.tr(A::LinearMap{T}, t::Int = 10) where {T}
    # Approximate trace using mat-vec's with basis vectors
    N = size(A, 2)
    t = min(t, N)
    x = zeros(T, N)
    tr = zero(T)
    for ix in StatsBase.sample(1:N, t; replace = false)
        x[ix] = one(T)
        tr += (A*x)[ix]
        x[ix] = zero(T)
    end
    return tr * (N/t)
end

# `norm`, `opnorm`, and `normAm`
Expmv.normAm(A::LinearMap, p::Real, t::Int = 10) = (Normest1.normest1(A^p, t)[1], 0)
LinearAlgebra.norm(A::LinearMap, p::Real, t::Int = 10) = expmv_norm(A, p, t)
LinearAlgebra.opnorm(A::LinearMap, p::Real, t::Int = 10) = expmv_norm(A, p, t)

# ---------------------------------------------------------------------------- #
# expmv and related functions
# ---------------------------------------------------------------------------- #

# Custom norm for calling expmv
function expmv_norm(A, p::Real=1, t::Int=10)
    !(size(A,1) == size(A,2)) && error("Matrix A must be square")
    !(p == 1 || p == Inf) && error("Only p=1 or p=Inf supported")
    p == Inf && (A = A')
    t = min(t, size(A,2))
    return Normest1.normest1(A, t)[1]
end
# Default fallback for vectors
expmv_norm(x::AbstractVector, p::Real=2, t::Int=10) = LinearAlgebra.norm(x, p)

function diffeq_solver(domain; abstol = 1e-8, reltol = 1e-8, linear_solver = :GMRES)
    function solver!(U, A, tspan, U0)
        signal, callbackfun = IntegrationCallback(U0, tspan[1], domain)
        prob = ODEProblem((du,u,p,t)->mul!(du,p[1],u), U0, tspan, (A,));
        sol = solve(prob, CVODE_BDF(linear_solver = linear_solver);
                    abstol = abstol,
                    reltol = reltol,
                    saveat = tspan,
                    alg_hints = :stiff,
                    callback = callbackfun)
        copyto!(U, sol.u[end])
        return U
    end
    return solver!
end

function expokit_solver(domain; tol = 1e-8, m = 30, opnorm = expmv_norm)
    function solver!(U, A, tspan, U0)
        anorm = opnorm(A, Inf)
        Expokit.expmv!(U, tspan[end], A, U0;
            anorm = anorm,
            tol = tol,
            m = m)
        return U
    end
    return solver!
end

function expmv_solver(domain; prec = "single", opnorm = expmv_norm)
    function solver!(U, A, tspan, U0)
        M = Expmv.select_taylor_degree(A, U0; opnorm = opnorm)[1]
        Expmv.expmv!(U, tspan[end], A, U0;
            prec = prec,
            M = M,
            opnorm = opnorm)
        return U
    end
    return solver!
end

# ---------------------------------------------------------------------------- #
# DiscreteCallback for integrating system using DifferentialEquations.jl
# ---------------------------------------------------------------------------- #
struct SignalIntegrator{uDim,gDim,Nd,T,Nf}
    time::Vector{T}
    signal::Vector{Vec{uDim,T}}
    domain::ParabolicDomain{gDim,Nd,T,Nf}
end
function (p::SignalIntegrator)(int)
    push!(p.signal, integrate(int.u, p.domain))
    push!(p.time, int.t)
    u_modified!(int, false)
end
function IntegrationCallback(u0, t0, domain)
    intial_signal = integrate(u0, domain)
    signalintegrator! = SignalIntegrator([t0], [intial_signal], domain)
    discretecallback = DiscreteCallback((u,t,int) -> true, signalintegrator!, save_positions = (false, false))
    return signalintegrator!, discretecallback
end

gettime(p::SignalIntegrator) = p.time
getsignal(p::SignalIntegrator) = p.signal
signalnorm(p::SignalIntegrator) = norm.(p.signal)
relativesignalnorm(p::SignalIntegrator) = signalnorm(p)./norm(getsignal(p)[1])
relativesignal(p::SignalIntegrator) = (S = getsignal(p); return S./norm(S[1]))

function Base.show(io::IO, p::SignalIntegrator)
    compact = get(io, :compact, false)
    nsignals = length(p.signal)
    ntimes = length(p.time)
    plural_s = nsignals == 1 ? "" : "s"
    print(io, "$(typeof(p))")
    if compact || !compact
        print(io, " with $nsignals stored signal", plural_s)
    else
        print(io, "\n    time: $ntimes-element ", typeof(p.time))
        print(io, "\n  signal: $nsignals-element ", typeof(p.signal))
        print(io, "\n  domain: "); show(IOContext(io, :compact => true), p.domain)
    end
end

# ---------------------------------------------------------------------------- #
# Local frequency perturbation map functions
# ---------------------------------------------------------------------------- #

struct OmegaDerivedConstants{T}
    ω₀::T
    s²::T
    c²::T
end
function OmegaDerivedConstants(p::BlochTorreyParameters{T}) where {T}
    γ, B₀, θ = p.gamma, p.B0, p.theta
    ω₀ = γ * B₀
    s², c² = sin(θ)^2, cos(θ)^2
    return OmegaDerivedConstants{T}(ω₀, s², c²)
end

@inline function omega_tissue(x::Vec{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χI, χA, ri², ro² = p.ChiI, p.ChiA, radius(c_in)^2, radius(c_out)^2
    dx = x - origin(c_in)
    r² = dx⋅dx
    cos2ϕ = (dx[1]-dx[2])*(dx[1]+dx[2])/r² # cos2ϕ == (x²-y²)/r² == (x-y)(x+y)/r²

    tmp = b.s² * cos2ϕ * ((ro² - ri²)/r²) # Common calculation
    I = χI/2 * tmp # isotropic component
    A = χA/8 * tmp # anisotropic component
    return b.ω₀ * (I + A)
end

@inline function omega_myelin(x::Vec{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χI, χA, ri², ro = p.ChiI, p.ChiA, radius(c_in)^2, radius(c_out)
    dx = x - origin(c_in)
    r² = dx⋅dx
    cos2ϕ = (dx[1]-dx[2])*(dx[1]+dx[2])/r² # cos2ϕ == (x²-y²)/r² == (x-y)(x+y)/r²
    r = √r²

    I = χI/2 * (b.c² - 1/3 - b.s² * cos2ϕ * ri² / r²) # isotropic component
    A = χA * (b.s² * (-5/12 - cos2ϕ/8 * (1 + ri²/r²) + 3/4 * log(ro/r)) - b.c²/6) # anisotropic component
    return b.ω₀ * (I + A)
end

@inline function omega_axon(x::Vec{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χA, ri, ro = p.ChiA, radius(c_in), radius(c_out)
    A = 3/4 * χA * b.s² * log(ro/ri) # anisotropic (and only) component
    return b.ω₀ * A
end

# ---------------------------------------------------------------------------- #
# Global frequency perturbation functions: calculate ω(x) due to entire domain
# ---------------------------------------------------------------------------- #

# Calculate ω(x) inside region number `domain`, which is assumed to be tissue
function omega_tissue(x::Vec{2}, domain::MyelinDomain, params::BlochTorreyParameters)
    constants = OmegaDerivedConstants(params)
    ω = zero(eltype(x))
    @inbounds for i in 1:numfibres(domain)
        ω += omega_tissue(x, params, constants, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

# Calculate ω(x) inside region number `domain`, which is assumed to be myelin
function omega_myelin(x::Vec{2}, domain::MyelinDomain, params::BlochTorreyParameters, region::Int)
    constants = OmegaDerivedConstants(params)
    ω = omega_myelin(x, params, constants, getinnercircle(domain, region), getoutercircle(domain, region))
    @inbounds for i in Iterators.flatten((1:region-1, region+1:numfibres(domain)))
        ω += omega_tissue(x, params, constants, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

# Calculate ω(x) inside region number `domain`, which is assumed to be axonal
function omega_axon(x::Vec{2}, domain::MyelinDomain, params::BlochTorreyParameters, region::Int)
    constants = OmegaDerivedConstants(params)
    ω = omega_axon(x, params, constants, getinnercircle(domain, region), getoutercircle(domain, region))
    @inbounds for i in Iterators.flatten((1:region-1, region+1:numfibres(domain)))
        ω += omega_tissue(x, params, constants, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

# struct FreqMapParams{T}
#     ω₀::T
#     s²::T
#     c²::T
# end
#
# function FreqMapParams(p::BlochTorreyParameters{T}) where {T}
#     γ, B₀, θ = p.gamma, p.B0, p.theta
#     ω₀ = γ * B₀
#     s², c² = sin(θ)^2, cos(θ)^2
#     return FreqMapParams{T}(ω₀, s², c²)
# end
#
# # Calculate ω(x) inside region number `region`, which is assumed to be myelin
# function omega_myelin(
#     x::Vec{2,T},
#     domain::MyelinDomain{dim,Nd,T,Nf},
#     btparams::BlochTorreyParameters{T},
#     region::Int) where {dim,Nd,T,Nf}
#     freqparams = FreqMapParams(btparams)
#     ω = omega_myelin(x, freqparams, btparams, getinnercircle(domain, region), getoutercircle(domain, region))
#     @inbounds for i in IterTools.chain(1:region-1, region+1:numfibres(domain))
#         ω += omega_tissue(x, freqparams, btparams, getinnercircle(domain, i), getoutercircle(domain, i))
#     end
#     return ω
# end
#
# # Calculate ω(x) inside region number `region`, which is assumed to be axon
# function omega_axon(
#     x::Vec{2,T},
#     domain::MyelinDomain{dim,Nd,T,Nf},
#     btparams::BlochTorreyParameters{T},
#     region::Int) where {dim,Nd,T,Nf}
#     freqparams = FreqMapParams(btparams)
#     ω = omega_axon(x, freqparams, btparams, getinnercircle(domain, region), getoutercircle(domain, region))
#     @inbounds for i in IterTools.chain(1:region-1, region+1:numfibres(domain))
#         ω += omega_tissue(x, freqparams, btparams, getinnercircle(domain, i), getoutercircle(domain, i))
#     end
#     return ω
# end
#
# # Calculate ω(x) inside region number `region`, which is assumed to be tissue
# function omega_tissue(
#     x::Vec{2,T},
#     domain::MyelinDomain{dim,Nd,T,Nf},
#     btparams::BlochTorreyParameters{T}) where {dim,Nd,T,Nf}
#     freqparams = FreqMapParams(btparams)
#     ω = zero(T)
#     @inbounds for i in 1:numfibres(domain)
#         ω += omega_tissue(x, freqparams, btparams, getinnercircle(domain, i), getoutercircle(domain, i))
#     end
#     return ω
# end
#
# @inline function omega_isotropic_tissue(x::Vec{2},
#     b::FreqMapParams,
#     p::BlochTorreyParameters,
#     c_inner::Circle{2},
#     c_outer::Circle{2})
#     χI, ri², ro² = p.ChiI, radius(c_inner)^2, radius(c_outer)^2
#     dx = x - origin(c_inner)
#     r² = dx⋅dx
#     cos2ϕ = ((dx[1]-dx[2])*(dx[1]+dx[2]))/r²
#     return b.ω₀ * χI * b.s²/2 * cos2ϕ * (ro² - ri²)/r²
# end
#
# @inline function omega_anisotropic_tissue(x::Vec{2},
#     b::FreqMapParams,
#     p::BlochTorreyParameters,
#     c_inner::Circle{2},
#     c_outer::Circle{2})
#     χA, ri², ro² = p.ChiA, radius(c_inner)^2, radius(c_outer)^2
#     dx = x - origin(c_inner)
#     r² = dx⋅dx
#     cos2ϕ = ((dx[1]-dx[2])*(dx[1]+dx[2]))/r²
#     return b.ω₀ * χA * b.s²/8 * cos2ϕ * (ro² - ri²)/r²
# end
#
# @inline function omega_isotropic_myelin(x::Vec{2},
#     b::FreqMapParams,
#     p::BlochTorreyParameters,
#     c_inner::Circle{2},
#     c_outer::Circle{2})
#     χI, ri² = p.ChiI, radius(c_inner)^2
#     dx = x - origin(c_inner)
#     r² = dx⋅dx
#     cos2ϕ = ((dx[1]-dx[2])*(dx[1]+dx[2]))/r²
#     return b.ω₀ * χI * ( b.c² - 1/3 - b.s² * cos2ϕ * ri² / r² )/2
# end
#
# @inline function omega_anisotropic_myelin(x::Vec{2},
#     b::FreqMapParams,
#     p::BlochTorreyParameters,
#     c_inner::Circle{2},
#     c_outer::Circle{2})
#     χA, ri², ro = p.ChiA, radius(c_inner)^2, radius(c_outer)
#     dx = x - origin(c_inner)
#     r² = dx⋅dx
#     r = √r²
#     cos2ϕ = ((dx[1]-dx[2])*(dx[1]+dx[2]))/r²
#     return b.ω₀ * χA * ( b.s² * (-5/12 - cos2ϕ/8 * (1 + ri²/r²) + 3/4 * log(ro/r)) - b.c²/6 )
# end
#
# @inline function omega_anisotropic_axon(x::Vec{2},
#     b::FreqMapParams,
#     p::BlochTorreyParameters,
#     c_inner::Circle{2},
#     c_outer::Circle{2})
#     χA, ri, ro = p.ChiA, radius(c_inner), radius(c_outer)
#     return b.ω₀ * χA * 3b.s²/4 * log(ro/ri)
# end
#
# # Sum components to omega in the tissue region
# @inline function omega_tissue(x::Vec{2},
#     b::FreqMapParams,
#     p::BlochTorreyParameters,
#     c_inner::Circle{2},
#     c_outer::Circle{2})
#     return omega_isotropic_tissue(x,b,p,c_inner,c_outer) + omega_anisotropic_tissue(x,b,p,c_inner,c_outer)
# end
#
# # Sum components to omega in the myelin sheath region
# @inline function omega_myelin(x::Vec{2},
#     b::FreqMapParams,
#     p::BlochTorreyParameters,
#     c_inner::Circle{2},
#     c_outer::Circle{2})
#     return omega_isotropic_myelin(x,b,p,c_inner,c_outer) + omega_anisotropic_myelin(x,b,p,c_inner,c_outer)
# end
#
# # Sum components to omega in the axonal region
# @inline function omega_axon(x::Vec{2},
#     b::FreqMapParams,
#     p::BlochTorreyParameters,
#     c_inner::Circle{2},
#     c_outer::Circle{2})
#     return omega_anisotropic_axon(x,b,p,c_inner,c_outer)
# end

nothing
