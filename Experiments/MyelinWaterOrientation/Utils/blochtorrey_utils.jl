# ---------------------------------------------------------------------------- #
# BlochTorreyUtils
# ---------------------------------------------------------------------------- #

module BlochTorreyUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

include("normest1.jl")
using .Normest1: normest1
using ..GeometryUtils
using ..MeshUtils
using Parameters: @with_kw
using JuAFEM
using LinearAlgebra
using SparseArrays
using StatsBase
using LinearMaps
using Distributions

# ---------------------------------------------------------------------------- #
# Exported Methods
# ---------------------------------------------------------------------------- #
export doassemble!, addquadweights!, interpolate, interpolate!, integrate
export normest1_norm, radiidistribution
export axongrid, tissuegrid, myelingrid, packingdensity, factorize!, numfibres
export numsubdomains, numtissuedomains, nummyelindomains, numaxondomains, getsubdomain
export getgrid, gettissuedomain, getmyelindomains, getaxondomains
export getdofhandler, getcellvalues, getfacevalues,
       getmass, getmassfact, getstiffness, getquadweights,
       getsubdomain, getsubdomains, getboundary,
       getoutercircles, getinnercircles, getoutercircle, getinnercircle, getouterradius, getinnerradius

# ---------------------------------------------------------------------------- #
# Exported Types
# ---------------------------------------------------------------------------- #

export BlochTorreyParameters,
       AbstractParabolicProblem,
       MyelinProblem,
       BlochTorreyProblem,
       AbstractDomain,
       ParabolicDomain,
       MyelinDomain,
       ParabolicLinearMap

# Convenience definitions
const MyelinBoundary{gDim,T} = Union{<:Circle{gDim,T}, <:Rectangle{gDim,T}, <:Ellipse{gDim,T}}
const VectorOfVectors{T} = Vector{Vector{T}}

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

# AbstractParabolicProblem type
abstract type AbstractParabolicProblem{T} end

# MyelinProblem: holds a `BlochTorreyParameters` set of parameters
struct MyelinProblem{T} <: AbstractParabolicProblem{T}
    params::BlochTorreyParameters{T}
end

# BlochTorreyProblem: holds the only parameters necessary to solve the Bloch-
# Torrey equation, naming the Dcoeff, Rdecay, and Omega functions of position
struct BlochTorreyProblem{T,D,R,W} <: AbstractParabolicProblem{T}
    Dcoeff::D # Function which takes a Vec `x` and outputs Dcoeff(x)
    Rdecay::R # Function which takes a Vec `x` and outputs Rdecay(x)
    Omega::W # Function which takes a Vec `x` and outputs Omega(x)
    BlochTorreyProblem{T}(d::D,r::R,w::W) where {T,D,R,W} = new{T,D,R,W}(d,r,w)
end

# Abstract domain type. The type parameters are:
#   `uDim`:  Dimension of `u`
#   `gDim`:  Spatial dimension of domain
#   `T`:    Float type used
#   `Nd`:   Number of nodes per finite element
#   `Nf`:   Number of faces per finite element
abstract type AbstractDomain{uDim,gDim,T,Nd,Nf} end

# ParabolicDomain: generic domain type which holds this information necessary to
# solve a parabolic FEM problem M*du/dt = K*u
mutable struct ParabolicDomain{uDim,gDim,T,Nd,Nf} <: AbstractDomain{uDim,gDim,T,Nd,Nf}
    grid::Grid{gDim,Nd,T,Nf}
    dh::DofHandler{gDim,Nd,T,Nf}
    cellvalues::CellValues{gDim,T}
    facevalues::FaceValues{gDim,T}
    M::Symmetric{T,<:SparseMatrixCSC{T}}
    Mfact::Union{Factorization{T},Nothing}
    K::SparseMatrixCSC{T}
    w::Vector{T}
    function ParabolicDomain(
        grid::Grid{gDim,Nd,T,Nf}, ::Val{uDim} = Val(2);
        refshape = RefTetrahedron,
        quadorder = 3,
        funcinterporder = 1,
        geominterporder = 1) where {uDim,gDim,Nd,T,Nf}

        @assert uDim == 2 #TODO: where is this assumption? likely, assume dim(u) == dim(grid) somewhere

        # Quadrature and interpolation rules and corresponding cellvalues/facevalues
        func_interp = Lagrange{gDim, refshape, funcinterporder}()
        geom_interp = Lagrange{gDim, refshape, geominterporder}()
        quadrule = QuadratureRule{gDim, refshape}(quadorder)
        quadrule_face = QuadratureRule{gDim-1, refshape}(quadorder)
        cellvalues = CellVectorValues(T, quadrule, func_interp, geom_interp)
        facevalues = FaceVectorValues(T, quadrule_face, func_interp, geom_interp)
        # cellvalues = CellScalarValues(T, quadrule, func_interp, geom_interp)
        # facevalues = FaceScalarValues(T, quadrule_face, func_interp, geom_interp)

        # Degree of freedom handler
        dh = DofHandler(grid)
        push!(dh, :u, uDim, func_interp)
        close!(dh)

        # Mass matrix, inverse mass matrix, stiffness matrix, and weights vector
        M = create_symmetric_sparsity_pattern(dh)
        K = create_sparsity_pattern(dh)
        w = zeros(T, ndofs(dh))
        Mfact = nothing

        new{uDim,gDim,T,Nd,Nf}(grid, dh, cellvalues, facevalues, M, Mfact, K, w)
    end
end

# MyelinDomain: generic domain type which holds this information necessary to
# solve a parabolic FEM problem M*du/dt = K*u on a number of subdomains which
# represent different parts of the Myelin tissue. This subdomains are
# represented themselves as ParabolicDomain's
mutable struct MyelinDomain{uDim,gDim,T,Nd,Nf} <: AbstractDomain{uDim,gDim,T,Nd,Nf}
    fullgrid::Grid{gDim,Nd,T,Nf}
    outercircles::Vector{Circle{gDim,T}}
    innercircles::Vector{Circle{gDim,T}}
    domainboundary::MyelinBoundary{gDim,T}
    tissuedomain::ParabolicDomain{uDim,gDim,T,Nd,Nf}
    myelindomains::Vector{ParabolicDomain{uDim,gDim,T,Nd,Nf}}
    axondomains::Vector{ParabolicDomain{uDim,gDim,T,Nd,Nf}}
    function MyelinDomain(
            fullgrid::Grid{gDim,Nd,T,Nf},
            outercircles::Vector{Circle{gDim,T}},
            innercircles::Vector{Circle{gDim,T}},
            domainboundary::MyelinBoundary{gDim,T},
            tissuegrid::Grid{gDim,Nd,T,Nf},
            myelingrids::Vector{Grid{gDim,Nd,T,Nf}},
            axongrids::Vector{Grid{gDim,Nd,T,Nf}},
            ::Val{uDim} = Val(2);
            kwargs...) where {uDim,gDim,T,Nd,Nf}
        return new{uDim,gDim,T,Nd,Nf}(
            fullgrid, outercircles, innercircles, domainboundary,
            ParabolicDomain(tissuegrid, Val(uDim); kwargs...),
            ParabolicDomain.(myelingrids, Val(uDim); kwargs...),
            ParabolicDomain.(axongrids, Val(uDim); kwargs...))
    end
end

# ParabolicLinearMap: create a LinearMaps subtype which wrap the action of
# Mfact\K in a LinearMap object
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

# ---------------------------------------------------------------------------- #
# BlochTorreyParameters methods
# ---------------------------------------------------------------------------- #

radiidistribution(p::BlochTorreyParameters) = Distributions.Gamma(p.R_shape, p.R_mu/p.R_shape)

# ---------------------------------------------------------------------------- #
# AbstractDomain methods
# ---------------------------------------------------------------------------- #

# Interpolation is done by simply creating a `Dirichlet` constraint on every
# face of the domain and applying it to the vector `u`. This is really quite
# slow and wasteful, and there is almost definitely a better way to implement
# this, but it just isn't a bottleneck and this is easy.
function interpolate!(u::Vector{T},
                      f::Function,
                      domain::AbstractDomain{uDim,gDim,T}) where {uDim,gDim,T}
    ch = ConstraintHandler(getdofhandler(domain))
    ∂Ω = getfaces(getgrid(domain))
    dbc = JuAFEM.Dirichlet(:u, ∂Ω, (x,t) -> f(x), collect(1:uDim))
    add!(ch, dbc)
    close!(ch)
    update!(ch, zero(T)) # time zero
    apply!(u, ch)
    return u
end
interpolate(f::Function, domain::AbstractDomain) = interpolate!(zeros(ndofs(getdofhandler(domain))), f, domain)

# Optimization for when we can guarantee that the degrees of freedom `u` are
# purely nodal and we just want to assign a constant vector `u0` to each node
function interpolate!(u::Vector{T}, u0::Vec{uDim,T}, domain::AbstractDomain{uDim}) where {uDim,T}
    # Check that `u` has the correct length
    @assert length(u) == ndofs(getdofhandler(domain))
    if length(u) == uDim * getnnodes(getgrid(domain))
        # degrees of freedom are nodal; can efficiently assign directly
        u = reinterpret(Vec{uDim,T}, u)
        u .= (u0,)
        u = reinterpret(T, u)
    else
        # degrees of freedom are not nodal; call general projection
        interpolate!(u, x->u0, domain)
    end
    return u
end
interpolate(u0::Vec, domain::AbstractDomain) = interpolate!(zeros(ndofs(getdofhandler(domain))), u0, domain)

function integrate(u::Vector{Tu}, domain::AbstractDomain{uDim,gDim,T}) where {Tu,uDim,gDim,T}
    @assert length(u) == ndofs(getdofhandler(domain))
    u = reinterpret(Vec{uDim,Tu}, u)
    w = reinterpret(Vec{uDim,T}, getquadweights(domain))
    # Integrate. ⊙ == hadamardproduct is the Hadamard product of the Vec's.
    S = u[1] ⊙ w[1]
    @inbounds for i in 2:length(u)
        S += u[i] ⊙ w[i]
    end
    return S
end

# ---------------------------------------------------------------------------- #
# ParabolicDomain methods
# ---------------------------------------------------------------------------- #

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

# Quad weights are vectors of Vec{uDim,T} and are the same for each component;
# just taking the sum of the first component will return the area
GeometryUtils.area(d::ParabolicDomain{uDim}) where {uDim} = sum(@views getquadweights(d)[1:uDim:end])

# Show methods
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
function addquadweights!(domain::ParabolicDomain{uDim,gDim,T}) where {uDim,gDim,T}
    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    we = zeros(T, n_basefuncs)
    fill!(getquadweights(domain), zero(T))

    @inbounds for cell in CellIterator(getdofhandler(domain))
        # Reset element residual and reinit cellvalues
        fill!(we, zero(T))
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
function doassemble!(domain::ParabolicDomain{uDim,gDim,T}) where {uDim,gDim,T}
    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    Ke = zeros(T, n_basefuncs, n_basefuncs)
    Me = zeros(T, n_basefuncs, n_basefuncs)
    we = zeros(T, n_basefuncs)

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
        fill!(Ke, zero(T))
        fill!(Me, zero(T))
        fill!(we, zero(T))

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
# MyelinDomain methods
# ---------------------------------------------------------------------------- #

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
@inline gettissuedomain(m::MyelinDomain) = m.tissuedomain
@inline getmyelindomains(m::MyelinDomain) = m.myelindomains
@inline getaxondomains(m::MyelinDomain) = m.axondomains
@inline getoutercircles(m::MyelinDomain) = m.outercircles
@inline getinnercircles(m::MyelinDomain) = m.innercircles
@inline getboundary(m::MyelinDomain) = m.domainboundary
@inline tissuegrid(m::MyelinDomain) = getgrid(m.tissuedomain)
@inline myelingrid(m::MyelinDomain, i::Int) = getgrid(m.myelindomains[i])
@inline axongrid(m::MyelinDomain, i::Int) = getgrid(m.axondomains[i])

@inline getoutercircle(m::MyelinDomain, i::Int) = m.outercircles[i]
@inline getinnercircle(m::MyelinDomain, i::Int) = m.innercircles[i]
@inline getouterradius(m::MyelinDomain, i::Int) = radius(getoutercircle(m,i))
@inline getinnerradius(m::MyelinDomain, i::Int) = radius(getinnercircle(m,i))
@inline numfibres(m::MyelinDomain) = length(getoutercircles(m))

packingdensity(m::MyelinDomain) = estimate_density(getoutercircles(m))
GeometryUtils.area(m::MyelinDomain) = sum(d -> area(d), getsubdomains(m))
factorize!(m::MyelinDomain) = (map(factorize!, getsubdomains(m)); return m)

integrate(U::VectorOfVectors, domain::MyelinDomain) = sum((u,d) -> integrate(u,d), U, getsubdomains(domain))
interpolate!(U::VectorOfVectors, f::Function, domain::MyelinDomain) = map!((u,d) -> interpolate!(u, f, d), U, U, getsubdomains(domain))
interpolate!(U::VectorOfVectors, u0::Vec{uDim}, domain::MyelinDomain{uDim}) where {uDim} = map!((u,d) -> interpolate!(u, u0, d), U, U, getsubdomains(domain))

function interpolate(f::Function, domain::MyelinDomain{uDim,gDim,T}) where {uDim,gDim,T}
    U = [zeros(T, ndofs(getsubdomain(domain, i))) for i in 1:numsubdomains(domain)]
    return interpolate!(U, f, domain)
end
function interpolate(u0::Vec{uDim,T}, domain::MyelinDomain{uDim,gDim,T}) where {uDim,gDim,T}
    U = [zeros(T, ndofs(getsubdomain(domain, i))) for i in 1:numsubdomains(domain)]
    return interpolate!(U, u0, domain)
end

function LinearAlgebra.norm(U::VectorOfVectors, domain::MyelinDomain)
    @assert length(U) == numsubdomains(domain)
    return √sum(i -> norm(U[i], getsubdomain(domain,i))^2, 1:numsubdomains(domain))
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
# Assemble mass and stiffness matrices for MyelinProblem and BlochTorreyProblem
# ---------------------------------------------------------------------------- #

function doassemble!(prob::MyelinProblem{T}, domain::MyelinDomain) where {T}
    # Exterior region
    Rdecay = (x) -> prob.params.R2_Tissue # Tissue R2 #TODO which to use?
    # Rdecay = (x) -> prob.params.R2_lp # Large pool R2
    Dcoeff = (x) -> prob.params.D_Tissue # Dcoeff of tissue
    Omega = (x) -> omega_tissue(x, domain, prob.params) # Tissue region
    doassemble!(BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega), domain.tissuedomain)

    # Myelin sheath region
    Rdecay = (x) -> prob.params.R2_sp # Small pool R2
    Dcoeff = (x) -> prob.params.D_Sheath # Dcoeff of sheath
    for i in 1:numfibres(domain)
        Omega = (x) -> omega_myelin(x, domain, prob.params, i) # Sheath region
        doassemble!(BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega), domain.myelindomains[i])
    end

    # Axon region
    Rdecay = (x) -> prob.params.R2_lp # Large pool R2
    Dcoeff = (x) -> prob.params.D_Axon # Dcoeff of axonal region
    for i in 1:numfibres(domain)
        Omega = (x) -> omega_axon(x, domain, prob.params, i) # Axonal region
        doassemble!(BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega), domain.axondomains[i])
    end

    return domain
end

# Assemble the `BlochTorreyProblem` system $M u_t = K u$ on the domain `domain`.
function doassemble!(prob::BlochTorreyProblem, domain::ParabolicDomain{uDim,gDim,T}) where {uDim,gDim,T}
    # This assembly function is only for CellVectorValues
    @assert typeof(getcellvalues(domain)) <: CellVectorValues

    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    n_basefuncs = getnbasefunctions(getcellvalues(domain))
    Ke = zeros(T, n_basefuncs, n_basefuncs)
    Me = zeros(T, n_basefuncs, n_basefuncs)
    we = zeros(T, n_basefuncs)

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
        fill!(Ke, zero(T))
        fill!(Me, zero(T))
        fill!(we, zero(T))

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
                we[i] += sum(v) * dΩ
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
function surface_integral!(Ke, facevalues::FaceVectorValues, cell, q_point, coords, func::Function)
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
# ParabolicLinearMap methods: LinearMap's for solving M*du/dt = K*u ODE systems
# ---------------------------------------------------------------------------- #

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
            return Mfact\X
        elseif 2*size(Mfact,2) == size(X,1)
            x, y = X[1:2:end,:], X[2:2:end,:]
            return reassemble!(similar(X), Mfact\x, Mfact\y)
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
function normest1_norm(A::LinearMap, p::Real=1, t::Int=10)
    !(size(A,1) == size(A,2)) && error("Matrix A must be square")
    !(p == 1 || p == Inf) && error("Only p=1 or p=Inf supported")
    p == Inf && (A = A')
    t = min(t, size(A,2))
    return Normest1.normest1(A, t)[1]
end
LinearAlgebra.norm(A::LinearMap, p::Real, t::Int = 10) = normest1_norm(A, p, t)
LinearAlgebra.opnorm(A::LinearMap, p::Real, t::Int = 10) = normest1_norm(A, p, t)

# ---------------------------------------------------------------------------- #
# Local frequency perturbation map functions
# ---------------------------------------------------------------------------- #
struct OmegaDerivedConstants{T}
    ω₀::T
    s²::T
    c²::T
    function OmegaDerivedConstants(p::BlochTorreyParameters{T}) where {T}
        γ, B₀, θ = p.gamma, p.B0, p.theta
        ω₀ = γ * B₀
        s, c = sincos(θ)
        return new{T}(ω₀, s^2, c^2)
    end
end

@inline function omega_tissue(x::Vec{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χI, χA, ri², ro² = p.ChiI, p.ChiA, radius(c_in)^2, radius(c_out)^2
    dx = x - origin(c_in)
    r² = dx⋅dx
    cos2ϕ = (dx[1]-dx[2])*(dx[1]+dx[2])/r² # cos2ϕ == (x²-y²)/r² == (x-y)*(x+y)/r²

    tmp = b.s² * cos2ϕ * ((ro² - ri²)/r²) # Common calculation
    I = χI/2 * tmp # isotropic component
    A = χA/8 * tmp # anisotropic component
    return b.ω₀ * (I + A)
end

@inline function omega_myelin(x::Vec{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χI, χA, ri², ro = p.ChiI, p.ChiA, radius(c_in)^2, radius(c_out)
    dx = x - origin(c_in)
    r² = dx⋅dx
    cos2ϕ = (dx[1]-dx[2])*(dx[1]+dx[2])/r² # cos2ϕ == (x²-y²)/r² == (x-y)*(x+y)/r²
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

# Calculate ω(x) inside region number `region`, which is assumed to be tissue
function omega_tissue(x::Vec{2}, domain::MyelinDomain, params::BlochTorreyParameters)
    constants = OmegaDerivedConstants(params)
    ω = zero(eltype(x))
    @inbounds for i in 1:numfibres(domain)
        ω += omega_tissue(x, params, constants, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

# Calculate ω(x) inside region number `region`, which is assumed to be myelin
function omega_myelin(x::Vec{2}, domain::MyelinDomain, params::BlochTorreyParameters, region::Int)
    constants = OmegaDerivedConstants(params)
    ω = omega_myelin(x, params, constants, getinnercircle(domain, region), getoutercircle(domain, region))
    @inbounds for i in Iterators.flatten((1:region-1, region+1:numfibres(domain)))
        ω += omega_tissue(x, params, constants, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

# Calculate ω(x) inside region number `region`, which is assumed to be axonal
function omega_axon(x::Vec{2}, domain::MyelinDomain, params::BlochTorreyParameters, region::Int)
    constants = OmegaDerivedConstants(params)
    ω = omega_axon(x, params, constants, getinnercircle(domain, region), getoutercircle(domain, region))
    @inbounds for i in Iterators.flatten((1:region-1, region+1:numfibres(domain)))
        ω += omega_tissue(x, params, constants, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

end # module BlochTorreyUtils

nothing
