# ---------------------------------------------------------------------------- #
# BlochTorreyUtils
# ---------------------------------------------------------------------------- #

module BlochTorreyUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using Normest1
using GeometryUtils
using MeshUtils
using DistMesh
using JuAFEM
using LinearAlgebra
using SparseArrays
using SuiteSparse # need to define ldiv! for SuiteSparse.CHOLMOD.Factor
using StatsBase
using LinearMaps
using Parameters: @with_kw, @unpack

import Distributions
import Lazy

# ---------------------------------------------------------------------------- #
# Exported Methods
# ---------------------------------------------------------------------------- #
export normest1_norm, radiidistribution
export doassemble!, addquadweights!, factorize!, interpolate, interpolate!, integrate
export getgrid, getdomain, numfibres, createmyelindomains, omegamap
export getdofhandler, getcellvalues, getfacevalues,
       getmass, getmassfact, getstiffness, getquadweights,
       getregion, getoutercircles, getinnercircles, getoutercircle, getinnercircle, getouterradius, getinnerradius
export testproblem

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
       AbstractRegion,
       AbstractRegionUnion,
       AxonRegion,
       MyelinRegion,
       TissueRegion,
       PermeableInterfaceRegion,
       ParabolicLinearMap,
       DiffEqParabolicLinearMapWrapper

# Convenience definitions
const MyelinBoundary{gDim,T} = Union{<:Circle{gDim,T}, <:Rectangle{gDim,T}, <:Ellipse{gDim,T}}
const VectorOfVectors{T} = AbstractVector{<:AbstractVector{T}}
const MaybeSymmSparseMatrixCSC{T} = Union{<:SparseMatrixCSC{T}, <:Symmetric{T,<:SparseMatrixCSC{T}}}
# const MaybeNothingFactorization{T} = Union{Nothing, <:Factorization{T}}

const MassType{T} = MaybeSymmSparseMatrixCSC{T}
# const MassFactType{T} = Factorization{T}
const StiffnessType{T} = SparseMatrixCSC{T}

# Struct of BlochTorreyParameters. T is the float type.
@with_kw struct BlochTorreyParameters{T}
    B0::T             =    T(-3.0)          # External magnetic field (z-direction) [T]
    gamma::T          =    T(2.67515255e8)  # Gyromagnetic ratio [rad/s/T]
    theta::T          =    T(π)/2           # Main magnetic field angle w.r.t B0 [rad]
    g_ratio::T        =    T(0.8370)        # g-ratio (original 0.71), 0.84658 for healthy, 0.8595 for MS.
    R2_sp::T          =    T(inv(15e-3))    # #TODO (play with these?) Relaxation rate of small pool [s^-1] (Myelin) (Xu et al. 2017) (15e-3s)
    R2_lp::T          =    T(inv(63e-3))    # #TODO (play with these?) 1st attempt was 63E-3. 2nd attempt 76 ms
    R2_Tissue::T      =    T(inv(63e-3))    # #TODO (was 14.5Hz; changed to match R2_lp) Relaxation rate of tissue [s^-1]
    R2_water::T       =    T(inv(2.2))      # Relaxation rate of pure water
    D_Tissue::T       =    T(1500.0)        # #TODO (reference?) Diffusion coefficient in tissue [um^2/s]
    D_Sheath::T       =    T(250.0)         # #TODO (reference?) Diffusion coefficient in myelin sheath [um^2/s]
    D_Axon::T         =    T(2000.0)        # #TODO (reference?) Diffusion coefficient in axon interior [um^2/s]
    D_Blood::T        =    T(3037.0)        # Diffusion coefficient in blood [um^2/s]
    D_Water::T        =    T(3037.0)        # Diffusion coefficient in water [um^2/s]
    K_perm            =    T(1.0e-3)        # #TODO (reference?) Interface permeability constant [um/s]
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
    R2_Fe::T          =    T(inv(1e-6))     # Relaxation rate of iron in ferritin. Assumed to be really high.
    R2_WM::T          =    T(inv(70e-3))    # Relaxation rate of frontal WM. This is empirical;taken from literature. (original 58.403e-3) (patient 58.4717281111171e-3)
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
const VectorOfDomains{uDim,gDim,T,Nd,Nf} = AbstractVector{<:AbstractDomain{uDim,gDim,T,Nd,Nf}}

# ParabolicDomain: generic domain type which holds this information necessary to
# solve a parabolic FEM problem M*du/dt = K*u
mutable struct ParabolicDomain{uDim,gDim,T,Nd,Nf,MType<:MassType{T},KType<:StiffnessType{T}} <: AbstractDomain{uDim,gDim,T,Nd,Nf}
    grid::Grid{gDim,Nd,T,Nf}
    dh::DofHandler{gDim,Nd,T,Nf}
    cellvalues::CellValues{gDim,T}
    facevalues::FaceValues{gDim,T}
    refshape::Type{<:JuAFEM.AbstractRefShape}
    quadorder::Int
    funcinterporder::Int
    geominterporder::Int
    M::MType
    Mfact::Union{Nothing,<:Factorization}
    K::KType
    w::Vector{T}
end

function ParabolicDomain(
        grid::Grid{gDim,Nd,T,Nf},
        uDim::Int = 2;
        refshape = RefTetrahedron,
        quadorder::Int = 3,
        funcinterporder::Int = 1,
        geominterporder::Int = 1
    ) where {gDim,Nd,T,Nf}

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

    # Assign dof ordering to be such that node number `n` corresponds to dof's `uDim*n-(uDim-1):uDim*n`
    # NOTE: this is somewhat wasteful as nodes are visited multiple times, but it's easy
    if ndofs(dh) == uDim * getnnodes(grid)
        perm = zeros(Int, ndofs(dh))
        for cell in CellIterator(dh)
            for (i,n) in enumerate(cell.nodes)
                for d in uDim-1:-1:0
                    perm[cell.celldofs[uDim*i-d]] = uDim*n-d
                end
            end
        end
        renumber!(dh, perm)
    end

    # Mass matrix, inverse mass matrix, stiffness matrix, and weights vector
    M = create_sparsity_pattern(dh)
    # M = create_symmetric_sparsity_pattern(dh)
    K = create_sparsity_pattern(dh)
    w = zeros(T, ndofs(dh))
    Mfact = nothing

    ParabolicDomain{uDim,gDim,T,Nd,Nf,typeof(M),typeof(K)}(
        grid, dh, cellvalues, facevalues,
        refshape, quadorder, funcinterporder, geominterporder,
        M, Mfact, K, w
    )
end

abstract type AbstractRegion end
abstract type AbstractRegionUnion <: AbstractRegion end

struct AxonRegion <: AbstractRegion
    parent_circle_idx::Int
end
struct MyelinRegion <: AbstractRegion
    parent_circle_idx::Int
end
struct TissueRegion <: AbstractRegion end
struct PermeableInterfaceRegion <: AbstractRegionUnion end

# MyelinDomain:
# Generic domain type which holds the information necessary to solve a parabolic
# FEM problem M*du/dt = K*u on a domain which represents a region containing
# or in close proximity to myelin. The complete domain is represented as a
# ParabolicDomain, which stores the underlying grid, mass matrix M, stiffness
# matrix K, etc.
mutable struct MyelinDomain{R<:AbstractRegion,uDim,gDim,T,Nd,Nf,DType<:ParabolicDomain{uDim,gDim,T,Nd,Nf}} <: AbstractDomain{uDim,gDim,T,Nd,Nf}
    region::R
    domain::DType
    outercircles::Vector{Circle{2,T}}
    innercircles::Vector{Circle{2,T}}
    ferritins::Vector{Vec{3,T}}
end

# Constructor given a `Grid` and kwargs in place of a `ParabolicDomain`
function MyelinDomain(
        region::R,
        grid::Grid{gDim,Nd,T,Nf},
        outercircles::Vector{Circle{2,T}},
        innercircles::Vector{Circle{2,T}},
        ferritins::Vector{Vec{3,T}} = Vec{3,T}[],
        uDim::Int = 2; #::Val{uDim} = Val(2);
        kwargs...
    ) where {R,gDim,T,Nd,Nf} #{R,uDim,gDim,T,Nd,Nf}

    domain = ParabolicDomain(grid, uDim; kwargs...)
    return MyelinDomain{R,uDim,gDim,T,Nd,Nf,typeof(domain)}(
        region, domain, outercircles, innercircles, ferritins
    )
end

# # Copy constructor for new ParabolicDomain keyword arguments
# function MyelinDomain(m::MyelinDomain; kwargs...)
#     return MyelinDomain(
#         m.region,
#         getgrid(m.domain),
#         m.outercircles,
#         m.innercircles,
#         m.ferritins;
#         kwargs...
#     )
# end

# Create BlochTorreyProblem from a MyelinProblem and a MyelinDomain
function BlochTorreyProblem(p::MyelinProblem{T}, m::MyelinDomain) where {T}
    @inline Dcoeff(x...) = dcoeff(x..., p, m) # Dcoeff function
    @inline Rdecay(x...) = rdecay(x..., p, m) # R2 function
    @inline Omega(x...) = omega(x..., p, m) # Omega function
    return BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega)
end

# Copy constructor for creating a ParabolicDomain from a MyelinDomain
# ParabolicDomain(m::MyelinDomain) = deepcopy(m.domain)

# ParabolicLinearMap: create a LinearMaps subtype which wrap the action of
# Mfact\K in a LinearMap object. Does not make copies of M, Mfact, or K;
# simply is a light wrapper for them
struct ParabolicLinearMap{T,MType<:AbstractMatrix{T},KType<:AbstractMatrix{T}} <: LinearMap{T}
    M::MType
    Mfact::Union{Nothing,<:Factorization}
    K::KType
    function ParabolicLinearMap(M::AbstractMatrix{T}, Mfact::Union{Nothing,<:Factorization}, K::AbstractMatrix{T}) where {T}
        @assert (size(M) == size(K)) && (size(M,1) == size(M,2))
        # @assert (size(M) == size(Mfact) || size(M) == 2 .* size(Mfact)) # for factoring only M[1:2:end,1:2:end]
        new{T,typeof(M),typeof(K)}(M, Mfact, K)
    end
end
ParabolicLinearMap(d::ParabolicDomain) = ParabolicLinearMap(getmass(d), getmassfact(d), getstiffness(d))

# ---------------------------------------------------------------------------- #
# BlochTorreyParameters methods
# ---------------------------------------------------------------------------- #

# By default, truncate the radii distribution (of Gamma type) so as to prevent the
# generated radii being too large/small for the purpose of mesh generation
# With the default lower = -1.5 (in units of sigma) and upper = 2.5 and the default
# BlochTorreyParameters, this corresponds to rmean/rmin ~ 2.7 and rmax/rmin ~ 2.0,
# and captures ~95% of the pdf mass.
function radiidistribution(p::BlochTorreyParameters, lower = -1.5, upper = 2.5)
    k, θ = p.R_shape, p.R_scale # shape and scale parameters
    d = Distributions.Gamma(k, θ)
    if !(lower == -Inf && upper == Inf)
        μ, σ = k*θ, √k*θ
        d = Truncated(d, μ + lower*σ, μ + upper*σ)
    end
    return d
end
@inline radiidistribution(p::MyelinProblem) = radiidistribution(p.params)

# ---------------------------------------------------------------------------- #
# AbstractDomain methods
# ---------------------------------------------------------------------------- #

# Interpolation is done by simply creating a `Dirichlet` constraint on every
# face of the domain and applying it to the vector `u`. This is really quite
# slow and wasteful, and there is almost definitely a better way to implement
# this, but it just isn't a bottleneck and this is easy.
function interpolate!(
        u::AbstractVector{T},
        f::Function,
        domain::AbstractDomain{uDim,gDim,T}
    ) where {uDim,gDim,T}

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
function interpolate!(u::AbstractVector{T}, u0::Vec{uDim,T}, domain::AbstractDomain{uDim}) where {uDim,T}
    # Check that `u` has the correct length
    @assert length(u) == ndofs(getdofhandler(domain))
    if length(u) == uDim * getnnodes(getgrid(domain))
        # degrees of freedom are nodal; can efficiently assign directly
        u = reinterpret(Vec{uDim,T}, u)
        u .= Ref(u0)
        u = copy(reinterpret(T, u))
    else
        # degrees of freedom are not nodal; call general projection
        interpolate!(u, x->u0, domain)
    end
    return u
end
interpolate(u0::Vec, domain::AbstractDomain) = interpolate!(zeros(ndofs(getdofhandler(domain))), u0, domain)

function integrate(u::AbstractVector{Tu}, domain::AbstractDomain{uDim,gDim,T}) where {Tu,uDim,gDim,T}
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

# "Vectorized" versions of functions for convenience
integrate(U::VectorOfVectors, domains::VectorOfDomains) = sum(map((u,d) -> integrate(u,d), U, domains))
interpolate!(U::VectorOfVectors, f::Function, domains::VectorOfDomains) = map!((u,d) -> interpolate!(u, f, d), U, U, domains)
interpolate!(U::VectorOfVectors, u0::Vec{uDim}, domains::VectorOfDomains{uDim}) where {uDim} = map!((u,d) -> interpolate!(u, u0, d), U, U, domains)
interpolate(f::Function, domains::VectorOfDomains) = map(d -> interpolate!(zeros(ndofs(d)), f, d), domains)
interpolate(u0::Vec{uDim}, domains::VectorOfDomains{uDim}) where {uDim} = map(d -> interpolate!(zeros(ndofs(d)), u0, d), domains)

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
GeometryUtils.area(d::ParabolicDomain{uDim}) where {uDim} = area(getgrid(d)) # just calculate area of grid directly

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
        ::Val{uDim} = Val(2); #uDim::Int = 2;
        kwargs...
    ) where {uDim,gDim,T,Nd,Nf} #{gDim,T,Nd,Nf}

    @assert length(outercircles) == length(innercircles) == length(myelingrids) == length(axongrids)

    isgridempty(g::Grid) = (getnnodes(g) == 0 || getncells(g) == 0)

    Mtype = MyelinDomain{R,uDim,gDim,T,Nd,Nf} where {R}
    ms = Vector{Mtype}()

    for (i, a) in enumerate(axongrids)
        isgridempty(a) && continue
        push!(ms, MyelinDomain(AxonRegion(i), a, outercircles, innercircles, ferritins, uDim; kwargs...))
    end

    for (i, m) in enumerate(myelingrids)
        isgridempty(m) && continue
        push!(ms, MyelinDomain(MyelinRegion(i), m, outercircles, innercircles, ferritins, uDim; kwargs...))
    end

    for t in tissuegrids
        isgridempty(t) && continue
        push!(ms, MyelinDomain(TissueRegion(), t, outercircles, innercircles, ferritins, uDim; kwargs...))
    end

    return ms
end

# General type signature is MyelinDomain{R<:AbstractRegion,uDim,gDim,T,Nd,Nf},
# so here we restrict it to 2D geometry (gDim) with triangular elements (number
# of nodes per elem Nd = 3, number of faces per elem Nf = 3).
const TriangularMyelinDomain{R,uDim,T,DType} = MyelinDomain{R,uDim,2,T,3,3,DType}

# Create interface domain from vector of MyelinDomain's which are all assumed
# to have the same outercircles, innercircles, and ferritins
function MyelinDomain(
        region::PermeableInterfaceRegion,
        prob::MyelinProblem,
        ms::AbstractVector{<:TriangularMyelinDomain{R,uDim,T} where R}
    ) where {uDim,T}
    domain = ParabolicDomain(region, prob, ms)
    myelindomain = TriangularMyelinDomain{PermeableInterfaceRegion,uDim,T,typeof(domain)}(
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
        ms::AbstractVector{<:TriangularMyelinDomain{R,uDim,T} where {R}}
    ) where {uDim,T}

    # Construct one large ParabolicDomain containing all grids
    gDim, Nd, Nf = 2, 3, 3 # Triangular 2D domain
    grid = Grid(getgrid.(ms)) # combine grids into single large grid
    domain = ParabolicDomain(grid, uDim;
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
    boundaryfaceset = getfaceset(getgrid(domain), "boundary")
    nodepairs = NTuple{gDim,Int}[JuAFEM.faces(cells[f[1]])[f[2]] for f in boundaryfaceset] # pairs of node indices
    nodecoordpairs = NTuple{gDim,Vec{uDim,T}}[(getcoordinates(nodes[n[1]]), getcoordinates(nodes[n[2]])) for n in nodepairs] # pairs of node coordinates

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
    Se = T(-κ/6) .* T[ 2  1 -1 -2   # Minus sign in front since we build the negative stiffness matrix
                       1  2 -2 -1   # `Se` represents the local stiffness matrix of a zero volume (line segment) interface element
                      -1 -2  2  1   # The segment interfaces between the pairs of nodes (A1,B1) and (A2,B2), where A nodes and B nodes have the same coordinates
                      -2 -1  1  2 ] # `Se` is ordered such that it acts on [A1,B1,B2,A2]
    _Se = similar(Se) # temp matrix for storing ck * Se

    # S matrix global indices
    Is, Js, Ss = Vector{Int}(), Vector{Int}(), Vector{T}()
    sizehint!(Is, length(Se) * uDim * length(interfaceindices))
    sizehint!(Js, length(Se) * uDim * length(interfaceindices))
    sizehint!(Ss, length(Se) * uDim * length(interfaceindices))

    for idx in interfaceindices
        ck = norm(getcoordinates(nodes[idx[1]]) - getcoordinates(nodes[idx[2]])) # length of edge segment
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
# ParabolicLinearMap methods: LinearMap's for solving M*du/dt = K*u ODE systems
# ---------------------------------------------------------------------------- #

# Properties
Base.size(A::ParabolicLinearMap) = size(A.K)
LinearAlgebra.issymmetric(A::ParabolicLinearMap) = false
LinearAlgebra.ishermitian(A::ParabolicLinearMap) = false
LinearAlgebra.isposdef(A::ParabolicLinearMap) = false

# AdjointMap and TransposeMap wrapper definitions (shouldn't need these definitions since ParabolicLinearMap <: LinearMap)
# LinearAlgebra.adjoint(A::ParabolicLinearMap) = LinearMaps.AdjointMap(A)
# LinearAlgebra.transpose(A::ParabolicLinearMap) = LinearMaps.TransposeMap(A)

# LinearMaps doesn't define A_mul_B etc. for abstract LinearMap's for some reason... Need to explicitly define them here
LinearMaps.A_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector)  = mul!(y, A, x)
LinearMaps.A_mul_B!(y::AbstractMatrix, A::ParabolicLinearMap, x::AbstractMatrix)  = mul!(y, A, x)
LinearMaps.At_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector) = mul!(y, transpose(A), x)
LinearMaps.At_mul_B!(y::AbstractMatrix, A::ParabolicLinearMap, x::AbstractMatrix) = mul!(y, transpose(A), x)
LinearMaps.Ac_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector) = mul!(y, adjoint(A), x)
LinearMaps.Ac_mul_B!(y::AbstractMatrix, A::ParabolicLinearMap, x::AbstractMatrix) = mul!(y, adjoint(A), x)

# LinearAlgebra.ldiv! is not defined for SuiteSparse.CHOLMOD.Factor; define in terms of \ for simplicity
LinearAlgebra.ldiv!(y::AbstractVecOrMat, A::SuiteSparse.CHOLMOD.Factor, x::AbstractVecOrMat) = copyto!(y, A\x)
LinearAlgebra.ldiv!(A::SuiteSparse.CHOLMOD.Factor, x::AbstractVecOrMat) = copyto!(x, A\x)

# Multiplication action
Minv_K_mul_u!(Y, X, K, Mfact) = (mul!(Y, K, X); ldiv!(Mfact, Y); return Y)
Kt_Minv_mul_u!(Y, X, K, Mfact) = (mul!(Y, transpose(K), Mfact\X); return Y)
Kc_Minv_mul_u!(Y, X, K, Mfact) = (mul!(Y, adjoint(K), Mfact\X); return Y)

# Multiplication with Vector or Matrix
LinearAlgebra.mul!(Y::AbstractVector, A::ParabolicLinearMap, X::AbstractVector) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
LinearAlgebra.mul!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
LinearAlgebra.mul!(Y::AbstractVector, A::LinearMaps.TransposeMap{T, <:ParabolicLinearMap{T}}, X::AbstractVector) where {T} = Kt_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
LinearAlgebra.mul!(Y::AbstractMatrix, A::LinearMaps.TransposeMap{T, <:ParabolicLinearMap{T}}, X::AbstractMatrix) where {T} = Kt_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
LinearAlgebra.mul!(Y::AbstractVector, A::LinearMaps.AdjointMap{T, <:ParabolicLinearMap{T}}, X::AbstractVector) where {T} = Kc_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)
LinearAlgebra.mul!(Y::AbstractMatrix, A::LinearMaps.AdjointMap{T, <:ParabolicLinearMap{T}}, X::AbstractMatrix) where {T} = Kc_Minv_mul_u!(Y, X, A.lmap.K, A.lmap.Mfact)

function Base.show(io::IO, d::ParabolicLinearMap)
    compact = get(io, :compact, false)
    if compact
        print(io, size(d,1), "×", size(d,2), " ", typeof(d))
    else
        print(io, "$(typeof(d)) with:")
        print(io, "\n     M: "); _compact_show_sparse(io, d.M)
        print(io, "\n Mfact: "); _compact_show_factorization(io, d.Mfact)
        print(io, "\n     K: "); _compact_show_sparse(io, d.K)
    end
end

# ---------------------------------------------------------------------------- #
# LinearMap methods: helper functions for LinearMap's
# ---------------------------------------------------------------------------- #

# For taking literal powers of LinearMaps, e.g. A^2
Base.to_power_type(A::Union{<:LinearMaps.AdjointMap, <:LinearMaps.TransposeMap, <:LinearMap}) = A

function LinearAlgebra.tr(A::LinearMap{T}, t::Int = 10) where {T}
    # Approximate trace using mat-vec's with basis vectors
    N = size(A, 2)
    t = min(t, N)
    x = zeros(T, N)
    y = similar(x)
    est = zero(T)
    for ix in StatsBase.sample(1:N, t; replace = false)
        x[ix] = one(T)
        y = mul!(y, A, x)
        est += y[ix]
        x[ix] = zero(T)
    end
    return N * (est/t)
end

# Default to p = 2 for consistency with Base, even though it would throw an error
LinearAlgebra.opnorm(A::LinearMap, p::Real = 2, t::Int = 10) = normest1_norm(A, p, t)
# LinearAlgebra.norm(A::LinearMap, p::Real = 2, t::Int = 10) = normest1_norm(A, p, t)

# ---------------------------------------------------------------------------- #
# DiffEqParabolicLinearMapWrapper methods: Effectively a simplified LinearMap,
# but subtypes AbstractMatrix so that it can be passed to DiffEq* solvers
# ---------------------------------------------------------------------------- #

struct DiffEqParabolicLinearMapWrapper{T,Atype} <: AbstractMatrix{T}
    A::Atype
    DiffEqParabolicLinearMapWrapper(A::Atype) where {Atype} = new{eltype(A), Atype}(A)
end
LinearAlgebra.adjoint(A::DiffEqParabolicLinearMapWrapper) = DiffEqParabolicLinearMapWrapper(A.A')
LinearAlgebra.transpose(A::DiffEqParabolicLinearMapWrapper) = DiffEqParabolicLinearMapWrapper(transpose(A.A))

# NOTE: purposefully not forwarding getindex; wrapper type should behave like a LinearMap
Lazy.@forward DiffEqParabolicLinearMapWrapper.A (Base.size, LinearAlgebra.tr, LinearAlgebra.issymmetric, LinearAlgebra.ishermitian, LinearAlgebra.isposdef)

# TODO: Lazy.@forward gives ambiguity related errors when trying to forward these methods: mul!, norm, opnorm
LinearAlgebra.mul!(Y::AbstractVector, A::DiffEqParabolicLinearMapWrapper, X::AbstractVector) = mul!(Y, A.A, X)
LinearAlgebra.mul!(Y::AbstractMatrix, A::DiffEqParabolicLinearMapWrapper, X::AbstractMatrix) = mul!(Y, A.A, X)
LinearAlgebra.opnorm(A::DiffEqParabolicLinearMapWrapper, p::Real, t::Int = 10) = normest1_norm(A.A, p, t)
# LinearAlgebra.norm(A::DiffEqParabolicLinearMapWrapper, p::Real, t::Int = 10) = normest1_norm(A.A, p, t)

Base.show(io::IO, A::DiffEqParabolicLinearMapWrapper) = print(io, "$(typeof(A))")
Base.show(io::IO, ::MIME"text/plain", A::DiffEqParabolicLinearMapWrapper) = print(io, "$(size(A,1)) × $(size(A,2)) $(typeof(A))")

# ---------------------------------------------------------------------------- #
# Local frequency perturbation map functions
# ---------------------------------------------------------------------------- #
struct OmegaDerivedConstants{T}
    ω₀::T
    s²::T
    c²::T
    function OmegaDerivedConstants(p::MyelinProblem{T}) where {T}
        γ, B₀, θ = p.params.gamma, p.params.B0, p.params.theta
        ω₀ = γ * B₀
        s, c = sincos(θ)
        return new{T}(ω₀, s^2, c^2)
    end
end

@inline function omega_tissue(x::Vec{2}, p::MyelinProblem, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χI, χA, ri², ro² = p.params.ChiI, p.params.ChiA, radius(c_in)^2, radius(c_out)^2
    dx = x - origin(c_in)
    r² = dx⋅dx
    cos2ϕ = (dx[1]-dx[2])*(dx[1]+dx[2])/r² # cos2ϕ == (x²-y²)/r² == (x-y)*(x+y)/r²

    tmp = b.s² * cos2ϕ * ((ro² - ri²)/r²) # Common calculation
    I = χI/2 * tmp # isotropic component
    A = χA/8 * tmp # anisotropic component
    return b.ω₀ * (I + A)
end

@inline function omega_myelin(x::Vec{2}, p::MyelinProblem, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χI, χA, E, ri², ro = p.params.ChiI, p.params.ChiA, p.params.E, radius(c_in)^2, radius(c_out)
    dx = x - origin(c_in)
    r² = dx⋅dx
    cos2ϕ = (dx[1]-dx[2])*(dx[1]+dx[2])/r² # cos2ϕ == (x²-y²)/r² == (x-y)*(x+y)/r²
    r = √r²

    I = χI * (b.c² - 1/3 - b.s² * cos2ϕ * ri² / r²)/2 # isotropic component
    A = χA * (b.s² * (-5/12 - cos2ϕ/8 * (1 + ri²/r²) + 3/4 * log(ro/r)) - b.c²/6) # anisotropic component
    return b.ω₀ * (I + A + E)
end

@inline function omega_axon(x::Vec{2}, p::MyelinProblem, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χA, ri, ro = p.params.ChiA, radius(c_in), radius(c_out)
    A = 3χA/4 * b.s² * log(ro/ri) # anisotropic (and only) component
    return b.ω₀ * A
end

# ---------------------------------------------------------------------------- #
# Global frequency perturbation functions: calculate ω(x) due to entire domain
# ---------------------------------------------------------------------------- #

# Calculate ω(x) inside region number `region`, which is assumed to be tissue
function omega(x::Vec{2}, p::MyelinProblem, region::TissueRegion, outercircles::Vector{C}, innercircles::Vector{C}) where {C<:Circle{2}}
    
    # If there are no structures, then there is no frequency shift ω
    (isempty(outercircles) && isempty(innercircles)) && return zero(eltype(x))
        constants = OmegaDerivedConstants(p)

    ω = sum(eachindex(outercircles, innercircles)) do i
        @inbounds ωi = omega_tissue(x, p, constants, innercircles[i], outercircles[i])
        return ωi
    end

    return ω
end
@inline omega(x::Vec{2}, p::MyelinProblem, domain::MyelinDomain{TissueRegion}) = omega(x, p, getregion(domain), getoutercircles(domain), getinnercircles(domain))

# Calculate ω(x) inside region number `region`, which is assumed to be myelin
function omega(x::Vec{2}, p::MyelinProblem, region::MyelinRegion, outercircles::Vector{C}, innercircles::Vector{C}) where {C<:Circle{2}}
        
    # If there are no structures, then there is no frequency shift ω
    (isempty(outercircles) && isempty(innercircles)) && return zero(eltype(x))
    constants = OmegaDerivedConstants(p)

    ω = sum(eachindex(outercircles, innercircles)) do i
        @inbounds ωi = if i == region.parent_circle_idx
            omega_myelin(x, p, constants, innercircles[i], outercircles[i])
        else
            omega_tissue(x, p, constants, innercircles[i], outercircles[i])
        end
        return ωi
    end

    return ω
end
@inline omega(x::Vec{2}, p::MyelinProblem, domain::MyelinDomain{MyelinRegion}) = omega(x, p, getregion(domain), getoutercircles(domain), getinnercircles(domain))

# Calculate ω(x) inside region number `region`, which is assumed to be axonal
function omega(x::Vec{2}, p::MyelinProblem, region::AxonRegion, outercircles::Vector{C}, innercircles::Vector{C}) where {C<:Circle{2}}
    
    # If there are no structures, then there is no frequency shift ω
    (isempty(outercircles) && isempty(innercircles)) && return zero(eltype(x))
    constants = OmegaDerivedConstants(p)

    ω = sum(eachindex(outercircles, innercircles)) do i
        @inbounds ωi = if i == region.parent_circle_idx
            omega_axon(x, p, constants, innercircles[i], outercircles[i])
        else
            omega_tissue(x, p, constants, innercircles[i], outercircles[i])
        end
        return ωi
    end

    return ω
end
@inline omega(x::Vec{2}, p::MyelinProblem, domain::MyelinDomain{AxonRegion}) = omega(x, p, getregion(domain), getoutercircles(domain), getinnercircles(domain))

# Calculate ω(x) by searching for the region which `x` is contained in
function omega(
        x::Vec{2},
        p::MyelinProblem,
        outercircles::Vector{C},
        innercircles::Vector{C},
        outer_bdry_point_type = :myelin, # `:tissue` or `:myelin`
        inner_bdry_point_type = :myelin, # `:myelin` or `:axon`
        thresh_outer = outer_bdry_point_type == :myelin ?  √eps(eltype(x)) : -√eps(eltype(x)),
        thresh_inner = inner_bdry_point_type == :myelin ? -√eps(eltype(x)) :  √eps(eltype(x))
    ) where {C <: Circle{2}}
    
    # If there are no structures, then there is no frequency shift ω
    (isempty(outercircles) && isempty(innercircles)) && return zero(eltype(x))

    # - Positive `thresh_outer` interprets `outercircles` boundary points as being part of
    #   the myelin region; negative interprets boundary points as in the tissue region
    # - Similarly, negative `thresh_inner` interprets `innercircles` boundary
    #   points as being within the myelin region, and axon region for positive
    i_outer = findfirst(c -> is_in_circle(x, c, thresh_outer), outercircles)
    i_inner = findfirst(c -> is_in_circle(x, c, thresh_inner), innercircles)

    region = if i_outer == nothing
        TissueRegion() # not in outercircles -> tissue region
    elseif i_inner != nothing
        AxonRegion(i_inner) # in innercircles -> axon region
    else
        MyelinRegion(i_outer) # in outercircles but not innercircles -> myelin region
    end

    return omega(x, p, region, outercircles, innercircles)
end
@inline omega(x::Vec{2}, p::MyelinProblem, domain::MyelinDomain{PermeableInterfaceRegion}) = omega(x, p, getoutercircles(domain), getinnercircles(domain))

# Individual coordinate input
@inline omega(x, y, p::MyelinProblem, domain::MyelinDomain) = omega(Vec{2}((x, y)), p, domain)

# Return a vector of vectors of nodal values of ω(x) evaluated on each MyelinDomain
function omegamap(p::MyelinProblem, m::MyelinDomain)
    ω = BlochTorreyProblem(p, m).Omega # Get omega function for domain m
    return map(getnodes(getgrid(m))) do node
        ω(getcoordinates(node)) # Map over grid nodes, returning ω(x) for each node
    end
end

# ---------------------------------------------------------------------------- #
# Global dcoeff/rdecay functions on each region
# ---------------------------------------------------------------------------- #

#TODO: re-write to take in plain vectors of inner/outer circles/ferritins which
# can be called on its own, and wrap with a method that takes a MyelinDomain,
# just like omega(x,p,outercircles,innercircles,...) above

@inline dcoeff(x, p::MyelinProblem, m::MyelinDomain{TissueRegion}) = p.params.D_Tissue
@inline dcoeff(x, p::MyelinProblem, m::MyelinDomain{MyelinRegion}) = p.params.D_Sheath
@inline dcoeff(x, p::MyelinProblem, m::MyelinDomain{AxonRegion}) = p.params.D_Axon
@inline dcoeff(x, y, p::MyelinProblem, m::MyelinDomain) = dcoeff(Vec{2}((x, y)), p, m)

@inline rdecay(x, p::MyelinProblem, m::MyelinDomain{TissueRegion}) = p.params.R2_Tissue
@inline rdecay(x, p::MyelinProblem, m::MyelinDomain{MyelinRegion}) = p.params.R2_sp
@inline rdecay(x, p::MyelinProblem, m::MyelinDomain{AxonRegion}) = p.params.R2_lp
@inline rdecay(x, y, p::MyelinProblem, m::MyelinDomain) = rdecay(Vec{2}((x, y)), p, m)


# ============================================================================ #
#
# Stiffness matrix and mass matrix assembly
#
# ============================================================================ #


# ---------------------------------------------------------------------------- #
# Assembly on a ParabolicDomain of a BlochTorreyProblem
# ---------------------------------------------------------------------------- #

# Assemble the `BlochTorreyProblem` system $M u_t = K u$ on the domain `domain`.
function doassemble!(
        domain::ParabolicDomain{uDim,gDim,T,Nd,Nf},
        prob::BlochTorreyProblem{T}
    ) where {uDim,gDim,T,Nd,Nf}

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
    Ke = zeros(T, n_basefuncs, n_basefuncs)
    Me = zeros(T, n_basefuncs, n_basefuncs)
    we = zeros(T, n_basefuncs)

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

    # # Now, allocate local interface element matrices.
    # n_basefuncs = getnbasefunctions(getfacevalues(domain))
    # Se = zeros(T, 2*n_basefuncs, 2*n_basefuncs)
    # @show size(Se)
    #
    # # Loop over the edges of the cell for interface contributions to `Ke`.
    # # For example, if "Neumann Boundary" is a subset of boundary points, use:
    # #   `onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "Neumann Boundary")`
    # for face in 1:nfaces(cell) && (cellid(cell), face) ∈ getfaceset(grid, "Interface")
    #     if onboundary(cell, face)
    #         # Initialize face values
    #         JuAFEM.reinit!(getfacevalues(domain), cell, face)
    #
    #         for q_point in 1:getnquadpoints(facevalues)
    #             dΓ = getdetJdV(facevalues, q_point)
    #             coords_qp = spatial_coordinate(facevalues, q_point, coords)
    #
    #             # calculate the heat conductivity and heat source at point `coords_qp`
    #             f = func(coords_qp)
    #             fdΓ = f * dΓ
    #
    #             for i in 1:getnbasefunctions(facevalues)
    #                 n = getnormal(facevalues, q_point)
    #                 v = shape_value(facevalues, q_point, i)
    #                 vfdΓ = v * fdΓ
    #                 for j in 1:n_basefuncs
    #                     ∇u = shape_gradient(facevalues, q_point, j)
    #                     Ke[i,j] += (∇u⋅n) * vfdΓ
    #                 end
    #             end
    #         end
    #     end
    # end
    # function surface_integral!(Ke, facevalues::FaceVectorValues, cell, q_point, coords, func::Function)
    # end

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

end # module BlochTorreyUtils

nothing
