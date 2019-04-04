# Convenience definitions
const MyelinBoundary{gDim,T} = Union{<:Circle{gDim,T}, <:Rectangle{gDim,T}, <:Ellipse{gDim,T}}
const VectorOfVectors{T} = AbstractVector{<:AbstractVector{T}}
const MaybeSymmSparseMatrixCSC{T} = Union{<:SparseMatrixCSC{T}, <:Symmetric{T,<:SparseMatrixCSC{T}}}
# const MaybeNothingFactorization{T} = Union{Nothing, <:Factorization{T}}
const MassType{T} = MaybeSymmSparseMatrixCSC{T}
# const MassFactType{T} = Factorization{T}
const StiffnessType{T} = SparseMatrixCSC{T}
const TriangularGrid{T} = Grid{2,3,T,3}

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
mutable struct ParabolicDomain{
        uDim, gDim, T, Nd, Nf, MType <: MassType{T}, MfactType <: Factorization{T}, KType <: StiffnessType{T}
        } <: AbstractDomain{uDim,gDim,T,Nd,Nf}
    grid::Grid{gDim,Nd,T,Nf}
    dh::DofHandler{gDim,Nd,T,Nf}
    cellvalues::CellValues{gDim,T}
    facevalues::FaceValues{gDim,T}
    refshape::Type{<:JuAFEM.AbstractRefShape}
    quadorder::Int
    funcinterporder::Int
    geominterporder::Int
    M::MType
    Mfact::Union{Nothing,MfactType}
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

    # Mass and stiffness matrices, and weights vector
    M = create_sparsity_pattern(dh)
    # M = create_symmetric_sparsity_pattern(dh)
    K = create_sparsity_pattern(dh)
    w = zeros(T, ndofs(dh))

    # Initialize Mfact to nothing
    Mfact = nothing
    MfactType = SuiteSparse.CHOLMOD.Factor{T}

    ParabolicDomain{uDim,gDim,T,Nd,Nf,typeof(M),MfactType,typeof(K)}(
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

# For convenience, define a 2D triangular myelin domain type:
#   grid dimension `gDim` = 2
#   number of nodes per elem `Nd` = 3
#   number of faces per elem `Nf` = 3
const TriangularMyelinDomain{R,uDim,T,DType} = MyelinDomain{R,uDim,2,T,3,3,DType}

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
struct ParabolicLinearMap{T, MType<:AbstractMatrix{T}, MfactType <: Factorization{T}, KType<:AbstractMatrix{T}} <: LinearMap{T}
    M::MType
    Mfact::MfactType
    K::KType
    function ParabolicLinearMap(M::AbstractMatrix{T}, Mfact::Factorization{T}, K::AbstractMatrix{T}) where {T}
        @assert (size(M) == size(Mfact) == size(K)) && (size(M,1) == size(M,2))
        new{T, typeof(M), typeof(Mfact), typeof(K)}(M, Mfact, K)
    end
end
ParabolicLinearMap(d::ParabolicDomain) = ParabolicLinearMap(getmass(d), getmassfact(d), getstiffness(d))