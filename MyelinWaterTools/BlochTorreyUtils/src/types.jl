# Convenience definitions
const FieldType{T} = Union{<:Vec{dim,T} where dim, Complex{T}} where {T<:Real}
const DofType{T} = Union{T, <:FieldType{T}} where {T<:Real}
const AbstractMaybeCplxMat{T} = AbstractMatrix{Tc} where {Tc<:Union{T,Complex{T}}}
const MassType{T} = Union{<:SparseMatrixCSC{T}, <:Symmetric{T,<:SparseMatrixCSC{T}}}
const MassFactType{T} = Factorization{T}
const StiffnessType{T} = SparseMatrixCSC{Tc} where {Tc<:Union{T,Complex{T}}}
const VectorOfVectors{T} = AbstractVector{<:AbstractVector{T}}
const TriangularGrid{T} = Grid{2,3,T,3}
const MyelinBoundary{gDim,T} = Union{Circle{gDim,T}, Rectangle{gDim,T}, Ellipse{gDim,T}}

# ---------------------------------------------------------------------------- #
# BlochTorreyParameters
#   Physical parameters needed for Bloch-Torrey simulations
# ---------------------------------------------------------------------------- #
@with_kw struct BlochTorreyParameters{T}
    B0::T           = T(-3.0) # ................................. [T]        External magnetic field (z-direction)
    gamma::T        = T(2.67515255e8) # ......................... [rad/s/T]  Gyromagnetic ratio
    theta::T        = T(π)/2 # .................................. [rad]      Main magnetic field angle w.r.t B0
    R1_sp::T        = T(inv(200e-3)) # .......................... [1/s]      1/T1 relaxation rate of small pool (myelin) water [REF: TODO]
    R1_lp::T        = T(inv(1084e-3)) # ......................... [1/s]      1/T1 relaxation rate of large pool (intra-cellular/axonal) water [REF: TODO]
    R1_Tissue::T    = T(inv(1084e-3)) # ......................... [1/s]      1/T1 relaxation rate of white matter tissue [REF: 1084 ± 45; https://www.ncbi.nlm.nih.gov/pubmed/16086319]
    R2_sp::T        = T(inv(15e-3)) # ........................... [1/s]      TODO (play with these?) Relaxation rate of small pool (Myelin) (Xu et al. 2017) (15e-3s)
    R2_lp::T        = T(inv(63e-3)) # ........................... [1/s]      TODO (play with these?) 1st attempt was 63E-3. 2nd attempt 76 ms
    R2_Tissue::T    = T(inv(63e-3)) # ........................... [1/s]      TODO (was 14.5Hz; changed to match R2_lp) Relaxation rate of tissue
    R2_water::T     = T(inv(2.2)) # ............................. [1/s]      Relaxation rate of pure water
    D_Water::T      = T(3037.0) # ............................... [um^2/s]   Diffusion coefficient in water
    D_Blood::T      = T(3037.0) # ............................... [um^2/s]   Diffusion coefficient in blood
    D_Tissue::T     = T(1500.0) # ............................... [um^2/s]   TODO (reference?) Diffusion coefficient in tissue
    D_Sheath::T     = T(1000.0) # ............................... [um^2/s]   TODO (reference?) Mean diffusivity coefficient in myelin sheath
    D_Axon::T       = T(2000.0) # ............................... [um^2/s]   TODO (reference?) Diffusion coefficient in axon interior
    FRD_Sheath::T   = T(0.5) # .................................. [unitless] TODO (reference?) Fractional radial diffusivity within myelin sheath; FRD ∈ [0,1] where 0 is purely polar, 0.5 is isotropic, and 1 is purely radial
    K_perm          = T(1.0e-3) # ............................... [um/s]     TODO (reference?) Interface permeability constant
    K_Axon_Sheath   = T(K_perm) # ............................... [um/s]     Axon-Myelin interface permeability
    K_Tissue_Sheath = T(K_perm) # ............................... [um/s]     Tissue-Myelin interface permeability
    R_mu::T         = T(0.46) # ................................. [um]       Axon mean radius (taken to be outer radius)
    R_shape::T      = T(5.7) # .................................. [unitless] Axon shape parameter for Gamma distribution (Xu et al. 2017)
    R_scale::T      = T(R_mu/R_shape) # ......................... [um]       Axon scale parameter for Gamma distribution (Xu et al. 2017)
    PD_sp::T        = T(0.5) # .................................. [unitless] Relative proton density (Myelin)
    PD_lp::T        = T(1.0) # .................................. [unitless] Relative proton density (Intra Extra)
    PD_Fe::T        = T(1.0) # .................................. [unitless] Relative proton density (Ferritin)
    g_ratio::T      = T(0.8370) # ............................... [um/um]    g-ratio (originally 0.71; 0.84658 for healthy, 0.8595 for MS)
    AxonPDensity::T = T(0.83) # ................................. [unitless] Axon packing density based region in white matter (Xu et al. 2017) (originally 0.83)
    MVF::T          = T(AxonPDensity*(1-g_ratio^2)) # ........... [unitless] Myelin volume fraction, assuming periodic circle packing and constant g_ratio
    MWF::T          = T(PD_sp*MVF/(PD_lp-(PD_lp-PD_sp)*MVF)) # .. [unitless] Myelin water fraction, assuming periodic circle packing and constant g_ratio
    ChiI::T         = T(-60e-9) # ............................... [unitless] Isotropic susceptibility of myelin (TODO check how to get it) (Xu et al. 2017)
    ChiA::T         = T(-120e-9) # .............................. [unitless] Anisotropic Susceptibility of myelin (Xu et al. 2017)
    E::T            = T(10e-9) # ................................ [unitless] Exchange component to resonance freqeuency (Wharton and Bowtell 2012)
    R2_Fe::T        = T(inv(1e-6)) # ............................ [1/s]      Relaxation rate of iron in ferritin (assumed extremely high)
    R2_WM::T        = T(inv(70e-3)) # ........................... [1/s]      Relaxation rate of frontal WM (empirical and taken from literature; original 58.403e-3; patient 58.472e-3)
    R_Ferritin::T   = T(4.0e-3) # ............................... [um]       Ferritin mean radius
    R_conc::T       = T(0.0424) # ............................... [mg/g]     Concentration of iron in the frontal white matter (0.0424 in frontal WM; 0.2130 in globus pallidus deep grey matter)
    Rho_tissue::T   = T(1.073) # ................................ [g/ml]     White matter tissue density
    ChiTissue::T    = T(-9.05e-6) # ............................. [unitless] Isotropic susceptibility of tissue
    ChiFeUnit::T    = T(1.4e-9) # ............................... [ug/g]     TODO (check units) Susceptibility of iron per ppm/(ug/g) weight fraction of iron.
    ChiFeFull::T    = T(520.0e-6) # ............................. [ug/g]     TODO (check units) Susceptibility of iron for ferritin particle FULLY loaded with 4500 iron atoms. (use volume of FULL spheres) (from Contributions to magnetic susceptibility)
    Rho_Iron::T     = T(7.874) # ................................ [g/cm^3]   Iron density
end

function BlochTorreyParameters(d::Dict{Symbol,T}) where {T}
    d_ = deepcopy(d)
    f = fieldnames(BlochTorreyParameters)
    for (k, v) ∈ zip(collect(keys(d_)), collect(values(d_)))
        # Deprecations, e.g.:
        # if k == :K_perm
        #     delete!(d_, k)
        #     d_[:K_Axon_Sheath] = v
        #     d_[:K_Tissue_Sheath] = v
        # end
        if k ∉ f
            delete!(d_, k)
            @warn "Key $k not found a field of BlochTorreyParameters"
        end
    end
    return BlochTorreyParameters{T}(;d_...)
end
Base.Dict(p::BlochTorreyParameters{T}) where {T} = Dict{Symbol,T}(f => getfield(p,f) for f in fieldnames(typeof(p)))

@inline GeometryUtils.floattype(p::BlochTorreyParameters{T}) where {T} = T

# ---------------------------------------------------------------------------- #
# AbstractParabolicProblem
#   Subtypes of this abstract type hold all necessary information to solve a
#   given problem on a corresponding AbstractDomain.
# ---------------------------------------------------------------------------- #
abstract type AbstractParabolicProblem{T} end

# BlochTorreyProblem: holds the only parameters necessary to solve the Bloch-
# Torrey equation, naming the Dcoeff, Rdecay, and Omega functions of position
struct BlochTorreyProblem{T,D,R,W} <: AbstractParabolicProblem{T}
    Dcoeff::D # Function which takes a Vec `x` and outputs Dcoeff(x)
    Rdecay::R # Function which takes a Vec `x` and outputs Rdecay(x)
    Omega::W # Function which takes a Vec `x` and outputs Omega(x)
    BlochTorreyProblem{T}(d::D,r::R,w::W) where {T,D,R,W} = new{T,D,R,W}(d,r,w)
end

# MyelinProblem: holds a `BlochTorreyParameters` set of parameters
struct MyelinProblem{T} <: AbstractParabolicProblem{T}
    params::BlochTorreyParameters{T}
end

@inline GeometryUtils.floattype(p::MyelinProblem{T}) where {T} = T

# ---------------------------------------------------------------------------- #
# AbstractDomain
#   Abstract type with the most generic information on the underlying problem:
#       Tu:     Bottom float type used for underlying function, e.g. Float64
#       uType:  Vector type of unknown function, e.g. Vec{2,Tu}, Complex{Tu}
#       gDim:   Spatial dimension of domain
# ---------------------------------------------------------------------------- #
abstract type AbstractDomain{Tu,uType,gDim} end
const VectorOfDomains{Tu,uType,gDim} = AbstractVector{<:AbstractDomain{Tu,uType,gDim}}

# ---------------------------------------------------------------------------- #
# ParabolicDomain
# ---------------------------------------------------------------------------- #
mutable struct ParabolicDomain{Tu,uType<:FieldType{Tu},gDim} <: AbstractDomain{Tu,uType,gDim}
    grid::JuAFEM.Grid{gDim}
    dh::JuAFEM.DofHandler{gDim}
    refshape::JuAFEM.AbstractRefShape
    cellvalues::Union{<:JuAFEM.CellValues{gDim}, <:Tuple{Vararg{<:JuAFEM.CellValues{gDim}}}}
    facevalues::Union{<:JuAFEM.FaceValues{gDim}, <:Tuple{Vararg{<:JuAFEM.FaceValues{gDim}}}}
    quadorder::Int
    funcinterporder::Int
    geominterporder::Int
    M::MassType{Tu}
    Mfact::Union{Nothing, <:MassFactType{Tu}}
    K::StiffnessType{Tu}
    metadata::Dict{Any,Any}
end

# ---------------------------------------------------------------------------- #
# AbstractRegion
#   Along with it's subtypes, allows for dispatching on the different regions
#   which an underlying grid, function, etc. may be represented on
# ---------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------- #
# MyelinDomain <: AbstractDomain
#   Generic domain type which holds the information necessary to solve a
#   parabolic FEM problem M*du/dt = K*u on a domain which represents a region
#   containing or in close proximity to myelin. The complete domain is
#   represented as a ParabolicDomain, which stores the underlying grid, mass
#   matrix M, stiffness matrix K, etc.
# ---------------------------------------------------------------------------- #
mutable struct MyelinDomain{R<:AbstractRegion,Tu,uType,gDim} <: AbstractDomain{Tu,uType,gDim}
    region::R
    domain::ParabolicDomain{Tu,uType,gDim}
    outercircles::Vector{Circle{2,Tu}}
    innercircles::Vector{Circle{2,Tu}}
    ferritins::Vector{Vec{3,Tu}}
end

# TriangularMyelinDomain is a MyelinDomain with grid dimension gDim = 2,
# nodes per finite element Nd = 3, and faces per finite element Nf = 2
const TriangularMyelinDomain{R,Tu,uType} = MyelinDomain{R,Tu,uType,2} #TODO FIXME

# ---------------------------------------------------------------------------- #
# ParabolicLinearMap <: LinearMap
#   Lightweight wrapper over mass matrix M, factorized mass matrix Mfact, and
#   stiffness matrix K. Acts on vectors and matrices as Mfact\K.
# ---------------------------------------------------------------------------- #
struct ParabolicLinearMap{T, MType<:AbstractMatrix, MfactType <: MassFactType, KType<:AbstractMatrix} <: LinearMap{T}
    M::MType
    Mfact::MfactType
    K::KType
    function ParabolicLinearMap(M::AbstractMatrix, Mfact::MassFactType, K::AbstractMatrix)
        @assert (size(M) == size(Mfact) == size(K)) && (size(M,1) == size(M,2))
        T = promote_type(eltype(M), eltype(Mfact), eltype(K))
        new{T, typeof(M), typeof(Mfact), typeof(K)}(M, Mfact, K)
    end
end

# ---------------------------------------------------------------------------- #
# LinearOperatorWrapper <: AbstractMatrix
#   Effectively a simplified LinearMap, but subtypes AbstractMatrix so that it
#   can be passed on to DiffEq* solvers, etc.
# ---------------------------------------------------------------------------- #
struct LinearOperatorWrapper{T,Atype} <: AbstractMatrix{T}
    A::Atype
    LinearOperatorWrapper(A::Atype) where {Atype} = new{eltype(A), Atype}(A)
end

# ---------------------------------------------------------------------------- #
# Constructors
# ---------------------------------------------------------------------------- #

# Create BlochTorreyProblem from a MyelinProblem and a MyelinDomain
function BlochTorreyProblem(p::MyelinProblem{T}, m::MyelinDomain) where {T}
    @inline Dcoeff(x...) = dcoeff(x..., p, m) # Dcoeff function
    @inline Rdecay(x...) = (r1decay(x..., p, m), r2decay(x..., p, m)) # R2 function
    @inline Omega(x...) = omega(x..., p, m) # Omega function
    return BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega)
end

# Construct a ParabolicDomain from a Grid and interpolation/integration settings
function ParabolicDomain(
        grid::Grid{gDim},
        ::Type{uType} = Vec{2,floattype(grid)};
        refshape::JuAFEM.AbstractRefShape = RefTetrahedron(),
        quadorder::Int = 3,
        funcinterporder::Int = 1,
        geominterporder::Int = 1
    ) where {gDim,Tu,uType<:FieldType{Tu}}

    @assert 1 <= fielddim(uType) <= 3
    @assert floattype(grid) == floattype(uType)
    uDim = fielddim(uType)

    # Quadrature and interpolation rules and corresponding cellvalues/facevalues
    func_interp = Lagrange{gDim, typeof(refshape), funcinterporder}()
    geom_interp = Lagrange{gDim, typeof(refshape), geominterporder}()
    quadrule = QuadratureRule{gDim, typeof(refshape)}(quadorder)
    quadrule_face = QuadratureRule{gDim-1, typeof(refshape)}(quadorder)

    if uDim == 1
        cellvalues = CellScalarValues(Tu, quadrule, func_interp, geom_interp)
        facevalues = FaceScalarValues(Tu, quadrule_face, func_interp, geom_interp)
    elseif uDim == 2
        cellvalues = CellVectorValues(Tu, quadrule, func_interp, geom_interp)
        facevalues = FaceVectorValues(Tu, quadrule_face, func_interp, geom_interp)
    elseif uDim == 3
        cellvalues = (CellVectorValues(Tu, quadrule, func_interp, geom_interp), CellScalarValues(Tu, quadrule, func_interp, geom_interp))
        facevalues = (FaceVectorValues(Tu, quadrule_face, func_interp, geom_interp), FaceScalarValues(Tu, quadrule_face, func_interp, geom_interp))
    end

    # Degree of freedom handler
    dh = DofHandler(grid)
    if uDim == 1 || uDim == 2
        push!(dh, :u, uDim, func_interp)
    elseif uDim == 3
        push!(dh, :u, 2, func_interp)
        push!(dh, :uz, 1, func_interp)
    end
    close!(dh)

    # Assign dof ordering to be such that node number `n` corresponds to dof's `uDim*n-(uDim-1):uDim*n`,
    # i.e. each node's DOFs are consecutive, and are in order of node number
    #   NOTE: this is somewhat wasteful as nodes are visited multiple times, but it's easy
    @assert ndofs(dh) == uDim * getnnodes(grid)
    perm = zeros(Int, ndofs(dh))
    if uDim == 1 || uDim == 2
        dr_u = dof_range(dh, :u)
        for cell in CellIterator(dh)
            for (i,n) in enumerate(cell.nodes)
                for d in uDim-1:-1:0
                    perm[cell.celldofs[dr_u[uDim*i-d]]] = uDim * n - d
                end
            end
        end
    elseif uDim == 3
        dr_u, dr_uz = dof_range(dh, :u), dof_range(dh, :uz)
        for cell in CellIterator(dh)
            for (i,n) in enumerate(cell.nodes)
                perm[cell.celldofs[dr_u[2i-1]]] = 3n-2 # transverse component
                perm[cell.celldofs[dr_u[2i  ]]] = 3n-1 # transverse component
                perm[cell.celldofs[dr_uz[i  ]]] = 3n   # longitudinal component
            end
        end
    end
    renumber!(dh, perm)

    # Mass and stiffness matrices
    M = create_sparsity_pattern(dh) #create_symmetric_sparsity_pattern(dh)
    K = uType <: Complex ? complex(create_sparsity_pattern(dh)) : create_sparsity_pattern(dh)
    Mfact = nothing

    ParabolicDomain{Tu,uType,gDim}(
        grid, dh, refshape, cellvalues, facevalues, quadorder, funcinterporder, geominterporder,
        M, Mfact, K, Dict{Any,Any}()
    )
end

# Construct MyelinDomain from a Region and Grid. Internally, a ParabolicDomain
# is constructed, and so keyword arguments are forwarded to that constructor
function MyelinDomain(
        region::R,
        grid::Grid{gDim},
        outercircles::Vector{<:Circle{2}},
        innercircles::Vector{<:Circle{2}},
        ferritins::Vector{<:Vec{3}} = Vec{3,floattype(grid)}[],
        ::Type{uType} = Vec{2,floattype(grid)};
        kwargs...
    ) where {R,gDim,Tu,uType<:FieldType{Tu}}
    domain = ParabolicDomain(grid, uType; kwargs...)
    return MyelinDomain{R,Tu,uType,gDim}(region, domain, outercircles, innercircles, ferritins)
end

# Construct a ParabolicLinearMap from a ParabolicDomain
ParabolicLinearMap(d::ParabolicDomain) = ParabolicLinearMap(getmass(d), getmassfact(d), getstiffness(d))

# Construct a LinearOperatorWrapper from a ParabolicDomain
LinearOperatorWrapper(d::ParabolicDomain) = LinearOperatorWrapper(ParabolicLinearMap(d))
