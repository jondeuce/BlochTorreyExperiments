# ---------------------------------------------------------------------------- #
# Bloch-Torrey parameters type
# ---------------------------------------------------------------------------- #

# Struct of BlochTorreyParameters. T is the float type.
#TODO: rewrite to use inner constructor
struct BlochTorreyParameters{T}
    params::Dict{Symbol,T}
end
function BlochTorreyParameters(::Type{T} = Float64; kwargs...) where {T}
    # default parameters
    default_params = Dict{Symbol,T}(
        :B0             =>    3.0,          # External magnetic field [T]
        :gamma          =>    2.67515255e8, # Gyromagnetic ratio [rad/(T*s)]
        :theta          =>    π/2,          # Main magnetic field angle w.r.t B0 [rad/(T*s)]
        :g_ratio        =>    0.8370,       # g-ratio (original 0.71) ,0.84658 for healthy, 0.8595 for MS.
        :R2_sp          =>    1.0/15e-3,    # #TODO play with these? Relaxation rate of small pool [s^-1] (Myelin) (Xu et al. 2017) (15e-3s)
        :R2_lp          =>    1.0/63e-3,    # #TODO play with these? 1st attempt was 63E-3. 2nd attempt 76 ms
        :R2_Tissue      =>    14.5,         # Relaxation rate of tissue [s^-1]
        :R2_water       =>    1.0/2.2,      # Relaxation rate of pure water
        :D_Tissue       =>    2000.0,       # #TODO reference? Diffusion coefficient in tissue [um^2/s]
        :D_Sheath       =>    1000.0,       # #TODO reference? Diffusion coefficient in myelin sheath [um^2/s]
        :D_Axon         =>    2500.0,       # #TODO reference? Diffusion coefficient in axon interior [um^2/s]
        :D_Blood        =>    3037.0,       # Diffusion coefficient in blood [um^2/s]
        :D_Water        =>    3037.0,       # Diffusion coefficient in water [um^2/s]
        :R_mu           =>    0.46,         # Axon mean radius [um] ; this is taken to be outer radius.
        :R_shape        =>    5.7,          # Axon shape parameter for Gamma distribution (Xu et al. 2017)
        :R_scale        =>    0.46/5.7,     # Axon scale parameter for Gamma distribution (Xu et al. 2017)
        :AxonPDensity   =>    0.83,         # Axon packing density based region in white matter. (Xu et al. 2017) (originally 0.83)
        :AxonPDActual   =>    0.64,         # The actual axon packing density you're aiming for.
        :PD_sp          =>    0.5,          # Relative proton density (Myelin)
        :PD_lp          =>    1.0,          # Relative proton density (Intra Extra)
        :PD_Fe          =>    1.0,          # Relative proton density (Ferritin)
        :ChiI           =>   -60e-9,        # Isotropic susceptibility of myelin [ppb] (check how to get it) (Xu et al. 2017)
        :ChiA           =>   -120e-9,       # Anisotropic Susceptibility of myelin [ppb] (Xu et al. 2017)
        :E              =>    10e-9,        # Exchange component to resonance freqeuency [ppb] (Wharton and Bowtell 2012)
        :R2_Fe          =>    1.0/1e-6,     # Relaxation rate of iron in ferritin. Assumed to be really high.
        :R2_WM          =>    1.0/70e-3,    # Relaxation rate of frontal WM. This is empirical;taken from literature. (original 58.403e-3) (patient 58.4717281111171e-3)
        :R_Ferritin     =>    4.0e-3,       # Ferritin mean radius [um].
        :R_conc         =>    0.0,          # Conntration of iron in the frontal white matter. [mg/g] (0.0424 in frontal WM) (0.2130 in globus pallidus; deep grey matter)
        :Rho_tissue     =>    1.073,        # White matter tissue density [g/ml]
        :ChiTissue      =>   -9.05e-6,      # Isotropic susceptibility of tissue
        :ChiFeUnit      =>    1.4e-9,       # Susceptibility of iron per ppm/ (ug/g) weight fraction of iron.
        :ChiFeFull      =>    520.0e-6,     # Susceptibility of iron for ferritin particle FULLY loaded with 4500 iron atoms. (use volume of FULL spheres) (from Contributions to magnetic susceptibility)
        :Rho_Iron       =>    7.874         # Iron density [g/cm^3]
        )

    # Get input paramaters and collect as a dictionary
    input_params = Dict{Symbol,T}(kwargs)

    # Check that input params are valid
    @assert all(keys(input_params)) do k
        iskey = k ∈ keys(default_params)
        ~iskey && warn("$k is not a valid key")
        return iskey
    end

    # Merge input params into defaults and return
    return BlochTorreyParameters{T}(merge(default_params, input_params))
end

Base.getindex(p::BlochTorreyParameters, s::Symbol) = p.params[s]
Base.setindex!(p::BlochTorreyParameters, v, s::Symbol) = error("Parameters are immutable")
Base.display(p::BlochTorreyParameters) = (println("$(typeof(p)) with parameters:\n"); display(p.params))

radiidistribution(p::BlochTorreyParameters) = Distributions.Gamma(p[:R_shape], p[:R_mu]/p[:R_shape])

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
    M::SparseMatrixCSC{T}
    Mfact::Union{Factorization{T},Void}
    K::SparseMatrixCSC{T}
    w::Vector{T}
end

function ParabolicDomain(grid::Grid{dim,Nd,T,Nf};
    udim = 2,
    refshape = RefTetrahedron,
    quadorder = 1,
    funcinterporder = 1,
    geominterporder = 1) where {dim,Nd,T,Nf}

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
    @assert udim == 2 #TODO: where is this assumption? likely, assume dim(u) == dim(grid)
    dh = DofHandler(grid)
    push!(dh, :u, udim, func_interp)
    close!(dh)

    # Mass matrix, inverse mass matrix, stiffness matrix, and weights vector
    M = create_sparsity_pattern(dh)
    Mfact = nothing
    K = copy(M)
    w = zeros(T, ndofs(dh))

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

factorize!(d::ParabolicDomain) = (d.Mfact = cholfact(getmass(d)); return d)
Base.norm(u, domain::ParabolicDomain) = √dot(u, getmass(domain) * u)

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
                    Ke[i, j] -= (∇v ⊡ ∇u) * dΩ
                    Me[i, j] += (v ⋅ u) * dΩ
                end
            end
        end

        # The last step in the element loop is to assemble `Ke` and `Me`
        # into the global `K` and `M` with `assemble!`.
        assemble!(assembler_M, celldofs(cell), Me)
        assemble!(assembler_K, celldofs(cell), Ke, we)
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

function Base.norm(U::Vector{Vector{T}}, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    @assert length(U) == numsubdomains(domain)
    return √sum(i->norm(U[i], getsubdomain(domain,i))^2, 1:numsubdomains(domain))
end

# ---------------------------------------------------------------------------- #
# Assmeble mass and stiffness matrices for MyelinProblem and BlochTorreyProblem
# ---------------------------------------------------------------------------- #

function doassemble!(prob::MyelinProblem{T},
                     domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    # Exterior region
    Rdecay = (x) -> prob.params[:R2_lp]
    Dcoeff = (x) -> prob.params[:D_Tissue]
    Omega = (x) -> omega_tissue(x, domain, prob.params)
    doassemble!(BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega), domain.tissuedomain)

    # Myelin sheath region
    Rdecay = (x) -> prob.params[:R2_sp]
    Dcoeff = (x) -> prob.params[:D_Sheath]
    for i in 1:numfibres(domain)
        Omega = (x) -> omega_myelin(x, domain, prob.params, i)
        doassemble!(BlochTorreyProblem{T}(Dcoeff, Rdecay, Omega), domain.myelindomains[i])
    end

    # Axon region
    Rdecay = (x) -> prob.params[:R2_lp]
    Dcoeff = (x) -> prob.params[:D_Axon]
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
function Minv_K_mul_u!(Y, X, K, Mfact)
   A_mul_B!(Y, K, X)
   copy!(Y, Mfact\Y)
   return Y
end

function Kt_Minv_mul_u!(Y, X, K, Mfact)
   At_mul_B!(Y, K, Mfact\X)
   return Y
end

function paraboliclinearmap(K, Mfact)
   @assert (size(K) == size(Mfact)) && (size(K,1) == size(K,2))
   fwd_mul! = (Y, X) -> Minv_K_mul_u!(Y, X, K, Mfact);
   trans_mul! = (Y, X) -> Kt_Minv_mul_u!(Y, X, K, Mfact);
   return LinearMap(fwd_mul!, trans_mul!, size(K)...;
      ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)
end
paraboliclinearmap(d::ParabolicDomain) = paraboliclinearmap(getstiffness(d), getmassfact(d))

#TODO: Probably shouldn't define these; would only be used for normest1 which is
# definitely not a bottleneck, and could silently break something?
#TODO: Could define own LinearMap subtype however? Worth it?
# import Base.LinAlg: A_mul_B!, At_mul_B!, Ac_mul_B!
# A_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.f!(Y,X);
# At_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.fc!(Y,X);
# Ac_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.fc!(Y,X);

struct ParabolicLinearMap{T} <: LinearMap{T}
    M::AbstractMatrix{T}
    Mfact::Factorization{T}
    K::AbstractMatrix{T}
    function ParabolicLinearMap(M::AbstractMatrix{T}, Mfact::Factorization{T}, K::AbstractMatrix{T}) where {T}
        @assert (size(M) == size(Mfact) == size(K)) && (size(M,1) == size(M,2))
        new{T}(M, Mfact, K)
    end
end

# properties
Base.size(A::ParabolicLinearMap) = size(A.K)
Base.issymmetric(A::ParabolicLinearMap) = false
Base.ishermitian(A::ParabolicLinearMap) = false
Base.isposdef(A::ParabolicLinearMap) = false

# multiplication with Vector
Base.LinAlg.A_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector) = Minv_K_mul_u!(y, x, A.K, A.Mfact)
Base.LinAlg.At_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector) = Kt_Minv_mul_u!(y, x, A.K, A.Mfact)
Base.LinAlg.Ac_mul_B!(y::AbstractVector, A::ParabolicLinearMap, x::AbstractVector) = Kt_Minv_mul_u!(y, x, A.K, A.Mfact)
Base.:(*)(A::ParabolicLinearMap, x::AbstractVector) = A_mul_B!(similar(x), A, x)

# multiplication with Matrix
Base.LinAlg.A_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Minv_K_mul_u!(Y, X, A.K, A.Mfact)
Base.LinAlg.At_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Kt_Minv_mul_u!(Y, X, A.K, A.Mfact)
Base.LinAlg.Ac_mul_B!(Y::AbstractMatrix, A::ParabolicLinearMap, X::AbstractMatrix) = Kt_Minv_mul_u!(Y, X, A.K, A.Mfact)
Base.:(*)(A::ParabolicLinearMap, X::AbstractMatrix) = A_mul_B!(similar(X), A, X)

function Base.LinAlg.trace(A::ParabolicLinearMap{T}, t::Int = 10) where {T}
    # Approximate trace, given by the trace corresponding to the lumped mass matrix M[i,i] = Σ_j M[i,j]
    # return sum(diag(A.K)./sum(A.M,2))

    # Approximate trace using mat-vec's with basis vectors
    N = size(A,2)
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

# normAm
Expmv.normAm(A::LinearMap, m::Int, t::Int = 10) = (est = normest1(A^m, t)[1]; return (est, 0))
Base.norm(A::ParabolicLinearMap, args...) = expmv_norm(A, args...)

# ---------------------------------------------------------------------------- #
# expmv and related functions
# ---------------------------------------------------------------------------- #

# Custom norm for calling expmv
function expmv_norm(A, p::Real=1, t::Int=10)
    !(p == 1 || p == Inf) && error("Only p=1 or p=Inf supported")
    p == Inf && (A = A')
    return normest1(A, t)[1]
end
# Default fallback for vectors
expmv_norm(x::AbstractVector, p::Real=2, args...) = Base.norm(x, p, args...)

nothing

# ---------------------------------------------------------------------------- #
# Local frequency perturbation map functions
# ---------------------------------------------------------------------------- #

struct OmegaDerivedConstants{T}
    ω₀::T
    s²::T
    c²::T
end
function OmegaDerivedConstants(p::BlochTorreyParameters{T}) where {T}
    γ, B₀, θ = p[:gamma], p[:B0], p[:theta]
    ω₀ = γ * B₀
    s², c² = sin(θ)^2, cos(θ)^2
    return OmegaDerivedConstants{T}(ω₀, s², c²)
end

@inline function omega_tissue(x::Vec{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χI, χA, ri², ro² = p[:ChiI], p[:ChiA], radius(c_in)^2, radius(c_out)^2
    dx = x - origin(c_in)
    r² = dx⋅dx
    cos2ϕ = (dx[1]^2-dx[2]^2)/r²

    tmp = b.s² * cos2ϕ * ((ro² - ri²)/r²) # Common calculation
    I = χI/2 * tmp # isotropic component
    A = χA/8 * tmp # anisotropic component
    return b.ω₀ * (I + A)
end

@inline function omega_myelin(x::Vec{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χI, χA, ri², ro = p[:ChiI], p[:ChiA], radius(c_in)^2, radius(c_out)
    dx = x - origin(c_in)
    r² = dx⋅dx
    r = √r²
    cos2ϕ = (dx[1]^2-dx[2]^2)/r²

    I = χI/2 * (b.c² - 1/3 - b.s² * cos2ϕ * ri² / r²) # isotropic component
    A = χA * (b.s² * (-5/12 - cos2ϕ/8 * (1 + ri²/r²) + 3/4 * log(ro/r)) - b.c²/6) # anisotropic component
    return b.ω₀ * (I + A)
end

@inline function omega_axon(x::Vec{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, c_in::Circle{2}, c_out::Circle{2})
    χA, ri, ro = p[:ChiA], radius(c_in), radius(c_out)
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
    @inbounds for i in IterTools.chain(1:region-1, region+1:numfibres(domain))
        ω += omega_tissue(x, params, constants, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

# Calculate ω(x) inside region number `domain`, which is assumed to be axonal
function omega_axon(x::Vec{2}, domain::MyelinDomain, params::BlochTorreyParameters, region::Int)
    constants = OmegaDerivedConstants(params)
    ω = omega_axon(x, params, constants, getinnercircle(domain, region), getoutercircle(domain, region))
    @inbounds for i in IterTools.chain(1:region-1, region+1:numfibres(domain))
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
#     γ, B₀, θ = p[:gamma], p[:B0], p[:theta]
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
#     χI, ri², ro² = p[:ChiI], radius(c_inner)^2, radius(c_outer)^2
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
#     χA, ri², ro² = p[:ChiA], radius(c_inner)^2, radius(c_outer)^2
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
#     χI, ri² = p[:ChiI], radius(c_inner)^2
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
#     χA, ri², ro = p[:ChiA], radius(c_inner)^2, radius(c_outer)
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
#     χA, ri, ro = p[:ChiA], radius(c_inner), radius(c_outer)
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
