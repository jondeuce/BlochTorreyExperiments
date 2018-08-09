# ---------------------------------------------------------------------------- #
# Bloch-Torrey parameters type
# ---------------------------------------------------------------------------- #

# Struct of BlochTorreyParameters. T is the float type
struct BlochTorreyParameters{T}
    params::Dict{Symbol,T}
end

Base.getindex(p::BlochTorreyParameters, s::Symbol) = p.params[s]
Base.setindex!(p::BlochTorreyParameters, v, s::Symbol) = error("Parameters are immutable")
Base.display(p::BlochTorreyParameters) = (println("$(typeof(p)) with parameters:\n"); display(p.params))

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

# ---------------------------------------------------------------------------- #
# Myelin grid type
# ---------------------------------------------------------------------------- #

# Abstract domain type. The spatial dimension is `dim`, the nodes per element is
# `Nd`, the float type is `T`, and the number of faces per element is `Nf`.
abstract type AbstractDomain{dim,Nd,T,Nf} end

# ---------------------------------------------------------------------------- #
# Generic parabolic domain grid type
# ---------------------------------------------------------------------------- #

mutable struct ParabolicDomain{dim,Nd,T,Nf} <: AbstractDomain{dim,Nd,T,Nf}
    grid::Grid{dim,Nd,T,Nf}
    dh::DofHandler{dim,Nd,T,Nf}
    cellvalues::CellValues{dim,T}
    facevalues::FaceValues{dim,T}
    M::SparseMatrixCSC{T}
    Minv::Union{Factorization{T},Void}
    K::SparseMatrixCSC{T}
    w::Vector{T}
end

@inline function _check_is_nodal(u::Vector, d::AbstractDomain{dim}) where {dim}
    @assert (length(u) == ndofs(getdofhandler(d)) == dim*getnnodes(getgrid(d)))
end

function ParabolicDomain(grid::Grid{dim,Nd,T,Nf};
                         shape = RefTetrahedron,
                         quadorder = 1,
                         funcinterporder = 1,
                         geominterporder = 1) where {dim,Nd,T,Nf}
    # Quadrature and interpolation rules and corresponding cellvalues/facevalues
    func_interp = Lagrange{dim, shape, funcinterporder}()
    geom_interp = Lagrange{dim, shape, geominterporder}()
    quadrule = QuadratureRule{dim, shape}(quadorder)
    quadrule_face = QuadratureRule{dim-1, shape}(quadorder)
    cellvalues = CellVectorValues(T, quadrule, func_interp, geom_interp)
    facevalues = FaceVectorValues(T, quadrule_face, func_interp, geom_interp)

    # Degree of freedom handler
    dh = DofHandler(grid)
    push!(dh, :u, dim, func_interp)
    close!(dh)

    # Mass matrix, inverse mass matrix, stiffness matrix, and weights vector
    M = create_sparsity_pattern(dh)
    Minv = nothing
    K = copy(M)
    w = zeros(T, ndofs(dh))

    ParabolicDomain{dim,Nd,T,Nf}(grid, dh, cellvalues, facevalues, M, Minv, K, w)
end

@inline numsubdomains(d::ParabolicDomain) = 1
@inline getsubdomain(d::ParabolicDomain, i::Int) = i == 1 ? d : error("i=$i, but ParabolicDomain has only 1 subdomain.")
@inline getgrid(d::ParabolicDomain) = d.grid
@inline getdofhandler(d::ParabolicDomain) = d.dh
@inline cellvalues(d::ParabolicDomain) = d.cellvalues
@inline facevalues(d::ParabolicDomain) = d.facevalues
@inline getmass(d::ParabolicDomain) = d.M
@inline getmassinv(d::ParabolicDomain) = d.Minv
@inline getstiffness(d::ParabolicDomain) = d.K
@inline getquadweights(d::ParabolicDomain) = d.w
@inline JuAFEM.ndofs(d::ParabolicDomain) = JuAFEM.ndofs(getdofhandler(d))

factorize!(d::ParabolicDomain) = (d.Minv = cholfact(getmass(d)); return d)

# These interpolation methods seem overly constrictive:
#   TODO: How to do this without assuming nodal values only?
function interpolate!(u::Vector{T}, u0::Vec{dim,T}, domain::ParabolicDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    # _check_is_nodal(u, domain)
    # u = reinterpret(Vec{dim,T}, u)
    # fill!(u, u0)
    # return reinterpret(T, u)
    return interpolate!(u, x->u0, domain)
end

function interpolate!(u::Vector{T}, func::Function, domain::ParabolicDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    ch = ConstraintHandler(getdofhandler(domain))
    ∂Ω = getfaces(getgrid(domain))
    dbc = JuAFEM.Dirichlet(:u, ∂Ω, (x,t) -> func(x), collect(1:dim))
    add!(ch, dbc)
    close!(ch)
    update!(ch, 0.0)
    apply!(u, ch)
    return u

    # _check_is_nodal(u, domain)

    # u = reinterpret(Vec{dim,T}, u)
    # @inbounds for i in 1:getnnodes(getgrid(domain))
    #     x = getcoordinates(getnodes(getgrid(domain), i))
    #     u[i] = func(x)
    # end
    # return reinterpret(T, u)

    # for f in 1:JuAFEM.nfields(getdofhandler(domain))
    #     field_dim = getdofhandler(domain).field_dims[f]
    #     # space_dim = field_dim == 2 ? 3 : field_dim
    #     # data = fill(0.0, dim, getnnodes(getdofhandler(domain).grid))
    #     offset = JuAFEM.field_offset(getdofhandler(domain), getdofhandler(domain).field_names[f])
    #     for cell in CellIterator(getdofhandler(domain))
    #         _celldofs = celldofs(cell)
    #         counter = 1
    #         for node in getnodes(cell)
    #             x = getcoordinates(getnodes(getgrid(domain), node))
    #             fval = func(x)
    #             for d in 1:getdofhandler(domain).field_dims[f]
    #                 # data[d, node] = u[_celldofs[counter + offset]]
    #                 u[_celldofs[counter + offset]] = fval[d]
    #                 counter += 1
    #             end
    #         end
    #     end
    # end
    # return u

    # fill!(u,0)
    # n_basefuncs = getnbasefunctions(cellvalues(domain))
    # ue = zeros(n_basefuncs)
    # @inbounds for cell in CellIterator(getdofhandler(domain))
    #     fill!(ue, 0)
    #     coords = getcoordinates(cell)
    #     JuAFEM.reinit!(cellvalues(domain), cell)
    #     for q_point in 1:getnquadpoints(cellvalues(domain))
    #         # dΩ = getdetJdV(cellvalues(domain), q_point)
    #         coords_qp = spatial_coordinate(cellvalues(domain), q_point, coords)
    #         for i in 1:n_basefuncs
    #             v = shape_value(cellvalues(domain), q_point, i)
    #             # @show (q_point, i, dΩ, coords_qp, v)
    #             ue[i] += func(coords_qp) ⋅ v# * dΩ
    #         end
    #     end
    #     # _celldofs = celldofs(cell)
    #     # error("exiting...")
    #     # for i in 1:length(_celldofs)
    #     #     u[_celldofs[i]] += ue[i]
    #     # end
    #     assemble!(u, celldofs(cell), ue)
    # end
    # return u
end

function integrate(u::Vector{T}, domain::ParabolicDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    # _check_is_nodal(u, domain)
    u = reinterpret(Vec{dim,T}, u)
    w = reinterpret(Vec{dim,T}, getquadweights(domain))

    # Integrate. ⊙ === hadamardproduct is the Hadamard product of the vectors.
    S = zero(Vec{dim,T})
    @inbounds for i in 1:length(u)
        S += u[i] ⊙ w[i]
    end
    # S = sum(u .⊙ w) # Equivalent, but slower due to allocation of temporary

    return S
end

Base.norm(u, domain::ParabolicDomain) = √dot(u, getmass(domain) * u)

# ---------------------------------------------------------------------------- #
# Myelin grid type
# ---------------------------------------------------------------------------- #

mutable struct MyelinDomain{dim,Nd,T,Nf} <: AbstractDomain{dim,Nd,T,Nf}
    fullgrid::Grid{dim,Nd,T,Nf}
    outercircles::Vector{Circle{dim,T}}
    innercircles::Vector{Circle{dim,T}}
    domainboundary::Union{Circle{dim,T}, Rectangle{dim,T}}
    tissuedomain::ParabolicDomain{dim,Nd,T,Nf}
    myelindomains::Vector{ParabolicDomain{dim,Nd,T,Nf}}
    axondomains::Vector{ParabolicDomain{dim,Nd,T,Nf}}
end

function MyelinDomain(
        fullgrid::Grid{dim,Nd,T,Nf},
        outercircles::Vector{Circle{dim,T}},
        innercircles::Vector{Circle{dim,T}},
        domainboundary::Rectangle{dim,T},
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
@inline getsubdomains(m::MyelinDomain) = getsubdomain.(m, 1:numsubdomains(m))

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
factorize!(m::MyelinDomain) = map(i->factorize!(getsubdomain(m,i)), 1:numsubdomains(m))

function integrate(U::Vector{Vector{T}}, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    @assert length(U) == numsubdomains(domain)
    Σ = zero(Vec{dim,T})
    @inbounds for i in 1:numsubdomains(domain)
        subdomain = getsubdomain(domain, i)
        @assert length(U[i]) == ndofs(getdofhandler(subdomain))
        Σ += integrate(U[i], subdomain)
    end
    return Σ
end

function interpolate!(U::Vector{Vector{T}}, u0::Vec{dim,T}, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    @assert length(U) == numsubdomains(domain)
    @inbounds for i in 1:length(U)
        interpolate!(U[i], u0, getsubdomain(domain, i))
    end
    return U
end
function interpolate(u0::Vec{dim,T}, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    U = [zeros(ndofs(getsubdomain(domain, i))) for i in 1:numsubdomains(domain)]
    return interpolate!(U, u0, domain)
end

function interpolate!(U::Vector{Vector{T}}, func::Function, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    @assert length(U) == numsubdomains(domain)
    @inbounds for i in 1:length(U)
        interpolate!(U[i], func, getsubdomain(domain, i))
    end
    return U
end
function interpolate(func::Function, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    U = [zeros(ndofs(getsubdomain(domain, i))) for i in 1:numsubdomains(domain)]
    return interpolate!(U, func, domain)
end

function Base.norm(U::Vector{Vector{T}}, domain::MyelinDomain{dim,Nd,T,Nf}) where {dim,Nd,T,Nf}
    @assert length(U) == numsubdomains(domain)
    return √sum(i->norm(U[i], getsubdomain(domain,i))^2, 1:numsubdomains(domain))
end

# ---------------------------------------------------------------------------- #
# Assmeble mass and stiffness matrices
# ---------------------------------------------------------------------------- #

function doassemble!(domain::MyelinDomain{dim,Nd,T,Nf},
                     btparams::BlochTorreyParameters{T}) where {dim,Nd,T,Nf}
    # Exterior region
    Rdecay = (x) -> btparams[:R2_lp]
    Dcoeff = (x) -> btparams[:D_Tissue]
    Omega = (x) -> omega_tissue(x, domain, btparams)
    doassemble!(domain.tissuedomain, Rdecay, Dcoeff, Omega)

    # Myelin sheath region
    Rdecay = (x) -> btparams[:R2_sp]
    Dcoeff = (x) -> btparams[:D_Sheath]
    for i in 1:numfibres(domain)
        Omega = (x) -> omega_myelin(x, domain, btparams, i)
        doassemble!(domain.myelindomains[i], Rdecay, Dcoeff, Omega)
    end

    # Axon region
    Rdecay = (x) -> btparams[:R2_lp]
    Dcoeff = (x) -> btparams[:D_Axon]
    for i in 1:numfibres(domain)
        Omega = (x) -> omega_axon(x, domain, btparams, i)
        doassemble!(domain.axondomains[i], Rdecay, Dcoeff, Omega)
    end

    return domain
end

### Assembling the linear system
# Assemble the linear system, $K u = f$. `doassemble` takes `cellvalues`, the
# sparse matrices, a `DofHandler`, and functions Rdecay, Dcoeff, and Omega
# as input arguments. The assembled mass and stiffness matrices are returned.
function doassemble!(domain::ParabolicDomain{dim,Nd,T,Nf},
                     Rdecay::Function,
                     Dcoeff::Function,
                     Omega::Function) where {dim,Nd,T,Nf}
    # This assembly function is only for CellVectorValues
    @assert typeof(cellvalues(domain)) <: CellVectorValues

    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    n_basefuncs = getnbasefunctions(cellvalues(domain))
    Ke = zeros(n_basefuncs, n_basefuncs)
    Me = zeros(n_basefuncs, n_basefuncs)
    we = zeros(n_basefuncs)

    # Next we create assemblers for the stiffness matrix `K` and the mass
    # matrix `M`. The assemblers are just thin wrappers around `K` and `M`
    # and some extra storage to make the assembling faster.
    assembler_K = start_assemble(domain.K, domain.w)
    assembler_M = start_assemble(domain.M)

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
        JuAFEM.reinit!(cellvalues(domain), cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `Me`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`, and the quadrature
        # coordinate can be queried from `cellvalues` by `spatial_coordinate`
        for q_point in 1:getnquadpoints(cellvalues(domain))
            dΩ = getdetJdV(cellvalues(domain), q_point)
            coords_qp = spatial_coordinate(cellvalues(domain), q_point, coords)

            # calculate the heat conductivity and heat source at point `coords_qp`
            R = Rdecay(coords_qp)
            D = Dcoeff(coords_qp)
            ω = Omega(coords_qp)

            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            for i in 1:n_basefuncs
                v  = shape_value(cellvalues(domain), q_point, i)
                ∇v = shape_gradient(cellvalues(domain), q_point, i)
                we[i] += sum(v) * dΩ # v[1] and v[2] are never non-zero together
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues(domain), q_point, j)
                    ∇u = shape_gradient(cellvalues(domain), q_point, j)
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
# Freqeuency perturbation map functions
# ---------------------------------------------------------------------------- #
struct FreqMapParams{T}
    ω₀::T
    s²::T
    c²::T
end

function FreqMapParams(p::BlochTorreyParameters{T}) where {T}
    γ, B₀, θ = p[:gamma], p[:B0], p[:theta]
    ω₀ = γ * B₀
    s², c² = sin(θ)^2, cos(θ)^2
    return FreqMapParams{T}(ω₀, s², c²)
end

# Calculate ω(x) inside region number `region`, which is assumed to be myelin
function omega_myelin(
        x::Vec{2,T},
        domain::MyelinDomain{dim,Nd,T,Nf},
        btparams::BlochTorreyParameters{T},
        region::Int) where {dim,Nd,T,Nf}
    freqparams = FreqMapParams(btparams)
    ω = omega_myelin(x, freqparams, btparams, getinnercircle(domain, region), getoutercircle(domain, region))
    @inbounds for i in IterTools.chain(1:region-1, region+1:numfibres(domain))
        ω += omega_tissue(x, freqparams, btparams, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

# Calculate ω(x) inside region number `region`, which is assumed to be axon
function omega_axon(
        x::Vec{2,T},
        domain::MyelinDomain{dim,Nd,T,Nf},
        btparams::BlochTorreyParameters{T},
        region::Int) where {dim,Nd,T,Nf}
    freqparams = FreqMapParams(btparams)
    ω = omega_axon(x, freqparams, btparams, getinnercircle(domain, region), getoutercircle(domain, region))
    @inbounds for i in IterTools.chain(1:region-1, region+1:numfibres(domain))
        ω += omega_tissue(x, freqparams, btparams, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

# Calculate ω(x) inside region number `region`, which is assumed to be tissue
function omega_tissue(
        x::Vec{2,T},
        domain::MyelinDomain{dim,Nd,T,Nf},
        btparams::BlochTorreyParameters{T}) where {dim,Nd,T,Nf}
    freqparams = FreqMapParams(btparams)
    ω = zero(T)
    @inbounds for i in 1:numfibres(domain)
        ω += omega_tissue(x, freqparams, btparams, getinnercircle(domain, i), getoutercircle(domain, i))
    end
    return ω
end

@inline function omega_isotropic_tissue(x::Vec{2},
                                        b::FreqMapParams,
                                        p::BlochTorreyParameters,
                                        c_inner::Circle{2},
                                        c_outer::Circle{2})
    χI, ri², ro² = p[:ChiI], radius(c_inner)^2, radius(c_outer)^2
    dx = x - origin(c_inner)
    r² = dx⋅dx
    cos2ϕ = ((dx[1]-dx[2])*(dx[1]+dx[2]))/r²
    return b.ω₀ * χI * b.s²/2 * cos2ϕ * (ro² - ri²)/r²
end

@inline function omega_anisotropic_tissue(x::Vec{2},
                                          b::FreqMapParams,
                                          p::BlochTorreyParameters,
                                          c_inner::Circle{2},
                                          c_outer::Circle{2})
    χA, ri², ro² = p[:ChiA], radius(c_inner)^2, radius(c_outer)^2
    dx = x - origin(c_inner)
    r² = dx⋅dx
    cos2ϕ = ((dx[1]-dx[2])*(dx[1]+dx[2]))/r²
    return b.ω₀ * χA * b.s²/8 * cos2ϕ * (ro² - ri²)/r²
end

@inline function omega_isotropic_myelin(x::Vec{2},
                                        b::FreqMapParams,
                                        p::BlochTorreyParameters,
                                        c_inner::Circle{2},
                                        c_outer::Circle{2})
    χI, ri² = p[:ChiI], radius(c_inner)^2
    dx = x - origin(c_inner)
    r² = dx⋅dx
    cos2ϕ = ((dx[1]-dx[2])*(dx[1]+dx[2]))/r²
    return b.ω₀ * χI * ( b.c² - 1/3 - b.s² * cos2ϕ * ri² / r² )/2
end

@inline function omega_anisotropic_myelin(x::Vec{2},
                                          b::FreqMapParams,
                                          p::BlochTorreyParameters,
                                          c_inner::Circle{2},
                                          c_outer::Circle{2})
    χA, ri², ro = p[:ChiA], radius(c_inner)^2, radius(c_outer)
    dx = x - origin(c_inner)
    r² = dx⋅dx
    r = √r²
    cos2ϕ = ((dx[1]-dx[2])*(dx[1]+dx[2]))/r²
    return b.ω₀ * χA * ( b.s² * (-5/12 - cos2ϕ/8 * (1 + ri²/r²) + 3/4 * log(ro/r)) - b.c²/6 )
end

@inline function omega_anisotropic_axon(x::Vec{2},
                                        b::FreqMapParams,
                                        p::BlochTorreyParameters,
                                        c_inner::Circle{2},
                                        c_outer::Circle{2})
    χA, ri, ro = p[:ChiA], radius(c_inner), radius(c_outer)
    return b.ω₀ * χA * 3b.s²/4 * log(ro/ri)
end

# Sum components to omega in the tissue region
@inline function omega_tissue(x::Vec{2},
                              b::FreqMapParams,
                              p::BlochTorreyParameters,
                              c_inner::Circle{2},
                              c_outer::Circle{2})
        return omega_isotropic_tissue(x,b,p,c_inner,c_outer) + omega_anisotropic_tissue(x,b,p,c_inner,c_outer)
end

# Sum components to omega in the myelin sheath region
@inline function omega_myelin(x::Vec{2},
                              b::FreqMapParams,
                              p::BlochTorreyParameters,
                              c_inner::Circle{2},
                              c_outer::Circle{2})
    return omega_isotropic_myelin(x,b,p,c_inner,c_outer) + omega_anisotropic_myelin(x,b,p,c_inner,c_outer)
end

# Sum components to omega in the axonal region
@inline function omega_axon(x::Vec{2},
                            b::FreqMapParams,
                            p::BlochTorreyParameters,
                            c_inner::Circle{2},
                            c_outer::Circle{2})
    return omega_anisotropic_axon(x,b,p,c_inner,c_outer)
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
paraboliclinearmap(d::ParabolicDomain) = paraboliclinearmap(getstiffness(d), getmassinv(d))

#TODO: Probably don't need to define these; would only be used for normest1
# which is definitely not a bottleneck, and this clearly could come back to bite
# me at some unknown time...
# import Base.LinAlg: A_mul_B!, At_mul_B!, Ac_mul_B!
# A_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.f!(Y,X);
# At_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.fc!(Y,X);
# Ac_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.fc!(Y,X);

# ---------------------------------------------------------------------------- #
# expmv and related functions
# ---------------------------------------------------------------------------- #

# Custom norm for calling expmv
expmv_norm(x::AbstractVector, p::Real=2, args...) = Base.norm(x, p, args...) #fallback
function expmv_norm(A, p::Real=1, t::Int=10)
    if p == 1
        return normest1(A, t)[1]
    elseif p == Inf
        return normest1(A', t)[1]
    else
        error("Only p=1 or p=Inf supported")
    end
end

nothing
