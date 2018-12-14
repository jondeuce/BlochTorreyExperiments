function testbtfindiff2D(opts::BlochTorreyParameters{T};
        Npts::Int           = 100,                  # Number of points per dimension
        Domain::NTuple{2,T} = opts.R_mu .* (-2,2),  # Bounds for side of square domain
        Router::T           = opts.R_mu,            # Outer radius; defaults to R_mu
        Time::T             = 60e-3,                # Simulation time
        IsPermeable::Bool   = false,                # Permeability of myelin
        Plot::Bool          = true                  # Plot resulting magnitude and phase
    ) where {T}

    @assert opts.D_Axon == opts.D_Tissue == opts.D_Sheath

    D = opts.D_Tissue
    a, b = Domain
    h = (b-a)/(Npts-1)
    pts = range(a, stop = b, length = Npts)

    Gamma(x,y) = complex(r2decay(x,y,opts,Router), omega(x,y,opts,Router))
    G = Complex{T}[Gamma(x,y) for x in pts, y in pts]

    g  = opts.g_ratio # 0.8
    ro = Router #0.5
    ri = g*ro
    m_int = IsPermeable ? falses(Npts,Npts) : Bool[x^2 + y^2 < ri^2 for x in pts, y in pts]
    m_ext = IsPermeable ?  trues(Npts,Npts) : Bool[x^2 + y^2 > ro^2 for x in pts, y in pts]

    tmp = similar(G)
    function A(du,u,isadj::Bool=false)
        # du .= D .* lap(u, h, h, m_int, m_ext) .- G .* u
        u = reshape(u, Npts, Npts)
        du = reshape(du, Npts, Npts)
        lap!(du, u, h, h, m_int, m_ext)
        du .*= D
        isadj ? (tmp .= conj.(G) .* u) : (tmp .= G .* u)
        axpy!(-one(real(T)), tmp, du)
        return du
    end
    Amap = LinearMaps.FunctionMap{Complex{T}}(
        (du,u) -> A(du,u,false), (du,u) -> A(du,u,true), Npts^2;
        issymmetric = true, ishermitian = false, ismutating = true)

    U0 = fill(one(T)*im, Npts^2)
    U = similar(U0)
    @time U = Expokit.expmv!(U, Time, Amap, U0)

    U0 = reshape(U0, Npts, Npts)
    U = reshape(U, Npts, Npts)

    # unwrap = @(x) sliceND(unwrapLap(repmat(x,1,1,11)),6,3)
    # figure, imagesc(rot90(unwrap(angle(U)))); axis image; title phase; colorbar
    # figure, imagesc(rot90(abs(U))); axis image; title magnitude; colorbar

    # Return a named tupled of geometry structures
    geom = (
        pts = pts,
        m_int = m_int,
        m_ext = m_ext,
        Amap = Amap,
        U0 = U0
    )

    return U, geom
end

function omega(
        x::T,
        y::T,
        opts::BlochTorreyParameters{T} = BlochTorreyParameters{T}(),
        Router::T = opts.R_mu
    ) where {T}

    B0    = opts.B0    # -3.0         # External magnetic field (z-direction) [T]
    gamma = opts.gamma # 2.67515255e8 # Gyromagnetic ratio [rad/s/T]
    w0    = gamma * B0 # Resonance frequency [rad/s]
    th    = opts.theta # pi/2   # Main magnetic field angle w.r.t B0 [rad]
    c2    = cos(th)^2
    s2    = sin(th)^2
    ChiI  = opts.ChiI # -60e-9  # Isotropic susceptibility of myelin [ppb] (check how to get it) (Xu et al. 2017)
    ChiA  = opts.ChiA # -120e-9 # Anisotropic Susceptibility of myelin [ppb] (Xu et al. 2017)
    E     = opts.E    #  10e-9  # Exchange component to resonance freqeuency [ppb] (Wharton and Bowtell 2012)

    g  = opts.g_ratio # 0.8
    ro = Router #0.5
    ri = g*ro
    ri2 = ri^2
    ro2 = ro^2

    r2 = x^2 + y^2
    r = sqrt(r2)
    t = atan(y,x)

    w = if (r < ri)
        w0 * ChiA * 3*s2/4 * log(ro/ri)
    elseif (ri <= r && r <= ro)
        (w0 * ChiI/2) * (c2 - 1/3 - s2 * cos(2*t) * (ri2/r2)) +
        (w0 * E) +
        (w0 * ChiA) * (s2 * (-5/12 - cos(2*t)/8 * (1+ri2/r2) + (3/4) * log(ro/r)) - c2/6)
    else
        (w0 * ChiI * (s2/2)) * cos(2*t) * (ro2 - ri2) / r2 +
        (w0 * ChiA * (s2/8)) * cos(2*t) * (ro2 - ri2) / r2
    end

    return w
end

function r2decay(
        x::T,
        y::T,
        opts::BlochTorreyParameters{T} = BlochTorreyParameters{T}(),
        Router::T = opts.R_mu # Outer radius relative to R_mu
    ) where {T}

    R2_sp = opts.R2_sp # 1/15e-3 # Relaxation rate of small pool [s^-1] (Myelin) (Xu et al. 2017) (15e-3s)
    R2_lp = opts.R2_lp # 1/63e-3 # Relaxation rate of large pool [s^-1] (Intra/Extra-cellular)

    g  = opts.g_ratio # 0.8
    ro = Router #0.5
    ri = g*ro
    ri2 = ri^2
    ro2 = ro^2
    r2 = x^2 + y^2

    R2 = ri2 <= r2 && r2 <= ro2 ? R2_sp : R2_lp

    return R2
end

# struct BTOp{T,Tr,N}
#     G::AbstractArray{T,N}
#     h::NTuple{N,Tr}
#     m_int::AbstractArray{Bool,N}
#     m_ext::AbstractArray{Bool,N}
# end
# function LinearAlgebra.mul!(du::AbstractArray{T,2}, A::BTOp{T}, u::AbstractArray{T,2}) where {T}
#     # return lap(u, A.h[1], A.h[2], A.m_int, A.m_ext) .- A.G .* u
#     lap!(du, u, A.h[1], A.h[2], A.m_int, A.m_ext)
#     axpy!(-one(real(T)), A.G .* u, du)
#     return du
# end
# Base.size(A::BTOp) = (length(A.G), length(A.G))
# Base.size(A::BTOp, d::Int) = (d == 1 || d == 2) ? length(A.G) : 1
# Base.eltype(A::BTOp{T}) where {T} = T
# Base.:*(a::Number, A::BTOp) = BTOp(a.*A.G, A.h./sqrt(a), A.m_int, A.m_ext)
# LinearAlgebra.opnorm(A::BTOp{T}, p) where {T} = one(T)
