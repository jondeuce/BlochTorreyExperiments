# ---------------------------------------------------------------------------- #
# Bloch-Torrey parameters type
# ---------------------------------------------------------------------------- #
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
        :R2_sp          =>    1.0/15e-3,    # Relaxation rate of small pool [s^-1] (Myelin) (Xu et al. 2017) (15e-3s)
        :R2_lp          =>    1.0/63e-3,    # 1st attempt was 63E-3. 2nd attempt 76 ms
        :R2_Tissue      =>    14.5,         # Relaxation rate of tissue [s^-1]
        :R2_water       =>    1.0/2.2,      # Relaxation rate of pure water
        :D_Tissue       =>    2000.0,       # Diffusion coefficient in tissue [um^2/s]
        :D_Sheath       =>    2000.0,       # Diffusion coefficient in myelin sheath [um^2/s]
        :D_Axon         =>    2000.0,       # Diffusion coefficient in axon interior [um^2/s]
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
# Freqeuency perturbation map functions
# ---------------------------------------------------------------------------- #
struct FreqMapBuffer{T}
    ω₀::T
    s²::T
    c²::T
    r::T
    r²::T
    cos2ϕ::T
end

function FreqMapBuffer(p::BlochTorreyParameters, x::Vec{2})
    ω₀ = p[:gamma] * p[:B0]
    s², c² = sin(θ)^2, cos(θ)^2
    r² = x⋅x
    r = √r²
    cos2ϕ = x[1]^2 - x[2]^2
    return FreqMapBuffer(ω₀, s², c², r, r², cos2ϕ)
end

@inline function omega_isotropic_tissue(p::BlochTorreyParameters,
                                        b::FreqMapBuffer,
                                        c_inner::Circle{2},
                                        c_outer::Circle{2})
    χI, ri², ro² = p[:ChiI], radius(c_inner)^2, radius(c_outer)^2
    return b.ω₀ * χI * b.s²/2 * b.cos2ϕ * (ro² - ri²)/b.r²
end

@inline function omega_anisotropic_tissue(p::BlochTorreyParameters,
                                          b::FreqMapBuffer,
                                          c_inner::Circle{2},
                                          c_outer::Circle{2})
    χA, ri², ro² = p[:ChiA], radius(c_inner)^2, radius(c_outer)^2
    return b.ω₀ * χA * b.s²/8 * b.cos2ϕ * (ro² - ri²)/b.r²
end

@inline function omega_isotropic_sheath(p::BlochTorreyParameters,
                                        b::FreqMapBuffer,
                                        c_inner::Circle{2},
                                        c_outer::Circle{2})
    χI, ri² = p[:ChiI], radius(c_inner)^2
    return b.ω₀ * χI * ( b.c² - 1/3 - b.s² * b.cos2ϕ * ri² / b.r² )/2
end

@inline function omega_anisotropic_sheath(p::BlochTorreyParameters,
                                          b::FreqMapBuffer,
                                          c_inner::Circle{2},
                                          c_outer::Circle{2})
    χA, ri², ro = p[:ChiA], radius(c_inner)^2, radius(c_outer)
    return b.ω₀ * χA * ( b.s² * (-5/12 - b.cos2ϕ/8 * (1 + ri²/b.r²) + 3/4 * log(ro/b.r)) - b.c²/6 )
end

@inline function omega_anisotropic_axon(p::BlochTorreyParameters,
                                        b::FreqMapBuffer,
                                        c_inner::Circle{2},
                                        c_outer::Circle{2})
    χA, ri, ro = p[:ChiA], radius(c_inner), radius(c_outer)
    return b.ω₀ * χA * 3b.s²/4 * log(ro/ri)
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
function get_mass_and_stifness_map(K, Mfact)
   @assert (size(K) == size(Mfact)) && (size(K,1) == size(K,2))
   fwd_mul! = (Y, X) -> Minv_K_mul_u!(Y, X, K, Mfact);
   trans_mul! = (Y, X) -> Kt_Minv_mul_u!(Y, X, K, Mfact);
   return LinearMap(fwd_mul!, trans_mul!, size(K)...;
      ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)
end

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
