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
# Global dcoeff/r1decay/r2decay functions on each region
# ---------------------------------------------------------------------------- #

#TODO: re-write to take in plain vectors of inner/outer circles/ferritins which
# can be called on its own, and wrap with a method that takes a MyelinDomain,
# just like omega(x,p,outercircles,innercircles,...) above

@inline dcoeff(x, p::MyelinProblem, m::MyelinDomain{TissueRegion}) = p.params.D_Tissue
@inline dcoeff(x, p::MyelinProblem, m::MyelinDomain{MyelinRegion}) = p.params.D_Sheath
@inline dcoeff(x, p::MyelinProblem, m::MyelinDomain{AxonRegion}) = p.params.D_Axon
@inline dcoeff(x, y, p::MyelinProblem, m::MyelinDomain) = dcoeff(Vec{2}((x, y)), p, m)

@inline r1decay(x, p::MyelinProblem, m::MyelinDomain{TissueRegion}) = p.params.R1_Tissue
@inline r1decay(x, p::MyelinProblem, m::MyelinDomain{MyelinRegion}) = p.params.R1_sp
@inline r1decay(x, p::MyelinProblem, m::MyelinDomain{AxonRegion}) = p.params.R1_lp
@inline r1decay(x, y, p::MyelinProblem, m::MyelinDomain) = r1decay(Vec{2}((x, y)), p, m)

@inline r2decay(x, p::MyelinProblem, m::MyelinDomain{TissueRegion}) = p.params.R2_Tissue
@inline r2decay(x, p::MyelinProblem, m::MyelinDomain{MyelinRegion}) = p.params.R2_sp
@inline r2decay(x, p::MyelinProblem, m::MyelinDomain{AxonRegion}) = p.params.R2_lp
@inline r2decay(x, y, p::MyelinProblem, m::MyelinDomain) = r2decay(Vec{2}((x, y)), p, m)