# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

module NBodyCirclePacking

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using ..CirclePackingUtils
using ..CirclePackingUtils: d², ∇d², ∇²d²
using GeometryUtils
using OrdinaryDiffEq, SteadyStateDiffEq
using Parameters: @unpack

export pack

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #

function pack(
        radii::AbstractVector;
        initial_origins::AbstractVector{V} = initialize_origins(radii; distribution = :uniformsquare),
        domain::Rectangle{2} = initialize_domain(radii, initial_origins),
        tspan = (0.0, 100.0), # timespan
        init_speed = 5.0, # initial speed
        min_speed = 1e-2, # minimum speed
        damping = 0.5, # friction
        distancescale = mean(radii),
        epsilon::Real = zero(eltype(radii)),
        autodiff::Bool = false,
        alg = BS3(),
        # SSalg = DynamicSS(alg; reltol = 1e-3*distancescale, abstol = 0)
        # SSalg = SSRootfind(alg; reltol = 1e-3*distancescale, abstol = 0),
    ) where {V <: Vec{2}}
    
    # Initial state
    Rmax = maximum(radii)
    u0 = zeros(eltype(V), 4*length(radii))
    U0 = reinterpret(NTuple{2,V}, u0)
    for i in eachindex(U0)
        x0 = initial_origins[i]
        v0 = randn(V)
        v0 = (init_speed * Rmax) * (v0 / norm(v0))
        U0[i] = (x0, v0)
    end
    
    # Callback
    cb = CallbackSet(
        termination_callback(min_speed * Rmax),
        # any_circle_overlap_callback()
        [circle_pair_overlap_callback(i,j) for i in 2:length(radii) for j in 1:i-1]...
    )
    
    # @info "Min speed: $min_speed"
    # @info "Alg = $alg"
    # @info "Friction: β = $damping"
    # @info "tspan = $tspan"

    p = (r = radii, R = Rmax, P = widths(domain), β = damping, vmin = min_speed * Rmax)
    prob = ODEProblem{true}(coulomb!, u0, tspan, p)
    sol = solve(prob, alg; callback = cb, reltol = 1e-3, abstol = 0)
    # prob = SteadyStateProblem{true}(coulomb!, u0, p)
    # sol = solve(prob, SSalg; callback = cb)#, dt = 0.1)

    # Extract circles
    U = reinterpret(NTuple{2,V}, sol.u[end])
    circles = [Circle(U[1], radii[i]) for (i,U) in enumerate(U)]
    circles = periodic_unique_circles(circles, domain)

    return (sol = sol, circles = circles, domain = domain)
end

function pack(c::AbstractVector{Circle{2,T}}; kwargs...) where {T}
    o, r = tovectors(c)
    o = reinterpret(Vec{2,T}, o) |> copy
    return pack(r; initial_origins = o, kwargs...)
end

# N-body attractive Coulomb force function
function coulomb!(du, u, p, t)
    @unpack r, R, P, β, vmin = p
    Vdu, Vu = Vec{2,eltype(du)}, Vec{2,eltype(u)}
    dU, U = reinterpret(NTuple{2,Vdu}, du), reinterpret(NTuple{2,Vu}, u)
    @inbounds for i in eachindex(dU, U)
        xi, vi = U[i]
        mi = (r[i]/R)^2
        Fci = zero(Vdu)
        for j in eachindex(dU, U)
            (i == j) && continue
            xj, vj = U[j]
            mj = (r[j]/R)^2
            dx = periodic_diff(xj, xi, P)
            # dx = xj - xi
            L = norm(dx)
            # Fci += (mi * mj / (L/R)^2) * (dx/L) # Coulomb attraction
            Fci -= (mi * mj / (L/R)^2) * (dx/L) # Coulomb repulsion
        end
        Ffi = -β * (vi/R) # Friction
        γ = tanh(norm2(vi/vmin)) # Coulomb weighting
        ai = (γ * Fci + Ffi)/mi # Acceleration
        dU[i] = (vi, ai)
    end
    return du
end

function termination_callback(vmin)
    function condition(u,t,integrator)
        U = reinterpret(Vec{2,eltype(u)}, u)
        return all(v -> norm2(v) < vmin^2, @views(U[2:2:end]))
    end
    affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(condition, affect!)
end

function circle_pair_overlap_callback(i,j)
    function condition(u,t,integrator)
        @unpack r, P = integrator.p
        Vu = Vec{2,eltype(u)}
        U = reinterpret(NTuple{2,Vu}, u)
        ri, rj = r[i], r[j]
        xi, xj = U[i][1], U[j][1]
        dx = periodic_diff(xj, xi, P)
        # dx = xj - xi
        d = norm(dx) - ri - rj
        return d
    end

    function affect_neg!(integrator)
        @unpack P = integrator.p
        u = integrator.u
        Vu = Vec{2,eltype(u)}
        U = reinterpret(NTuple{2,Vu}, u)
        xi, vi = U[i]
        xj, vj = U[j]
        
        n = periodic_diff(xj, xi, P)
        # n = xj - xi
        n /= norm(n) # unit normal
        t = Vu((-n[2], n[1])) # unit tangent
        vi = (vi ⋅ t) * t - (vi ⋅ n) * n # normal component of velocity is flipped
        vj = (vj ⋅ t) * t - (vj ⋅ n) * n # normal component of velocity is flipped
        
        U[i] = (xi, vi)
        U[j] = (xj, vj)
        return u_modified!(integrator, true)
    end
    
    affect! = nothing # do nothing on upcrossings (negative to positive)
    
    return ContinuousCallback(condition, affect!, affect_neg!)
end

function any_circle_overlap_callback()
    imin, jmin = 0, 0
    function condition(u,t,integrator)
        @unpack r, P = integrator.p
        Vu = Vec{2,eltype(u)}
        U = reinterpret(NTuple{2,Vu}, u)

        d = eltype(u)(Inf)
        for i in 2:length(r), j in 1:i-1
            ri, rj = r[i], r[j]
            xi, xj = U[i][1], U[j][1]
            dx = periodic_diff(xj, xi, P)
            # dx = xj - xi
            dij = norm(dx) - ri - rj
            if dij < d
                d = min(d, dij)
                imin, jmin = i, j
            end
        end
        return d
    end

    function affect_neg!(integrator)
        @unpack r, P = integrator.p
        u = integrator.u
        Vu = Vec{2,eltype(u)}
        U = reinterpret(NTuple{2,Vu}, u)
        i,j = imin, jmin
        xi, vi = U[i]
        xj, vj = U[j]

        dx = periodic_diff(xj, xi, P)
        # dx = xj - xi
        n = dx / norm(dx) # unit normal
        t = Vu((-n[2], n[1])) # unit tangent
        vi = (vi ⋅ t) * t - (vi ⋅ n) * n # normal component of velocity is flipped
        vj = (vj ⋅ t) * t - (vj ⋅ n) * n # normal component of velocity is flipped

        U[i] = (xi, vi)
        U[j] = (xj, vj)
        return u_modified!(integrator, true)
    end

    affect! = nothing # do nothing on upcrossings (negative to positive)
    
    return ContinuousCallback(condition, affect!, affect_neg!)
end

end # module NBodyCirclePacking
