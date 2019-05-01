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

export pack

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #

function pack(
        radii::AbstractVector;
        initial_origins::AbstractVector{V} = initialize_origins(radii; distribution = :uniformsquare),
        distancescale = mean(radii),
        epsilon::Real = zero(eltype(radii)),
        autodiff::Bool = false,
        alg = Tsit5(),
        SSalg = DynamicSS(alg; reltol = 1e-3*distancescale, abstol = 0)
        # SSalg = SSRootfind(alg; reltol = 1e-3*distancescale, abstol = 0),
    ) where {V <: Vec{2}}
    
    # Initialize
    origins = copy(initial_origins)
    origins .-= Ref(mean(origins))
    origins .*= 1.1
    Rmax = maximum(radii)
    xmin, xmax = extrema(x[1] for x in origins)
    ymin, ymax = extrema(x[2] for x in origins)
    domain = Rectangle(V((xmin - Rmax, ymin - Rmax)), V((xmax + Rmax, ymax + Rmax)))

    # N-body attractive Coulomb force function
    function f!(du, u, p, t)
        r, R, P, β = p
        Vdu, Vu = Vec{2,eltype(du)}, Vec{2,eltype(u)}
        dU, U = reinterpret(NTuple{2,Vdu}, du), reinterpret(NTuple{2,Vu}, u)
        @inbounds for i in eachindex(dU, U)
            xi, vi = U[i]
            mi = (r[i]/R)^2
            Fi = zero(Vdu)
            for j in eachindex(dU, U)
                (i == j) && continue
                xj, vj = U[j]
                mj = (r[j]/R)^2
                dx = periodic_diff(xj, xi, P) # xj - xi
                d = norm(dx)
                Fi += (mi * mj / (d/R)^2) * (dx/d) # Coulomb attraction
                # Fi -= (mi * mj / (d/R)^2) * (dx/d) # Coulomb repulsion
            end
            Fi /= mi
            Fi -= β * (vi/R) # Friction
            dU[i] = (vi, Fi)
        end
        return du
    end

    function callback_factory(i,j)
        condition = (u,t,integrator) -> begin
            r, _, P, _ = integrator.p
            Vu = Vec{2,eltype(u)}
            U = reinterpret(NTuple{2,Vu}, u)
            ri, rj = r[i], r[j]
            xi, xj = U[i][1], U[j][1]
            
            dx = periodic_diff(xj, xi, P) # xj - xi
            
            @show norm(dx) - ri - rj

            error("exiting condition...")

            return norm(dx) - ri - rj
        end
        affect! = (integrator) -> begin
            _, _, P, _ = integrator.p
            u = integrator.u
            Vu = Vec{2,eltype(u)}
            U = reinterpret(NTuple{2,Vu}, u)
            xi, vi = U[i]
            xj, vj = U[j]
            
            n = periodic_diff(xj, xi, P) # xj - xi
            n /= norm(n) # unit normal
            t = Vu((-n[2], n[1])) # unit tangent
            
            @info "before"
            @show vi ⋅ n
            @show vj ⋅ n
            @show vi ⋅ t
            @show vj ⋅ t

            vi = (vi ⋅ t) * t - (vi ⋅ n) * n # normal component of velocity is flipped
            vj = (vj ⋅ t) * t - (vj ⋅ n) * n # normal component of velocity is flipped
            U[i] = (xi, vi)
            U[j] = (xj, vj)
            
            @info "after"
            @show vi ⋅ n
            @show vj ⋅ n
            @show vi ⋅ t
            @show vj ⋅ t

            error("exiting affect!...")

            return integrator
        end
        return ContinuousCallback(condition, affect!)
    end

    # Initial state
    u0 = zeros(eltype(V), 4*length(radii))
    U0 = reinterpret(NTuple{2,V}, u0)
    for i in eachindex(U0)
        x0 = origins[i]
        v0 = zero(V)
        U0[i] = (x0, v0)
    end

    @show u0
    
    # Callback
    cb = CallbackSet([callback_factory(i,j) for i in 2:length(radii) for j in 1:i-1]...)
    
    β = 1.0 # friction
    p = (radii, Rmax, widths(domain), β)
    prob = SteadyStateProblem{true}(f!, u0, p)
    sol = solve(prob, SSalg; callback = cb)

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

end # module NBodyCirclePacking
