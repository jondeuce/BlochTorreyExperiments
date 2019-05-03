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
        distancescale = mean(radii),
        epsilon::Real = zero(eltype(radii)),
        autodiff::Bool = false,
        alg = Euler(),#Tsit5(),
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

    # Initial state
    u0 = zeros(eltype(V), 4*length(radii))
    U0 = reinterpret(NTuple{2,V}, u0)
    for i in eachindex(U0)
        x0 = origins[i]
        v0 = rand(V)
        v0 = Rmax * v0 / norm(v0)
        U0[i] = (x0, v0)
    end

    # Callback
    cb = CallbackSet([circle_pair_overlap_callback(i,j) for i in 2:length(radii) for j in 1:i-1]...)
    # cb = any_circle_overlap_callback()

    β = 0.5 # friction
    @info "Friction: β = $β"
    p = (r = radii, R = Rmax, P = widths(domain), β = β)
    prob = SteadyStateProblem{true}(coulomb!, u0, p)
    sol = solve(prob, SSalg; callback = cb, dt = 0.1)

    # Extract circles
    U = reinterpret(NTuple{2,V}, sol.u[end])
    circles = [Circle(U[1], radii[i]) for (i,U) in enumerate(U)]
    # circles = periodic_unique_circles(circles, domain)

    return (sol = sol, circles = circles, domain = domain)
end

function pack(c::AbstractVector{Circle{2,T}}; kwargs...) where {T}
    o, r = tovectors(c)
    o = reinterpret(Vec{2,T}, o) |> copy
    return pack(r; initial_origins = o, kwargs...)
end

# N-body attractive Coulomb force function
function coulomb!(du, u, p, t)
    @unpack r, R, P, β = p
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
            # dx = periodic_diff(xj, xi, P)
            dx = xj - xi
            L = norm(dx)
            Fci += (mi * mj / (L/R)^2) * (dx/L) # Coulomb attraction
            # Fci -= (mi * mj / (L/R)^2) * (dx/L) # Coulomb repulsion
        end
        Ffi = -β * (vi/R) # Friction
        γ = tanh(norm2(vi/R)) # Coulomb weighting
        ai = (γ * Fci + Ffi)/mi # Acceleration
        dU[i] = (vi, ai)
    end
    return du
end

function circle_pair_overlap_callback(i,j)
    cond_count = 0
    aff_count = 0
    II, JJ = 8, 4
    condition = (u,t,integrator) -> begin
        @unpack r, P = integrator.p
        Vu = Vec{2,eltype(u)}
        U = reinterpret(NTuple{2,Vu}, u)
        ri, rj = r[i], r[j]
        xi, xj = U[i][1], U[j][1]
        # dx = periodic_diff(xj, xi, P)
        dx = xj - xi
        d = norm(dx) - ri - rj

        # if (cond_count <= 100)# && (i==II && j==JJ)
        #     if d < 0
        #         @info "Circles $i and $j overlap:  d = $d"
        #     else
        #         @info "Circles $i and $j distance: d = $d"
        #     end
        # end
        
        if mod(cond_count,100) == 0
            if (cond_count <= 1000)# && (i==II && j==JJ)
                v = maximum(U->norm(U[2]), U)
                @info "Maximum velocity norm (t = $t): v = $v"
            end
        end

        cond_count += 1
        # cond_count == 100 && error("exiting at callback...")

        return d
    end

    affect_neg! = (integrator) -> begin
        @unpack P = integrator.p
        u = integrator.u
        Vu = Vec{2,eltype(u)}
        U = reinterpret(NTuple{2,Vu}, u)
        xi, vi = U[i]
        xj, vj = U[j]
        
        # n = periodic_diff(xj, xi, P)
        n = xj - xi
        n /= norm(n) # unit normal
        t = Vu((-n[2], n[1])) # unit tangent
        
        # @info "Circles $i and $j collided"
        # @info "before"
        # @show vi ⋅ n
        # @show vj ⋅ n
        # @show vi ⋅ t
        # @show vj ⋅ t

        vi = (vi ⋅ t) * t - (vi ⋅ n) * n # normal component of velocity is flipped
        vj = (vj ⋅ t) * t - (vj ⋅ n) * n # normal component of velocity is flipped
        U[i] = (xi, vi)
        U[j] = (xj, vj)
        
        # @info "after"
        # @show vi ⋅ n
        # @show vj ⋅ n
        # @show vi ⋅ t
        # @show vj ⋅ t

        if aff_count <= 10# && (i==II && j==JJ)
            @info "Circles $i and $j collided"
            aff_count += 1
        end

        # error("exiting affect!...")

        return u_modified!(integrator, true)
    end
    
    affect! = nothing # do nothing on upcrossings (negative to positive)
    
    return ContinuousCallback(condition, affect!, affect_neg!)
end

function any_circle_overlap_callback()
    cond_count = 0
    aff_count = 0
    imin, jmin = 0, 0
    condition = (u,t,integrator) -> begin
        @unpack r, P = integrator.p
        Vu = Vec{2,eltype(u)}
        U = reinterpret(NTuple{2,Vu}, u)

        d = eltype(u)(Inf)
        for i in 2:length(r), j in 1:i-1
            ri, rj = r[i], r[j]
            xi, xj = U[i][1], U[j][1]
            # dx = periodic_diff(xj, xi, P)
            dx = xj - xi
            dij = norm(dx) - ri - rj
            if dij < d
                d = min(d, dij)
                imin, jmin = i, j
            end
        end

        if mod(cond_count,100) == 0
            if (cond_count <= 1000)
                v = maximum(U->norm(U[2]), U)
                @info "Maximum velocity norm (t = $t): v = $v"
            end
        end

        cond_count += 1
        # cond_count == 100 && error("exiting at callback...")

        return d
    end

    affect_neg! = (integrator) -> begin
        @unpack r, P = integrator.p
        u = integrator.u
        Vu = Vec{2,eltype(u)}
        U = reinterpret(NTuple{2,Vu}, u)

        i,j = imin, jmin
        xi, vi = U[i]
        xj, vj = U[j]
        # dx = periodic_diff(xj, xi, P)
        dx = xj - xi
        n = dx / norm(dx) # unit normal
        t = Vu((-n[2], n[1])) # unit tangent

        vi = (vi ⋅ t) * t - (vi ⋅ n) * n # normal component of velocity is flipped
        vj = (vj ⋅ t) * t - (vj ⋅ n) * n # normal component of velocity is flipped
        U[i] = (xi, vi)
        U[j] = (xj, vj)

        @info "Circles $i and $j collided (t = $(integrator.t)): d = $(norm(dx) - r[i] - r[j])"
        aff_count += 1

        return u_modified!(integrator, true)
    end

    affect! = nothing # do nothing on upcrossings (negative to positive)
    
    return ContinuousCallback(condition, affect!, affect_neg!)
end

end # module NBodyCirclePacking
