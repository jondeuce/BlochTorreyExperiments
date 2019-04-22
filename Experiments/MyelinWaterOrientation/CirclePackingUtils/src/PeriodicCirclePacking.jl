# ============================================================================ #
# Tools for creating a set of packed circles
# ============================================================================ #

module PeriodicCirclePacking

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using ..CirclePackingUtils
using ..CirclePackingUtils: d², ∇d², ∇²d²
using ..CirclePackingUtils: expbarrier, ∇expbarrier, ∇²expbarrier
using ..CirclePackingUtils: softplusbarrier, ∇softplusbarrier#, ∇²softplusbarrier
using ..CirclePackingUtils: genericbarrier, ∇genericbarrier, ∇²genericbarrier
using GeometryUtils
using LinearAlgebra, Statistics
using DiffResults, Optim, LineSearches, ForwardDiff
using Tensors

export pack

# ---------------------------------------------------------------------------- #
# Create circle packing from set of radii
# ---------------------------------------------------------------------------- #

function pack(
        radii::AbstractVector;
        initial_origins::AbstractVector{V} = initialize_origins(radii; distribution = :uniformsquare),
        distancescale = mean(radii),
        epsilon::Real = 0.1*distancescale,
        autodiff::Bool = false,
        chunksize::Int = 10,
        secondorder::Bool = false,
        Alg = secondorder ? Newton(linesearch = LineSearches.BackTracking(order=3))
                          : LBFGS(linesearch = LineSearches.BackTracking(order=3)),
        Opts = Optim.Options(iterations = 100_000,
                             x_tol = 1e-6*distancescale,
                             g_tol = 1e-12,
                             allow_f_increases = true)
    ) where {V <: Vec{2}}

    # Initial unknowns
    Rmax = maximum(radii)
    xmin, xmax = extrema(x[1] for x in initial_origins)
    ymin, ymax = extrema(x[2] for x in initial_origins)
    W0, H0 = (xmax - xmin) + 2*Rmax, (ymax - ymin) + 2*Rmax # initial width/height of box
    os = initial_origins .- Ref(V((xmin - Rmax, ymin - Rmax))) # ensure all circles are in box
    x0 = [reinterpret(eltype(V), os); W0; H0] # initial unknowns

    # Create energy function and gradient/hessian
    if !autodiff && secondorder
        @warn "Hessian not implemented; set autodiff = true for second order. Defaulting to first order."
        secondorder = false
    end

    if autodiff
        energy = autodiff_barrier_energy(radii, epsilon)
        ∇energy! = nothing
    else
        energy, ∇energy!, _ = barrier_energy(radii, epsilon)
    end

    # Form (*)Differentiable object
    opt_obj =
        (autodiff && secondorder) ? TwiceDifferentiable(energy, x0; autodiff = :forward) :
        (autodiff && !secondorder) ? OnceDifferentiable(energy, x0; autodiff = :forward) :
        OnceDifferentiable(energy, ∇energy!, x0)

    # Optimize and get results
    result = optimize(opt_obj, x0, Alg, Opts)

    # Extract results
    x = copy(Optim.minimizer(result))
    widths = V((abs(x[end-1]), abs(x[end])))
    origins = reinterpret(V, x[1:end-2]) |> copy
    origins .= periodic_mod.(origins, Ref(widths)) # force origins to be contained in [0,W] x [0,H]
    packed_circles = Circle.(origins, radii)
    boundary_rectangle = Rectangle(zero(V), widths)

    # Return named tuple
    geom = (circles = packed_circles, domain = boundary_rectangle)

    return geom
end

function pack(c::AbstractVector{Circle{2,T}}; kwargs...) where {T}
    o, r = tovectors(c)
    o = reinterpret(Vec{2,T}, o) |> copy
    return pack(r; initial_origins = o, kwargs...)
end

function barrier_energy(r::AbstractVector, ϵ::Real)
    # Mutual distance and overlap distance squared functions, gradients, and hessians

    # @info "expbarrier"
    # @inline get_b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); d²(dx,r1,r2,ϵ) + expbarrier(dx,r1,r2,ϵ))
    # @inline get_∇b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); ∇d²(dx,r1,r2,ϵ) + ∇expbarrier(dx,r1,r2,ϵ))
    # @inline get_∇²b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); ∇²d²(dx,r1,r2,ϵ) + ∇²expbarrier(dx,r1,r2,ϵ))

    # @info "softplusbarrier"
    # @inline get_b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); d²(dx,r1,r2,ϵ) + softplusbarrier(dx,r1,r2,ϵ))
    # @inline get_∇b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); ∇d²(dx,r1,r2,ϵ) + ∇softplusbarrier(dx,r1,r2,ϵ))
    # @inline get_∇²b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); ∇²d²(dx,r1,r2,ϵ) + ∇²softplusbarrier(dx,r1,r2,ϵ))

    # @info "log genericbarrier"
    # α = 1e-6 * (ϵ/maximum(r))^2
    # b   = d -> d > 0 ? -α * log(d/ϵ) : typeof(d)(Inf)
    # ∂b  = d -> d > 0 ? -α / d        : typeof(d)(0)#(-Inf)
    # ∂²b = d -> d > 0 ?  α / d^2      : typeof(d)(0)#(Inf)
    
    # @info "exp genericbarrier"
    μ = 2 * maximum(r)
    α = -2 * log(ϵ/μ) / ϵ
    b = d -> μ^2 * exp(-α * (d + ϵ))
    ∂b = d -> -α * μ^2 * exp(-α * (d + ϵ))
    ∂²b = d -> α^2 * μ^2 * exp(-α * (d + ϵ))

    @inline get_b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); d²(dx,r1,r2,ϵ) + genericbarrier(b,dx,r1,r2,ϵ))
    @inline get_∇b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); ∇d²(dx,r1,r2,ϵ) + ∇genericbarrier(∂b,dx,r1,r2,ϵ))
    @inline get_∇²b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); ∇²d²(dx,r1,r2,ϵ) + ∇²genericbarrier(∂b,∂²b,dx,r1,r2,ϵ))

    # Energy function/gradient/hessian
    N = length(r)
    λ = 0.0 * N*(N+1)/2

    function energy(x)
        o, P = @views(x[1:end-2]), Vec{2}((abs(x[end-1]), abs(x[end])))
        F = pairwise_sum(get_b(P), o, r) # mutual distance penalty term
        F += λ * P[1] * P[2] # box area penalty term
    end

    function ∇energy!(g, x)
        o, P = @views(x[1:end-2]), Vec{2}((abs(x[end-1]), abs(x[end])))
        @views pairwise_grad!(g[1:end-2], get_∇b(P), o, r)
        p = Vec{2}((x[end-1], x[end]))
        @views g[end-1:end] .= Tensors.gradient(p -> (P = Vec{2}((abs(p[1]), abs(p[2]))); pairwise_sum(get_b(P), o, r) + λ * P[1] * P[2]), p)
        return g
    end

    # function ∇²energy!(h, x)
    #     o, P = @views(x[1:end-2]), Vec{2}((x[end-1], x[end]))
    #     pairwise_hess!(@views(h[1:end-2, 1:end-2]), get_∇²b(P), o, r)
    #     @views h[end-1:end, end-1:end] .= Tensors.hessian(P -> pairwise_sum(get_b(P), o, r), P)
    #     @views h[1:end-2, end-1:end] .= 0 #TODO this is an incorrect assumption
    #     @views h[end-1:end, 1:end-2] .= 0 #TODO this is an incorrect assumption
    #     return h
    # end
    ∇²energy! = nothing #TODO

    return energy, ∇energy!, ∇²energy!
end

function autodiff_barrier_energy(r::AbstractVector, ϵ::Real)
    @inline get_b(P) = (o1,o2,r1,r2) -> (dx = periodic_diff(o1,o2,P); d²(dx,r1,r2,ϵ) + softplusbarrier(dx,r1,r2,ϵ))
    function energy(x)
        o, P = @views(x[1:end-2]), Vec{2}((x[end-1], x[end]))
        return pairwise_sum(get_b(P), o, r)
    end
end

end # module PeriodicCirclePacking

# # Testing BlackBoxOptim
# function test()
#     function getgeom(x,r)
#         circles = Circle.(Vec{2}.(tuple.(x[1:2:end-2], x[2:2:end-2])), r)
#         domain = Rectangle(zero(Vec{2}), Vec{2}((x[end-1],x[end])))
#         geom = (circles = periodic_circles(circles, domain), domain = domain)
#         return geom
#     end
#     Ncircles = 50;
#     r = rand(radiidistribution(btparams), Ncircles);
#     ϵ = 0.01 * btparams.R_mu;
#     SearchRange = [(0.0, 10.0) for _ in 1:2*length(r)+2];
#     energy = PeriodicCirclePacking.autodiff_barrier_energy(r,ϵ);
#     opt_energy = (x) -> energy(x) + 100 * div(length(x)-2,2)^2 * x[end-1]*x[end];
#     res = bboptimize(opt_energy; SearchRange = SearchRange, NumDimensions = 2*length(r)+2, TraceMode = :silent, MaxTime = 5.0);
#     geom0 = getgeom(best_candidate(res), r);
#     geom = PeriodicCirclePacking.pack(geom0.circles);
#     fig = plot(periodic_circles(geom...)); plot!(fig, geom.domain); display(fig)
#     @show periodic_density(geom...)
# end