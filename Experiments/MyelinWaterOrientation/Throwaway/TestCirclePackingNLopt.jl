using NLopt
using Tensors

import DiffResults, ForwardDiff
using DiffResults: DiffResult, JacobianResult, GradientResult, value
using ForwardDiff: Chunk, GradientConfig, JacobianConfig, jacobian!, gradient!

using GeometryUtils

function obj(x::Vector, r::Vector)
    T = eltype(x)
    N = div(length(x), 2) # number of circles
    os = reinterpret(Vec{2,T}, x)

    out = zero(T)
    @inbounds for i in 1:N-1
        oi, ri = os[i], r[i]
        @inbounds for j in i+1:N
            oj, rj = os[j], r[j]
            out += signed_edge_distance(oi, ri, oj, rj)^2
        end
    end

    # return out + x[1]^2 + x[2]^2 + x[3]^2
    return out
end

function con!(c::Vector, x::Vector, r::Vector, t = 1e-3)
    T = eltype(x)
    N = div(length(x), 2) # number of circles
    os = reinterpret(Vec{2,T}, x)

    idx = 0 # constraint count
    @inbounds for i in 1:N-1
        oi, ri = os[i], r[i]
        @inbounds for j in i+1:N
            oj, rj = os[j], r[j]
            idx += 1
            c[idx] = t - signed_edge_distance(oi, ri, oj, rj)
        end
    end

    return c
end

# Initialize
T = Float64
Nx, Ny = 10, 10
N = Nx*Ny
Ndof = 2N
Ncon = N*(N-1)÷2

# Initial circles
r = ones(T, N) # radii
t = 1e-2 # minimum allowed distance
xp, yp = 2.5*maximum(r) .* (0:Nx-1), 2.5*maximum(r) .* (0:Ny-1)
x0 = collect(reinterpret(T, collect(Iterators.product(xp, yp))[:]))
# x0 = 100N .* (2.0 .* rand(T, Ndof) .- 1.0) # initial origins
c0 = zeros(T, Ncon)
jac = zeros(T, Ncon, Ndof)
grad = zeros(T, Ndof)

# Wrap functions
chnk = Chunk(x0)
confun! = (c,x) -> con!(c,x,r,t)
objfun = (x) -> obj(x,r)
jcfg = JacobianConfig(confun!, c0, x0, chnk)
gcfg = GradientConfig(objfun, x0, chnk)

# Objective in place
function objfun!(x::Vector, grad::Vector)
    # return length(grad) > 0 ? gradient!(grad, objfun, x, gcfg) : objfun(x)
    # return length(grad) > 0 ? gradient!(grad, objfun, x) : objfun(x)
    if length(grad) > 0
        res = DiffResult(zero(eltype(x)), grad)
        gradient!(res, objfun, x, gcfg)
        # gradient!(res, objfun, x)
        return value(res)
    else
        return objfun(x)
    end
end
objfun!(x0, grad)

# Contraint initialization
function ∇confun!(c::Vector, x::Vector, jac::Matrix)
    # return length(jac) > 0 ? jacobian!(jac, confun!, c, x) : confun!(c, x)
    # return length(jac) > 0 ? jacobian!(jac, confun!, c, x, jcfg) : confun!(c, x)
    if length(jac) > 0
        res = DiffResult(c, jac)
        jacobian!(res, confun!, c, x, jcfg)
        # jacobian!(res, confun!, c, x)
        return value(res)
    else
        confun!(c, x)
        return c
    end
end
∇confun!(c0, x0, jac)

# Optimizaiton initialization
opt = Opt(:LD_MMA, Ndof)
ATOL = 1e-2 * t
xtol_abs!(opt, ATOL)
inequality_constraint!(opt, ∇confun!, ATOL*ones(Ncon))
min_objective!(opt, objfun!)

(minf, minx, ret) = optimize(opt, x0)
println("optfun(minx) = $minf < $(objfun(x0)) after $(opt.numevals) iterations (returned $ret)")

copt = confun!(similar(c0), minx)
@show minimum(t .- copt)

function minimum_distances(cs::Vector{C}) where {C<:Circle{dim,T}} where {dim,T}
    out = T[]
    for i in 1:length(cs)
        min_dist = T(Inf)
        for j in Iterators.flatten((1:i-1, i+1:length(cs)))
            min_dist = min(min_dist, signed_edge_distance(cs[i], cs[j]))
        end
        @show min_dist
        push!(out, min_dist)
    end
    out
end

cs = [Circle{2,Float64}(Vec{2,Float64}((minx[2i-1],minx[2i])), r[i]) for i in 1:length(r)]
@show minimum_signed_edge_distance(cs)
minimum_distances(cs)

# ---------------------------------------------------------------------------- #
# Tutorial
# ---------------------------------------------------------------------------- #

using NLopt

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end
    return sqrt(x[2])
end

function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_MMA, 2)
lower_bounds!(opt, [-Inf, 0.])
xtol_rel!(opt,1e-4)

min_objective!(opt, myfunc)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

x0 = [1.234, 5.678]
(minf,minx,ret) = optimize(opt, x0)
nevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $nevals iterations (returned $ret)")
