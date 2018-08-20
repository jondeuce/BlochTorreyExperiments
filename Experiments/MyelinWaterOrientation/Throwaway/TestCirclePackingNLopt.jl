using NLopt
using Tensors
using DiffResults: DiffResult, JacobianResult, GradientResult, value
using ForwardDiff
using ForwardDiff: Chunk, GradientConfig, JacobianConfig, jacobian!, gradient!

include("../Geometry/geometry_utils.jl")
using .GeometryUtils

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
N = 10
Ndof = 2N
Ncon = N*(N-1)÷2

# Initial circles
r = ones(T, N) # radii
t = 1e-3 # min. distance
x0 = 100N .* (2.0 .* rand(T, Ndof) .- 1.0) # initial origins
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
        # gradient!(res, objfun, x, gcfg)
        gradient!(res, objfun, x)
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
        # jacobian!(res, confun!, c, x, jcfg)
        jacobian!(res, confun!, c, x)
        return value(res)
    else
        confun!(c, x)
        return c
    end
end
∇confun!(c0, x0, jac)

# Optimizaiton initialization
opt = Opt(:LD_MMA, Ndof)
xtol_rel!(opt, 1e-8)
min_objective!(opt, objfun!)
inequality_constraint!(opt, ∇confun!, 1e-4*ones(Ncon))

(minf, minx, ret) = optimize(opt, x0)
nevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $nevals iterations (returned $ret)")

copt = confun!(similar(c0), minx)
@show minimum(t .- copt)

circles = [Circle{2,Float64}(Vec{2,Float64}((minx[2i-1],minx[2i])), r[i]) for i in 1:length(r)]
@show minimum_signed_edge_distance(circles)

function minimum_distances(circles::Vector{C}) where {C<:Circle{dim,T}} where {dim,T}
    out = []
    for i in 1:length(circles)
        push!(out, minimum(j->signed_edge_distance(circles[i],circles[j]), Iterators.flatten((1:i-1,i+1:length(circles)))))
    end
    out
end

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
