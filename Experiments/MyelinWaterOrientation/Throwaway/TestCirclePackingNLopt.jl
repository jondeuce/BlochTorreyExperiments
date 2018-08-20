using NLopt
using Tensors
using DiffResults: JacobianResult, GradientResult, value
using ForwardDiff
using ForwardDiff: Chunk, GradientConfig, JacobianConfig, jacobian!, gradient!

include("../Geometry/geometry_utils.jl")
using .GeometryUtils

function obj(x::AbstractVector)
    T = eltype(x)
    N = div(length(x), 3) # number of circles
    cs = reinterpret(Circle{2,T}, x)

    out = zero(T)
    @inbounds for i in 1:N-1
        ci = cs[i]
        @inbounds for j in i+1:N
            cj = cs[j]
            out += signed_edge_distance(ci,cj)^2
        end
    end

    # return out + x[1]^2 + x[2]^2 + x[4]^2
    return out
end

function c!(result::Vector, x::Vector, t = 1e-3)
    T = eltype(x)

    # N = div(length(x)-1, 3) # number of circles
    # t = x[end] # distance threshold

    N = div(length(x), 3) # number of circles
    cs = reinterpret(Circle{2,T}, x)

    idx = 0 # constraint count
    @inbounds for i in 1:N-1
        # ci = Circle(Vec{2,T}((x[3i-2], x[3i-1])), x[3i])
        ci = cs[i]
        @inbounds for j in i+1:N
            idx += 1
            # cj = Circle(Vec{2,T}((x[3j-2], x[3j-1])), x[3j])
            cj = cs[j]
            result[idx] = t - signed_edge_distance(ci,cj)
        end
    end

    return result
end

# Initialize
T = Float64
N = 3
Ndof = 3N # + 1
Ncon = N*(N-1)÷2

# Initial circles
x0 = 1.0 .+ rand(T, Ndof)
y0 = zeros(T, Ncon)
jac = zeros(T, Ncon, Ndof)
grad = zeros(T, Ndof)

chnk = Chunk(x0)
jcfg = JacobianConfig(c!, y0, x0, chnk)
gcfg = GradientConfig(obj, x0, chnk)

# Objective in place
function obj!(x::Vector, grad::Vector)
    # return length(grad) > 0 ? gradient!(grad, obj, x, gcfg) : obj(x)
    # return length(grad) > 0 ? gradient!(grad, obj, x) : obj(x)
    if length(grad) > 0
        res = GradientResult(grad)
        # gradient!(res, obj, x, gcfg)
        gradient!(res, obj, x)
        return value(res)
    else
        return obj(x)
    end
end
obj!(x0, grad)

# Contraint initialization
function ∇c!(y::Vector, x::Vector, jac::Matrix)
    # return length(jac) > 0 ? jacobian!(jac, c!, y, x) : c!(y, x)
    # return length(jac) > 0 ? jacobian!(jac, c!, y, x, jcfg) : c!(y, x)
    if length(jac) > 0
        res = JacobianResult(y, x)
        # jacobian!(res, c!, y, x, jcfg)
        jacobian!(res, c!, y, x)
        return value(res)
    else
        c!(y, x)
        return y
    end
end
∇c!(y0, x0, jac)

# Optimizaiton initialization
opt = Opt(:LD_MMA, Ndof)
# xtol_rel!(opt, 1e-4)

min_objective!(opt, obj!)
# inequality_constraint!(opt, ∇c!, 1e-4*ones(Ncon))

(minf, minx, ret) = optimize(opt, x0)
nevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $nevals iterations (returned $ret)")

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
