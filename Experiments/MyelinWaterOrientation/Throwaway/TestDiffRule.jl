using ForwardDiff: Dual, value, partials
using DiffRules: @define_diffrule
using Base.Test
using BenchmarkTools

# module Fmod
#     export f, df
#     @inline f(x) = sin(cos(x))
#     @inline df(x) = -cos(cos(x))*sin(x)
# end
# g(x) = sin(cos(x))
# @define_diffrule Fmod.f(x) = :(Fmod.df($x))
# # @inline f(x::Dual{T}) where {T} = Dual{T}(f(value(x)), df(x)*partials(x))

module Fmod
    # export f, df_dx, df_dy
    @inline g(x,y) = (x*sqrt(1-x^2) + asin(x))/2 - x*y
    @inline f(x,y) = (x*sqrt(1-x^2) + asin(x))/2 - x*y
    @inline df_dx(x,y) = sqrt(1-x^2) - y
    @inline df_dy(x,y) = -x
end
@define_diffrule Fmod.f(x,y) = :(Fmod.df_dx($x, $y)), :(Fmod.df_dy($x, $y))


# ---------------------------------------------------------------------------- #
# Derivative test

function derivative_test()
    g = x -> Fmod.g(clamp(x, -1/√2, 1/√2), 1/√2)
    f = x -> Fmod.f(clamp(x, -1/√2, 1/√2), 1/√2)

    ∇g = (x) -> ForwardDiff.derivative(g, x)
    ∇f = (x) -> ForwardDiff.derivative(f, x)

    x0 = rand()
    @test ∇g(x0) ≈ ∇f(x0)
end

# ---------------------------------------------------------------------------- #
# Gradient test

function gradient_test()
    g(x) = Fmod.g(clamp(x, -1/√2, 1/√2), 1/√2)
    f(x) = Fmod.f(clamp(x, -1/√2, 1/√2), 1/√2)
    F(x) = sum(abs2, f.(x))
    G(x) = sum(abs2, g.(x))

    x0 = rand(100)
    Fconfig = ForwardDiff.GradientConfig(F, x0, ForwardDiff.Chunk{10}());
    Gconfig = ForwardDiff.GradientConfig(G, x0, ForwardDiff.Chunk{10}());

    ∇G = (x) -> ForwardDiff.gradient(G, x, Gconfig)
    ∇F = (x) -> ForwardDiff.gradient(F, x, Fconfig)

    @test ∇G(x0) ≈ ∇F(x0)
    display(@benchmark $∇G($x0))
    display(@benchmark $∇F($x0))
end

derivative_test()
gradient_test()


# ---------------------------------------------------------------------------- #
# circle_packing gradient test

using StaticArrays
using JuAFEM
using JuAFEM: vertices, faces, edges
using MATLAB
using LinearMaps
using Optim
using Cuba
using Distributions
using Calculus

function circle_packing_test(N::Int = 1000)

    const dim = 2
    const T = Float64

    c = rand(Circle{2,Float64})
    x0 = [-rand(), -rand(), rand(), rand()]
    f = x -> intersect_area(c, x[1], x[2], x[3], x[4])

    # Shared buffer
    const buffer = similar(x0)

    # ForwardDiff
    Fwd_g!, Fwd_fg! = wrap_gradient(f, x0; isforward = true)
    Fwd_g = x -> Fwd_g!(buffer, x)

    # ReverseDiff
    Rev_g!, Rev_fg! = wrap_gradient(f, x0; isforward = false, isdynamic = true)
    Rev_g = x -> Rev_g!(buffer, x)

    # ReverseDiff gradient (pre-recorded config; slower, but dynamic)
    const cfg = ReverseDiff.GradientConfig(x0)
    Rev_g = x -> ReverseDiff.gradient!(buffer, f, x, cfg)

    @show (Fwd_g(x0), Rev_g(x0))

    iter = 0
    while iter < N
        x0 = [-rand(), -rand(), rand(), rand()]
        df = Calculus.gradient(f, x0)
        df_fwd = Fwd_g(x0)
        df_rev = Rev_g(x0)

        @assert isapprox(df_fwd, df_rev; atol=1e-12) @show (Fwd_g(x0), Rev_g(x0))

        if ~isapprox(df, df_fwd; rtol = 1e-6, atol=1e-6)
            @show @. log10(max(abs(df - df_fwd),1e-6))
        end

        iter +=1
    end
    @show iter

end

circle_packing_test(100_000)
@enter circle_packing_test()

nothing
