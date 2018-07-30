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

    x₀ = rand()
    @test ∇g(x₀) ≈ ∇f(x₀)
end

# ---------------------------------------------------------------------------- #
# Gradient test

function gradient_test()
    g(x) = Fmod.g(clamp(x, -1/√2, 1/√2), 1/√2)
    f(x) = Fmod.f(clamp(x, -1/√2, 1/√2), 1/√2)
    F(x) = sum(abs2, f.(x))
    G(x) = sum(abs2, g.(x))

    x₀ = rand(100)
    Fconfig = ForwardDiff.GradientConfig(F, x₀, ForwardDiff.Chunk{10}());
    Gconfig = ForwardDiff.GradientConfig(G, x₀, ForwardDiff.Chunk{10}());

    ∇G = (x) -> ForwardDiff.gradient(G, x, Gconfig)
    ∇F = (x) -> ForwardDiff.gradient(F, x, Fconfig)

    @test ∇G(x₀) ≈ ∇F(x₀)
    display(@benchmark $∇G($x₀))
    display(@benchmark $∇F($x₀))
end

derivative_test()
gradient_test()

nothing
