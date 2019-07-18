"""
    opt_to_cpu(opt)

Move all IdDict fields of optimiser `opt` onto the cpu. Does not remove tracking.
"""
function opt_to_cpu(opt, params)
    newopt = deepcopy(opt)
    for field ∈ fieldnames(typeof(opt))
        if getfield(opt, field) isa IdDict
            state = getfield(opt, field)
            newstate = getfield(newopt, field)
            empty!(newstate)
            for p in params
                newstate[Flux.cpu(deepcopy(p))] = Flux.cpu(deepcopy(state[p]))
            end
        end
    end
    return newopt
end
opt_to_cpu(o::Flux.Optimiser, params) = Flux.Optimiser(map(o -> opt_to_cpu(o, params), o.os)...)

"""
    AdaBound(η = 0.001, β = (0.9, 0.999), γ = 0.001, clip = 0.1)

[AdaBound](https://openreview.net/forum?id=Bkg3g2R9FX) optimiser.
"""
mutable struct AdaBound
    eta   :: Float64
    beta  :: Tuple{Float64, Float64}
    gamma :: Float64
    clip  :: Float64
    state :: IdDict
end
AdaBound(η = 0.001, β = (0.9, 0.999), γ = 0.001, clip = 0.1) =
AdaBound(η, β, γ, clip, IdDict())

function Flux.Optimise.apply!(o::AdaBound, x, Δ)
    ϵ = 1e-8
    η, β, γ, clip = o.eta, o.beta, o.gamma, o.clip
    mt, vt, βp, n = get!(o.state, x, (zero(x), zero(x), β, 1))
    lb = clip * (1 - 1/(γ * n + 1))
    ub = clip * (1 + 1/(γ * n))
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ  = mt * clamp(η / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ), lb, ub)
    o.state[x] = (mt, vt, βp .* β, n+1)
    return Δ
end
