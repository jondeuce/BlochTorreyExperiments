####
#### Functors
####

struct WalkState{V, K, P}
    value::V
    key::K
    parent::P
    depth::Int
    isleaf::Bool
end

function fwalk_(f, x, isleaf = Functors.isleaf)
    root_state = WalkState(x, missing, missing, 0, isleaf(x))
    f(root_state)
    fwalk_(f, root_state, isleaf)
end

function fwalk_(f, d::Dict, isleaf = Functors.isleaf)
    root_state = WalkState(d, missing, missing, 0, false)
    for (k, v) in d
        child_state = WalkState(v, k, root_state, 1, isleaf(v))
        f(child_state)
        fwalk_(f, child_state, isleaf)
    end
end

function fwalk_(f, parent_state::WalkState, isleaf)
    func, _ = Flux.functor(parent_state.value)
    foreach(pairs(func)) do (child_key, child_value)
        child_state = WalkState(child_value, child_key, parent_state, parent_state.depth + 1, isleaf(child_value))
        f(child_state)
        fwalk_(f, child_state, isleaf)
    end
    nothing
end

function find_model_param(m, x)
    keychain = nothing
    found = false
    fwalk_(m) do state
        found && return
        if x === state.value && state.isleaf # ensure leaf value, since we are looking for leaf parameters
            found = true
            keychain = Any[]
            while state.depth != 0
                pushfirst!(keychain, state.key)
                state = state.parent
            end
        end
    end
    return keychain
end

####
#### Optimisers
####

walk(o::Flux.Optimise.Optimiser) = mapfoldl(walk, vcat, o; init = Any[])
walk(o::AbstractArray) = walk(Flux.Optimise.Optimiser(convert(Vector{Any}, vec(o))))
walk(o) = Any[o]

function update_optimizers!(f!, opts, args...; field::Symbol)
    for opt in walk(opts)
        if hasfield(typeof(opt), field)
            f!(opt, args...) # update optimizer
        else
            continue
        end
    end
end

function update_optimizers!(f!, optimizers::AbstractDict, args...; field::Symbol)
    for (name, opts) in optimizers
        update_optimizers!(f!, opts, name, args...; field)
    end
end
