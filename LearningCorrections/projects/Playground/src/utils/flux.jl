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

function _test_find_model_param()
    m_aliased = Flux.Dense(2,2)
    m = Flux.Chain(Flux.Chain(Flux.Dense(1,2), m_aliased, m_aliased), Flux.Dense(2,1))
    @assert find_model_param(m, m[1][1].weight) == [1, 1, :weight]
    @assert find_model_param(m, m[1][2].bias) == [1, 2, :bias]
    @assert find_model_param(m, m[2].bias) == [2, :bias]
    @assert find_model_param(m, m[1][2].bias) == find_model_param(m, m[1][3].bias) == [1, 2, :bias] # aliased params returns first found
    @assert find_model_param(m, m[1][2].weight) == find_model_param(m, m[1][3].weight) == [1, 2, :weight] # aliased params returns first found
    @assert find_model_param(Dict(:m => m), m[1][1].weight) == [:m, 1, 1, :weight]
end

function fmap_(f, x, isleaf = Functors.isleaf; cache = IdDict())
    haskey(cache, x) && return cache[x]
    cache[x] = isleaf(x) ? f(x) : begin
        func, re = Flux.functor(x)
        re(map(y -> fmap_(f, y, isleaf; cache), func))
    end
end

function fmap_trainables(f, x, args...; kwargs...)
    θ = IdDict(Flux.params(x) .=> true)
    fmap_trainables_inner(y) = haskey(θ, y) ? f(y) : y
    fmap_(fmap_trainables_inner, x, args...; kwargs...)
end

function allparams!(ps, x, isleaf = Functors.isleaf)
    isleaf(x) && x isa AbstractArray{<:Number} ?
        push!(ps, x) :
        foreach(y -> allparams!(ps, y, isleaf), Flux.functor(x)[1])
    return ps
end
allparams(x, args...) = allparams!(Any[], x, args...)

function accum∇(m, ∇m)
    ps, ∇ps = allparams(m), allparams(∇m)
    θs = IdDict(Flux.params(m) .=> true)
    order = IdDict()
    ∇ = Any[]
    for (i, (p, ∇p)) in enumerate(zip(ps, ∇ps))
        haskey(θs, p) || continue # trainables only
        if !haskey(order, p)
            order[p] = i # record location, in case of weight-sharing
            push!(∇, copy(∇p))
        else
            ∇[order[p]] .+= ∇p # accumulate gradient
        end
    end
    return ∇
end

function restructure(m, ps::AbstractVecOrMat)
    i = 0
    fmap_trainables(m) do x
        if ps isa AbstractVector
            len = length(x)
            x = reshape(ps[i.+(1:len)], size(x))
        else
            basesize = size(x)[1:end-1]
            len = prod(basesize)
            x = reshape(ps[i.+(1:len), ..], basesize..., size(ps,2))
        end
        i += len
        return x
    end
end

Zygote.@adjoint function restructure(m, ps::AbstractVecOrMat)
    function ∇restructure(∇m)
        ∇ps = mapreduce(vcat, accum∇(m, ∇m)) do ∇p
            ps isa AbstractVector ? vec(∇p) : reshape(∇p, :, batchsize(∇p))
        end
        return (nothing, reshape(∇ps, size(ps)))
    end
    return restructure(m, ps), ∇restructure
end

function _hypernet_test()
    nparams(m) = mapreduce(length, +, Flux.params(m); init=0)
    lossfun(m, xs...) = () -> √mean(abs2, m(xs...))
    gradcheck(ℓ, m) = modelgradcheck(ℓ, m; verbose = false, subset = :random, rtol = 1e-3, atol = 1e-4)

    local template, hyper, hypernet
    let
        c1 = x->Flux.normalise(x;dims=1)
        d1 = Flux.Diagonal(2)
        d2 = Flux.Dense(2,3,x->x^2) # smooth activation function for gradient test
        d3 = Flux.Diagonal(3)
        d4 = Flux.Dense(3,2,x->Flux.softplus(abs(x))) # smooth activation function for gradient test
        template = Flux.Chain(c1, d1, Flux.Chain(d2, d3), d3, NotTrainable(d4), d2) # must include shared weights
        hyper = Flux.Chain(Flux.Dense(3,32,Flux.relu), Flux.Dense(32,nparams(template)))
        hypernet = hypernet_from_template(hyper, template)
        @assert hypernet.frame[3][1] === hypernet.frame[6] && hypernet.frame[3][2] === hypernet.frame[4] # test that weight-sharing patterns are preserved
    end

    # vector inputs
    let x = rand(Float32, 2)
        ps = mapreduce(vec, vcat, Flux.params(template))
        m1 = template
        m2 = restructure(hypernet.frame, ps)
        y1 = m1(x)
        y2 = m2(x)
        @assert y1 ≈ y2
        @assert gradcheck(lossfun(m2, x), m2)
    end

    # repeated batched inputs w/ same model per input
    let x = repeat(rand(Float32, 2), 1, 5)
        ps = mapreduce(vec, vcat, Flux.params(template))
        m1 = _x -> reduce(hcat, [template(_x[:,j]) for j in 1:size(_x,2)]) # template(x[:,j]) across columns
        m2 = restructure(hypernet.frame, repeat(ps, 1, batchsize(x)))
        y1 = m1(x)
        y2 = m2(x)
        @assert y1 ≈ y2 && y1[:,1:end-1] ≈ y1[:,2:end]
        @assert gradcheck(lossfun(m2, x), m2)
    end

    # random batched inputs, random models
    let x = rand(Float32, 2, 5), z = rand(Float32, 3, 5)
        ps = hyper(z)
        m1s = map(j -> restructure(template, ps[:,j]), 1:batchsize(x)) # new model per column
        @assert ps ≈ reduce(hcat, [mapreduce(vec, vcat, Flux.params(_m1)) for _m1 in m1s])
        m1 = _x -> reduce(hcat, [_m1(x[:,j]) for (j,_m1) in enumerate(m1s)]) # m1s[j](x[:,j]) across columns
        m2 = restructure(hypernet.frame, ps)
        m3 = hypernet(z)
        y1 = m1(x)
        y2 = m2(x)
        y3 = m3(x)
        @assert y1 ≈ y2 && y2 ≈ y3
        @assert gradcheck(lossfun(m2, x), m2)
        @assert gradcheck(lossfun(x -> hypernet(z)(x), x), hypernet)
    end

    hypernet
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

function update_optimizers!(f!, optimizers::AbstractDict{<:AbstractString,Any}, args...; field::Symbol)
    for (name, opts) in optimizers
        update_optimizers!(f!, opts, name, args...; field)
    end
end
