using Flux, Zygote, Optim, FluxOptTools, Statistics
m      = Chain(Dense(1,3,tanh) , Dense(3,1))
x      = LinRange(-pi,pi,100)'
y      = sin.(x)
loss() = mean(abs2, m(x) .- y)
Zygote.refresh()
# pars   = Flux.params(m)
# lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
# res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))

struct OptimState1{M,O,S}
    method::M
    options::O
    state::S
end
function OptimState1(
        loss,
        ps0::Flux.Params,
        method::Optim.LBFGS = LBFGS(),
        options = Optim.Options(;Optim.default_options(method)..., iterations = 1000, store_trace = true)
    )

    lossfun, gradfun, fg!, initial_x = optfuns(loss, ps0)
    initial_x = copy(initial_x)
    d = Optim.OnceDifferentiable(Optim.only_fg!(fg!), initial_x)
    state = Optim.initial_state(method, options, d, initial_x)

    return OptimState1{typeof(method), typeof(options), typeof(state)}(method, options, state)
end

const OptimState = OptimState1

function _update!(opt::OptimState, loss, ps)
    Zygote.refresh()
    lossfun, gradfun, fg!, p0 = optfuns(loss, ps)
    # res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))
    res = Optim.optimize(
        Optim.OnceDifferentiable(Optim.only_fg!(fg!), p0), #TODO p0[1]
        p0,
        opt.method,
        opt.options,
        opt.state)
    copyto!(ps, Optim.minimizer(res))
    return res #TODO
end

function _train!(loss, ps, data, opt::OptimState; cb = () -> ())
    ps = Flux.Params(ps)
    cb = Flux.Optimise.runall(cb)
    Flux.@progress for d in data
        try
            return _update!(opt, () -> loss(d...), ps)
        catch ex
            if ex isa Flux.Optimise.StopException
                break
            else
                rethrow(ex)
            end
        end
    end
end
