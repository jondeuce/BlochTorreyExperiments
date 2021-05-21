"""
Convert Julia callback function to Python function.
Julia functions can already by passed directly via pycall
but it's via a wrapper type that will error if Python tries
to inspect the Julia function too closely, e.g. counting
the number of arguments, etc.

First argument is assumed to be the `engine` object,
which is used to terminate training in an error or
user interrupt occurs.
"""
macro j2p(f)
    local wrapped_f = :(terminate_on_error($(esc(f))))
    local jlfun2pyfun = esc(:(PyCall.jlfun2pyfun))
    quote
        $jlfun2pyfun($wrapped_f)
    end
end

function terminate_on_error(f; msg = "")
    function terminate_on_error_inner(engine, args...)
        try
            f(engine, args...)
        catch e
            if e isa InterruptException
                @info "User interrupt"
            else
                !isempty(msg) && @warn msg
                @warn sprint(showerror, e, catch_backtrace())
            end
            engine.terminate()
        end
    end
end

# Initialize WandBLogger object
function init_wandb_logger(settings; activate = nothing, dryrun = true, wandb_dir = DrWatson.projectdir())
    if activate isa Bool
        use_wandb_logger[] = activate
    end
    if use_wandb_logger[]
        # Collect WandB environment variables from settings file (unless already set) and copy to python environment
        WandBEnv = filter(((k,v),) -> startswith(k, "WANDB"), ENV)
        get!(WandBEnv, "WANDB_MODE", ifelse(dryrun, "dryrun", "run"))
        get!(WandBEnv, "WANDB_DIR", wandb_dir)

        DefaultWandBEnv = TOML.parsefile(DrWatson.datadir("default_wandb_env.toml"))
        for (k,v) in DefaultWandBEnv
            get!(WandBEnv, k, v)
        end

        PyEnv = PyDict(pyimport("os").environ)
        merge!(PyEnv, WandBEnv)

        WandBLogger = ignite.contrib.handlers.wandb_logger.WandBLogger
        WandBLogger(config = flatten_dict(settings))
    else
        nothing
    end
end

function on_bad_params_or_gradients(f, models, ps::Zygote.Params, gs::Zygote.Grads; verbose = true)
    # Perform quick initial check: summing all parameters/gradients will return Inf or NaN if and only if any of the elements are Inf or NaN
    pg_sum    = sum([sum(p) for p in ps])
    pg_sum    = sum([sum(g) for g in gs if g !== nothing]; init = pg_sum)
    found_bad = isnan(pg_sum) || isinf(pg_sum)
    !found_bad && return false

    # Evidently, the check haskey(gs, p) is very expensive relative to any(isnan, ...) or any(isinf, ...), hence the above fast check before reaching this loop
    for p in ps
        if length(p) > 0
            found_nan_grad = found_inf_grad = found_nan_param = found_inf_param = false
            (found_nan_param = any(isnan, p)) && verbose && @info "Found NaN parameter: $(find_model_param(models, p))"
            (found_inf_param = any(isinf, p)) && verbose && @info "Found Inf parameter: $(find_model_param(models, p))"
            if haskey(gs, p)
                g = gs[p]
                if g !== nothing
                    (found_nan_grad = any(isnan, g)) && verbose && @info "Found NaN gradient: $(find_model_param(models, p))"
                    (found_inf_grad = any(isinf, g)) && verbose && @info "Found Inf gradient: $(find_model_param(models, p))"
                end
            end
            found_bad |= found_nan_grad || found_inf_grad || found_nan_param || found_inf_param
        end
    end
    found_bad && f()
    return found_bad
end
on_bad_params_or_gradients(f, models, ps::Zygote.Params; kwargs...) = on_bad_params_or_gradients(f, models, ps, Zygote.Grads(IdDict(ps .=> nothing), ps); kwargs...)
on_bad_params_or_gradients(f, models; kwargs...) = on_bad_params_or_gradients(f, models, Flux.params(models); kwargs...)
on_bad_params_or_gradients(f, models::Dict; kwargs...) = on_bad_params_or_gradients(f, models, Flux.params(values(models)...); kwargs...)

# Throttle even to run every `period` seconds
function throttler_event_filter(period = 0.0)
    last_time = Ref(-Inf)
    function throttler_event_filter_internal(engine, event)
        now = time()
        if now - last_time[] >= period
            last_time[] = now
            return true
        else
            return false
        end
    end
end

# Timeout run after `period` seconds
function timeout_event_filter(timeout = Inf)
    start_time = time()
    function timeout_event_filter_internal(engine, event)
        now = time()
        return now - start_time >= timeout
    end
end

# Run arbitrary user input code
function user_input_event(Mod = Main)
    function user_input_event_inner(engine)
        while true
            println("Enter valid Julia code:")
            s = chomp(readline())
            try
                display(Mod.eval(Meta.parse(s)))
            catch e
                @warn sprint(showerror, e, catch_backtrace())
            end
            println("\nContinue? [Y/n]:")
            if lowercase(chomp(readline())) == "n"
                return
            end
        end
    end
end

# Run `f` if `file` is found, or else if predicate `pred(engine)` is true
function file_event(f, pred = (engine) -> false; file::AbstractString)
    function file_event_inner(engine)
        if isfile(file)
            try
                f(engine)
            finally
                mv(file, file * ".fired"; force = true)
            end
        elseif pred(engine)
            f(engine)
        end
    end
end

function terminate_file_event(; file::AbstractString)
    file_event(file = file) do engine
        @info "Exiting: found file $file"
        engine.terminate()
    end
end

function droplr_file_event(optimizers::AbstractDict; file::AbstractString, lrdrop, lrthresh)
    file_event(; file = file) do engine
        update_optimizers!(optimizers; field = :eta) do opt, name
            new_eta = max(opt.eta / lrdrop, lrthresh)
            if new_eta > lrthresh
                @info "$(engine.state.epoch): Dropping $name optimizer learning rate to $new_eta"
            else
                @info "$(engine.state.epoch): Learning rate reached minimum value $lrthresh for $name optimizer"
            end
            opt.eta = new_eta
        end
    end
end
