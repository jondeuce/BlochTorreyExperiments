module Ignite

using ArgParse
using ..PyTools
import ..TOML
import ..Flux
import ..JL_WANDB_LOGGER

# Initialize WandBLogger object
function init_wandb_logger(settings)
    WandBLogger = ignite.contrib.handlers.wandb_logger.WandBLogger
    return JL_WANDB_LOGGER[] ? WandBLogger(config = flatten_dict(settings)) : nothing
end

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
            ret = nothing
            try
                ret = Mod.eval(Meta.parse(s))
            catch e
                @warn sprint(showerror, e, catch_backtrace())
            end
            println("Continue? [Y/n]:")
            if lowercase(chomp(readline())) == "n"
                return ret
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

function droplr_file_event(optimizers::AbstractDict{<:AbstractString,Any}; file::AbstractString, lrrate::Int, lrdrop, lrthresh)
    pred(engine) = engine.state.epoch > 1 && mod(engine.state.epoch-1, lrrate) == 0
    file_event(pred; file = file) do engine
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

# Flatten settings dictionary
flatten_dict(d::AbstractDict{<:AbstractString, Any}, prefix = "", delim = ".") = _recurse_insert!(Dict{String,Any}(), d, prefix, delim)
function _recurse_insert!(dout::AbstractDict{<:AbstractString, Any}, d::AbstractDict{<:AbstractString, Any}, prefix = "", delim = ".")
    maybeprefix(k) = isempty(prefix) ? k : prefix * delim * k
    for (k, v) in d
        if v isa AbstractDict{<:AbstractString, Any}
            _recurse_insert!(dout, v, maybeprefix(k), delim)
        else
            dout[maybeprefix(k)] = deepcopy(v)
        end
    end
    return dout
end

# Set `d[k]` to `new` if its current value is `default`, else do nothing
function compare_and_set!(d::AbstractDict, k, default, new)
    if isequal(d[k], default)
        d[k] = deepcopy(new)
    end
    return d[k]
end

# Nested dict access
nestedaccess(d::AbstractDict, args...) = isempty(args) ? d : nestedaccess(d[first(args)], Base.tail(args)...)
nestedaccess(d) = d

# Save and print settings file
function save_and_print(settings::AbstractDict; filename = nothing)
    if (filename !== nothing)
        @assert endswith(filename, ".toml")
        mkpath(dirname(filename))
        open(filename; write = true) do io
            TOML.print(io, settings)
        end
    end
    TOML.print(stdout, settings)
    return settings
end

function breadth_first_iterator(tree::AbstractDict)
    iter = Pair{<:Union{Nothing, <:AbstractDict}, <:Pair{<:Union{Nothing, <:AbstractString}, <:AbstractDict}}[nothing => (nothing => tree)]
    oldleafs = 1
    while true
        newleafs = 0
        for i in oldleafs:length(iter)
            parent, (_, leaf) = iter[i]
            oldleafs += 1
            for (k,v) in leaf
                if v isa AbstractDict
                    push!(iter, leaf => (k => v))
                    newleafs += 1
                end
            end
        end
        newleafs == 0 && break
    end
    return iter
end

# Settings parsing
function parse_command_line!(
        settings::AbstractDict{<:AbstractString, Any},
        args = isinteractive() ? String[] : ARGS,
    )

    # Fields "INHERIT" with value "%PARENT%" specify that all fields from (and only from) the immediate parent
    # should be copied into the child, unless that key is already present in the child
    for (parent, (key, leaf)) in reverse(breadth_first_iterator(settings))
        if (parent !== nothing) && get(leaf, "INHERIT", "") == "%PARENT%"
            for (k,v) in parent
                (v isa AbstractDict) && continue
                !haskey(leaf, k) && (leaf[k] = deepcopy(parent[k]))
            end
            delete!(leaf, "INHERIT")
        else
            continue
        end
    end

    # Fields with value "%PARENT%" take default values from the corresponding field of their parent
    for (parent, (key, leaf)) in breadth_first_iterator(settings)
        (parent === nothing) && continue
        for (k,v) in leaf
            (v == "%PARENT%") && (leaf[k] = deepcopy(parent[k]))
        end
    end

    # Generate arg parser
    function populate_arg_table!(parser, leaf_settings, root_settings = leaf_settings)
        for (k,v) in leaf_settings
            if v isa AbstractDict
                populate_arg_table!(parser, Dict{String,Any}(k * "." * kin => deepcopy(vin) for (kin, vin) in v), root_settings)
            else
                props = Dict{Symbol,Any}(:default => deepcopy(v))
                if v isa AbstractVector
                    props[:arg_type] = eltype(v)
                    props[:nargs] = length(v)
                else
                    props[:arg_type] = typeof(v)
                end
                add_arg_table!(parser, "--" * k, props)
            end
        end
        return parser
    end
    parser = populate_arg_table!(ArgParseSettings(), settings, settings)

    # Parse and merge into settings
    for (k,v) in parse_args(args, parser)
        ksplit = String.(split(k, "."))
        din = nestedaccess(settings, ksplit[begin:end-1]...)
        din[ksplit[end]] = deepcopy(v)
    end

    return settings
end

end # module
