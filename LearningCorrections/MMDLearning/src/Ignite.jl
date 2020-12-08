module Ignite

using ArgParse
using ..PyTools
import ..TOML
import ..Flux
import ..CUDA

export todevice, to32, to64, @j2p

# Default to running on CPU; init() will override to GPU if available
todevice(x) = Flux.cpu(x)
to32(x) = Flux.fmap(xi -> xi isa AbstractArray ? convert(AbstractArray{Float32}, xi) : xi, todevice(x))
to64(x) = Flux.fmap(xi -> xi isa AbstractArray ? convert(AbstractArray{Float64}, xi) : xi, todevice(x))

const JL_CUDA_FUNCTIONAL = Ref(false)
const JL_CUDA_DEVICE     = Ref(-1)
const JL_ZERO_SUBNORMALS = Ref(true)
const JL_WANDB_LOGGER    = Ref(false)

function init()
    JL_CUDA_FUNCTIONAL[] = get(ENV, "JL_DISABLE_GPU", "0") != "1" && CUDA.functional()
    JL_CUDA_DEVICE[]     = JL_CUDA_FUNCTIONAL[] ? parse(Int, get(ENV, "JL_CUDA_DEVICE", "0")) : -1
    JL_ZERO_SUBNORMALS[] = get(ENV, "JL_ZERO_SUBNORMALS", "1") == "1"
    JL_WANDB_LOGGER[]    = get(ENV, "JL_WANDB_LOGGER", "0") == "1"

    # CUDA settings
    if JL_CUDA_FUNCTIONAL[]
        @eval todevice(x) = Flux.gpu(x)
        CUDA.allowscalar(false)
        CUDA.device!(JL_CUDA_DEVICE[])
    end

    # Treat subnormals as zero
    if JL_ZERO_SUBNORMALS[]
        Threads.@threads for i in 1:Threads.nthreads()
            set_zero_subnormals(true)
        end
    end

    return nothing
end

# Initialize WandBLogger object
function init_wandb_logger(settings)
    WandBLogger = ignite.contrib.handlers.wandb_logger.WandBLogger
    return JL_WANDB_LOGGER[] ? WandBLogger(config = flatten_dict(settings)) : nothing
end

# Convert Julia callback function to Python function.
# Julia functions can already by passed directly via pycall
# but it's via a wrapper type that will error if Python tries
# to inspect the Julia function too closely, e.g. counting
# the number of arguments, etc.
# 
# First argument is assumed to be the `engine` object,
# which is used to terminate training in an error or
# user interrupt occurs.
macro j2p(f)
    local wrapped_f = :(wrap_catch_interrupt($(esc(f))))
    local jlfun2pyfun = esc(:(PyCall.jlfun2pyfun))
    quote
        $jlfun2pyfun($wrapped_f)
    end
end

function wrap_catch_interrupt(f; msg = "")
    function wrap_catch_interrupt_inner(engine, args...)
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

function droplr_file_event(optimizers::AbstractDict{<:AbstractString,Any}; file::AbstractString, lrrate::Int, lrdrop, lrthresh)
    pred(engine) = engine.state.epoch > 1 && mod(engine.state.epoch-1, lrrate) == 0
    file_event(pred; file = file) do engine
        for (name, opt) in optimizers, o in opt
            !hasfield(typeof(o), :eta) && continue
            new_eta = max(o.eta / lrdrop, lrthresh)
            if new_eta > lrthresh
                @info "$(engine.state.epoch): Dropping $name optimizer learning rate to $new_eta"
            else
                @info "$(engine.state.epoch): Learning rate reached minimum value $lrthresh for $name optimizer"
            end
            o.eta = new_eta
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
    if !isnothing(filename)
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
    iter = Pair{<:Union{Nothing, <:AbstractDict}, <:AbstractDict}[nothing => tree]
    oldleafs = 1
    while true
        newleafs = 0
        for i in oldleafs:length(iter)
            parent, leaf = iter[i]
            oldleafs += 1
            for (k,v) in leaf
                if v isa AbstractDict
                    push!(iter, leaf => v)
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
    for (parent, leaf) in reverse(breadth_first_iterator(settings))
        if !isnothing(parent) && haskey(leaf, "INHERIT") && leaf["INHERIT"] == "%PARENT%"
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
    for (parent, leaf) in breadth_first_iterator(settings)
        isnothing(parent) && continue
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
