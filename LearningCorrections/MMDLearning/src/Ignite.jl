module Ignite

using ArgParse
import ..TOML
import ..Flux
import ..CUDA

export to32, to64, todevice, @j2p, event_throttler, run_timeout

const JL_CUDA_FUNCTIONAL = Ref(get(ENV, "JL_DISABLE_GPU", "0") != "1" && CUDA.functional())
const JL_CUDA_DEVICE = Ref(parse(Int, get(ENV, "JL_CUDA_DEVICE", "0")))
const JL_ZERO_SUBNORMALS = Ref(get(ENV, "JL_ZERO_SUBNORMALS", "1") == "1")
const JL_WANDB_LOGGER = Ref(get(ENV, "JL_WANDB_LOGGER", "0") == "1")

# Send array to CPU or GPU
if JL_CUDA_FUNCTIONAL[]
    @eval todevice(x) = Flux.gpu(x)
else
    @eval todevice(x) = Flux.cpu(x)
end
to32(x) = x |> Flux.f32 |> todevice
to64(x) = x |> Flux.f64 |> todevice

# Initialize WandBLogger object
function init_wandb_logger(settings)
    WandBLogger = Main.ignite.contrib.handlers.wandb_logger.WandBLogger
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

# Convert row major Torch array to column major Julia array
reversedims(x::AbstractArray{T,N}) where {T,N} = permutedims(x, ntuple(i -> N-i+1, N))
array(x) = reversedims(x.detach().cpu().numpy())
array(x::AbstractArray) = x # fallback for Julia arrays

# Throttle even to run every `period` seconds
function event_throttler(period = 0.0)
    last_time = Ref(-Inf)
    function event_throttler_internal(engine, event)
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
function run_timeout(timeout = Inf)
    start_time = time()
    function run_timeout_internal(engine, event)
        now = time()
        return now - start_time >= timeout
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

# Nested dict access
nestedaccess(d::AbstractDict, args...) = isempty(args) ? d : nestedaccess(d[first(args)], Base.tail(args)...)
nestedaccess(d) = d

# Settings parsing
function parse_command_line!(
        defaults::AbstractDict{<:AbstractString, Any},
    )

    # Generate arg parser
    settings = ArgParseSettings()

    function _add_arg_table!(d, dglobal = d)
        for (k,v) in d
            if v isa AbstractDict
                _add_arg_table!(Dict{String,Any}(k * "." * kin => deepcopy(vin) for (kin, vin) in v), dglobal)
            else
                # Fields with value "%PARENT%" take default values of their parent field
                ksplit = String.(split(k, "."))
                depth = 0
                while v == "%PARENT%"
                    depth += 1
                    v = deepcopy(nestedaccess(dglobal, ksplit[begin:end-1-depth]..., ksplit[end]))
                end
                props = Dict{Symbol,Any}(:default => deepcopy(v))
                if v isa AbstractVector
                    props[:arg_type] = eltype(v)
                    props[:nargs] = length(v)
                else
                    props[:arg_type] = typeof(v)
                end
                add_arg_table!(settings, "--" * k, props)
            end
        end
    end
    _add_arg_table!(defaults)

    # Parse and merge into defaults
    args = isinteractive() ? String[] : ARGS
    for (k, v) in parse_args(args, settings)
        ksplit = String.(split(k, "."))
        din = nestedaccess(defaults, ksplit[begin:end-1]...)
        din[ksplit[end]] = deepcopy(v)
    end

    return defaults
end

# Save and print settings file
function save_and_print(settings::AbstractDict; outpath, filename)
    @assert endswith(filename, ".toml")
    !isdir(outpath) && mkpath(outpath)
    open(joinpath(outpath, filename); write = true) do io
        TOML.print(io, settings)
    end
    TOML.print(stdout, settings)
    return settings
end

# Set `d[k]` to `new` if its current value is `default`, else do nothing
function compare_and_set!(d::AbstractDict, k, default, new)
    if isequal(d[k], default)
        d[k] = deepcopy(new)
    end
    return d[k]
end

function __init__()
    # CUDA settings
    if JL_CUDA_FUNCTIONAL[]
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

end # module
