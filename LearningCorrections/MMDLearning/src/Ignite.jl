module Ignite

using ArgParse
using ..TOML

export @j2p, event_throttler, run_timeout

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
    local PyCall = __module__.PyCall
    local wrapped_f = :(wrap_catch_interrupt($(esc(f))))
    :($PyCall.jlfun2pyfun($wrapped_f))
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

# Settings parsing
function parse_command_line!(
        defaults::Dict{<:AbstractString, Any},
        args = isinteractive() ? String[] : ARGS,
    )

    # Generate arg parser
    settings = ArgParseSettings()

    function _add_arg_table!(d)
        for (k,v) in d
            if v isa Dict
                _add_arg_table!(Dict{String,Any}(k * "." * kin => deepcopy(vin) for (kin, vin) in v))
            else
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

    # Parse and merge into defaults
    _add_arg_table!(defaults)
    user_settings = parse_args(args, settings)

    for (k, v) in user_settings
        ksplit = String.(split(k, "."))
        din = defaults[ksplit[1]]
        for kin in ksplit[2:end-1]
            din = din[kin]
        end
        din[ksplit[end]] = deepcopy(v)
    end

    return defaults
end

# Save and print settings file
function save_and_print(settings::Dict; outpath, filename)
    @assert endswith(filename, ".toml")
    !isdir(outpath) && mkpath(outpath)
    open(joinpath(outpath, filename); write = true) do io
        TOML.print(io, settings)
    end
    TOML.print(stdout, settings)
    return settings
end

# Set `d[k]` to `new` if its current value is `default`, else do nothing
function compare_and_set!(d::Dict, k, default, new)
    if isequal(d[k], default)
        d[k] = deepcopy(new)
    end
    return d[k]
end

end # module
