module Ignite

export @j2p, array, event_throttler, run_timeout

# Convert Julia function to Python function.
# Julia functions can already by passed directly via pycall
# but it's via a wrapper type that will error if Python tries
# to inspect the Julia function too closely, e.g. counting
# the number of arguments, etc.
macro j2p(f)
    local PyCall = __module__.PyCall
    :($PyCall.jlfun2pyfun($(esc(f))))
end

# Convert row major Torch array to column major Julia array
reversedims(x::AbstractArray{T,N}) where {T,N} = permutedims(x, ntuple(i -> N-i+1, N))
array(x) = reversedims(x.detach().cpu().numpy())

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

end # module
