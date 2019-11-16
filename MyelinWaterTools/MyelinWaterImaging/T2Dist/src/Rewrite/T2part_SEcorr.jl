"""
    T2partOptions structure for T2map_SEcorr
"""
@with_kw struct T2partOptions{T} @deftype T
    T2Range::NTuple{2,T} = (0.015, 2.0)
    @assert 0.001 <= T2Range[1] < T2Range[2] <= 10.0

    nT2::Int = 40
    @assert nT2 >= 1
    
    spwin::NTuple{2,T} = (0.015, 0.040)
    @assert spwin[1] < spwin[2]
    
    mpwin::NTuple{2,T} = (0.040, 0.200)
    @assert mpwin[1] < mpwin[2]
    
    Sigmoid::Union{T,Nothing} = nothing
    @assert Sigmoid isa Nothing || Sigmoid > 0
end
T2partOptions(args...; kwargs...) = T2partOptions{Float64}(args...; kwargs...)

"""
maps = T2part_SEcorr(T2distributions; kwargs...)

Description:
  Analyzes T2 distributions produced by T2map_SEcorr to produce data maps
  of a series of parameters.

Inputs:
  T2distributions: 4-D array with data as (row,column,slice,T2 Amplitude)
  ...: A series of optional Property/Value pairs to modify settings.
    Defaults are given in brackets:
      "nT2":     Number of T2 values in distribution (size(T2distributions, 4))
      "T2Range": Min and Max T2 values of distribution ([0.015,2.000])
      "spwin":   Min and Max of the short peak window ([0.015,0.040])
      "mpwin":   Min and Max of the middle peak window ([0.040,0.200])
      "Sigmoid": Apply sigmoidal weighting to the upper limit of the 
                 short peak window. Value is the delta-T2 parameter 
                 (distance in seconds on either side of the spwin upper 
                 limit where sigmoid curve reaches 10% and 90%). (Default
                 is no sigmoid weighting)

Ouputs:
  maps: a named tuple containing the following 3D data maps as fields:
      -sfr (small pool (myelin water) fraction)
      -sgm (small pool (myelin water) geometric mean)
      -mfr (medium pool (intra/extra water) fraction)
      -mgm (medium pool (intra/extra water) geometric mean)

External Calls:
  none

Created by Thomas Prasloski
email: tprasloski@gmail.com
Ver. 1.2, August 2012
"""
function T2part_SEcorr(T2distributions::Array{T,4}; kwargs...) where {T}
    reset_timer!(TIMER)
    out = @timeit_debug TIMER "T2part_SEcorr" begin
        _T2part_SEcorr(T2distributions, T2partOptions{T}(;nT2 = size(T2distributions, 4), kwargs...))
    end
    if timeit_debug_enabled()
        println("\n"); show(TIMER); println("\n")
    end
    return out
end

function _T2part_SEcorr(T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T}

    nrows, ncols, nslices, nT2 = size(T2distributions)
    maps = Dict(
        "sfr" => fill(T(NaN), nrows, ncols, nslices),
        "sgm" => fill(T(NaN), nrows, ncols, nslices),
        "mfr" => fill(T(NaN), nrows, ncols, nslices),
        "mgm" => fill(T(NaN), nrows, ncols, nslices),
    )
    thread_buffers = [thread_buffer_maker(opts) for _ in 1:Threads.nthreads()]

    Threads.@threads for row in 1:nrows
        thread_buffer = thread_buffers[Threads.threadid()]
        @inbounds for col in 1:ncols, slice in 1:nslices
            for i in 1:opts.nT2
                thread_buffer.dist[i] = T2distributions[row,col,slice,i]
            end
            calc_maps!(thread_buffer, maps, opts, row, col, slice)
        end
    end

    return maps
end

# =========================================================
# Make thread-local buffers
# =========================================================
function thread_buffer_maker(o::T2partOptions{T}) where {T}
    dist = zeros(T, o.nT2)
    T2_times = logrange(o.T2Range..., o.nT2)
    sp = (findfirst(t -> t >= o.spwin[1], T2_times), findlast(t -> t <= o.spwin[2], T2_times))
    mp = (findfirst(t -> t >= o.mpwin[1], T2_times), findlast(t -> t <= o.mpwin[2], T2_times))
    logT2_times = log.(T2_times)
    logT2_times_sp = logT2_times[sp[1]:sp[2]]
    logT2_times_mp = logT2_times[mp[1]:mp[2]]
    weights = make_weights(o)
    return @ntuple(dist, T2_times, sp, mp, logT2_times_sp, logT2_times_mp, weights)
end

function make_weights(o::T2partOptions{T}) where {T}
    if !(o.Sigmoid === nothing)
        # Curve reaches 50% at T2_50perc and is (k and 1-k)*100 percent at T2_50perc +/- T2_kperc  
        k = T(0.1)
        T2_kperc = o.Sigmoid
        T2_50perc = o.spwin[2]
        sigma = abs(T2_kperc / (sqrt(T(2)) * erfinv(2*k-1)))
        1 .- normcdf.(o.T2_times, T2_50perc, sigma)
    else
        nothing
    end
end

# =========================================================
# Save thread local results to output maps
# =========================================================
function calc_maps!(thread_buffer, maps, o::T2partOptions, i...)
    @unpack dist, T2_times, sp, mp, logT2_times_sp, logT2_times_mp, weights = thread_buffer
    @unpack sfr, sgm, mfr, mgm = maps
    @views dist_sp, dist_mp = dist[sp[1]:sp[2]], dist[mp[1]:mp[2]]

    if !(o.Sigmoid === nothing)
        sfr[i...] = dot(dist, weights) / sum(dist) # Use sigmoidal weighting function
    else
        sfr[i...] = sum(dist_sp) / sum(dist)
    end
    sgm[i...] = exp(dot(dist_sp, logT2_times_sp) / sum(dist_sp))
    mfr[i...] = sum(dist_mp) / sum(dist)
    mgm[i...] = exp(dot(dist_mp, logT2_times_mp) / sum(dist_mp))

    return nothing
end

# Micro-optimizations for above map calculations. Timing shows very little speedup;
# probably not worth sacrificing readability
function calc_maps_opt!(thread_buffer, maps, o::T2partOptions{T}, i...) where {T}
    @unpack dist, T2_times, sp, mp, logT2_times_sp, logT2_times_mp, weights = thread_buffer
    @unpack sfr, sgm, mfr, mgm = maps

    Σ_dist, Σ_dist_sp, Σ_dist_mp = zero(T), zero(T), zero(T)
    dot_sp, dot_mp = zero(T), zero(T)
    @inbounds for i in 1:length(dist)
        Σ_dist += dist[i]
    end
    @inbounds for i in sp[1]:sp[2]
        dot_sp += dist[i] * logT2_times_sp[i]
        Σ_dist_sp += dist[i]
    end
    @inbounds for i in mp[1]:mp[2]
        dot_mp += dist[i] * logT2_times_mp[i]
        Σ_dist_mp += dist[i]
    end

    if !(o.Sigmoid === nothing)
        sfr[i...] = dot(dist, weights) / Σ_dist # Use sigmoidal weighting function
    else
        sfr[i...] = Σ_dist_sp / Σ_dist
    end
    sgm[i...] = exp(dot_sp / Σ_dist_sp)
    mfr[i...] = Σ_dist_mp / Σ_dist
    mgm[i...] = exp(dot_mp / Σ_dist_mp)

    return nothing
end
