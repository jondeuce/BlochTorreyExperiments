"""
    T2partOptions structure for T2mapSEcorr
"""
@with_kw struct T2partOptions{T} @deftype T
    # Size of first 3 dimensions of input 4D T2 distribution. Inferred automatically.
    GridSize::NTuple{3,Int}
    @assert all(GridSize .>= 1)

    # Number of T2 values in distribution. Inferred automatically as size(T2distributions, 4)
    nT2::Int
    @assert nT2 > 1

    # Min and Max T2 values of distribution
    T2Range::NTuple{2,T} = (0.015, 2.0)
    @assert 0.001 <= T2Range[1] < T2Range[2] <= 10.0
    
    # Min and Max of the short peak window
    SPWin::NTuple{2,T} = (0.014, 0.040)
    @assert SPWin[1] < SPWin[2]
    
    # Min and Max of the middle peak window
    MPWin::NTuple{2,T} = (0.040, 0.200)
    @assert MPWin[1] < MPWin[2]
    
    # Apply sigmoidal weighting to the upper limit of the short peak window.
    # Value is the delta-T2 parameter (distance in seconds on either side of
    # the SPWin upper limit where sigmoid curve reaches 10% and 90%).
    # Default is no sigmoid weighting.
    Sigmoid::Union{T,Nothing} = nothing
    @assert Sigmoid isa Nothing || Sigmoid > 0.0
end
T2partOptions(args...; kwargs...) = T2partOptions{Float64}(args...; kwargs...)

"""
maps = T2partSEcorr(T2distributions; kwargs...)

Description:
  Analyzes T2 distributions produced by T2mapSEcorr to produce data maps
  of a series of parameters.

Inputs:
  T2distributions: 4-D array with data as (row, column, slice, T2 Amplitude)
  kwargs: A series of optional keyword argument settings; see T2partOptions

Ouputs:
  maps: a dictionary containing the following 3D data maps as fields:
      "sfr": small pool (myelin water) fraction
      "sgm": small pool (myelin water) geometric mean
      "mfr": medium pool (intra/extra water) fraction
      "mgm": medium pool (intra/extra water) geometric mean

External Calls:
  none

Created by Thomas Prasloski
email: tprasloski@gmail.com
Ver. 1.2, August 2012

Adapted for Julia by Jonathan Doucette
email: jdoucette@phas.ubc.ca
Nov 2019
"""
function T2partSEcorr(T2distributions::Array{T,4}; kwargs...) where {T}
    reset_timer!(TIMER)
    out = @timeit_debug TIMER "T2partSEcorr" begin
        _T2partSEcorr(T2distributions, T2partOptions{T}(;
            GridSize = size(T2distributions)[1:3],
            nT2 = size(T2distributions, 4),
            kwargs...
        ))
    end
    if timeit_debug_enabled()
        println("\n"); show(TIMER); println("\n")
    end
    return out
end

function _T2partSEcorr(T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T}
    @assert size(T2distributions) == (opts.GridSize..., opts.nT2)
    maps = Dict{String, Array{T}}()
    maps["sfr"] = fill(T(NaN), opts.GridSize...)
    maps["sgm"] = fill(T(NaN), opts.GridSize...)
    maps["mfr"] = fill(T(NaN), opts.GridSize...)
    maps["mgm"] = fill(T(NaN), opts.GridSize...)
    thread_buffers = [thread_buffer_maker(opts) for _ in 1:Threads.nthreads()]

    LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads
    for slice in 1:opts.GridSize[3]
        Threads.@threads for col in 1:opts.GridSize[2]
            thread_buffer = thread_buffers[Threads.threadid()]
            @inbounds for row in 1:opts.GridSize[1]
                if any(isnan, @views(T2distributions[row, col, slice, :]))
                    continue
                end
                for i in 1:opts.nT2
                    thread_buffer.dist[i] = T2distributions[row, col, slice, i]
                end
                update_maps!(thread_buffer, maps, opts, row, col, slice)
            end
        end
    end
    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return maps
end

# =========================================================
# Make thread-local buffers
# =========================================================
function thread_buffer_maker(o::T2partOptions{T}) where {T}
    dist = zeros(T, o.nT2)
    T2_times = logrange(o.T2Range..., o.nT2)
    sp = (findfirst(t -> t >= o.SPWin[1], T2_times), findlast(t -> t <= o.SPWin[2], T2_times))
    mp = (findfirst(t -> t >= o.MPWin[1], T2_times), findlast(t -> t <= o.MPWin[2], T2_times))
    logT2_times = log.(T2_times)
    logT2_times_sp = logT2_times[sp[1]:sp[2]]
    logT2_times_mp = logT2_times[mp[1]:mp[2]]
    weights = sigmoid_weights(o)
    return @ntuple(dist, T2_times, sp, mp, logT2_times_sp, logT2_times_mp, weights)
end

function sigmoid_weights(o::T2partOptions{T}) where {T}
    if !(o.Sigmoid === nothing)
        # Curve reaches 50% at T2_50perc and is (k and 1-k)*100 percent at T2_50perc +/- T2_kperc  
        k, T2_kperc, T2_50perc = T(0.1), o.Sigmoid, o.SPWin[2]
        sigma = abs(T2_kperc / (sqrt(T(2)) * erfinv(2*k-1)))
        normccdf.((logrange(o.T2Range..., o.nT2) .- T2_50perc) ./ sigma)
    else
        nothing
    end
end

# =========================================================
# Save thread local results to output maps
# =========================================================
function update_maps!(thread_buffer, maps, o::T2partOptions, i...)
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
function update_maps_opt!(thread_buffer, maps, o::T2partOptions{T}, i...) where {T}
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
