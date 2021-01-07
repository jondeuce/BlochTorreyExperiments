####
#### Saving and formatting
####

getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
savebson(filename, data::AbstractDict) = @elapsed BSON.bson(filename, data)
# gitdir() = realpath(DrWatson.projectdir(".."))

function handleinterrupt(e; msg = "Error")
    if e isa InterruptException
        @info "User interrupt"
    elseif e isa Flux.Optimise.StopException
        @info "Training stopped Flux callback"
    else
        !isempty(msg) && @warn msg
        @warn sprint(showerror, e, catch_backtrace())
    end
    return nothing
end

function saveprogress(savedata::AbstractDict; savefolder, prefix = "", suffix = "")
    !isdir(savefolder) && mkpath(savefolder)
    for (key, data) in savedata
        try
            BSON.bson(
                joinpath(savefolder, "$(prefix)$(key)$(suffix).bson"),
                Dict{String,Any}(string(key) => deepcopy(Flux.cpu(data))),
            )
        catch e
            handleinterrupt(e; msg = "Error saving progress")
        end
    end
end

function saveplots(plothandles::AbstractDict; savefolder, prefix = "", suffix = "", ext = ".png")
    !isdir(savefolder) && mkpath(savefolder)
    for (name, p) in plothandles
        (p === nothing) && continue
        try
            savefig(p, joinpath(savefolder, prefix * string(name) * suffix * ext))
        catch e
            handleinterrupt(e; msg = "Error saving plot ($name)")
        end
    end
end

function capture_stdout(f)
    let original_stdout = stdout
        read_pipe, write_pipe = redirect_stdout()
        try
            f()
        finally
            close(write_pipe)
            redirect_stdout(original_stdout)
        end
        read(read_pipe, String)
    end
end

####
#### Callbacks
####

function log10ticks(a, b; baseticks = 1:10)
    l, u = floor(Int, log10(a)), ceil(Int, log10(b))
    ticks = unique!(vcat([10.0^x .* baseticks for x in l:u]...))
    return filter!(x -> a ≤ x ≤ b, ticks)
end

function slidingindices(epoch, window = 100)
    i1_window = findfirst(e -> e ≥ epoch[end] - window + 1, epoch)
    i1_first  = findfirst(e -> e ≥ window, epoch)
    (i1_first === nothing) && (i1_first = 1)
    return min(i1_window, i1_first) : length(epoch)
end

####
#### Histogram
####

function fast_hist_1D(y, edges; normalize = nothing)
    #=
    @assert !isempty(y) && length(edges) >= 2
    _y = sort!(copy(y))
    @assert _y[1] >= edges[1]
    h = Histogram((edges,), zeros(Int, length(edges)-1), :left)
    j, done = 1, false
    @inbounds for i = 1:length(_y)
        while _y[i] >= edges[j+1]
            j += 1
            done = (j+1 > length(edges))
            done && break
        end
        done && break
        h.weights[j] += 1
    end
    =#

    # Faster just to call fit(Histogram, ...) in parallel
    hist(w) = Histogram((copy(edges),), w, :left)
    hs = [hist(zeros(Int, length(edges)-1)) for _ in 1:Threads.nthreads()]
    Is = collect(Iterators.partition(1:length(y), div(length(y), Threads.nthreads(), RoundUp)))
    Threads.@threads for I in Is
        @inbounds begin
            yi = view(y, I) # chunk of original y vector
            hi = hs[Threads.threadid()]
            hi = fit(Histogram, yi, UnitWeights{Float64}(length(yi)), hi.edges[1]; closed = :left)
            hs[Threads.threadid()] = hi
        end
    end
    h = hist(sum([hi.weights for hi in hs]))

    (normalize !== nothing) && (h = Plots.normalize(h, mode = normalize))
    return h
end

function fast_hist_1D(y, edges::AbstractRange; normalize = nothing)
    @assert length(edges) >= 2
    lo, hi, dx, n = first(edges), last(edges), step(edges), length(edges)
    hist(w) = Histogram((copy(edges),), w, :left)
    hs = [hist(zeros(Int, n-1)) for _ in 1:Threads.nthreads()]
    Threads.@threads for i in eachindex(y)
        @inbounds begin
            j = 1 + floor(Int, (y[i] - lo) / dx)
            (1 <= j <= n-1) && (hs[Threads.threadid()].weights[j] += 1)
        end
    end
    h = hist(sum([hi.weights for hi in hs]))
    (normalize !== nothing) && (h = Plots.normalize(h, mode = normalize))
    return h
end

function _fast_hist_test()
    _make_hist(x, edges) = fit(Histogram, x, UnitWeights{Float64}(length(x)), edges; closed = :left)
    for _ in 1:100
        n = rand([1:10; 32; 512; 1024])
        x = 100 .* rand(n)
        edges = rand(Bool) ? [0.0; sort(100 .* rand(n))] : (rand(1:10) : rand(1:10) : 100-rand(1:10))
        try
            @assert fast_hist_1D(x, edges) == _make_hist(x, edges)
        catch e
            if e isa InterruptException
                break
            else
                fast_hist_1D(x, edges).weights' |> x -> (display(x); display((first(x), last(x), sum(x))))
                _make_hist(x, edges).weights' |> x -> (display(x); display((first(x), last(x), sum(x))))
                rethrow(e)
            end
        end
    end

    x = rand(10^7)
    for ne in [4, 64, 1024]
        edges = range(0, 1; length = ne)
        @info "range (ne = $ne)"; @btime fast_hist_1D($x, $edges)
        @info "array (ne = $ne)"; @btime fast_hist_1D($x, $(collect(edges)))
        @info "plots (ne = $ne)"; @btime $_make_hist($x, $edges)
    end
end

_ChiSquared(Pi::T, Qi::T) where {T} = ifelse(Pi + Qi <= eps(T), zero(T), (Pi - Qi)^2 / (2 * (Pi + Qi)))
_KLDivergence(Pi::T, Qi::T) where {T} = ifelse(Pi <= eps(T) || Qi <= eps(T), zero(T), Pi * log(Pi / Qi))
ChiSquared(P::Histogram, Q::Histogram) = sum(_ChiSquared.(Working.unitsum(P.weights), Working.unitsum(Q.weights)))
KLDivergence(P::Histogram, Q::Histogram) = sum(_KLDivergence.(Working.unitsum(P.weights), Working.unitsum(Q.weights)))
CityBlock(P::Histogram, Q::Histogram) = sum(abs, Working.unitsum(P.weights) .- Working.unitsum(Q.weights))
Euclidean(P::Histogram, Q::Histogram) = sqrt(sum(abs2, Working.unitsum(P.weights) .- Working.unitsum(Q.weights)))

function signal_histograms(Y::AbstractMatrix; nbins = nothing, edges = nothing, normalize = nothing)
    make_edges(x) = ((lo,hi) = extrema(vec(x)); return range(lo, hi; length = nbins)) # mid = median(vec(x)); length = ceil(Int, (hi - lo) * nbins / max(hi - mid, mid - lo))
    hists = Dict{Int, Histogram}()
    hists[0] = fast_hist_1D(vec(Y), (edges === nothing) ? make_edges(Y) : edges[0]; normalize)
    for i in 1:size(Y,1)
        hists[i] = fast_hist_1D(Y[i,:], (edges === nothing) ? make_edges(Y[i,:]) : edges[i]; normalize)
    end
    return hists
end
