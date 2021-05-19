####
#### Array
####

const AbstractTensor3D{T} = AbstractArray{T,3}
const AbstractTensor4D{T} = AbstractArray{T,4}
const CuTensor3D{T} = CuArray{T,3}
const CuTensor4D{T} = CuArray{T,4}

const PairOfTuples{N} = Pair{<:Tuple{Vararg{Any,N}}, <:Tuple{Vararg{Any,N}}}
const NTupleOfNamedTuples{Ks,M,N} = NTuple{N,<:NamedTuple{Ks,<:NTuple{M}}}
const NamedTupleOfNTuples{Ks,M,N} = NamedTuple{Ks, <:NTuple{M, <:NTuple{N}}}

const Dual64 = ForwardDiff.Dual{Tag, Float64} where {Tag}
const MaybeDualF64 = Union{Float64, <:Dual64}
const VecOrTupleF64 = Union{<:AbstractVector{Float64}, <:Tuple{Vararg{Float64}}}
const VecOrTupleMaybeDualF64 = Union{<:AbstractVector{<:MaybeDualF64}, <:Tuple{Vararg{<:MaybeDualF64}}}

# Recursively adapt storage type
fmap_adapt(::Type{T}, x) where {T <: AbstractArray} = Flux.fmap(y -> _fmap_adapt(T, y), x)
fmap_adapt(::Type{T}, x::AbstractArray) where {T <: AbstractArray} = _fmap_adapt(T, x)
_fmap_adapt(::Type{T}, x::AbstractArray) where {T <: AbstractArray} = arr_similar(T, x)
_fmap_adapt(::Type{T}, x) where {T <: AbstractArray} = x #fallback

# Similar to Flux.cpu and Flux.gpu, but won't force conversion to Float32
cpu(T, x) = fmap_adapt(Array{T}, x)
gpu(T, x) = Flux.use_cuda[] ? fmap_adapt(CuArray{T}, x) : cpu(T, x)
cpu(x) = cpu(Float64, x)
gpu(x) = gpu(Float32, x)
cpu64(x) = cpu(Float64, x)
cpu32(x) = cpu(Float32, x)
gpu64(x) = gpu(Float64, x)
gpu32(x) = gpu(Float32, x)

# Extend for non-fmap-able collections
for f in [:cpu, :gpu]
    @eval $f(T, d::AbstractDict) = Dict(k => $f(T, v) for (k,v) in d)
end

arr_similar(x::AbstractArray, y::AbstractArray) = arr_similar(typeof(x), y)
arr_similar(::Type{<:AbstractArray{T}}, y::AbstractArray) where {T} = CUDA.adapt(Array{T}, y) # fallback
arr_similar(::Type{<:CuArray{T}}, y::AbstractArray) where {T} = CUDA.adapt(CuArray{T}, y)
Zygote.@adjoint arr_similar(::Type{Tx}, y::Ty) where {Tx <: AbstractArray, Ty <: AbstractArray} = arr_similar(Tx, y), Δ -> (nothing, arr_similar(Ty, Δ)) # preserve input type on backward pass

function _test_arr_similar()
    for f1 in [rand, CUDA.rand], f2 in [rand, CUDA.rand], T1 in [Float32, Float64], T2 in [Float32, Float64], sz in [(1,), (1,1), (1,1,1)]
        x = f1(T1, sz)
        y = f2(T2, sz)
        @assert typeof(arr_similar(x, y)) == typeof(x)
        @assert typeof(arr_similar(y, x)) == typeof(y)
        @assert isapprox(cpu(arr_similar(x, y)), cpu(y); atol = T1 == T2 ? 0 : max(eps(T1), eps(T2)))
        @assert isapprox(cpu(arr_similar(y, x)), cpu(x); atol = T1 == T2 ? 0 : max(eps(T1), eps(T2)))
    end
end

# rand_similar and randn_similar
for f in [:zeros, :ones, :rand, :randn]
    f_similar = Symbol(f, :_similar)
    @eval $f_similar(x::AbstractArray) = $f_similar(typeof(x), size(x)...)
    @eval $f_similar(x::AbstractArray, sz...) = $f_similar(typeof(x), sz...)
    @eval $f_similar(::Type{<:AbstractArray{T}}, sz...) where {T} = Zygote.ignore(() -> Base.$f(T, sz...)) # fallback
    @eval $f_similar(::Type{<:CuArray{T}}, sz...) where {T} = Zygote.ignore(() -> CUDA.$f(T, sz...)) # CUDA
end

fill_similar(x::AbstractArray, v) = fill_similar(typeof(x), v, size(x)...)
fill_similar(x::AbstractArray, v, sz...) = fill_similar(typeof(x), v, sz...)
fill_similar(::Type{<:AbstractArray{T}}, v, sz...) where {T} = Base.fill(T(v), sz...) # fallback
fill_similar(::Type{<:CuArray{T}}, v, sz...) where {T} = CUDA.fill(T(v), sz...) # CUDA

@inline ofeltypefloat(x::Number, y) = oftypefloat(x, y)
@inline ofeltypefloat(x, y) = convert(float(eltype(x)), y)
@inline oftypefloat(x, y) = oftype(float(x), y)
@inline epseltype(x) = eps(float(eltype(x)))

# Unzip array of structs into struct of arrays
unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

# Display and return
disp_ret(x; type = true, val = true) = (type && display(typeof(x)); val && display(x); (type || val) && println(""); x)

# Check if dx approximately divides x
is_approx_multiple_of(x, dx) = (n = round(Int, x/dx); x ≈ n * dx)

# make_kwargs from dictionary of settings
make_kwargs(settings, keys...) = Any[Symbol(k) => v for (k,v) in foldl(getindex, string.(keys); init = settings)]

# Draw `samples` items from `x` with optional shuffling. Sampling is done with replacement iff `samples > length(x)`
function sample_maybeshuffle(x; shuffle = true, samples = Colon(), seed = 0)
    (samples === Colon()) && (samples = length(x))
    I = shuffle ?
        sample(MersenneTwister(seed), 1:length(x), samples; replace = samples > length(x)) :
        1:min(length(x), samples)
    return x[I], I
end

####
#### Dict, (Named)Tuples
####

# Find locations of all `needles` within `haystack`. Will error if not found
function findall_within(haystack::AbstractArray, needles::AbstractArray)
    hashmap = Dict(haystack .=> eachindex(haystack))
    (I -> hashmap[I]).(needles)
end

# Recursively try to access collection[keys[1]][...][keys[end]], returning `nothing` if any key ∈ keys does not exist
function recursive_try_get(collection, keys)
    foldl(keys; init = collection) do acc, key
        acc === missing ? acc : get(acc, key, missing)
    end
end

# Map over dictionary values
map_dict(f, d::Dict{K,V}) where {K,V} = Dict(map(((k,v),) -> k => f(v), collect(d)))

# Differentiable summing of dictionary values
sum_dict(d::Dict{K,V}) where {K,V} = sum(values(d))

Zygote.@adjoint function sum_dict(d::Dict{K,V}) where {K,V}
    sum_dict(d), function (Δ)
        grad = Zygote.grad_mut(__context__, d)
        for k in keys(d)
            grad[k] = Zygote.accum(get(grad, k, nothing), Δ)
        end
        return (grad,)
    end
end

@generated function mask_tuple(tup::NamedTuple{keys, NTuple{N,T}}, ::Val{mask}) where {keys,N,T,mask}
    ex = [:(keys[$i] => getproperty(tup, keys[$i])) for i in 1:N if mask[i]]
    return :((; $(ex...)))
end

@generated function mask_tuple(tup::NTuple{N,T}, ::Val{mask}) where {N,T,mask}
    ex = [:(tup[$i]) for i in 1:N if mask[i]]
    return :(($(ex...),))
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

####
#### Multi-threading utils
####

function foreach_with_progress(f, foreach_, input::AbstractRange; dt = 1.0, checkpoint_cb = nothing, checkpoint_freq::Int = typemax(Int))

    completed = falses(length(input))
    completed_copy = copy(completed)
    checkpoint_count = 1
    ℓ = Threads.SpinLock()
    p = ProgressMeter.Progress(length(input), dt)

    @async foreach_(enumerate(input)) do (c,i)
        f(i)
        Threads.lock(ℓ) do
            completed[c] = true
        end
    end

    while true
        sleep(dt)
        Threads.lock(ℓ) do
            completed_copy .= completed
        end
        if checkpoint_cb !== nothing && checkpoint_freq <= length(input)
            while checkpoint_count <= length(input)
                checkpoint_range      = checkpoint_count : min(checkpoint_count + checkpoint_freq - 1, length(input))
                checkpoint_statuses   = @views completed_copy[checkpoint_range]
                completed_inputs      = @views input[checkpoint_range]
                if all(checkpoint_statuses)
                    checkpoint_count = checkpoint_range[end] + 1
                    Threads.lock(ℓ) do
                        checkpoint_cb(completed_inputs)
                    end
                else
                    break
                end
            end
        end
        total_completed = sum(completed_copy)
        ProgressMeter.update!(p, total_completed)
        total_completed === length(input) && break
    end

    return nothing
end
foreach_with_progress(f, input; kwargs...) = foreach_with_progress(f, ThreadPools.qbforeach, input; kwargs...)

####
#### IO, error handling
####

getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")

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

# Save and print settings file
function save_settings(settings::AbstractDict; filename = nothing, verbose = true)
    if (filename !== nothing)
        @assert endswith(filename, ".toml")
        mkpath(dirname(filename))
        open(filename; write = true) do io
            TOML.print(io, settings)
        end
    end
    verbose && TOML.print(stdout, settings)
    return settings
end

function save_progress(savedata::AbstractDict; savefolder, ext, prefix = "", suffix = "")
    for (key, data) in savedata
        try
            filename = joinpath(mkpath(savefolder), prefix * string(key) * suffix * ext)
            data = Dict{String,Any}(string(key) => deepcopy(cpu(data)))
            FileIO.save(filename, data)
        catch e
            handle_interrupt(e; msg = "Error saving progress for data: $key")
        end
    end
end

function save_plots(plothandles::AbstractDict; savefolder, prefix = "", suffix = "", ext = ".png")
    for (name, p) in plothandles
        (p === nothing) && continue
        try
            savefig(p, joinpath(mkpath(savefolder), prefix * string(name) * suffix * ext))
        catch e
            handle_interrupt(e; msg = "Error saving plot ($name)")
        end
    end
end

function handle_interrupt(e; msg = "Error")
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

function load_module_from_project(; project_path::AbstractString, module_name::Symbol, sandbox_name = gensym())
    curr_project_path = dirname(Base.active_project())
    mod = sandbox = nothing
    try
        @suppress Pkg.activate(project_path)
        ex = :(
            module $sandbox_name # sandbox module to load `module_name` into
            using $module_name # import `module_name` and all its exported symbols
            end
        )
        Main.eval(ex)
        sandbox = getfield(Main, sandbox_name)
        mod = getfield(sandbox, module_name)
    finally
        @suppress Pkg.activate(curr_project_path)
    end
    return mod, sandbox
end

function load_bson_from_project(filename::AbstractString; kwargs...)
    _, sandbox = load_module_from_project(; kwargs...)
    return load_bson_from_project(filename, sandbox)
end

function load_bson_from_project(filename::AbstractString, sandbox::Module)
    @eval Main BSON.resolve(fs) = reduce((m, f) -> getfield(m, Symbol(f)), fs; init = $sandbox) # https://github.com/JuliaIO/BSON.jl/pull/28
    data = Base.invokelatest(BSON.load, filename)
    @eval Main BSON.resolve(fs) = reduce((m, f) -> getfield(m, Symbol(f)), fs; init = Main)
    return data
end

function _load_bson_from_project_test()
    OldProjPath = joinpath(@__DIR__, "../../old/Proj/")
    # BSON.load(joinpath(OldProjPath, "data/oldstruct.bson")) # fails
    mod, sandbox = load_module_from_project(project_path = OldProjPath, module_name = :Proj, sandbox_name = :Sandbox) # load old `Proj` module into toplevel module `Sandbox`
    data = load_bson_from_project(joinpath(OldProjPath, "data/oldstruct.bson"), sandbox) # load `oldstruct.bson` using definition contained in `Sandbox`
end

####
#### Misc.
####

function find_used_names(mod; dir = ".", incl = "*.jl", excl = "", excl_dir = "")
    mod_names = names(mod)
    name_counts = map(mod_names) do name
        cmd = ["grep", "-nr", "--word-regexp", "--color", "--max-count", "1", "--include", incl, "--exclude", excl, "--exclude-dir", excl_dir, String(name), dir]
        try
            count = parse(Int, chomp(read(pipeline(Cmd(cmd), `wc -l`), String)))
            (count > 0) && (@info name; run(Cmd(cmd)); println(""))
            return count
        catch e
            e isa ProcessFailedException ? (count = 0) : rethrow(e) # `grep` errors if no matches are found
        end
    end

    name_map = Dict{Module, Vector{Symbol}}()
    for (name, count) in zip(mod_names, name_counts)
        !(count > 0) && continue
        defining_mod = try
            which(mod, name) # why does this sometimes fail?
        catch e
            @warn sprint(showerror, e, catch_backtrace())
            continue
        end
        push!(get!(name_map, defining_mod, Symbol[]), name)
    end

    for (defining_mod, used_names) in name_map
        """
        @reexport using $defining_mod: $(join(used_names, ", "))
        """ |> println
    end

    return name_map
end

nothing
