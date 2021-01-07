####
#### Common utilities, not relying on custom types etc.
####

const AbstractTensor3D{T} = AbstractArray{T,3}
const AbstractTensor4D{T} = AbstractArray{T,4}
const CuTensor3D{T} = CUDA.CuArray{T,3}
const CuTensor4D{T} = CUDA.CuArray{T,4}

todevice(x) = Flux.use_cuda[] ? Flux.gpu(x) : Flux.cpu(x)
todevice(d::AbstractDict) = Dict(k => todevice(v) for (k,v) in d)
to32(x) = Flux.fmap(xi -> xi isa AbstractArray ? convert(AbstractArray{Float32}, xi) : xi, todevice(x))
to64(x) = Flux.fmap(xi -> xi isa AbstractArray ? convert(AbstractArray{Float64}, xi) : xi, todevice(x))

arr_similar(x::AbstractArray, y::AbstractArray) = arr_similar(typeof(x), y)
arr_similar(::Type{<:AbstractArray{T}}, y::AbstractArray) where {T} = convert(Array{T}, y)
arr_similar(::Type{<:AbstractArray{T}}, y::CUDA.CuArray{T}) where {T} = convert(Array{T}, y) #TODO: CuArray -> Array works directly if eltypes are equal
arr_similar(::Type{<:AbstractArray{T1}}, y::CUDA.CuArray{T2}) where {T1,T2} = convert(Array{T1}, y |> Flux.cpu) #TODO: CuArray -> Array falls back to scalar indexing with unequal eltypes
arr_similar(::Type{<:CUDA.CuArray{T1}}, y::CUDA.CuArray{T2}) where {T1,T2} = convert(CUDA.CuArray{T1}, y) #TODO: Needed for disambiguation
arr_similar(::Type{<:CUDA.CuArray{T}}, y::AbstractArray) where {T} = convert(CUDA.CuArray{T}, y)
Zygote.@adjoint arr_similar(::Type{Tx}, y::Ty) where {Tx <: AbstractArray, Ty <: AbstractArray} = arr_similar(Tx, y), Δ -> (nothing, arr_similar(Ty, Δ)) # preserve input type on backward pass

arr32(x::AbstractArray) = arr_similar(Array{Float32}, x)
arr64(x::AbstractArray) = arr_similar(Array{Float64}, x)

# rand_similar and randn_similar
for f in [:zeros, :ones, :rand, :randn]
    f_similar = Symbol(f, :_similar)
    @eval $f_similar(x::AbstractArray, sz...) = $f_similar(typeof(x), sz...)
    @eval $f_similar(::Type{<:AbstractArray{T}}, sz...) where {T} = Zygote.ignore(() -> Base.$f(T, sz...)) # fallback
    @eval $f_similar(::Type{<:CUDA.CuArray{T}}, sz...) where {T} = Zygote.ignore(() -> CUDA.$f(T, sz...)) # CUDA
end

fill_similar(x::AbstractArray, v, sz...) = fill_similar(typeof(x), v, sz...)
fill_similar(::Type{<:AbstractArray{T}}, v, sz...) where {T} = Base.fill(T(v), sz...) # fallback
fill_similar(::Type{<:CUDA.CuArray{T}}, v, sz...) where {T} = CUDA.fill(T(v), sz...) # CUDA

function find_used_names(mod;
        dir = ".",
        incl = "*.jl",
        excl = "",
        excl_dir = "",
        )
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

nothing
