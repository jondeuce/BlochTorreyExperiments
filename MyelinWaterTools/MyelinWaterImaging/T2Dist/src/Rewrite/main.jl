parse_or_convert(::Type{T}, s::AbstractString) where {T} = parse(T, s)
parse_or_convert(::Type{T}, x) where {T} = convert(T, x)
_strip_union_nothing(T::Type{Union{Tin, Nothing}}) where {Tin} = Tin
_get_tuple_type(T::Type{Union{Tup, Nothing}}) where {Tup <: Tuple} = Tup
_get_tuple_type(::Type{Tup}) where {Tup <: Tuple} = Tup

function filter_parsed_args(parsed_args, options_type)
    kwargs = deepcopy(parsed_args)
    fields, types = fieldnames(options_type), fieldtypes(options_type)
    typemap = Dict{Symbol,Any}(fields .=> types)
    for (k,v) in kwargs
        if k ∉ fields
            delete!(kwargs, k)
            continue
        end
        if v isa AbstractString
            T = typemap[k]
            if !(T <: AbstractString)
                kwargs[k] = parse_or_convert(_strip_union_nothing(T), v)
            end
        elseif v isa AbstractVector
            if isempty(v)
                delete!(kwargs, k)
            else
                T = _get_tuple_type(typemap[k])
                kwargs[k] = tuple(parse_or_convert.(fieldtypes(T), v)...)
            end
        end
    end
    return kwargs
end

function sorted_arg_table_entries(options...)
    fields = [Symbol[fieldnames(typeof(o))...] for o in options]
    types = [Any[fieldtypes(typeof(o))...] for o in options]
    values = [getfield.(Ref(o), f) for (f,o) in zip(fields, options)]
    fields, values, types = reduce(vcat, fields), reduce(vcat, values), reduce(vcat, types)
    args = []
    for (k, v, T) in sort(collect(zip(fields, values, types)); by = x -> x[1])
        if k === :nTE || k === :nT2 || k === :GridSize
            continue
        end
        push!(args, "--" * string(k))
        if T === Bool
            push!(args, Dict(:action => :store_true, :help => "optional flag (default: false)"))
        elseif T <: Union{<:Tuple, Nothing}
            fields = fieldtypes(_get_tuple_type(T))
            if v === nothing
                push!(args, Dict(:nargs => length(fields), :default => v))
            else
                push!(args, Dict(:nargs => length(fields), :default => [v...]))
            end
        else
            push!(args, Dict(:default => v))
        end
    end
    return args
end

function create_settings()
    settings = ArgParseSettings(
        fromfile_prefix_chars = "@",
        error_on_conflict = false,
    )

    @add_arg_table settings begin
        "input"
            help = "input filename(s), or directory containing input file(s)"
            required = true
            nargs = '+' # At least one input is required
        "--output", "-o"
            help = "output folder (default: input folder, or folder containing input file)"
            default = ""
        "--T2map"
            help = "compute T2 distribution from input image, a multi-echo 4D array"
            action = :store_true
        "--T2part"
            help = "analyze input T2 distribution to produce parameter maps; if --T2map flag is also passed, T2 distributions are first computed from the multi-echo input image"
            action = :store_true
        "--quiet", "-q"
            help = "print minimal information - such as current progress - to the terminal"
            action = :store_true
    end

    add_arg_group(settings,
        "T2mapSEcorr/T2partSEcorr arguments",
        "internal arguments",
    )
    map_opts = T2mapOptions{Float64}(nTE = 32, GridSize = (1,1,1))
    part_opts = T2partOptions{Float64}(nT2 = 40, GridSize = (1,1,1))
    add_arg_table(settings, sorted_arg_table_entries(map_opts, part_opts)...)

    return settings
end

const VALID_SUFFIXES = [".mat"]
filter_valid(filenames) = filter(s -> any(endswith.(s, VALID_SUFFIXES)), filenames)
chop_suffix(filename::AbstractString) = 
    endswith(filename, ".mat") ? filename[1:end-4] :
    error("Currently only .mat files are supported")

function load_images(parsed_args)
    @unpack input, output = parsed_args    

    inputfiles = if length(input) == 1 && isdir(input[1])
        joinpath.(input[1], filter_valid(readdir(input[1])))
    else
        filter_valid(input)
    end

    if isempty(inputfiles)
        error("No valid files were found for processing. Note that currently only .mat files are supported")
    end

    outputfolders = if isempty(output)
        dirname.(inputfiles)
    else
        [output for _ in 1:length(inputfiles)]
    end

    function load_image(filename)
        print("Loading input file: $filename... ")
        t = @elapsed image = if endswith(filename, ".mat")
            data = MAT.matread(filename)
            key = findfirst(x -> x isa AbstractArray{T,4} where {T}, data)
            data[key]
        else
            error("Currently, only .mat files are supported")
        end
        println("Done ($(round(t; digits = 2)) seconds)")
        return image
    end
    images = load_image.(inputfiles)

    return images, inputfiles, outputfolders
end

function main()
    settings = create_settings()
    parsed_args = parse_args(settings; as_symbols = true)
    
    println("Parsed args:")
    for (arg, val) in parsed_args
        @show arg, val
    end

    println("T2mapOptions args:")
    t2map_kwargs = filter_parsed_args(parsed_args, T2mapOptions{Float64})
    for (arg, val) in t2map_kwargs
        @show arg, val
    end

    println("T2partOptions args:")
    t2part_kwargs = filter_parsed_args(parsed_args, T2partOptions{Float64})
    for (arg, val) in t2part_kwargs
        @show arg, val
    end

    images, inputfiles, outputfolders = load_images(parsed_args)
    for (image, filename, folder) in zip(images, inputfiles, outputfolders)
        choppedfilename = chop_suffix(basename(filename))

        dist = if parsed_args[:T2map]
            maps, distributions = T2mapSEcorr(image; t2map_kwargs...)
            MAT.matwrite(joinpath(folder, choppedfilename * ".t2maps.mat"), maps)
            MAT.matwrite(joinpath(folder, choppedfilename * ".t2dist.mat"), Dict("dist" => distributions))
            distributions
        else
            image
        end

        if parsed_args[:T2part]
            parts = T2partSEcorr(dist; t2part_kwargs...)
            MAT.matwrite(joinpath(folder, choppedfilename * ".t2parts.mat"), parts)
        end
    end
end
