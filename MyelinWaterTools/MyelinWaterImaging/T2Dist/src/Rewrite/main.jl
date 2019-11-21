# Helper functions
_parse_or_convert(::Type{T}, s::AbstractString) where {T} = parse(T, s)
_parse_or_convert(::Type{T}, x) where {T} = convert(T, x)
_strip_union_nothing(T::Type{Union{Tin, Nothing}}) where {Tin} = Tin
_strip_union_nothing(T::Type) = T
_get_tuple_type(T::Type{Union{Tup, Nothing}}) where {Tup <: Tuple} = Tup
_get_tuple_type(::Type{Tup}) where {Tup <: Tuple} = Tup

# Allowed file types for loading
const ALLOWED_FILE_SUFFIXES = [".mat"]
const ALLOWED_FILE_SUFFIXES_STRING = join(ALLOWED_FILE_SUFFIXES, ", ", ", and ")

filter_allowed(filenames) = filter(s -> any(endswith.(s, ALLOWED_FILE_SUFFIXES)), filenames)

function chop_allowed_suffix(filename::AbstractString)
    for suffix in ALLOWED_FILE_SUFFIXES
        if endswith(filename, suffix)
            return filename[1:end-length(suffix)]
        end
    end
    error("Currently only $ALLOWED_FILE_SUFFIXES_STRING files are supported")
end

function get_file_info(parsed_args)
    @unpack input, output = parsed_args    

    inputfiles = if length(input) == 1 && isdir(input[1])
        joinpath.(input[1], filter_allowed(readdir(input[1])))
    else
        filter_allowed(input)
    end
    isempty(inputfiles) && error("No valid files were found for processing. Note that currently only $ALLOWED_FILE_SUFFIXES_STRING files are supported")

    outputfolders = if isempty(output)
        dirname.(inputfiles)
    else
        [output for _ in 1:length(inputfiles)]
    end

    return @ntuple(inputfiles, outputfolders)
end

function load_image(filename)
    if endswith(filename, ".mat")
        data = MAT.matread(filename)
        key = findfirst(x -> x isa AbstractArray{T,4} where {T}, data)
        if key === nothing
            error("No 4D array was found in the input file: $filename")
        else
            data[key]
        end
    else
        error("Currently, only $ALLOWED_FILE_SUFFIXES_STRING files are supported")
    end
end

function get_parsed_args_subset(parsed_args, options_type)
    kwargs = deepcopy(parsed_args)
    fields = fieldnames(options_type)
    types = fieldtypes(options_type)
    typemap = Dict{Symbol,Type}(fields .=> types)
    for (k,v) in kwargs
        if k ∉ fields
            delete!(kwargs, k)
            continue
        end
        if v isa AbstractString # parse v to appropriate type, which may not be String
            T = typemap[k]
            if !(T <: AbstractString)
                kwargs[k] = _parse_or_convert(_strip_union_nothing(T), v)
            end
        elseif v isa AbstractVector # convert AbstractVector v to appropriate Tuple type
            if isempty(v)
                delete!(kwargs, k) # default v === nothing for a Tuple type results in an empty vector
            else
                T = _get_tuple_type(typemap[k])
                kwargs[k] = tuple(_parse_or_convert.(fieldtypes(T), v)...) # each element should be individually parsed
            end
        end
    end
    return kwargs
end

function sorted_arg_table_entries(options...)
    fields, types, values = Symbol[], Type[], Any[]
    for o in options, (f,T) in zip(fieldnames(typeof(o)), fieldtypes(typeof(o)))
        push!(fields, f); push!(types, T); push!(values, getfield(o, f))
    end
    args = []
    for (k, v, T) in sort!(collect(zip(fields, values, types)); by = first)
        (k === :nTE || k === :GridSize) && continue # Skip automatically determined parameters
        push!(args, "--" * string(k))
        if T <: Union{<:Tuple, Nothing}
            nargs = length(fieldtypes(_get_tuple_type(T)))
            if v === nothing
                push!(args, Dict(:nargs => nargs, :default => v))
            else
                push!(args, Dict(:nargs => nargs, :default => [v...]))
            end
        else
            push!(args, Dict(:default => v))
        end
    end
    return args
end

function create_argparse_settings()
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
            help = "analyze T2 distribution to produce parameter maps; if --T2map flag is also passed, T2 distributions are first computed from the multi-echo input"
            action = :store_true
        "--quiet", "-q"
            help = "print minimal information (such as current progress) to the terminal. Note: 1) errors are not silenced, and 2) this flag overrides --Verbose flag in T2mapSEcorr"
            action = :store_true
    end

    add_arg_group(settings,
        "T2mapSEcorr/T2partSEcorr arguments",
        "internal arguments",
    )
    t2map_opts = T2mapOptions{Float64}(nTE = 32, GridSize = (1,1,1))
    t2part_opts = T2partOptions{Float64}(nT2 = t2map_opts.nT2, GridSize = (1,1,1))
    tmp = sorted_arg_table_entries(t2map_opts, t2part_opts)
    add_arg_table(settings, tmp...)

    return settings
end

function main()
    settings = create_argparse_settings()
    parsed_args = parse_args(settings; as_symbols = true)
    t2map_kwargs = get_parsed_args_subset(parsed_args, T2mapOptions{Float64})
    t2part_kwargs = get_parsed_args_subset(parsed_args, T2partOptions{Float64})

    # Unpack parsed flags, overriding appropriate options fields
    @unpack T2map, T2part, quiet = parsed_args
    quiet && (t2map_kwargs[:Verbose] = !quiet) # Force Verbose == false
    !T2map && (delete!(t2part_kwargs, :nT2)) # Infer nT2 from input T2 distbn

    # Get input file list and output folder list
    inputfiles, outputfolders = get_file_info(parsed_args)

    for (filename, folder) in zip(inputfiles, outputfolders)
        try
            choppedfilename = chop_allowed_suffix(basename(filename))
            mkpath(folder)

            # Save settings files
            for settingsfile in filter(s -> startswith(s, "@"), ARGS)
                src = settingsfile[2:end]
                dst = joinpath(folder, choppedfilename * "." * basename(src))
                cp(src, dst; force = true)
            end

            # Load image(s)
            !quiet && print("\n\n* Loading input file: $filename ... ")
            t = @elapsed image = load_image(filename)
            !quiet && println("Done ($(round(t; digits = 2)) seconds)")

            dist = if T2map
                # Compute T2 distribution from input 4D multi-echo image
                !quiet && println("\n* Running T2mapSEcorr on file: $filename ... ")
                t = @elapsed maps, distributions = T2mapSEcorr(image; t2map_kwargs...)
                !quiet && println("* Done ($(round(t; digits = 2)) seconds)")
                
                # Save results to .mat files
                savefile = joinpath(folder, choppedfilename * ".t2dist.mat")
                !quiet && print("\n* Saving T2 distribution to file: $savefile ... ")
                t = @elapsed MAT.matwrite(savefile, Dict("dist" => distributions))
                !quiet && println("Done ($(round(t; digits = 2)) seconds)")
                
                savefile = joinpath(folder, choppedfilename * ".t2maps.mat")
                !quiet && print("\n* Saving T2 parameter maps to file: $savefile ... ")
                t = @elapsed MAT.matwrite(savefile, maps)
                !quiet && println("Done ($(round(t; digits = 2)) seconds)")
                
                distributions
            else
                # Input image is the T2 distribution
                image
            end

            if T2part
                # Analyze T2 distribution to produce parameter maps
                !quiet && (T2map ? print("\n* Running T2partSEcorr ... ") : print("\n* Running T2partSEcorr on file: $filename ... "))
                t = @elapsed parts = T2partSEcorr(dist; t2part_kwargs...)
                !quiet && println("Done ($(round(t; digits = 2)) seconds)")

                # Save results to .mat file
                savefile = joinpath(folder, choppedfilename * ".t2parts.mat")
                !quiet && print("\n* Saving T2 parts maps to file: $savefile ... ")
                t = @elapsed MAT.matwrite(savefile, parts)
                !quiet && println("Done ($(round(t; digits = 2)) seconds)")
            end
        catch e
            println("\n\n\n********************************************************************************\n")
            @warn "Error during processing of file: $filename"
            println("\n")
            @warn sprint(showerror, e, catch_backtrace())
            println("\n********************************************************************************\n")
        end
    end

    return nothing
end
