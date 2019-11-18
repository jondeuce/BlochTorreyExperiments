function parse_commandline()
    settings = ArgParseSettings(
        fromfile_prefix_chars = "@",
    )

    @add_arg_table settings begin
        "--input", "-i"
            help = "input filename(s), or directory containing input files"
            required = true
            nargs = '+' # At least one input is required
        "--T2map"
            help = "compute T2 distribution from input image, a multi-echo 4D array"
            action = :store_true
        "--T2part"
            help = "analyze input T2 distribution to produce parameter maps. If --T2map flag is also passed, T2 distributions are first computed from the multi-echo input image"
            action = :store_true
    end

    add_arg_group(settings,
        "T2mapSEcorr",
        "Settings for computing T2 distributions using NNLS in the presence of stimulated echos by optimizing the refocusing pulse flip angle"
    )

    opts = T2mapOptions{Float64}(nTE = 32, GridSize = (1,1,1))
    args = []
    for (k, type) in zip(fieldnames(typeof(opts)), fieldtypes(typeof(opts)))
        if k === :nTE || k === :GridSize
            continue
        end
        v = getfield(opts, k)
        push!(args, "--" * string(k))
        if v === nothing
            push!(args, Dict(:arg_type => type))
        else
            push!(args, Dict(:arg_type => type, :default => v))
        end
    end
    add_arg_table(settings, args...)

    return parse_args(settings)
    # return settings
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
end
