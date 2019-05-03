# Struct for saving MWF results (TODO: don't need to parameterize by T here)
@with_kw struct MWFResults{T}
    metadata::Dict{Symbol,Any}               = Dict()
    params::Vector{BlochTorreyParameters{T}} = []
    sols::Vector{Vector{ODESolution}}        = []
    mwfvalues::Vector{Dict{Symbol,T}}        = []
end

function CSV.write(results::MWFResults, i)
    curr_date = getnow()
    for (j,sol) in enumerate(results.sols[i])
        t = DataFrame(sol.t')
        u = DataFrame(sol)
        fname = curr_date * "__sol_$(i)__region_$(j).csv"
        CSV.write(fname, u)
    end
    return nothing
end

function CSV.write(results::MWFResults)
    for i in 1:length(results.sols)
        CSV.write(results, i)
    end
    return nothing
end

function mwfresults_from_csv(;
        basedir = ".",
        geomfilename = "geom.bson",
        timefirstrow = true,
        save = true
    )
    resultsdict = resultsdict_from_csv(
        basedir = basedir, geomfilename = geomfilename, timefirstrow = timefirstrow, save = save);
    results = MWFResults{Float64}(
        Dict{Symbol,Any}(
            :geom             => resultsdict[:geom],
            :params           => resultsdict[:params],
            :omegas           => resultsdict[:omegas],
            :domains          => [(myelinprobs=p, myelinsubdomains=s, myelindomains=d) for (p,s,d) in zip(resultsdict[:myelinprobs], resultsdict[:myelinsubdomains], resultsdict[:myelindomains])]
        ),
        resultsdict[:params],
        resultsdict[:sols],
        resultsdict[:mwfvalues]
    )
    return results
end

function resultsdict_from_csv(;
        basedir = ".",
        geomfilename = "geom.bson",
        timefirstrow = true,
        save = true
    )
    # Load geometry
    geom = loadgeometry(geomfilename)

    # Find btparam filenames and solution filenames
    paramfiles = filter(s -> endswith(s, "btparams.bson"), readdir(basedir))
    solfiles = filter(s -> endswith(s, ".csv"), readdir(basedir))
    numregions = length(solfiles) ÷ length(paramfiles)
    
    # Initialize results
    allparams = [convert(BlochTorreyParameters{Float64}, BSON.load(pfile)[:btparams]) for pfile in paramfiles]
    results = blank_results_dict()

    # unpack geometry and create myelin domains
    for (params, solfilebatch) in zip(allparams, Iterators.partition(solfiles, numregions))
        
        @info "Recreating myelin domains"
        myelinprob, myelinsubdomains, myelindomains = createdomains(params, geom.exteriorgrids, geom.torigrids, geom.interiorgrids, geom.outercircles, geom.innercircles)
        
        @info "Recreating frenquency fields"
        omega = calcomega(myelinprob, myelinsubdomains)
        
        @info "Creating solutions from .csv files"
        sols = []
        for (myelindomain, solfile) in zip(myelindomains, solfilebatch)
            @info "Reading solution file: " * solfile
            csv = CSV.read(solfile)

            t = if timefirstrow
                [csv[1,j] for j in 1:size(csv,2)]
            else
                # Have to guess the times
                @warn "No time points saved. Creating linearly spaced time vector with repeats on π-pulses"
                t = Float64[]
                len = 2 * ((size(csv,2) - 1) ÷ 3) + 1
                for (i,tt) in enumerate(range(0.0, 320e-3, length=len))
                    push!(t,tt); iseven(i) && push!(t,tt)
                end
                t
            end
            
            u = if timefirstrow
                [[csv[i,j] for i in 2:size(csv,1)] for j in 1:size(csv, 2)]
            else
                [[csv[i,j] for i in 1:size(csv,1)] for j in 1:size(csv, 2)]
            end
            
            prob = ODEProblem(myelindomain, u[1], (t[1], t[end]))
            alg = default_algorithm()(prob)

            sol = DiffEqBase.build_solution(prob, alg, t, u)
            push!(sols, sol)
        end

        @info "Computing MWF values"
        mwfvalues, signals = compareMWFmethods(sols, myelindomains, geom.outercircles, geom.innercircles, geom.bdry)
        
        push!(results[:params], params)
        push!(results[:myelinprobs], myelinprob)
        push!(results[:myelinsubdomains], myelinsubdomains)
        push!(results[:myelindomains], myelindomains)
        push!(results[:omegas], omega)
        push!(results[:sols], sols)
        push!(results[:signals], signals)
        push!(results[:mwfvalues], mwfvalues)
    end

    @info "Saving new MWFResults structure"
    if save
        try
            BSON.bson(getnow() * "__results.bson", Dict(:results => deepcopy(results)))
        catch e
            @warn "Error saving results!"
            @warn sprint(showerror, e, catch_backtrace())
        end
    end

    return results
end

function compareMWFmethods(results::MWFResults)
    @unpack outercircles, innercircles, bdry = results.metadata[:geom]
    domains = results.metadata[:domains]
    sols = results.sols
    return [compareMWFmethods(sols[i], domains[i].myelindomains, outercircles, innercircles, bdry) for i in 1:length(sols)]
end