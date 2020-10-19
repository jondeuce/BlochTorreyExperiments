import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Dates, Glob, MAT, BSON, MATLAB, Plots, LaTeXStrings, AxisArrays, NaNMath
# pyplot(size = (800,600))
pyplot(size = (1000,750))
# pyplot(size = (1600,900))

homedir() = "/project/st-arausch-1/jcd1994/code"
# sweepdir() = "/project/st-arausch-1/jcd1994/GRE-DSC-PVS-Orientation2020Fall/2020-09-11"
# sweepdir() = "/project/st-arausch-1/jcd1994/GRE-DSC-PVS-Orientation2020Fall/2020-09-16-static"
# sweepdir() = "/project/st-arausch-1/jcd1994/GRE-DSC-PVS-Orientation2020Fall/2020-09-16-dynamic"
sweepdir() = "/project/st-arausch-1/jcd1994/GRE-DSC-PVS-Orientation2020Fall/2020-09-23-nlopt"
saveplot(name) = p -> (foreach(ext -> savefig(p, name * ext), [".png", ".pdf"]); return p)

jobnum(jobdir) = parse(Int, match(r"Job-(\d+)_", jobdir)[1])
jobdirs(sweepdir = sweepdir()) = glob(glob"Job*", sweepdir) |> dirs -> Dict(jobnum.(dirs) .=> dirs)

mxcall(:addpath, 0, homedir())
mxcall(:addpath, 0, mxcall(:btpathdef, 1))

unzip(x::AbstractArray) = ntuple(i -> (xj -> xj[i]).(x), length(x[1]))
rowvcat(xs) = reduce(vcat, permutedims.(vec.(xs)))
getresult(r, f) = r["Results"][f]
getresult(r, f1, f2) = r["Results"][f1][f2]
getresults(rs, fs...) = getresult.(rs, fs...) |> x -> eltype(x) <: AbstractArray ? rowvcat(x) : x
lossfun(ydata, ymodel, weights, normfun) = mxcall(:perforientation_objfun, 1, zeros(1,3), zeros(size(ydata)...), ydata, ymodel, weights, normfun)
lossfun(rs, normfun) = lossfun(getresults(rs, "dR2_Data"), getresults(rs, "dR2"), getresults(rs, "args", "Weights"), normfun)

_nanreduce(f, A, ::Colon) = filter(!isnan, A) |> x -> isempty(x) ? float(eltype(A))(NaN) : f(x)
_nanreduce(f, A, dims) = mapslices(a -> _nanreduce(f, a, :), A, dims = dims)
nanreduce(f, A; dims = :) = dropdims(_nanreduce(f, A, dims); dims = dims)
function nanreduce(f, A::AxisArray; dims::Union{Symbol,NTuple{N,Symbol}}) where {N}
    if dims isa Symbol; dims = (dims,); end
    ax = filter(ax -> AxisArrays.axisname(ax) ∉ dims, AxisArrays.axes(A))
    dims = map(d -> axisdim(A, Axis{d}), dims)
    AxisArray(nanreduce(f, convert(Array, A); dims = dims), ax)
end

function save_iterations(jobdirs = jobdirs(); force = false)
    for (j,jobdir) in sort(jobdirs)
        @info "Starting... $(basename(jobdir))"
        if !force && all(isfile.(joinpath.(jobdir, ["stop.fired", "IterationsResults.mat"])))
            continue
        end

        @time begin
            iter_files, iter_results = nothing, nothing
            if isfile(joinpath(jobdir, "AllIterationsResults.mat"))
                iter_results = MAT.matread(joinpath(jobdir, "AllIterationsResults.mat"))["AllIterationsResults"]
            else
                iter_files = readdir(glob"2020**/*.mat", jobdir)
                !isempty(iter_files) && (iter_results = matread.(iter_files))
            end
            isnothing(iter_results) && continue

            d = Dict{String,Any}()
            d["timestamp"] = isnothing(iter_files) ? "" :
                iter_files .|> dirname .|> basename .|> s -> DateTime(s[1:25], "yyyy-mm-dd-T-HH-MM-SS-sss") .|> string
            for f in ["iBVF", "aBVF", "CA", "alpha_range", "dR2", "dR2_Data"]
                d[f] = getresults(iter_results, f)
            end
            d["BVF"] = d["iBVF"] + d["aBVF"]
            ps = getresults(iter_results, "params")
            d["Rmajor"] = ps[:,2]
            d["MinorExpansion"] = ps[:,3]
            for ℓ in ["L2W", "R2w", "AICc"]
                d[ℓ] = lossfun(iter_results, ℓ)
            end

            MAT.matwrite(joinpath(jobdir, "IterationsResults.mat"), deepcopy(d))
        end
    end
end
save_iterations(force = true)

function read_and_plot(jobdirs = jobdirs())
    function _read_and_plot(jobdir)
        res = matread(joinpath(jobdir, "IterationsResults.mat"))
        s(x) = string(round(x; sigdigits = 4))

        plot(
            plot(;title = basename(jobdir), grid = :none, ticks = :none, showaxis = false),
            plot(
                res["AICc"] |> y -> scatter(y; ylabel = "AICc", c = :blue, lab = "min = $(s(minimum(y)))\ncurr = $(s(y[end]))"),
                res["CA"] |> y -> scatter(y; ylabel = "CA", c = :red, lab = "curr = $(s(y[end]))"),
                100 * res["iBVF"] |> y -> scatter(y; ylabel = "iBVF", c = :green, lab = "curr = $(s(y[end]))"),
                100 * res["aBVF"] |> y -> scatter(y; ylabel = "aBVF", c = :purple, lab = "curr = $(s(y[end]))");
            ),
            layout = @layout([a{0.01h}; b{0.99h}]),
        ) |> saveplot(joinpath(jobdir, "Losses")) |> display

        plot(
            plot(;title = basename(jobdir), grid = :none, ticks = :none, showaxis = false),
            plot(map(sortperm(res["AICc"])[1:min(12,end)]) do i
            xdata, ydata, fdata = vec.((res["alpha_range"][i,:], res["dR2_Data"][i,:], res["dR2"][i,:]))
            title = "AICc = $(s(res["AICc"][i])), CA = $(s(res["CA"][i]))\naBVF = $(s(100 * res["aBVF"][i])), iBVF = $(s(100 * res["iBVF"][i])), BVF = $(s(100 * (res["aBVF"][i] + res["iBVF"][i])))"
            plot(
                xdata, [ydata fdata];
                xlab = L"$\alpha$ [degrees]", ylab = L"$\Delta R_2^*$ [Hz]", lab = ["Data" "Fit"], title = title, leg = :bottomright,
                lw = 2, titlefontsize = 8, labelfontsize = 8, legendfontsize = 6,
            )
            end...),
            layout = @layout([a{0.01h}; b{0.99h}]),
        ) |> saveplot(joinpath(jobdir, "Fits")) |> display
    end

    for (j,jobdir) in sort(jobdirs)
        if all(isfile.(joinpath.(jobdir, ("Geom.mat", "BBOptWorkspace.mat"))))
            if isfile(joinpath(jobdir, "stop.fired"))
                @info "Done ---- $(basename(jobdir))"
                continue
            else
                if all(isfile.(joinpath.(jobdir, "AllIterationsResults" .* (".mat", ".zip"))))
                    @info "Finished ---- $(basename(jobdir))"
                else
                    @info "Running ---- $(basename(jobdir))"
                end
            end
        else
            if any(isfile.(joinpath.(jobdir, "PBS-" .* ("error", "output") .* ".txt")))
                @info "---- Re-run! ---- $(basename(jobdir))"
            else
                @info "---- Queueing ---- $(basename(jobdir))"
            end
            continue
        end

        @time _read_and_plot(jobdir)
        sleep(0.1)
    end
end
read_and_plot()

function plot_jobfit(jobdir; nplots = 4)
    res = MAT.matread(joinpath(jobdir, "IterationsResults.mat"))
    isort = sortperm(res["AICc"])
    s(x) = string(round(x; sigdigits = 4))
    plot(
        plot(;title = basename(jobdir), grid = :none, ticks = :none, showaxis = false),
        plot(map(isort[begin:min(nplots,end)]) do i
            xdata, ydata, fdata = vec.((res["alpha_range"][i,:], res["dR2_Data"][i,:], res["dR2"][i,:]))
            title = "AICc = $(s(res["AICc"][i])), CA = $(s(res["CA"][i])), Rmajor = $(s(res["Rmajor"][i]))\naBVF = $(s(100 * res["aBVF"][i])), iBVF = $(s(100 * res["iBVF"][i])), BVF = $(s(100 * (res["aBVF"][i] + res["iBVF"][i])))"
            plot(
                xdata, [ydata fdata];
                xlab = L"$\alpha$ [degrees]", ylab = L"$\Delta R_2^*$ [Hz]", lab = ["Data" "Fit"], title = title, leg = :bottomright,
                lw = 2, titlefontsize = 10, labelfontsize = 10, legendfontsize = 10,
            )
            end...),
        layout = @layout([a{0.01h}; b{0.99h}]),
    ) |> saveplot(joinpath(sweepdir(), "ModelFits." * basename(jobdir))) |> display
end
# plot_jobfit.(readdir(glob"Job-34_*", sweepdir()))

function plot_sweepsummary(jobdirs = jobdirs())
    job_outputs = map(collect(sort(jobdirs))) do (j,jobdir)
        resfile = joinpath(jobdir, "IterationsResults.mat")
        sweepfile = joinpath(jobdir, "SweepSettings.mat")
        if isfile(resfile) && isfile(sweepfile)
            (results = matread(resfile), params = matread(sweepfile))
        else
            nothing
        end
    end
    job_outputs = filter(!isnothing, job_outputs)
    job_outputs = sort(job_outputs; by = o -> (o.params["Dtissue"], o.params["PVSvolume"], Int(o.params["Nmajor"])))
    results, params = unzip(job_outputs)

    Dtissue_all, PVSvolume_all, Nmajor_all = (p -> p["Dtissue"]).(params), (p -> p["PVSvolume"]).(params), (p -> Int(p["Nmajor"])).(params)
    Dtissue, PVSvolume, Nmajor = sort(unique(Dtissue_all)), sort(unique(PVSvolume_all)), sort(unique(Nmajor_all))
    data = AxisArray(fill(NaN, length.((Dtissue, PVSvolume, Nmajor))); Dtissue, PVSvolume, Nmajor)
    for (ps, rs) in zip(params, results)
        data[atvalue.((ps["Dtissue"], ps["PVSvolume"], ps["Nmajor"]))...] = minimum(rs["AICc"])
    end

    _get(d::Dict, ks) = getindex.(Ref(d), ks)
    allcolors = Dict(0.0 => :blue,   0.5 => :orange,    1.0 => :purple,    2.0 => :red,    3.0 => :green)
    allshapes = Dict(0.0 => :circle, 0.5 => :utriangle, 1.0 => :dtriangle, 2.0 => :square, 3.0 => :diamond)
    allstyles = Dict(0.0 => :solid,  0.5 => :dashdot,   1.0 => :dash,      2.0 => :dot,    3.0 => :dot)

    plot(
        sticks((r -> minimum(r["AICc"])).(results); xlab = "Simulation #", ylab = "AICc", title = "All Simulations", leg = :none, line = (2, _get(allcolors, Dtissue_all), _get(allstyles, PVSvolume_all)), marker = (3,:square,:black)),
        plot(PVSvolume, nanreduce(minimum, data; dims = :Nmajor)'; xlab = "PVSvolume", ylab = "AICc", title = "Minimum over Nmajor",    label = ("Dtissue = " .* string.(Dtissue')),     line = (2, :dash, _get(allcolors, Dtissue')),   marker = (5, _get(allshapes, Dtissue'), _get(allcolors, Dtissue'))),
        plot(Nmajor, nanreduce(minimum, data; dims = :Dtissue)';   xlab = "Nmajor",    ylab = "AICc", title = "Minimum over Dtissue",   label = ("PVSvolume = " .* string.(PVSvolume')), line = (2, :dash, _get(allcolors, PVSvolume')), marker = (5, _get(allshapes, PVSvolume'), _get(allcolors, PVSvolume'))),
        plot(Nmajor, nanreduce(minimum, data; dims = :PVSvolume)'; xlab = "Nmajor",    ylab = "AICc", title = "Minimum over PVSvolume", label = ("Dtissue = " .* string.(Dtissue')),     line = (2, :dash, _get(allcolors, Dtissue')),   marker = (5, _get(allshapes, Dtissue'), _get(allcolors, Dtissue'))),
    ) |> saveplot(joinpath(sweepdir(), "PerfOrientResultsSummary")) |> display
end
plot_sweepsummary()

# let _sweepdir = sweepdir()
#     jobdirs = glob(glob"Job*", _sweepdir) |> dirs -> Dict(jobnum.(dirs) .=> dirs)
#     for (j,dir) in sort(jobdirs)
#         (j > 48) && break
#         @eval jobdir() = $dir
#         @info basename(dir)
#         cleanup()
#     end
# end

# let
#     for (j,dir) in sort(jobdirs)
#         if isempty(readdir(glob"*-worker-*", dir)) #isfile(joinpath(dir, "AllIterationsResults.zip")) && isfile(joinpath(dir, "AllIterationsResults.mat"))
#             @info j, dir
#         end
#     end
# end

# let _dir = sweepdir()
#     for j in sort([37,38])
#         flds = readdir(Glob.GlobMatch("Job-$(j)_*/*-worker-*"), _dir) .|> basename
#         times = (flds .|> t -> DateTime(t[1:25], "yyyy-mm-dd-T-HH-MM-SS-sss")) |> sort
#         @show j, times[end]
#     end
# end

nothing
