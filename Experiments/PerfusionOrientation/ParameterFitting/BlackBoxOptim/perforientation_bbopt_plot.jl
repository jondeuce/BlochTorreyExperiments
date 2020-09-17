import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Dates, Glob, MAT, BSON, MATLAB, BlackBoxOptim, Plots, LaTeXStrings, AxisArrays, NaNMath

homedir() = "/project/st-arausch-1/jcd1994/code"
sweepdir() = "/project/st-arausch-1/jcd1994/GRE-DSC-PVS-Orientation2020Fall/2020-09-11"
saveplot(name) = p -> (foreach(ext -> savefig(p, name * ext), [".png", ".pdf"]); return p)

jobnum(jobdir) = (s = basename(jobdir); parse(Int, s[6] == '_' ? s[5:5] : s[5:6]))
jobdirs = glob(glob"Job*", sweepdir()) |> dirs -> Dict(jobnum.(dirs) .=> dirs)

mxcall(:addpath, 0, homedir())
mxcall(:addpath, 0, mxcall(:btpathdef, 1))

rowvcat(xs) = reduce(vcat, permutedims.(vec.(xs)))
getresult(r, f) = r["Results"][f]
getresult(r, f1, f2) = r["Results"][f1][f2]
getresults(rs, fs...) = getresult.(rs, fs...) |> x -> eltype(x) <: AbstractArray ? rowvcat(x) : x
lossfun(ydata, ymodel, weights, normfun) = mxcall(:perforientation_objfun, 1, zeros(1,3), zeros(size(ydata)...), ydata, ymodel, weights, normfun)
lossfun(rs, normfun) = lossfun(getresults(rs, "dR2_Data"), getresults(rs, "dR2"), getresults(rs, "args", "Weights"), normfun)

_nanreduce(f, A, ::Colon) = filter(!isnan, A) |> x -> isempty(x) ? eltype(A)(NaN) : f(x)
_nanreduce(f, A, dims) = mapslices(a->_nanreduce(f,a,:), A, dims=dims)
nanreduce(f, A; dims=:) = dropdims(_nanreduce(f, A, dims); dims=dims)
function nanreduce(f, A::AxisArray; dims::Union{Symbol,NTuple{N,Symbol} where N})
    if dims isa Symbol; dims = (dims,); end
    ax = filter(ax -> AxisArrays.axisname(ax) ∉ dims, AxisArrays.axes(A))
    dims = map(d -> axisdim(A, Axis{d}), dims)
    AxisArray(nanreduce(f, convert(Array, A); dims = dims), ax)
end

function unzip(x::AbstractArray{<:Tuple})
    ys = map(xj -> zeros(typeof(xj), size(x)), x[1])
    for I in eachindex(x), j in 1:length(ys)
        ys[j][I] = x[I][j]
    end
    return ys
end

function read_and_plot(jobdirs)
    for (j,jobdir) in sort(jobdirs)
        @time try
            @info "Starting... $(basename(jobdir))"

            if isfile(joinpath(jobdir, "BBOptWorkspace.mat"))
                if isfile(joinpath(jobdir, "done.txt"))
                    @info "Done"
                    continue
                else
                    if isfile(joinpath(jobdir, "BBOptResults.mat"))
                        @info "Finished"
                        touch(joinpath(jobdir, "done.txt"))
                    else
                        @info "Running"
                    end
                end
            else
                if isfile(joinpath(jobdir, "PBS-error.txt"))
                    @info "---- Re-run! ----"
                else
                    @info "Queueing"
                end
                continue
            end

            cd(jobdir)
            resfiles = readdir(glob"2020**/*.mat", jobdir)
            results = matread.(resfiles)
            times = resfiles .|> dirname .|> basename .|> s -> DateTime(s[1:25], "yyyy-mm-dd-T-HH-MM-SS-sss")
            losses = lossfun(results, "AICc")
            
            (length(resfiles) < 5) && continue
            s(x) = string(round(x; sigdigits = 4))

            plot(
                plot(;title = basename(jobdir), grid = :none, ticks = :none, showaxis = false), #titlefontsize = 12
                plot(
                    losses |> y -> scatter(y; ylabel = "AICc", c = :blue, lab = "min = $(s(minimum(y)))\ncurr = $(s(y[end]))"),
                    (r -> r["Results"]["CA"]).(results) |> y -> scatter(y; ylabel = "CA", c = :red, lab = "curr = $(s(y[end]))"),
                    (r -> 100 * r["Results"]["iBVF"]).(results) |> y -> scatter(y; ylabel = "iBVF", c = :green, lab = "curr = $(s(y[end]))"),
                    (r -> 100 * r["Results"]["aBVF"]).(results) |> y -> scatter(y; ylabel = "aBVF", c = :purple, lab = "curr = $(s(y[end]))");
                ),
                layout = @layout([a{0.01h}; b{0.99h}]), size = (1600,900),
            ) |> saveplot("Losses") |> display

            plot(
                plot(;title = basename(jobdir), grid = :none, ticks = :none, showaxis = false), #titlefontsize = 12
                plot(map(sortperm(losses)[1:min(12,end)]) do i
                res = results[i]["Results"]
                xdata, ydata, fdata = vec.((res["alpha_range"], res["dR2_Data"], res["dR2"]))
                title = "AICc = $(s(losses[i])), CA = $(s(res["CA"]))\naBVF = $(s(100 * res["aBVF"])), iBVF = $(s(100 * res["iBVF"])), BVF = $(s(100 * (res["aBVF"] + res["iBVF"])))"
                plot(
                    xdata, [ydata fdata];
                    xlab = L"$\alpha$ [degrees]", ylab = L"$\Delta R_2^*$ [Hz]", lab = ["Data" "Fit"], title = title, leg = :bottomright,
                    lw = 2, titlefontsize = 10, labelfontsize = 10, legendfontsize = 10,
                )
                end...),
                layout = @layout([a{0.01h}; b{0.99h}]), size = (1600,900),
            ) |> saveplot("Fits") |> display

            sleep(0.1)
        finally
            cd(sweepdir())
        end
    end
end
pyplot(size = (1600,900))
read_and_plot(jobdirs)

function save_iterations(jobdirs; force = false)
    for (j,jobdir) in sort(jobdirs)
        @info "Starting... $(basename(jobdir))"
        if !force && isfile(joinpath(jobdir, "done.txt"))
            continue
        end

        @time try
            cd(jobdir)
            resfiles = readdir(glob"2020**/*.mat", jobdir)
            results = matread.(resfiles)

            d = Dict{String,Any}()
            d["timestamp"] = resfiles .|> dirname .|> basename .|> s -> DateTime(s[1:25], "yyyy-mm-dd-T-HH-MM-SS-sss") .|> string

            for f in ["iBVF", "aBVF", "CA", "alpha_range", "dR2", "dR2_Data"]
                d[f] = getresults(results, f)
            end
            ps = getresults(results, "params")
            d["Rmajor"] = ps[:,2]
            d["MinorExpansion"] = ps[:,3]
            d["BVF"] = d["iBVF"] + d["aBVF"]

            for ℓ in ["L2W", "R2w", "AICc"]
                d[ℓ] = lossfun(results, ℓ)
            end

            # display(results[1]["Results"])
            # display(results[1]["Results"]["args"])
            # display(d)

            MAT.matwrite(joinpath(jobdir, "IterationsResults.mat"), deepcopy(d))
        finally
            cd(sweepdir())
        end
    end
end
save_iterations(jobdirs)

function plot_jobfit(jobdir; nplots = 4)
    res = MAT.matread(joinpath(jobdir, "IterationsResults.mat"))
    isort = sortperm(res["AICc"])
    s(x) = string(round(x; sigdigits = 4))
    plot(
        plot(;title = basename(jobdir), grid = :none, ticks = :none, showaxis = false), #titlefontsize = 12
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
# pyplot(size = (800,600))
# plot_jobfit.(readdir(glob"Job-40_*", sweepdir()))

function plot_sweepsummary(jobdirs)
    jobnums, resfiles, sweepfiles = Int[], String[], String[]
    for (j,jobdir) in sort(jobdirs)
        resfile = joinpath(jobdir, "IterationsResults.mat")
        sweepfile = joinpath(jobdir, "SweepSettings.mat")
        if isfile(resfile) && isfile(sweepfile)
            push!(jobnums, j)
            push!(resfiles, resfile)
            push!(sweepfiles, sweepfile)
        end
    end
    results = matread.(resfiles)
    params = matread.(sweepfiles)
    isort = sortperm(params; by = p -> (p["Dtissue"], p["PVSvolume"], Int(p["Nmajor"])))
    results, params = results[isort], params[isort]

    Dtissue_all, PVSvolume_all, Nmajor_all = (p -> p["Dtissue"]).(params), (p -> p["PVSvolume"]).(params), (p -> Int(p["Nmajor"])).(params)
    Dtissue, PVSvolume, Nmajor = sort(unique(Dtissue_all)), sort(unique(PVSvolume_all)), sort(unique(Nmajor_all))
    data = AxisArray(fill(NaN, length.((Dtissue, PVSvolume, Nmajor))); Dtissue, PVSvolume, Nmajor)
    for (ps, rs) in zip(params, results)
        data[atvalue.((ps["Dtissue"], ps["PVSvolume"], ps["Nmajor"]))...] = minimum(rs["AICc"])
    end
    Dcolors = Dict(0.0 => :blue, 2.0 => :red, 3.0 => :green)
    PVScolors = Dict(0.0 => :blue, 1.0 => :red, 3.0 => :green)
    PVSshapes = Dict(0.0 => :circle, 1.0 => :square, 3.0 => :diamond)
    PVSstyle = Dict(0.0 => :solid, 1.0 => :dash, 3.0 => :dot)

    plot(
        sticks((r -> minimum(r["AICc"])).(results); xlab = "Simulation #", ylab = "AICc", title = "All Simulations", leg = :none, lw = 1.5, lc = [Dcolors[D] for D in Dtissue_all], linestyle = [PVSstyle[P] for P in PVSvolume_all], marker = (2,:square,:black)), #markersize = 5, markershape = [PVSshapes[P] for P in PVSvolume_all], markercolor = [PVScolors[P] for P in PVSvolume_all]),
        plot(PVSvolume, nanreduce(minimum, data; dims = :Nmajor)'; xlab = "PVSvolume", ylab = "AICc", title = "Minimum over Nmajor", label = ("Dtissue = " .* string.(Dtissue')), markersize = 5, markershape = [:circle :square :diamond], line = (2,:dash)),
        plot(Nmajor, nanreduce(minimum, data; dims = :Dtissue)'; xlab = "Nmajor", ylab = "AICc", title = "Minimum over Dtissue", label = ("PVSvolume = " .* string.(PVSvolume')), markersize = 5, markershape = [:circle :square :diamond], line = (2,:dash)),
        plot(Nmajor, nanreduce(minimum, data; dims = :PVSvolume)'; xlab = "Nmajor", ylab = "AICc", title = "Minimum over PVSvolume", label = ("Dtissue = " .* string.(Dtissue')), markersize = 5, markershape = [:circle :square :diamond], line = (2,:dash)),
    ) |> saveplot(joinpath(sweepdir(), "PerfOrientResultsSummary")) |> display
end
# pyplot(size = (800,600))
# plot_sweepsummary(jobdirs)

# maybeconvergedjobs = [45]
# nonconvergedjobs = [24; 30]
# redojobs = [33]
# donejobs = [13; 18; 20; 35; 37; 46; 49; 52]
# jobpids = Dict(18 => 679847, 20 => 679849, 24 => 679851, 30 => 679852, 13 => 680764, 49 => 683665)
# for (j,pid) in jobpids
#     if j ∈ donejobs
#         touch(joinpath(jobdirs[j], "done.txt"))
#         @show run(Cmd(["qdel", string(pid)]))
#     end
# end

# for j in [21,26,27]
#     touch(joinpath(jobdirs[j], "done.txt"))
# end

# for (j,pid) in jobpids
#     if j >= 55
#         try
#             @show run(Cmd(["qdel", string(pid)]))
#         catch
#         end
#     end
# end

# for sweepsetdir in joinpath.(sweepdir(), ["2020-09-11", "2020-09-12"])
#     mergedir = joinpath(sweepdir(), "merged")
#     for dir in readdir(glob"Job*", sweepsetdir)
#         @show dir
#         @time for iterdir in readdir(glob"2020-09-*-worker-*", dir)
#             rm(iterdir, recursive=true, force=true)
#         end
#     end
# end

# let
#     sweepsetdir = sweepdir()
#     mergedir = joinpath(sweepsetdir[1:end-length(basename(sweepsetdir))], "merged")
#     for dir in sort(readdir(glob"Job*", sweepsetdir); by = jobnum)
#         newdir = joinpath(mergedir, "Job-$(jobnum(dir)+54)" * basename(dir)[end-34:end])
#         @show dir
#         @show newdir
#         mv(dir, newdir)
#     end
# end
