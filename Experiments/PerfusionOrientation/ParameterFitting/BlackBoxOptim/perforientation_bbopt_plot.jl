import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Dates, Glob, MAT, BSON, MATLAB, BlackBoxOptim, Plots, LaTeXStrings
pyplot(size = (1600,900))

sweepdir() = "/project/st-arausch-1/jcd1994/GRE-DSC-PVS-Orientation2020Fall/2020-09-11"
saveplot(name) = p -> (display(p); foreach(ext -> savefig(p, name * ext), [".png", ".pdf"]))

jobnum(jobdir) = (s = basename(jobdir); parse(Int, s[6] == '_' ? s[5:5] : s[5:6]))
jobdirs = glob(glob"Job*", sweepdir()) |> dirs -> Dict(jobnum.(dirs) .=> dirs)

for (j,jobdir) in sort(jobdirs)
    try
        @info "Starting... $(basename(jobdir))"

        cd(jobdir)
        !isfile(joinpath(jobdir, "BBOptWorkspace.mat")) && continue
        isfile(joinpath(jobdir, "done.txt")) && continue

        resfiles = readdir(glob"2020**/*.mat", jobdir)
        results = matread.(resfiles)
        diaries = joinpath.(dirname.(resfiles), "Diary.log")
        times = diaries .|> dirname .|> basename .|> s -> DateTime(s[1:25], "yyyy-mm-dd-T-HH-MM-SS-sss")
        losses = diaries .|> f -> readlines(f)[end] .|> l -> parse(Float64, l[8:end])
        s(x) = string(round(x; sigdigits = 4))
        
        (length(resfiles) < 5) && continue

        plot(
            plot(;title = basename(jobdir), grid = :none, ticks = :none, showaxis = false), #titlefontsize = 12
            plot(
                losses |> y -> scatter(y; ylabel = "AICc", c = :blue, lab = "min = $(s(minimum(y)))\ncurr = $(s(y[end]))"),
                (r -> r["Results"]["CA"]).(results) |> y -> scatter(y; ylabel = "CA", c = :red, lab = "curr = $(s(y[end]))"),
                (r -> 100 * r["Results"]["iBVF"]).(results) |> y -> scatter(y; ylabel = "iBVF", c = :green, lab = "curr = $(s(y[end]))"),
                (r -> 100 * r["Results"]["aBVF"]).(results) |> y -> scatter(y; ylabel = "aBVF", c = :purple, lab = "curr = $(s(y[end]))");
            ),
            layout = @layout([a{0.01h}; b{0.99h}]), size = (1600,900),
        ) |> saveplot("Losses")

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
        ) |> saveplot("Fits")

        sleep(1.0)
    finally
        cd(sweepdir())
    end
end
