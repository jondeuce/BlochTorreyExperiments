using CairoMakie, DataFrames, MAT, UnPack, Glob
set_theme!(
    theme_ggplot2();
    resolution = (800,600),
    font = "CMU Serif",
    fontsize = 24,
    linewidth = 3,
)

function load_results(dir)
    results_files = readdir(glob"**/*.mat", dir)
    results_matfiles = MAT.matread.(results_files)
    results_tuples = map(results_matfiles) do res
        @unpack Time, Signal, CPMGArgs, GammaArgs = res
        @unpack TE, nTE, dt, D_Blood, D_Tissue, D_VRS, VRSRelativeRad = CPMGArgs
        @unpack Angle_Deg, B0, Y, Ya = GammaArgs
        (;
            TE          = 1000 * TE |> Float64,
            nTE         = nTE |> Int,
            dt          = 1000 * dt |> Float64,
            Angle_Deg   = Angle_Deg |> Float64,
            Time        = 1000 .* vec(Time) |> Vector{Float64},
            Signal      = vec(Signal) ./ abs(Signal[1]) |> Vector{ComplexF64},
        )
    end
    return DataFrame(results_tuples)
end

function plot_signals(results)
    TEs = unique(sort(results.TE))
    TE_cmap = Dict(TEs .=> cgrad(:roma, length(TEs), categorical = true))
    for df in groupby(results, :Angle_Deg)
        fig = Figure()
        ax = fig[1, 1] = Axis(fig;
            yscale = log10, xlabel = "Time [ms]", ylabel = "Signal [a.u.]", title = "Field Angle = $(first(df.Angle_Deg)) deg"
        )
        for row in eachrow(df)
            lines!(row.Time, abs.(row.Signal); color = TE_cmap[row.TE], label = "TE = $(row.TE)")
        end
        fig[1,2] = Legend(fig, ax, "Echo Time [ms]")
        save("CPMG_Angle_$(first(df.Angle_Deg)).png", fig)
        display(fig)
    end

    Angles = unique(sort(results.Angle_Deg))
    Angle_cmap = Dict(Angles .=> cgrad(:roma, length(Angles), categorical = true))
    for df in groupby(results, :TE)
        fig = Figure()
        ax = fig[1, 1] = Axis(fig;
            yscale = log10, xlabel = "Time [ms]", ylabel = "Signal [a.u.]", title = "Echo Time = $(first(df.TE)) ms"
        )
        for row in eachrow(df)
            lines!(row.Time, abs.(row.Signal); color = Angle_cmap[row.Angle_Deg], label = "θ = $(row.Angle_Deg)")
        end
        fig[1,2] = Legend(fig, ax, "Field Angle [deg]")
        save("CPMG_TE_$(first(df.TE)).png", fig)
        display(fig)
    end
end
