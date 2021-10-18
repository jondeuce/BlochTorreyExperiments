using StatsPlots
using Plots
cd(@__DIR__)

for name in names(StatsPlots)
    if !Base.isexported(Plots, name) && ! Base.isexported(StatsPlots, name)
        error("not Plots or StatsPlots: $name")
    else
        mod = Base.binding_module(StatsPlots, name)
    end
    try
        ripgrep = readchomp(`rg $name --glob '*.jl' --word-regexp`)
        open("names-$(mod).txt"; append = true) do io
            println(io, "---- $name ----\n")
            println(io, ripgrep)
            println(io, "\n")
        end
    catch e
    end
end
