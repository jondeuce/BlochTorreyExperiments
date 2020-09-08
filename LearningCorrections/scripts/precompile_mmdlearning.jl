import Reexport
import SpecialFunctions
import Distributions
import DataFrames
import BenchmarkTools
import StatsPlots
import LoopVectorization
import Tullio
import BSON
import DrWatson
import Flux
import NNlib
import Zygote
import BlackBoxOptim
import Optim
import ForwardDiff

StatsPlots.pyplot(size=(800,600))

# Plots
let
    p = StatsPlots.plot(rand(5), rand(5))
    display(p)
end

# Zygote
let
    f(x) = sum(abs2, x)
    Zygote.gradient(f, rand(10))
end

