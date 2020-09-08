using PackageCompiler

pkg_list = [
    :Revise,
    :OhMyREPL,
    :Reexport,
    :SpecialFunctions,
    :Distributions,
    :DataFrames,
    :BenchmarkTools,
    :StatsPlots,
    :LoopVectorization,
    :Tullio,
    :BSON,
    :DrWatson,
    :Flux,
    :NNlib,
    :Zygote,
    :BlackBoxOptim,
    :Optim,
    :ForwardDiff,
]

create_sysimage(
    pkg_list;
    sysimage_path = joinpath(@__DIR__, "sys_mmdlearning.so"),
    precompile_execution_file = joinpath(@__DIR__, "precompile_mmdlearning.jl"),
)

using Pkg
Pkg.pin(string.(pkg_list))
