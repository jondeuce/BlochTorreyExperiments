# MWI Model fitting types
abstract type AbstractMWIFittingModel end
struct NNLSRegression <: AbstractMWIFittingModel end
struct TwoPoolMagnToMagn <: AbstractMWIFittingModel end
struct ThreePoolMagnToMagn <: AbstractMWIFittingModel end
struct ThreePoolCplxToMagn <: AbstractMWIFittingModel end
struct ThreePoolCplxToCplx <: AbstractMWIFittingModel end
const TwoPoolMagnData = TwoPoolMagnToMagn
const TwoPoolModel = TwoPoolMagnToMagn
const ThreePoolMagnData = Union{ThreePoolMagnToMagn, ThreePoolCplxToMagn}
const ThreePoolModel = Union{ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx}

# Struct for saving MWF results
@with_kw struct MWFResults{T}
    metadata::Dict{Symbol,Any}               = Dict()
    params::Vector{BlochTorreyParameters{T}} = []
    sols::Vector{Vector{ODESolution}}        = []
    mwfvalues::Vector{Dict{Symbol,T}}        = []
end

# Standard date format
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")

function CSV.write(results::MWFResults, i)
    curr_date = getnow()
    for (j,sol) in enumerate(results.sols[i])
        fname = curr_date * "__sol_$(i)__region_$(j).csv"
        CSV.write(fname, DataFrame(sol))
    end
    return nothing
end

function CSV.write(results::MWFResults)
    for i in 1:length(results.sols)
        CSV.write(results, i)
    end
    return nothing
end