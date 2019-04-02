# MWI Model fitting types
abstract type AbstractMWIFittingModel end

@with_kw struct TwoPoolMagnToMagn <: AbstractMWIFittingModel
    TE::Float64       = 10e-3
    nTE::Int          = 32
    fitmethod::Symbol = :local
end

@with_kw struct ThreePoolMagnToMagn <: AbstractMWIFittingModel
    TE::Float64       = 10e-3
    nTE::Int          = 32
    fitmethod::Symbol = :local
end

@with_kw struct ThreePoolCplxToMagn <: AbstractMWIFittingModel
    TE::Float64       = 10e-3
    nTE::Int          = 32
    fitmethod::Symbol = :local
end

@with_kw struct ThreePoolCplxToCplx <: AbstractMWIFittingModel
    TE::Float64       = 10e-3
    nTE::Int          = 32
    fitmethod::Symbol = :local
end

@with_kw struct NNLSRegression <: AbstractMWIFittingModel
    TE::Float64              = 10e-3            # Measurement spacing
    nTE::Int                 = 32               # Number of echos; signals are assumed to be sampled at times t = TE.*(0:nTE)
    nT2::Int                 = 120              # Number of T2 bins used during fitting process, spaced logarithmically in `T2Range`
    Threshold::Float64       = 0.0              # First echo intensity cutoff for empty voxels
    RefConAngle::Float64     = 180.0            # Refocusing Pulse Control Angle (TODO: check value from scanner; old default is 165.0)
    T2Range::Vector{Float64} = [8e-3, 2.0]      # Min and Max T2 values used during fitting (typical for in-vivo)
    SPWin::Vector{Float64}   = [8e-3, 25e-3]    # short peak window (typically 1.5X echospacing to 40ms)
    MPWin::Vector{Float64}   = [25e-3, 200e-3]  # middle peak window
end

const TwoPoolMagnData = TwoPoolMagnToMagn
const TwoPoolModel = TwoPoolMagnToMagn
const ThreePoolMagnData = Union{ThreePoolMagnToMagn, ThreePoolCplxToMagn}
const ThreePoolModel = Union{ThreePoolMagnToMagn, ThreePoolCplxToMagn, ThreePoolCplxToCplx}

get_tspan(opts::AbstractMWIFittingModel) = opts.TE .* (0, opts.nTE)
get_tpoints(opts::AbstractMWIFittingModel) = opts.TE .* (0:opts.nTE)
get_T2vals(opts::NNLSRegression) = exp10.(range(log10(opts.T2Range[1]), stop=log10(opts.T2Range[2]), length=opts.nT2))

# Blank dictionary for storing results
blank_results_dict() = Dict{Symbol,Any}(
        :geom             => [],
        :params           => [],
        :myelinprobs      => [],
        :myelinsubdomains => [],
        :myelindomains    => [],
        :omegas           => [],
        :sols             => [],
        :signals          => [],
        :mwfvalues        => []
    )