# ============================================================================ #
# MWFLearning
# ============================================================================ #

module MWFLearning

using Reexport
@reexport using LinearAlgebra, Statistics, StatsBase
@reexport using Flux

import DrWatson, BSON, TOML, Dates
using DrWatson: @dict, @ntuple
using Parameters: @unpack

export DrWatson, BSON, TOML, Dates
export @dict, @ntuple, @unpack

export prepare_data, get_model
export normalize_signal, project_onto_exp
export log10range

include("src/utils.jl")
include("src/models.jl")

end # module MWFLearning