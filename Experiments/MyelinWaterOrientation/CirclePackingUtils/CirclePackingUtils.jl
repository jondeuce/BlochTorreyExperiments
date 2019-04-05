module CirclePackingUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using GeometryUtils
using LinearAlgebra, Statistics
using DiffResults, Optim, LineSearches, ForwardDiff, Roots
using Tensors

export estimate_density, opt_subdomain, scale_to_density, covariance_energy
export tocircles, tocircles!, tovectors, tovectors!, initialize_origins
export pairwise_sum, pairwise_grad!, pairwise_hess!
export wrap_gradient, check_density_callback

export GreedyCirclePacking
export EnergyCirclePacking

include("src/utils.jl")
include("src/GreedyCirclePacking.jl")
include("src/EnergyCirclePacking.jl")

end # module CirclePackingUtils