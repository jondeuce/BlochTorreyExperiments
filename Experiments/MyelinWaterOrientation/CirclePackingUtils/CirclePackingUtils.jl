module CirclePackingUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using Reexport
@reexport using GeometryUtils

using LinearAlgebra, Statistics, StatsFuns
using DiffResults, Optim, LineSearches, ForwardDiff, Roots

export estimate_density, opt_subdomain, scale_to_density, covariance_energy
export tocircles, tocircles!, tovectors, tovectors!, initialize_origins
export periodic_diff, periodic_mod
export periodic_circles, periodic_unique_circles, periodic_density
export periodic_scale_to_threshold, periodic_scale_to_density, periodic_subdomain
export pairwise_sum, pairwise_grad!, pairwise_hess!
export wrap_gradient, check_density_callback

export GreedyCirclePacking, EnergyCirclePacking, PeriodicCirclePacking

include("src/utils.jl")
include("src/pairwise_gradient.jl")
include("src/density_estimation.jl")
include("src/periodic_tools.jl")
include("src/GreedyCirclePacking.jl")
include("src/EnergyCirclePacking.jl")
include("src/PeriodicCirclePacking.jl")

end # module CirclePackingUtils