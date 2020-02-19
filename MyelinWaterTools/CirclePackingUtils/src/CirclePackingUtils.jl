module CirclePackingUtils

# ---------------------------------------------------------------------------- #
# Dependencies
# ---------------------------------------------------------------------------- #

using Reexport
@reexport using GeometryUtils

using LinearAlgebra, Random, Statistics, StatsFuns
using DiffResults, Optim, LineSearches, ForwardDiff, Roots
using OrdinaryDiffEq, SteadyStateDiffEq
using Parameters: @unpack

export estimate_density, opt_subdomain, scale_to_density, covariance_energy
export tocircles, tocircles!, tovectors, tovectors!, initialize_origins, initialize_domain
export periodic_diff, periodic_mod
export periodic_circles, periodic_unique_circles, periodic_circle_repeat, periodic_density
export periodic_scale_to_threshold, periodic_scale_to_density, periodic_subdomain
export pairwise_sum, pairwise_grad!, pairwise_hess!
export wrap_gradient, check_density_callback

export GreedyCirclePacking, EnergyCirclePacking, PeriodicCirclePacking, NBodyCirclePacking

include("utils.jl")
include("pairwise_gradient.jl")
include("density_estimation.jl")
include("periodic_tools.jl")
include("GreedyCirclePacking.jl")
include("EnergyCirclePacking.jl")
include("PeriodicCirclePacking.jl")
include("NBodyCirclePacking.jl")

end # module CirclePackingUtils