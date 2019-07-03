# ---------------------------------------------------------------------------- #
# BlochTorreyParameters methods
# ---------------------------------------------------------------------------- #

# By default, truncate the radii distribution (of Gamma type) so as to prevent the
# generated radii being too large/small for the purpose of mesh generation
# With the default lower = -1.5 (in units of sigma) and upper = 2.5 and the default
# BlochTorreyParameters, this corresponds to rmean/rmin ~ 2.7 and rmax/rmin ~ 2.0,
# and captures ~95% of the pdf mass.
function radiidistribution(p::BlochTorreyParameters, lower = -1.5, upper = 2.5)
    k, θ = p.R_shape, p.R_scale # shape and scale parameters
    d = Distributions.Gamma(k, θ)
    if !(lower == -Inf && upper == Inf)
        μ, σ = k*θ, √k*θ
        d = Distributions.Truncated(d, μ + lower*σ, μ + upper*σ)
    end
    return d
end
@inline radiidistribution(p::MyelinProblem) = radiidistribution(p.params)

# Functions for computing each of mwf, g_ratio, and packing density in terms of
# the other two parameters, assuming periodic circle packing and constant g_ratio
periodic_mwf(;density = 0.7, g_ratio = 0.8) = density * (1 - g_ratio^2)
periodic_g_ratio(;density = 0.7, mwf = 0.252) = √(1 - mwf / density)
periodic_packdensity(;mwf = 0.252, g_ratio = 0.8) = mwf / (1 - g_ratio^2)
