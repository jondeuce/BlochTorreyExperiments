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