# ---------------------------------------------------------------------------- #
# Generic geometry utilities for use within the JuAFEM.jl/Tensors.jl framework
# ---------------------------------------------------------------------------- #

# Compute the `skew product` between two 2-dimensional vectors. This is the same
# as computing the third component of the cross product if the vectors `a` and
# `b` were extended to three dimensions
@inline function skewprod(a::Vec{2,T}, b::Vec{2,T}) where T
    #@inbounds v = (a × b)[3] # believe it or not, this is just as fast...
    @inbounds v = a[1]*b[2] - b[1]*a[2]
    return v
end

nothing
