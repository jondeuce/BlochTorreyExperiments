# Copyright (C) 2004-2012 Per-Olof Persson. See COPYRIGHT.TXT for details.

# ---------------------------------------------------------------------------- #
# Distance functions
# ---------------------------------------------------------------------------- #

# Distance field for a block with bounds [x1, x2] x [y1, y2] x [z1, z2]
@inline function dblock(p::Vec{3}, x1, x2, y1, y2, z1, z2)
    # @inbounds d = -min(-z1 + p[3], z2 - p[3], -y1 + p[2], y2 - p[2], -x1 + p[1], x2 - p[1])
    @inbounds d = max(x1 - p[1], p[1] - x2, y1 - p[2], p[2] - y2, z1 - p[3], p[3] - z2) #equivalent
    return d
end

# Distance field for a block with lower bounds p1/upper bounds p2
@inline dblock(p::Vec{3}, p1::Vec{3}, p2::Vec{3}) = dblock(p, p1[1], p2[1], p1[2], p2[2], p1[3], p2[3])

# Distance field for a rectangle with bounds [x1, x2] x [y1, y2]
@inline function drectangle(p::Vec{2}, x1, x2, y1, y2)
    # @inbounds d = -min(-y1 + p[2], y2 - p[2], -x1 + p[1], x2 - p[1])
    @inbounds d = max(x1 - p[1], p[1] - x2, y1 - p[2], p[2] - y2) #equivalent
    return d
end

# Distance field for a rectangle with lower bounds p1/upper bounds p2
@inline drectangle(p::Vec{2}, p1::Vec{2}, p2::Vec{2}) = drectangle(p, p1[1], p2[1], p1[2], p2[2])

# Distance field for a rectangle with bounds [x1, x2] x [y1, y2]; more accurate
# version which accounts for corners
@inline function drectangle0(p::Vec{2}, x1, x2, y1, y2)
    d1 = y1 - p[2] # distance from bottom edge
    d2 = p[2] - y2 # distance from top edge
    d3 = x1 - p[1] # distance from left edge
    d4 = p[1] - x2 # distance from right edge
    
    # Most common case: all distances are negative, point is inside rectangle
    dmax = max(d1, d2, d3, d4)
    if dmax <= 0
        return dmax
    end

    d = if d1 > 0 && d3 > 0
        sqrt(d1^2 + d3^2) # bottom-left corner
    elseif d1 > 0 && d4 > 0
        sqrt(d1^2 + d4^2) # bottom-right corner
    elseif d2 > 0 && d3 > 0
        sqrt(d2^2 + d3^2) # top-left corner
    elseif d2 > 0 && d4 > 0
        sqrt(d2^2 + d4^2) # top-right corner
    else
        dmax # one of four remaining sides (note that in this case, exactly one of d1,...,d4 is positive, and so d == dmax)
    end

    return d
end
@inline drectangle0(p::Vec{2}, p1::Vec{2}, p2::Vec{2}) = drectangle0(p, p1[1], p2[1], p1[2], p2[2])

# Distance field for circle/sphere centered at p0 with radius r, or with p0 omitted
@inline dsphere(p::Vec, r) = norm(p) - r
@inline dsphere(p::Vec, p0::Vec, r) = norm(p - p0) - r
@inline dsphere(p::Vec{3}, xc, yc, zc, r) = dsphere(p, Vec{3}((xc, yc, zc)), r)

@inline dcircle(p::Vec, r) = dsphere(p, r)
@inline dcircle(p::Vec, p0::Vec, r) = dsphere(p, p0, r)
@inline dcircle(p::Vec{2}, xc, yc, r) = dsphere(p, Vec{2}((xc, yc)), r)

# ---------------------------------------------------------------------------- #
# Distance union, intersection, etc. functions
# ---------------------------------------------------------------------------- #

# Signed distance to the region formed by removing from region 1 its
# intersection with region 2.
@inline ddiff(d1, d2) = max(d1, -d2)

# Signed distance to the region formed by removing from region `din` the union
# of the regions `dout`
@inline ddiff(din, dout...) = -min(-din, dout...)  #NOTE: max(x, -y) == -min(-x, y)

# Signed distance to the region formed by the intersection of the two regions
@inline dintersect() = -Inf # value s.t. dintersect(-Inf, x) == x for all x
@inline dintersect(ds...) = max(ds...)

# Signed distance to the region formed by the union of the two regions
@inline dunion() = Inf # value s.t. dunion(Inf, x) == x for all x
@inline dunion(ds...) = min(ds...)

# ---------------------------------------------------------------------------- #
# Useful distances derived from the above functions
# ---------------------------------------------------------------------------- #

# Signed distance for a n-dimensional spherical shell with inner radius `a`
# and outer radius `b`
@inline dshell(p::Vec, a, b) = ddiff(dsphere(p, b), dsphere(p, a))
@inline dshell(p::Vec, p0::Vec, a, b) = ddiff(dsphere(p, p0, b), dsphere(p, p0, a))
