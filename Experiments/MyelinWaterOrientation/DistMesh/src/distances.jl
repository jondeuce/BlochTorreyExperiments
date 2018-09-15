# Copyright (C) 2004-2012 Per-Olof Persson. See COPYRIGHT.TXT for details.

# ---------------------------------------------------------------------------- #
# Distance functions
# ---------------------------------------------------------------------------- #

# Distance field for a block with bounds [x1,x2] x [y1,y2] x [z1,z2]
@inline function dblock(p::Vec{3},x1,x2,y1,y2,z1,z2)
    @inbounds d = -min(min(min(min(min(-z1+p[3],z2-p[3]),-y1+p[2]),y2-p[2]),-x1+p[1]),x2-p[1])
    return d
end

# Distance field for a block with lower bounds p1/upper bounds p2
@inline function dblock(p::Vec{3}, p1::Vec{3}, p2::Vec{3})
    @inbounds d = -min(min(min(min(min(-p1[3]+p[3],p2[3]-p[3]),-p1[2]+p[2]),p2[2]-p[2]),-p1[1]+p[1]),p2[1]-p[1])
    return d
end

# Distance field for a rectangle with bounds [x1,x2] x [y1,y2]
@inline function drectangle(p::Vec{2},x1,x2,y1,y2)
    @inbounds d = -min(min(min(-y1+p[2],y2-p[2]),-x1+p[1]),x2-p[1])
    return d
end

# Distance field for a rectangle with lower bounds p1/upper bounds p2
@inline function drectangle(p::Vec{2}, p1::Vec{2}, p2::Vec{2})
    @inbounds d = -min(min(min(-p1[2]+p[2],p2[2]-p[2]),-p1[1]+p[1]),p2[1]-p[1])
    return d
end

# Distance field for a rectangle with bounds [x1,x2] x [y1,y2]; more accurate
# version which accounts for corners
@inline function drectangle0(p::Vec{2},x1,x2,y1,y2)
    d1=y1-p[2];
    d2=-y2+p[2];
    d3=x1-p[1];
    d4=-x2+p[1];

    d5=sqrt(d1^2+d3^2);
    d6=sqrt(d1^2+d4^2);
    d7=sqrt(d2^2+d3^2);
    d8=sqrt(d2^2+d4^2);

    d=-min(min(min(-d1,-d2),-d3),-d4)


    if d1>0 && d3>0;
        d = d5
    end
    if d1>0 && d4>0;
        d = d6
    end
    if d2>0 && d3>0;
        d = d7
    end
    if d2>0 && d4>0;
        d = d8
    end

    return d
end
@inline drectangle0(p::Vec{2}, p1::Vec{2}, p2::Vec{2}) = drectangle0(p,p1[1],p2[2],p1[1],p2[2])

# Distance field for circle/sphere centered at p0 with radius r
@inline dsphere(p::Vec, p0::Vec, r) = norm(p - p0) - r
@inline dcircle(p::Vec, p0::Vec, r) = dsphere(p, p0, r)
@inline dcircle(p::Vec{2}, xc, yc, r) = dsphere(p, Vec{2}((xc,yc)), r)

# ---------------------------------------------------------------------------- #
# Distance union, intersection, etc. functions
# ---------------------------------------------------------------------------- #

# Signed distance to the region formed by removing from region 1 its
# intersection with region 2.
@inline ddiff(d1, d2) = max(d1, -d2)

# Signed distance to the region formed by the intersection of the two regions
@inline dintersect(d1, d2) = max(d1, d2)

# Signed distance to the region formed by the union of the two regions
@inline dunion(d1, d2) = min(d1, d2)
