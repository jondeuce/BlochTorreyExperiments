function [tmin, tmax, Lmin, Lmax] = raySphereIntersection( Origins, Directions, SphereRadius, SphereCenter )
%RAYBOXINTERSECTION [tmin, tmax, Lmin, Lmax, Pmin, Pmax] = raySphereIntersection( Origins, Directions, SphereRadius, SphereCenter )

[tmin, tmax, Lmin, Lmax] = raySphereIntersection_3D( Origins, Directions, SphereRadius, SphereCenter );

end

function [tmin,tmax,Lmin,Lmax] = raySphereIntersection_3D( p, v, R, C )
% https://www.wikiwand.com/en/Line%E2%80%93sphere_intersection

oc = bsxfun(@minus, p, C); % o - c
d0 = -sum(v .* oc, 1); % -dot(l, o - c)
dd = sqrt(d0.^2 - (sum(oc.^2, 1) - R^2));
tmin = d0 - dd;
tmax = d0 + dd;

Lmin = bsxfun( @plus, p, bsxfun( @times, tmin, v ) );
Lmax = bsxfun( @plus, p, bsxfun( @times, tmax, v ) );

end
