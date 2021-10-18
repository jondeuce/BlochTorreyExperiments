function [ L ] = raySphereIntersectionLength( Origins, Directions, SphereRadius, SphereCenter )
%RAYBOXINTERSECTIONLENGTH raySphereIntersectionLength( Origins, Directions, SphereRadius, SphereCenter )

[tmin, tmax, ~, ~] = raySphereIntersection( Origins, Directions, SphereRadius, SphereCenter );
L = tmax - tmin;

end
