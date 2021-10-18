function [ L ] = rayBoxIntersectionLength( Origins, Directions, BoxDims, BoxCenter )
%RAYBOXINTERSECTIONLENGTH 

[tmin, tmax] = rayBoxIntersection( Origins, Directions, BoxDims, BoxCenter );
L = tmax - tmin;

end











