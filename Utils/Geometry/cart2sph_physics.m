function [ th, phi, r ] = cart2sph_physics( x, y, z )
%CART2SPH_PHYSICS MATLAB's cart2sph function, but shifted to use the
%physicists convention:
%   th:  in [ 0,  pi]; angle from positive z-axis, aka polar angle
%   phi: in [-pi, pi]; angle from positive x-axis, aka azimuthal angle
%   r:   in [ 0, inf]; radial distance from the origin
% 
% Relation to cartesian coordinates:
%   x = r .* cos(phi) .* sin(th);
%   y = r .* sin(phi) .* sin(th);
%   z = r .* cos(th);

% note that phi, th output are swapped from documentation
[phi,th,r] = cart2sph(x,y,z);

% shift th from [-pi/2,pi/2] -> [0,pi]
th = pi/2 - th;

end

