function [th, phi, psi] = basis2Euler( R )
%SPHERICALCOORDINATES Basis vectors with z vector located at spherical
%angles (th,phi) and x, y vectors rotated as follows:
% 
% R = [ux uy uz] is constructed through rotations of I = [x y z] by three
% angles, th, phi, and psi:
%   -First, we rotate the local z directly to polar angles th and phi
%       -'phi' sets the angle between local z and global z
%       -'th' sets the angle between local z and local x, both projected
%        onto the global xy-plane
%       -'phi' and 'th' are polar coordinates in the body frame
%   -Then, we rotate by psi around global z
%       -'psi' sets the heading
% 
% The resulting matrix is the following:
%   R   =   [ cos(psi)*(cos(phi) - sin(th)^2*(cos(phi) - 1)) - cos(th)*sin(psi)*sin(th)*(cos(phi) - 1), cos(psi)*cos(th)*sin(th)*(cos(phi) - 1) - sin(psi)*(cos(phi) - cos(th)^2*(cos(phi) - 1)), cos(psi)*cos(th)*sin(phi) - sin(phi)*sin(psi)*sin(th)]
%           [ sin(psi)*(cos(phi) - sin(th)^2*(cos(phi) - 1)) + cos(psi)*cos(th)*sin(th)*(cos(phi) - 1), cos(psi)*(cos(phi) - cos(th)^2*(cos(phi) - 1)) + cos(th)*sin(psi)*sin(th)*(cos(phi) - 1), cos(psi)*sin(phi)*sin(th) + cos(th)*sin(phi)*sin(psi)]
%           [                                                                        -cos(th)*sin(phi),                                                                        -sin(phi)*sin(th),                                              cos(phi)]

phi = angle3D( [0;0;1], R(:,3) );
if phi > 1e-3
    th  = mod( atan2( -R(3,2), -R(3,1) ), 2*pi );
    psi = mod( atan2(  R(2,3),  R(1,3) ) - th, 2*pi );
else
    th  = 0;
    psi = mod( atan2(  R(2,1),  R(1,1) ), 2*pi );
end

end