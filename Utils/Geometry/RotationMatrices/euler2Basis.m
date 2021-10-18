function R = euler2Basis( th, phi, psi )
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

u = [ -sin(th); cos(th); 0 ];
k = [ 0; 0; 1 ];
R = axisAngle2Rot(k, psi) * axisAngle2Rot(u, phi);

end