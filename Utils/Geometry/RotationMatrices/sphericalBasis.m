function R = sphericalBasis( th, phi, psi, type )
%SPHERICALCOORDINATES Basis vectors with z vector located at spherical
%angles (th,phi) and x, y vectors rotated about z by angle psi
%   -'th' in (-pi,pi]
%   -'phi' in [0,pi]
%   -'psi' in (-pi,pi]
% 
% R = [ux uy uz] is constructed through rotations of I = [x y z] by three angles, th phi and psi:
%   -First, we rotate I about y by (-phi) to get R_phi
%       -phi is the unsigned angle between uz and z (standard spherical coordinates)
%   -Second, we rotate R_phi about z by (th) to get R_phi_theta
%       -th is the angle between uz projected onto the x-y plane and x (standard spherical coordinates)
%   -Last, we rotate R_phi_theta about uz by (psi) to get R
%       -psi is the angle about uz in which the other two unit vectors (ux and uy) have been rotated
% 
% The resulting matrix is the following:
%   R   =   [          cos(psi) - cos(psi)*cos(th)^2 - cos(th)*sin(psi)*sin(th) + cos(phi)*cos(psi)*cos(th)^2 + cos(phi)*cos(th)*sin(psi)*sin(th),          cos(th)^2*sin(psi) - sin(psi) - cos(phi)*cos(th)^2*sin(psi) - cos(psi)*cos(th)*sin(th) + cos(phi)*cos(psi)*cos(th)*sin(th), cos(th)*sin(phi)]
%           [ cos(phi)*sin(psi) + cos(th)^2*sin(psi) - cos(phi)*cos(th)^2*sin(psi) - cos(psi)*cos(th)*sin(th) + cos(phi)*cos(psi)*cos(th)*sin(th), cos(phi)*cos(psi) + cos(psi)*cos(th)^2 + cos(th)*sin(psi)*sin(th) - cos(phi)*cos(psi)*cos(th)^2 - cos(phi)*cos(th)*sin(psi)*sin(th), sin(th)*sin(phi)]
%           [                                                                                                             -cos(psi - th)*sin(phi),                                                                                                              sin(psi - th)*sin(phi),         cos(phi)]

if nargin < 4 || ~strcmpi(type, 'FUNCTION')
    R = RotationMatrix(th,phi,psi);
else
    R = @(th,phi,psi) RotationMatrix(th,phi,psi);
end

end

function Rot = RotationMatrix(th,phi,psi)

% minimum phi in radians
phi_thresh = 1e-4 * (pi/180);

if abs(phi) < phi_thresh
    Rot = Rw(th,phi,psi);
else
    Rot = Rw(th,phi,psi) * Rv(th,phi);
end

end

function Rot = Rv(th,phi)

% radial vector/second rotation vector
v = [-sin(phi)*sin(th); cos(th)*sin(phi); 0 ] / sin(phi);
% first rotation matrix
Rot = axisAngle2Rot( v, phi );

% Rot = [ cos(phi) - sin(th)^2*(cos(phi) - 1),    cos(th)*sin(th)*(cos(phi) - 1),         cos(th)*sin(phi);
%         cos(th)*sin(th)*(cos(phi) - 1),         cos(phi) - cos(th)^2*(cos(phi) - 1),    sin(phi)*sin(th);
%        -cos(th)*sin(phi),                      -sin(phi)*sin(th),                       cos(phi)        ];

end

function Rot = Rw(th,phi,psi)

% first rotation vector = cross([0;0;1],w)
w = [ cos(th)*sin(phi); sin(phi)*sin(th); cos(phi) ];
% second rotation matrix
Rot = axisAngle2Rot( w, psi );

% Rot = [ cos(psi) - cos(th)^2*sin(phi)^2*(cos(psi) - 1),                        -cos(phi)*sin(psi) - cos(th)*sin(phi)^2*sin(th)*(cos(psi) - 1),          sin(phi)*sin(psi)*sin(th) - cos(phi)*cos(th)*sin(phi)*(cos(psi) - 1);
%         cos(phi)*sin(psi) - cos(th)*sin(phi)^2*sin(th)*(cos(psi) - 1),          cos(psi) - sin(phi)^2*sin(th)^2*(cos(psi) - 1),                        -cos(th)*sin(phi)*sin(psi) - cos(phi)*sin(phi)*sin(th)*(cos(psi) - 1);
%        -sin(phi)*sin(psi)*sin(th) - cos(phi)*cos(th)*sin(phi)*(cos(psi) - 1),   cos(th)*sin(phi)*sin(psi) - cos(phi)*sin(phi)*sin(th)*(cos(psi) - 1),   cos(psi) - cos(phi)^2*(cos(psi) - 1)                                ];

end

