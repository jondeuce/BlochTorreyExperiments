function [ th, phi, psi ] = sphericalBasisInv( R )
%SPHERICALBASISINV Get spherical + psi angles of the rotation matrix R.
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
% 
% The important entries for inversion are:
%   -The third column (get phi and th):
%           z    =  [cos(th)sin(phi); sin(th)sin(phi); cos(phi)]
%       <=> phi  =  acos(z(3))
%           th   =  atan2(z(2),z(1))
%   -The third row of x and y (get psi given th)
%           x(3) = -cos(psi-th)sin(phi)
%           y(3) =  sin(psi-th)sin(phi)
%       <=> psi  =  atan2(y(3),-x(3)) + th

% z-vector is just polar coordinates
phi = angle3D( [0;0;1], R(:,3) );

% th, psi require special treatment
th  = get_th( R, phi );
psi = get_psi( R, th, phi );

end

function th = get_th( R, phi )

phi_thresh = 1e-4 * (pi/180);

% check for small phi
if abs(phi) < phi_thresh
    th = 0;
else
    th = atan2( R(2,3), R(1,3) );
end

end

function psi = get_psi( R, th, phi )

phi_thresh = 1e-4 * (pi/180);

% check for small phi
if phi < phi_thresh
    % z-axis is unchanged - psi is just polar angle
    psi = atan2(R(2,1),R(1,1));
else
    % from explicit form of final rotation matrix and solving for psi
    psi = atan2(R(3,2),-R(3,1)) + th;
end

end

