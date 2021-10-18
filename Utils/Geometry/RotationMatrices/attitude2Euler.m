function [psi, th, phi] = attitude2Euler( R )
%ATTITUDE2EULER Angles from attitude matrix based on definitions at
%   https://www.princeton.edu/~stengel/MAE331Lecture9.pdf

% H_I_to_B = ...
%     [                              cos(psi)*cos(th),                              cos(th)*sin(psi),         -sin(th)
%       cos(psi)*sin(phi)*sin(th) - cos(phi)*sin(psi), cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(th), cos(th)*sin(phi)
%       sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(th), cos(phi)*sin(psi)*sin(th) - cos(psi)*sin(phi), cos(phi)*cos(th) ];

psi = atan2(  R(1,2), R(1,1) );
th  = atan2( -R(1,3), sqrt( R(2,3)^2 + R(3,3)^2 ) );
phi = atan2(  R(2,3), R(3,3) );

end

