function Rot = axisAngle2Rot( u, th, flag )
%AXISANGLE2ROT Gets rotation matrix that rotates about u by angle th

if nargin < 3
    Rot = Ru(u(:)/norm(u),th);
else
    if ~flag
        Rot = Ru(u(:)/norm(u),th);
    else
        Rot = Ru_SmallAngle(u(:)/norm(u),th);
    end
end

end

function Rot = Ru(u,th)
% arbitrary rotation matrix
Rot = cos(th)*eye(3) + (1-cos(th))*kron(u,u') + sin(th)*skew(u);
end

function A = skew(u)
% skew symmetric matrix
A = [ 0,-u(3),u(2); u(3),0,-u(1); -u(2),u(1),0 ];
end

function Rot = Ru_SmallAngle(u,th)

c = 1;  % cos(th) ~ 1
s = th; % sin(th) ~ th
x = u(1); y = u(2); z = u(3);

Rot = [ c,     -z*s,    y*s;
        z*s,	c,     -x*s;
       -y*s,    x*s,	c   ];
   
% Rot = [    1,   -z*th,    y*th;
%         z*th,       1,   -x*th;
%        -y*th,    x*th,       1   ];

end

% Ru = @(th,phi) ...
% [ cos(phi) - sin(th)^2*(cos(phi) - 1),      cos(th)*sin(th)*(cos(phi) - 1), cos(th)*sin(phi);
%        cos(th)*sin(th)*(cos(phi) - 1), cos(phi) - cos(th)^2*(cos(phi) - 1), sin(phi)*sin(th);
%                     -cos(th)*sin(phi),                   -sin(phi)*sin(th),         cos(phi)]

