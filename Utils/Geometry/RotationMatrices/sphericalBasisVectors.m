function [ varargout ] = sphericalBasisVectors( theta, phi )
%SPHERICALBASISVECTORS Computes the spherical basis vectors at angles theta
%and phi based on the following definition:
% 
% 	phi_hat     :=  [ cos(theta)*cos(phi); sin(theta)*cos(phi);-sin(phi) ];
% 	theta_hat	:=	[-sin(theta); cos(theta); 0 ];
% 	r_hat       :=  [ cos(theta)*sin(phi); sin(theta)*sin(phi); cos(phi) ];
% 
% NOTE: Convention above is the convention used by:
%           http://mathworld.wolfram.com/SphericalCoordinates.html:

[ct,st,cp,sp]	=   deal( cos(theta), sin(theta), cos(phi), sin(phi) );

phi_hat     =	[	ct*cp;	st*cp; -sp	];
theta_hat	=	[  -st;     ct;     0	];
r_hat       =	[	ct*sp;	st*sp;	cp	];

if nargout <= 1
    varargout{1}        =   [ phi_hat, theta_hat, r_hat ];
else
    [varargout{1:3}]    =   deal( phi_hat, theta_hat, r_hat );
end

end

