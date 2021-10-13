function [ angle ] = minAngle3D( vec1, vec2, dim, isDegrees )
% MINANGLE3D Returns the minimum angle between the three dimensional
% vectors vec1 and vec2. That is:
% 
%   angle = min( angle(vec1,vec2), angle(vec1,-vec2) )
% 
% NOTE: angle is returned in radians, unless isDegrees is true

if nargin < 4
    isDegrees	=   false;
end
if nargin < 3
    dim	=   find( size(vec1) == 3, true, 'first' );
end

angle	=	min(    angle3D( vec1, vec2, dim, isDegrees ),	...
                    angle3D( vec1,-vec2, dim, isDegrees )	);

end

