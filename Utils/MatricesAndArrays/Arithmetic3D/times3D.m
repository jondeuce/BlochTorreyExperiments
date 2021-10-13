function [ F ] = times3D( x, y, z )
%TIMES3D Performs binary function '@plus' on 'x', 'y', and 'z' such that the
% result 'F' = x*y*z. If x, y, or z is a vector it will be reshaped along
% dimension 1, 2, or 3 respectively.

F	=   bsxfun3D( @times, x, y, z );

end

