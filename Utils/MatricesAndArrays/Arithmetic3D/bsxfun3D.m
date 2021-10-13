function [ F ] = bsxfun3D( fun, x, y, z )
%BSXFUN3D Performs binary function 'fun' on 'x', 'y', and 'z' such that the
% result 'F' = fun(fun(x,y),z). If x, y, or z is a vector it will be 
% reshaped along dimension 1, 2, or 3 respectively.

if ( length(x) == numel(x) ) && ( length(x) ~= size(x,1) )
    x	=   x(:);
end
if ( length(y) == numel(y) ) && ( length(y) ~= size(y,2) )
    y	=   y(:).';
end
if ( length(z) == numel(z) ) && ( length(z) ~= size(z,3) )
    z	=   reshape(z,1,1,[]);
end

F	=   bsxfun( fun, x, y );
F	=   bsxfun( fun, F, z );

end
