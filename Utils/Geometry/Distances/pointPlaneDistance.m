function [ d, signs ] = pointPlaneDistance( p, n, x0, dim )
%POINTPLANEDISTANCE Finds the minimum distance between the points 'p' and
%the plane defined by 'n' and 'x0', which is defined by:
%   
%       plane_equation(x) = dot( n, x - x0 ) = 0
% 

if nargin < 4;
    dim	= [];
end

% force vectors to be 3 x N
[ In, Out, Perm, p, n, x0 ] = formatVectors( dim, 1, 'trans', p, n, x0 );

% get distances
d           =   bsxfun( @times, n, bsxfun( @minus, p, x0 ) );
[~,~,~,d]	=   formatVectors( In(1), Out(1), Perm(1), d );
d           =   bsxfun( @rdivide, sum(d,In(1)), sqrt(sum(n.^2,1)) );

if nargout > 1
    signs	=   sign( d );
end

d	=   abs( d );

end

