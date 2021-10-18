function [ P, t ] = projPointLine( p, n, x0, dim )
%PROJPOINTLINE projects the points 'p' onto the line defined by 'n' and
%'x0', namely L = x0 + t*n

if nargin < 4;
    dim	= [];
end

% force vectors to be 3 x N
[ ~, ~, ~, p, n, x0 ] = formatVectors( dim, 1, 'trans', p, n, x0 );

% get projections
N	=   sqrt( sum( n.^2, 1 ) );
n	=	bsxfun( @rdivide, n, N );
P	=	bsxfun( @times, bsxfun( @minus, p, x0 ), n );
P	=   bsxfun( @times, sum(P,1), n );

if nargout > 1
    t           =	bsxfun( @rdivide, sqrt(sum(P.^2,1)), N );
    [~,~,~,t]	=   formatVectors( 1, dim, 'trans', t );
end

P           =	bsxfun( @plus, P, x0 );
[~,~,~,P]	=	formatVectors( 1, dim, 'trans', P );


end

