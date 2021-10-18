function [x] = linspacePeriodic( a, b, n, prec )
%LINSPACEPERIODIC Creates 'n' linearly spaced points 'x' in (a,b) such that
% the next n linearly spaced points on (b,b+(b-a)) is simply given by
% x+(b-a). No points between x and x+(b-a) will be the same.
% 
% INPUT ARGUMENTS
%   a:	lower bound
%   b:  upper bound
%   n:  number of points
% 
% OUTPUT ARGUMENTS
%   x:  linearly spaced points on (a,b) periodic with period b-a

% parse inputs
if nargin < 3 || isempty( n )
    n	=   100;
end

if nargin < 4 || isempty( prec )
    prec	=   'double';
end

% check for single precision (default to double)
switch upper(prec)
    case 'SINGLE'
        [a,b,n]	=   deal( single(a), single(b), single(n) );
    otherwise
        [a,b,n]	=   deal( double(a), double(b), double(n) );
end

% create x on (-0.5,0.5)
x	=	linspace(0,(n-1)/n,n) - 0.5*(n-1)/n;

% scale and shift to (a,b)
x	=   (b-a) * (x+0.5) + a;

end

