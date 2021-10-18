function [ y ] = linspaceVec(x1, x2, n, crop, isRow )
%LINSPACEVEC Returns an array y with linspace(x1(i),x2(i),n) along the rows
% or columns of y depending on the shape of x1 and x2. Default is row
% vectors if both x1, x2 are scalars, or if x1 and x2 are different shapes.
% 
% INPUT ARGUMENTS
%   x1:     Lower bounds (scalar or row/column vector)
%   x2:     Upper bounds (scalar or row/column vector)
%   n:      Number of points
%   isRow:  [T/F] Force to be row form or column form
%   crop:   [T/F] Include boundaries x1(i), x2(i) or not. May be scalar 
%                 ([T/F] to crop both ends), or a 2-element vector
%                 ([T/F, T/F] to specify which end to crop)
% 
% OUTPUT ARGUMENTS
%	y:      linspace output array
%           -> if x1, x2 are row vectors:
%               y(i,:) = linspace(x1(i),x2(i),n)
%           -> if x1, x2 are column vectors:
%               y(:,i) = linspace(x1(i),x2(i),n)'

if ~( isvector(x1) && isvector(x2) )
    error( 'x1 and x2 must be column vectors or row vectors' );
end

if isscalar(x1) && ~isscalar(x2)
    x1	=   repmat(x1,size(x2));
elseif isscalar(x2) && ~isscalar(x1)
    x2	=   repmat(x2,size(x1));
end

[L1,L2]	=   deal( length(x1), length(x2) );
if L1 ~= L2
    error( 'x1 and x2 must have the same length or be scalar' );
end

if nargin < 4 || isempty( crop )
    crop	=   false(1,2);
elseif isscalar(crop)
    crop	=   repmat(crop,[1,2]);
end

if nargin < 5 || isempty( isRow )
    [isRow1, isRow2]	=   deal( isrow(x1), isrow(x2) );
    if xor( isRow1, isRow2 )
        warning( 'x1 and x2 have different shapes; defaulting to rows' );
        [isRow1, isRow2]	=   deal( true );
    end
    isRow	=   isRow1 && isRow2;
end

y       =   zeros( L1, n );
[x1,x2]	=   deal( x1(:), x2(:) );

if crop(1) && crop(2)
    dx	=   (x2 - x1)/(n+1);
    [x1,x2]	=   deal( x1+dx, x2-dx );
elseif crop(1) && ~crop(2)
    x1	=   x1 + (x2 - x1)/n;
elseif crop(2) && ~crop(1)
    x2	=   x2 - (x2 - x1)/n;
end

for ii = 1:L1
    y(ii,:)	=   linspace( x1(ii), x2(ii), n );
end

if ~isRow
    y	=	y';
end

end

