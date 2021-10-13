function [ z ] = zipper( x, y, dim )
%ZIPPER Zippers the arrays x and y together along dimension dim.
% 
% E.g., zipper(1:5,17:21) = [ 1, 17, 2, 18, 3, 19, 4, 20, 5, 21 ]
% 
% If x and y are vectors with only 1 non-singleton dimension and have the
% same size, dim will be determined by dim = find(size(x) == length(x)).
% If they are the same length but of different shapes, the shape of z will
% correspond to the shape of x. If they are general arrays and dim is
% unspecified, an error will be thrown.

if nargin < 3
    if my_isvector(x) && my_isvector(y)
        dim	=   find( size(x) == length(x), true, 'first' );
        if ~isequal(size(x),size(y))
            y	=   reshape(y,size(x));
        end
    else
        error('Must specify dimension to zipper for arrays.');
    end
end

ndim	=   ndims(x);
if dim ~= 2
    perm	=   circshift(1:ndim,-dim+2,2);
    [x,y]	=   deal( permute(x,perm), permute(y,perm) );
end

z	=   cat(1,x,y);
z	=   reshape(z,size(x,1),size(x,2)+size(y,2));

if dim ~= 2
    perm	=   circshift(1:ndim,dim-2,2);
    z       =   permute(z,perm);
end

end

function b = my_isvector(x)
b	=   sum( size(x) ~= 1 ) == 1;
end