function y = Laplacian(x, h, gsize3D, iters)
%BLOCHTORREYACTION Laplacian operator w/ 2nd order central difference
% approximations on derivatives and periodic boundary conditions.
% 
% INPUTS:
%   x:     input array (3D (or flattened 1D) or 4D (or flattened 2D) complex double array)
%   h:     grid spacing (scalar double)
%   gsize3D: size of grid operated on (3-element array)
%   iters: Number of iterations to apply the operator to the input (default 1)

if nargin < 4 || isempty(iters); iters = 1; end
if ~( iters > 0 && iters == round(iters) ); error('iters must be a positive integer'); end

[ndim, ntime, gsize1D, gsize2D, gsize3D, gsize4D] = getGridSizes(x, gsize3D);

x = checkArray(x, gsize1D, gsize2D, gsize3D, gsize4D, false, true);
h = checkGridSpacing(h);

%----------------------------------------------------------------------

isSingle	=   isa(x,'single');
isDouble	=   isa(x,'double');
isReal      =   isreal(x);

if ~(isSingle || isDouble), error('x must be double or single.'); end
if isSingle, x = double(x); end %TODO single should be working, but returns zeros

if isReal
    y = Laplacian_d(x, h, gsize4D, ndim, iters);
else
    y = Laplacian_cd(x, h, gsize4D, ndim, iters);
end

if isSingle, y = single(y); end

end
