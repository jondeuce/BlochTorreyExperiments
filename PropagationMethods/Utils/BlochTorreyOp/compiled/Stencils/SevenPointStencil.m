function y = SevenPointStencil(x, kern, gsize3D, iters)
%SEVENPOINTSTENCIL General seven point stencil operator applied to the
%(complex) array x. That is,
% 
%   y(i,j,k) = kern(1)*x(i,j,k) + kern(2)*x(i-1,j,  k  ) + kern(3)*x(i+1,j,  k  )
%                               + kern(4)*x(i,  j-1,k  ) + kern(5)*x(i,  j+1,k  )
%                               + kern(6)*x(i,  j,  k-1) + kern(7)*x(i,  j,  k+1)
% 
% INPUTS:
%   x:       input array (3D (or flattened 1D) or 4D (or flattened 2D) complex double array)
%   kern:    7-point stecil; 7 element vector with kern(1) the center, kern(2:3)
%            the left/right (x-dir), kern(4:5) the up/down (ydir), and kern(6:7)
%            the back/forth (z-dir) weights, respectively
%   gsize3D: size of grid operated on (3-element array)
%   iters:   Number of iterations to apply the operator to the input (default 1)

if nargin < 4 || isempty(iters); iters = 1; end
if ~( iters > 0 && iters == round(iters) ); error('iters must be a positive integer'); end
if ~(isnumeric(kern) && numel(kern)==7); error('kern must be a length 7 numeric array'); end

[ndim, ntime, gsize1D, gsize2D, gsize3D, gsize4D] = getGridSizes(x, gsize3D);

x = checkArray(x, gsize1D, gsize2D, gsize3D, gsize4D, false, true);

%----------------------------------------------------------------------

isSingle	=   isa(x,'single');
isDouble	=   isa(x,'double');

if ~(isSingle || isDouble), error('x must be double or single.'); end
if isSingle, x = double(x); kern = double(kern); end %TODO: make single precision version

% x is used as a buffer internally, and since temp results will be complex
% anyways, it is easiest to just force x to complex here
if isreal(x) && ~isreal(kern) && iters > 1; x = complex(x); end

y = SevenPointStencil_cd(x, kern, gsize4D, ndim, iters);

if isSingle, y = single(y); end

end
