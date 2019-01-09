function y = SevenPointDifference_Brute(x, kern, gsize3D, iters, mask)
%SEVENPOINTDIFFERENCE_BRUTE SevenPointDifference_Brute(x, kern, gsize3D, iters)
% General seven point difference operator applied to the array x, with possible
% masking. The difference is weighted by the kernel, and only included if both
% boolean mask values are the same at each difference point. That is,
%
%   y(i,j,k) = kern(1) * x(i,j,k) +
%              kern(2) * (m(i,j,k) == m(i-1,j  ,k  )) * (x(i-1,j  ,k  ) - x(i,j,k)) +
%              kern(3) * (m(i,j,k) == m(i+1,j  ,k  )) * (x(i+1,j  ,k  ) - x(i,j,k)) +
%              kern(4) * (m(i,j,k) == m(i  ,j-1,k  )) * (x(i  ,j-1,k  ) - x(i,j,k)) +
%              kern(5) * (m(i,j,k) == m(i  ,j+1,k  )) * (x(i  ,j+1,k  ) - x(i,j,k)) +
%              kern(6) * (m(i,j,k) == m(i  ,j  ,k-1)) * (x(i  ,j  ,k-1) - x(i,j,k)) +
%              kern(7) * (m(i,j,k) == m(i  ,j  ,k+1)) * (x(i  ,j  ,k+1) - x(i,j,k))
% 
% For example, this can be used to perform the Laplacian with subvoxels of size
% [hx,hy,hz] and Neumann boundary conditions on mask edges using the kernel:
% 
%   kern = [0, 0.5/hx^2, 0.5/hx^2, 0.5/hy^2, 0.5/hy^2, 0.5/hz^2, 0.5/hz^2]
%
% INPUTS:
%   x:       input array. 3D/(or flattened 1D) or 4D/(or flattened 2D) real or
%            complex double array
%   kern:    7-point stencil; 7 element complex or real double array
%   gsize3D: size of grid (3-element array)
%   iters:   Number of iterations to apply the operator to the input (default 1)

if nargin == 0; runTests; return; end

if nargin < 5 || isempty(mask); mask = true(gsize3D); end
if nargin < 4 || isempty(iters); iters = 1; end
if ~(iters > 0 && iters == round(iters)); error('iters must be a positive integer'); end
if ~(isnumeric(kern) && numel(kern)==7); error('kern must be a length 7 numeric array'); end

[ndim, ntime, gsize1D, gsize2D, gsize3D, gsize4D] = getGridSizes(x, gsize3D);

x = checkArray(x, gsize1D, gsize2D, gsize3D, gsize4D, false, false);

%----------------------------------------------------------------------

y = sevenpointdiff_brute(x, kern, mask);
for ii = 2:iters
    y = sevenpointdiff_brute(y, kern, mask);
end

end

function y = sevenpointdiff_brute(x, kern, m)

y = kern(1) * x + ...
    kern(2) * (m == circshift(m, +1, 1)) .* (circshift(x, +1, 1) - x) + ...
    kern(3) * (m == circshift(m, -1, 1)) .* (circshift(x, -1, 1) - x) + ...
    kern(4) * (m == circshift(m, +1, 2)) .* (circshift(x, +1, 2) - x) + ...
    kern(5) * (m == circshift(m, -1, 2)) .* (circshift(x, -1, 2) - x) + ...
    kern(6) * (m == circshift(m, +1, 3)) .* (circshift(x, +1, 3) - x) + ...
    kern(7) * (m == circshift(m, -1, 3)) .* (circshift(x, -1, 3) - x);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function runTests

b = true;
% gsize = [3,4,5];
gsize = [5,5,2];
kern = randnc(7,1);
mask = rand(gsize) > 0.5;

x = randnc(gsize);
for it = 1:3
    b = runTestSet(x,kern,gsize,it,mask) && b;
end

x = randnc([gsize,4]);
for it = 1:3
    b = runTestSet(x,kern,gsize,it,mask) && b;
end

if b; fprintf('All tests passed.\n');
else; fprintf('Some tests failed.\n');
end

end

function b = runTestSet(x,kern,gsize,iters,mask)

b = true;

y = SevenPointDifferenceMasked(x,kern,gsize,iters,mask);
yb = SevenPointDifference_Brute(x,kern,gsize,iters,mask);
b = test_approx_eq(y,yb,'cplx-cplx') && b;

y = SevenPointDifferenceMasked(x,real(kern),gsize,iters,mask);
yb = SevenPointDifference_Brute(x,real(kern),gsize,iters,mask);
b = test_approx_eq(y,yb,'cplx-real') && b;

y = SevenPointDifferenceMasked(real(x),kern,gsize,iters,mask);
yb = SevenPointDifference_Brute(real(x),kern,gsize,iters,mask);
b = test_approx_eq(y,yb,'real-cplx') && b;

y = SevenPointDifferenceMasked(real(x),real(kern),gsize,iters,mask);
yb = SevenPointDifference_Brute(real(x),real(kern),gsize,iters,mask);
b = test_approx_eq(y,yb,'real-real') && b;

end

function str = errmsg(name, msg)
str = sprintf('Test failed: %s(test suite: %s)', msg, name);
end

function str = passedmsg(name, msg)
str = sprintf('Test passed: %s(test suite: %s)', msg, name);
end

function b = test_approx_eq(x,y,name,msg,tol)
if nargin < 5; tol = 10*max(eps(max(abs(x(:)))), eps(max(abs(y(:))))); end
if nargin < 4; msg = ''; end
if nargin < 3; name = 'N/A'; end
% tol = sqrt(tol);
maxdiff = max(abs(x(:)-y(:)));
b = (maxdiff <= tol);
if ~b; warning(errmsg(name,msg)); else; fprintf('%s\n',passedmsg(name,msg)); end
end