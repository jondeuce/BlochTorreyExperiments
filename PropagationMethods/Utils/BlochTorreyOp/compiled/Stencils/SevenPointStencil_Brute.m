function y = SevenPointStencil_Brute(x, kern, gsize3D, iters)
%SEVENPOINTSTENCIL_BRUTE SevenPointStencil_Brute(x, kern, gsize3D, iters)
% General seven point stencil operator applied to the array x. That is,
%
%   y(i,j,k) = kern(1)*x(i,j,k) + kern(2)*x(i-1,j,  k  ) + kern(3)*x(i+1,j,  k  )
%                               + kern(4)*x(i,  j-1,k  ) + kern(5)*x(i,  j+1,k  )
%                               + kern(6)*x(i,  j,  k-1) + kern(7)*x(i,  j,  k+1)
%
% INPUTS:
%   x:       input array. 3D/flattened 1D or 4D/flattened 2D real or
%            complex double array
%   kern:    7-point stecil; 7 element complex or real double array
%   gsize3D: size of grid (3-element array)
%   iters:   Number of iterations to apply the operator to the input (default 1)

if nargin == 0; runTests; return; end

if nargin < 4 || isempty(iters); iters = 1; end
if ~( iters > 0 && iters == round(iters) ); error('iters must be a positive integer'); end
if ~(isnumeric(kern) && numel(kern)==7); error('kern must be a length 7 numeric array'); end

[ndim, ntime, gsize1D, gsize2D, gsize3D, gsize4D] = getGridSizes(x, gsize3D);

x = checkArray(x, gsize1D, gsize2D, gsize3D, gsize4D, false, false);

%----------------------------------------------------------------------

y = sevenpointstencil_brute(x, kern);
for ii = 2:iters
    y = sevenpointstencil_brute(y, kern);
end

end

function y = sevenpointstencil_brute(x, kern)

y = kern(1)*x + ...
    kern(2)*circshift(x,1,1) + kern(3)*circshift(x,-1,1) + ...
    kern(4)*circshift(x,1,2) + kern(5)*circshift(x,-1,2) + ...
    kern(6)*circshift(x,1,3) + kern(7)*circshift(x,-1,3);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function runTests

b = true;
gsize = [3,4,5];
kern = randnc(7,1);

x = randnc(gsize);
for it = 1:2
    b = runTestSet(x,kern,gsize,it) && b;
end

x = randnc([gsize,4]);
for it = 1:2
    b = runTestSet(x,kern,gsize,it) && b;
end

if b; fprintf('All tests passed.\n');
else; fprintf('Some tests failed.\n');
end

end

function b = runTestSet(x,kern,gsize,iters)

b = true;

y = SevenPointStencil(x,kern,gsize,iters);
yb = SevenPointStencil_Brute(x,kern,gsize,iters);
b = test_approx_eq(y,yb,'cplx-cplx') && b;

y = SevenPointStencil(x,real(kern),gsize,iters);
yb = SevenPointStencil_Brute(x,real(kern),gsize,iters);
b = test_approx_eq(y,yb,'cplx-real') && b;

y = SevenPointStencil(real(x),kern,gsize,iters);
yb = SevenPointStencil_Brute(real(x),kern,gsize,iters);
b = test_approx_eq(y,yb,'real-cplx') && b;

y = SevenPointStencil(real(x),real(kern),gsize,iters);
yb = SevenPointStencil_Brute(real(x),real(kern),gsize,iters);
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