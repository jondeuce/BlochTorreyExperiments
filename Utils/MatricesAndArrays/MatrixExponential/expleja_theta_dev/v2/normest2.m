%Created:       15.08.2014 by Peter Kandolf
%Last edit: 	26.08.2014 by Peter Kandolf
%Version:       0.1
%Author:        Marco Caliari
%Remarks:
%
%Interface:
% [c,mv] = normest2(A,p,t,itermax,tol)
%
% estimate ||A^p||_2^(1/p), by at most itermax matrix A times t-column
% matrix products and itermax matrix A' times t-column matrix products.
% Actual used matrix times t-columns products are returned in mv.
%
%See also NORMEST, NORMEST1
%--------------------------------------------------------------------------
%Changes:
%   26.08.14 (PK):  changes from MC incorporated, switched from 2-norm 
%                   normalization to 1-norm normalization
%   20.08.14 (PK):  added some sort of termination criteria
%   15.08.14 (PK):  changes in version 0
%                   file created, changes for matlab syntax
%                   changed order of arguments to make it more convinient
function [c,mv] = normest2(A,p,t,itermax,tol)
if (nargin < 5), tol=1e-3; end
if (nargin < 4),  itermax = 100; end
if (nargin < 3),  t = 3; end
if (nargin < 2),  p = 1; end
n = length(A); %v = rng; rng(n); 
y = randn(n,t); c_old=inf; %x = x/diag(sqrt(sum(x.^2,1)));
for i = 1:itermax
    x = y/diag(sum(abs(y),1));%y/diag(sqrt(sum(y.^2,1)));
    y = A*x;
    for j = 2:p, y = A*y;  end
    y = A'*y;
    for j = 2:p, y = A'*y; end
    c = max(sum(abs(y)))^(1/(2*p)); 
    %c = max((real(sqrt(sum(conj(y).*x))).^(1/p))); % real is
    if abs(c-c_old)<tol*c || c == 0, break, end
    c_old=c;
end
mv=2*p*i*t;
%rng(v);
