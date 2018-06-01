function [ y ] = BlochTorreyAction_brute(x, h, D, f, iters, isdiag)
%BLOCHTORREYACTION_BRUTE Simple matlab implementation of istropic-diffusion
%Bloch-Torrey action:
%   y = D*lap(x,h)-Gamma*x
%
% NB: if isdiag == true, the input f is assumed to be f := -6*D/h^2 - Gamma
%     and not Gamma itself

if nargin < 6; isdiag = true; end
if nargin < 5; iters = 1; end

if isscalar(D)
    if ~isdiag; f = (-6*D/h^2) - f; end % convert Gamma -> Diag
    BTAction = @(x,h,D,f) BTAction_Scalar_D(x,h,D,f);
else
    if isdiag
        error('Only support f = Gamma for non-constant D');
    end
    BTAction = @(x,h,D,f) BTAction_Variable_D(x,h,D,f);
end

y = BTAction(x,h,D,f);
for ii = 1:iters-1
    y = BTAction(y,h,D,f);
end

end

function y = Lap(x,includecenter)
if nargin < 2; includecenter = true; end
y = circshift(x, 1, 1);
y = y + circshift(x,-1, 1);
y = y + circshift(x, 1, 2);
y = y + circshift(x,-1, 2);
y = y + circshift(x, 1, 3);
y = y + circshift(x,-1, 3);
if includecenter; y = y - 6*x; end
end

function g = GradDot(x,y)
g = 0.5 * ( ...
    (x - circshift(x, -1, 1)) .* (circshift(y, 1, 1) - y) + ...
    (x - circshift(x, -1, 2)) .* (circshift(y, 1, 2) - y) + ...
    (x - circshift(x, -1, 3)) .* (circshift(y, 1, 3) - y) + ...
    (circshift(x, 1, 1) - x) .* (y - circshift(y, -1, 1)) + ...
    (circshift(x, 1, 2) - x) .* (y - circshift(y, -1, 2)) + ...
    (circshift(x, 1, 3) - x) .* (y - circshift(y, -1, 3)) ...
    );
end

function y = BTAction_Scalar_D(x,h,D,f)
% f is the diagonal, not Gamma
y = (D/h^2) * Lap(x,false) + f.*x;
end

function y = BTAction_Variable_D(x,h,D,f)
% Isotropic term plus Flux term minus Gamma term (f is Gamma := R2 + i*dw)
y = (D.*Lap(x,true))/h^2 + GradDot(D,x)/h^2 - f.*x;
end

