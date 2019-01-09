function [ y ] = BlochTorreyAction_brute(x, h, D, f, iters, isdiag, m)
%BLOCHTORREYACTION_BRUTE Simple matlab implementation of istropic-diffusion
%Bloch-Torrey action:
%   y = D*lap(x,h)-Gamma*x
%
% NB: if isdiag == true, the input f is assumed to be f := -6*D/h^2 - Gamma
%     and not Gamma itself

if nargin < 7; m = logical([]); end
if nargin < 6; isdiag = true; end
if nargin < 5; iters = 1; end

if isscalar(D) && ~isempty(m)
    D = D*ones(size(m)); % scalar D function doesn't accept mask
end

if isscalar(D)
    if ~isdiag; f = (-6*D/h^2) - f; end % convert Gamma -> Diag
    BTAction = @(x,h,D,f) BTAction_Scalar_D(x,h,D,f);
else
    if isdiag
        error('Only support f = Gamma for non-constant D');
    end
    BTAction = @(x,h,D,f) BTAction_Variable_D(x,h,D,f,m);
end

y = BTAction(x,h,D,f);
for ii = 1:iters-1
    y = BTAction(y,h,D,f);
end

end

function y = fwd_diff(x,d)
y = circshift(x, -1, d) - x;
end

function y = bwd_diff(x,d)
y = x - circshift(x, 1, d);
end

function y = fwd_av(x,d)
y = 0.5 .* (circshift(x, -1, d) + x);
end

function y = bwd_av(x,d)
y = 0.5 .* (x + circshift(x, 1, d));
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

function z = GradDot(x,y)
z = 0.5 * ( ...
    bwd_diff(x,1) .* fwd_diff(y,1) + ...
    bwd_diff(x,2) .* fwd_diff(y,2) + ...
    bwd_diff(x,3) .* fwd_diff(y,3) + ...
    fwd_diff(x,1) .* bwd_diff(x,1) + ...
    fwd_diff(x,2) .* bwd_diff(x,2) + ...
    fwd_diff(x,3) .* bwd_diff(x,3) ...
    );
end

function z = DivDGrad(x,D)
% div(D*grad(x)) with backward divergence/forward gradient
z =     bwd_diff( D .* fwd_diff(x,1), 1 );
z = z + bwd_diff( D .* fwd_diff(x,2), 2 );
z = z + bwd_diff( D .* fwd_diff(x,3), 3 );
end

function z = FluxDiff(x,D)
% div(D*grad(x)) with backward divergence/forward gradient
phiF =  fwd_av(D,1) .* fwd_diff(x,1) + ...
        fwd_av(D,2) .* fwd_diff(x,2) + ...
        fwd_av(D,3) .* fwd_diff(x,3);
phiB =  bwd_av(D,1) .* bwd_diff(x,1) + ...
        bwd_av(D,2) .* bwd_diff(x,2) + ...
        bwd_av(D,3) .* bwd_diff(x,3);
z = phiF - phiB;
end

function z = FluxDiffNeumann(x,D,m)
% div(D*grad(x)) with backward divergence/forward gradient
phiF = (m == circshift(m, -1, 1)) .* fwd_av(D,1) .* fwd_diff(x,1) + ...
       (m == circshift(m, -1, 2)) .* fwd_av(D,2) .* fwd_diff(x,2) + ...
       (m == circshift(m, -1, 3)) .* fwd_av(D,3) .* fwd_diff(x,3);
phiB = (m == circshift(m, +1, 1)) .* bwd_av(D,1) .* bwd_diff(x,1) + ...
       (m == circshift(m, +1, 2)) .* bwd_av(D,2) .* bwd_diff(x,2) + ...
       (m == circshift(m, +1, 3)) .* bwd_av(D,3) .* bwd_diff(x,3);
z = phiF - phiB;
end

function y = BTAction_Scalar_D(x,h,D,f)
% f is the diagonal, not Gamma
y = (D/h^2) * Lap(x,false) + f.*x;
end

function y = BTAction_Variable_D(x,h,D,f,m)
% Flux difference minus Gamma term (f is Gamma := R2 + i*dw)
if isempty(m)
    y = FluxDiff(x,D)/h^2 - f.*x;
else
    y = FluxDiffNeumann(x,D,m)/h^2 - f.*x;
end

% Divergence of gradient minus Gamma term (f is Gamma := R2 + i*dw)
% y = DivDGrad(x,D)/h^2 - f.*x;

% Symmetrized expanded diffusion term minus Gamma term (f is Gamma := R2 + i*dw)
% y = (D.*Lap(x,true))/h^2 + GradDot(D,x)/h^2 - f.*x;
end

