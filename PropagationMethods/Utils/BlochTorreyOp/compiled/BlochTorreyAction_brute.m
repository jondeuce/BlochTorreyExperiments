function [ y ] = BlochTorreyAction_brute(x, h, D, f, iters)
%BLOCHTORREYACTION_BRUTE Simple matlab implementation of istropic-diffusion
%Bloch-Torrey action:
%   y = D*lap(x,h)-Gamma*x
%
% NB: f := -6*D/h^2 - Gamma is the input, NOT Gamma itself

if nargin < 5
    iters = 1;
end

y = BTAction(x,h,D,f);
for ii = 1:iters-1
    y = BTAction(y,h,D,f);
end

end

function y = BTAction(x,h,D,f)

y = circshift(x, 1, 1);
y = y + circshift(x,-1, 1);
y = y + circshift(x, 1, 2);
y = y + circshift(x,-1, 2);
y = y + circshift(x, 1, 3);
y = y + circshift(x,-1, 3);

y = (D/h^2) * y;
y = y + f.*x;

end