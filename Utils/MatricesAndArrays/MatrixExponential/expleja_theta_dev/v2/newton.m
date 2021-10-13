%Created:       28.08.2012 by Peter Kandolf
%Last edit: 	19.06.2015 by Peter Kandolf
%Version:       2.0
%Author:        Peter Kandolf
%Remarks:
%
%Interface:
% [Y, NORMERREST, INFO] = NEWTON (H, A, V, XI, DD, GAMMAFOC, ITER)
%
% Compute the Newton interpolation polynomial in real Leja points for the
% matrix function specified with the divided differences DD applied to the
% right hand side V of the operator A*H*V as Y=P_m(H*A)V. The nodes are in
% [-2,2]*gammafoc + d and the interpolation stops when it reached the
% maximal steps iter.
%
% The result is stored in Y the estimated error in NORMERREST and the
% number of steps in INFO. if the maximal number of iterations is reached
% but the desired error is not reached INFO contains -MAX_NUMBER_OF_STEPS.
%
%See also PHILEJA, NEWTON, NEWTONS, NEWTONS_FIX
%--------------------------------------------------------------------------
%Changes:
%   06.08.14 (PK):  correct usage of rel and abs tol for termination
%   14.11.12 (PK):  changes in version 0
%                   file created
function [y, normerrest, m] = newton(h, A, v, xi, dd,...
    abstol, reltol, nnorm, maxm)
w = v;    y = w * dd(1);   m=1;
normerrest=norm(y,nnorm);
while (normerrest > max(reltol * norm(y, nnorm),abstol)) && m<maxm
    m=m+1;
    wa = w;
    w = (A*w)*h - xi(m-1) * w;
    y = y + w * dd(m);
    normerrest=norm(w*dd(m),nnorm)+norm(wa*dd(m-1),nnorm);
end

