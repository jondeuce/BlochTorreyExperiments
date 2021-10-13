%Created:       28.08.2012 by Peter Kandolf
%Last edit: 	28.08.2014 by Peter Kandolf
%Version:       0
%Author:        Peter Kandolf
%Remarks:
%
%Interface:
% [Y, NORMERREST, INFO] = NEWTON_FIX (H, A, V, XI, DD, GAMMAFOC, ITER)
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
function [y, normerrest, m] = newton_proterror(h, A, v, xi, dd, ...
    abstol, reltol, nnorm, maxm)

w = v;    y(:,1) = w * dd(1);    m=1;
normerrest=norm(y(:,1),nnorm);
while (normerrest > -1 && m<maxm)
    m=m+1;
    wa = w;
    w = (A*w)*h - xi(m-1) * w;
    y(:,m) = y(:,m-1) + w * dd(m);
    normerrest=norm(w*dd(m),nnorm)+norm(wa*dd(m-1),nnorm);
end
