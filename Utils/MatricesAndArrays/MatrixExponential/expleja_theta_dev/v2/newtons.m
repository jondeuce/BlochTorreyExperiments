%Created:       28.08.2014 by Peter Kandolf
%Last edit: 	19.06.2014 by Peter Kandolf
%Version:       2.1
%Author:        Peter Kandolf
%Remarks:
%
%Interface:
% [Y, NORMERREST, INFO] = NEWTONS (H, A, V, XI, DD, GAMMAFOC,...
%                           ABSTOL,RELTOL,nnorm,ITER)
%
% Compute the Newton interpolation polynomial in conjugate pairs of  Leja
% points for the matrix function specified with the divided differences DD
% applied to the right hand side V of the operator A*V*H as Y=P_m(A*H)V.
% The nodes are in 1i*[-2,2]*gammafoc + d and the interpolation stops when
% it reached the maximal steps iter.
%
% The result is stored in Y the estimated error in NORMERREST and the
% number of steps in INFO. if the maximal number of iterations is reached
% but the desired error is not reached INFO contains -MAX_NUMBER_OF_STEPS.
%
%See also PHILEJA, NEWTON, NEWTONS, NEWTON_FIX
%--------------------------------------------------------------------------
%Changes:
%   06.08.14 (PK):  correct usage of rel and abs tol for termination
%   16.11.12 (PK):  changes in version 0
%                   file created
function [y, normerrest, m] = newtons (h, A, v, xi, dd,...
    abstol, reltol, nnorm, maxm)

w = v;    y = w * real (dd(1)); w = (A*w)*h - xi(1) * w; m=1;
normerrest=norm(y,nnorm);
while (normerrest > max(reltol * norm(y, nnorm),abstol)) && m+2<maxm
    m=m+2;
    wtilde = (A*w)*h  - real (xi(m-1)) * w;
    y = y + real(dd(m-1))*w+real(dd(m))*wtilde;
    w = (A*wtilde)*h - real (xi(m-1)) * wtilde + ...
        imag (xi(m-1))^2 * w;
    normerrest=norm(real(dd(m-1))*w+real(dd(m))*wtilde,nnorm);
end
