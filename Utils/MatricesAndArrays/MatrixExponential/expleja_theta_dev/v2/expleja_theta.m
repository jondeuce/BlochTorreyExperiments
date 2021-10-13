function [expAv,errest, info, c, m, extreigs, mu,gamma2]=expleja_theta(h, A, ...
    varargin)

% check consistency of matrix and vector sizes
v = varargin{1};
n=size (v,1);
if (n ~= size (A, 2)), size(v), size(A)
    error ('Inconsistent matrix and vector sizes')
end
%Check if tolerance is given and set to default otherwise
if (nargin >= 4)
    tol = varargin{2};
    if (length (tol) == 1), tol(2) = 0; tol(3) = Inf; tol(4)=2;
    elseif (length (tol) == 2), tol(3) = Inf; tol(4)=2;
    elseif (length (tol) == 3), tol(4)=2; 
    end
else %default value
    tol = [0,2^(-53),inf,inf];
end
% get spectral estimate
if nargin<8
    extreigs = gersh(A);
else
    extreigs=varargin{end};
end

% h and/or A are zero - computation is finished
if (h*(abs(extreigs.SR)+abs(extreigs.LR)+abs(extreigs.LI^2)) == 0) ||...
        norm(v,tol(3))==0
    expAv = v;     errest = 0;    info = 0;  c=0; m=0; extreigs=[];
    mu=0; gamma2=0; return
end

[nsteps, gamma2, xi, dd, A, mu, c, newt,  m]=...
    select_interp_para(h, A, v, extreigs, tol, 100, varargin{3},...
    0, varargin{4},varargin{5});
nsteps=double(nsteps);
expAv = v; errest = zeros(nsteps,1); info = zeros(nsteps,1);
eta=exp(mu*h/nsteps);
for j = 1:nsteps
    [pexpAv,perrest,pinfo] = newt(max(h)/nsteps,A,expAv(:,end),xi,dd,...
        tol(1)/nsteps,tol(2)/nsteps,tol(3), m);
    errest(j) = perrest; info(j) = pinfo; 
    if varargin{5}
        expAv(:,end+(1:size(pexpAv,2)))=pexpAv*eta;
    else
        expAv=pexpAv*eta;
    end
end

