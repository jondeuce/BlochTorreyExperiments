function [f,s,m,mv,mvd,unA] = expmv_diffuse3D(t,A,b,M,prec,shiftmat,bal,full_term,prnt,force_estm,tol,h,D,f,muA,normA)
%EXPMV_DIFFUSE3D Matrix exponential times vector or matrix.
%   [F,S,M,MV,MVD] = EXPMV(t,A,B,[],PREC) computes EXPM(t*A)*B without
%   explicitly forming EXPM(t*A). PREC is the required accuracy, 'single'
%   or 'double', and defaults to CLASS(A).
%   A total of MV products with A or A^* are used, of which MVD are
%   for norm estimation.
%   The full syntax is
%     [f,s,m,mv,mvd,unA] = expmv(t,A,b,M,prec,shift,bal,full_term,prnt).
%   unA = 1 if the alpha_p were used instead of norm(A).
%   If repeated invocation of EXPMV is required for several values of t
%   or B, it is recommended to provide M as an external parameter as
%   M = SELECT_TAYLOR_DEGREE(A,m_max,p_max,prec,shift,bal,true).
%   This also allows choosing different m_max and p_max.

%   Reference: A. H. Al-Mohy and N. J. Higham, Computing the action of
%   the matrix exponential, with an application to exponential
%   integrators. MIMS EPrint 2010.30, The University of Manchester, 2010.

%   Awad H. Al-Mohy and Nicholas J. Higham, March 19, 2010.

%==========================================================================
% Edited by JD, July 2016
%==========================================================================

%==========================================================================
% Input Parsing
%==========================================================================
% Handle case t = 0 before doing anything -jd
if t == 0
    [f,s,m,mv,mvd,unA]	=   deal( b, 1, 0, 0, 0, 1 );
    return
end
if isempty(t), t = 1; end

if nargin < 11, tol = []; end
if nargin < 10 || isempty(force_estm), force_estm = false; end
if nargin < 9 || isempty(prnt), prnt = false; end
if nargin < 8 || isempty(full_term), full_term = false; end
if nargin < 7 || isempty(bal), bal = false; end
if nargin < 6 || isempty(shiftmat), shiftmat = true; end
if nargin < 5 || isempty(prec), prec = []; end

% use diffuse3D function
normA	=	t.* normA;
g   =   (f+muA); %-f-mu*I = -(f+mu*I)
A	=   @(u) fmg_diffuse(u,h,D,g); % A*x
cg	=   conj(g); %(D*lap(u)-f)' = D*lap(u)-con
Ap	=   @(u) fmg_diffuse(u,h,D,cg); % A'*x
n   =	length(b);

% bal and shiftmat should always be false
[bal,shiftmat] = deal(false);
% if bal
%     [D,A,b] = balance(A,b);
%     if isempty(D), bal = false; end
% end
% if shiftmat
%     mu	=   trace(A)/length(A);
%     A	=	shift(A,mu);
% end

if nargin < 4 || isempty(M)
   tt = 1;
   [M,mvd,alpha,unA] = select_taylor_degree_diffuse3D(A,[],[],prec,false,false,force_estm,Ap,n,normA,t);
   clear At
   mv = mvd;
else
   tt = t; mv = 0; mvd = 0;
end

%==========================================================================
% Main Algorithm
%==========================================================================
if isempty(tol)
    tol = 2^(-53);
    if strcmpi(prec,'single'), tol = 2^(-24); end
end

% Case of t == 0 is now handled at beginning -jd
[m_max,p] = size(M);
U = diag(1:m_max);
C = ( (ceil(abs(tt)*M))'*U );
C (C == 0) = inf;

if p > 1
    [cost, m] = min(min(C)); % cost is the overall cost.
else
    [cost, m] = min(C);  % when C is one column. Happens if p_max = 2.
end
if cost == inf; cost = 0; end
s = max(cost/m,1);

eta = 1;
if shiftmat, eta = exp(t*mu/s); end
f = b;
% if prnt, fprintf('     m = %2.0f, s = %g\n', m, s), end
for i = 1:s
    
    c1    = infnorm(b);
    
    for k = 1:m
        
        if prnt, looptime = tic; end
        
        %b  =	A(b);
        %b  =   (t/(s*k)) * b;
        b   =   fmg_diffuse(b,h,D,g,t/(s*k));
        mv	=	mv + 1;
        f   =	f + b;
        c2  =	infnorm(b);
        fnorm = infnorm(f);
        
        if ~full_term
            if c1 + c2 <= tol*fnorm
                break
            end
            %if c1 + c2 <= tol*infnorm(f), break, end
            c1 = c2;
        end

        if prnt
            str = sprintf('i = %1d/%1d, k = %2d/%2d, -log10(c2) = %7.4f', i, s, k, m, -log10(c2));
            display_toc_time(toc(looptime),str);
        end
        
    end
    
    if eta ~= 1
        b = eta*f;
    else
        b = f;
    end
    
end
if bal, f = D*f; end

end

% function normx = infnorm(x)
% %faster version of norm(x(:),inf)
% normx = max(abs(x(:))); 
% end