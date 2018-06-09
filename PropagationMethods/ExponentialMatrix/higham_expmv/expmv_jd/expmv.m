function [f,s,m,mv,mvd,unA,m_min] = ...
    expmv(t,A,b,M,prec,shift,bal,full_term,prnt,m_min)
%EXPMV   Matrix exponential times vector or matrix.
%   [F,S,M,MV,MVD] = EXPMV(t,A,B,[],PREC) computes EXPM(t*A)*B without
%   explicitly forming EXPM(t*A). PREC is the required accuracy, 'double',
%   'single' or 'half', and defaults to CLASS(A).
%   A total of MV products with A or A^* are used, of which MVD are
%   for norm estimation.
%   The full syntax is
%     [f,s,m,mv,mvd,unA] = expmv(t,A,b,M,prec,shift,bal,full_term,prnt).
%   unA = 1 if the alpha_p were used instead of norm(A).
%   If repeated invocation of EXPMV is required for several values of t
%   or B, it is recommended to provide M as an external parameter as
%   M = SELECT_TAYLOR_DEGREE(A,b,m_max,p_max,prec,shift,bal,true).
%   This also allows choosing different m_max and p_max.

%   Reference: A. H. Al-Mohy and N. J. Higham. Computing the action of the
%   matrix exponential, with an application to exponential integrators.
%   SIAM J. Sci. Comput., 33(2):488--511, 2011.  Algorithm 3.2.

%   Awad H. Al-Mohy and Nicholas J. Higham, November 9, 2010.

%   Edited by JD: June 2017, June 2018

if nargin < 10 || isempty(m_min), m_min = []; end
if nargin < 9  || isempty(prnt), prnt = false; end
if nargin < 8  || isempty(full_term), full_term = false; end
if nargin < 7  || isempty(bal), bal = false; end
if bal
    [D,B] = balance(A);
    if norm(B,1) < norm(A,1), A = B; b = D\b; else; bal = false; end
end

if nargin < 6 || isempty(shift), shift = true; end
n = length(A);
if shift
    mu = full(trace(A)/n); % Much slower without full for sparse matrices
    A  = A - mu*eye(size(A),'like',A); %jd: speye(...) -> eye(...,'like',A)
end

if nargin < 5 || isempty(prec), prec = class(A); end
if nargin < 4 || isempty(M)
    tt = 1;
    [M,mvd,alpha,unA] = select_taylor_degree(t*A,b,[],[],prec,false,false);
    mv = mvd;
else
    tt = t; mv = 0; mvd = 0; unA = 1;
end

switch prec
    case 'double', tol = 2^(-53);
    case 'single', tol = 2^(-24);
    case 'half',   tol = 2^(-10);
    otherwise %jd
        tol = prec;
end

s = 1;
if t == 0
    m = 0;
else
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
end
eta = 1;
if shift, eta = exp(t*mu/s); end

skip_min = ~isempty(m_min);
if skip_min
    if length(m_min) < s; m_min(end+1:s) = 1;
    elseif length(m_min) > s; m_min = m_min(1:s);
    end
else
    m_min = zeros(1,s);
end

f = b;
for ii = 1:s
    c1 = Inf; c2 = Inf; % jd: junk values for printing
    if ~full_term && ~skip_min
        c1 = infnorm(b);
    end %jd
    
    for kk = 1:m
        
        is_first_min = (kk == m_min(ii));
        is_min = (kk < m_min(ii));
        
        if ~full_term && (skip_min && is_first_min)
            c1 = infnorm(b); % c1 hasn't been initialized; do so before b update
        end %jd
        
        b = A*b;
        b = (t/(s*kk))*b;
        f =  f + b;
        mv = mv + 1;
        
        if ~full_term && (~skip_min || (skip_min && ~is_min))
            c2 = infnorm(b);
        end %jd
        
        if prnt, print_iter(s,m,ii,kk,c1,c2,NaN,NaN), end %jd
        
        if ~full_term && (~skip_min || (skip_min && ~is_min))
            finf = infnorm(f); %jd
            if c1 + c2 <= tol*finf %jd
                m_min(ii) = kk;
                if prnt, print_iter(s,m,ii,kk,c1,c2,finf,tol), end
                break;
            end
        end
        c1 = c2;
    end
    
    f = eta*f;
    b = f;
end

if prnt, fprintf('\n'); end
if bal, f = D*f; end

end

function print_iter(s,m,ii,kk,c1,c2,finf,tol)

if ~(isinf(c1) || isinf(c2))
    if ~isnan(finf)
        % Final iteration; print resulting relative error
        fprintf('i = %2d/%2d, k = %2d/%2d, rel = %6e, tol = %6e\n', ...
            ii, s, kk, m, (c1+c2)/finf, tol);
    else
        % Intermediate iteration; print iteration numbers
        fprintf('i = %2d/%2d, k = %2d/%2d, err = %6e\n', ...
            ii, s, kk, m, c1+c2);
    end
else
    % c1 or c2 is not set
    fprintf('i = %2d/%2d, k = %2d/%2d, err = --\n', ii, s, kk, m);
end

end
