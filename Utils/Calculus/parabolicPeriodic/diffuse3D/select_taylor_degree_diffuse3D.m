function  [M,mv,alpha,unA] = ...
           select_taylor_degree_diffuse3D(A,m_max,p_max,prec,shiftmat,bal,force_estm,Ap,n,normA,t,len)
%SELECT_TAYLOR_DEGREE_DIFFUSE3D Select degree of Taylor approximation.
%   [M,MV,alpha,unA] = SELECT_TAYLOR_DEGREE(A,m_max,p_max) forms a matrix M
%   for use in determining the truncated Taylor series degree in EXPMV
%   and EXPMV_TSPAN, based on parameters m_max and p_max.
%   MV is the number of matrix-vector products with A or A^* computed.

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

if nargin < 12 || isempty(len); len = 1; end %length along 4th dimension
if nargin < 7, force_estm = false; end
if nargin < 4 || isempty(prec), prec = []; end
if nargin < 3 || isempty(p_max), p_max = 8; end
if nargin < 2 || isempty(m_max), m_max = 55; end
if p_max < 2 || m_max > 60 || m_max + 1 < p_max*(p_max - 1)
    error('>>> Invalid p_max or m_max.')
end
if nargin < 6 || isempty(bal), bal = false; end

% A is a function handle, with norm given
bal        = false;
shiftmat   = false;
force_estm = false;
Aclass     = prec;

if bal
    [D,A] = balance(A);
    if isempty(D), bal = false; end
end


switch upper(Aclass)
    case 'DOUBLE'
        load theta_taylor
    case 'SINGLE'
        load theta_taylor_single
end

if shiftmat
    mu	=   trace(A)/length(A);
    A	=	shift(A,mu);
end

mv = 0;
% normA given as argument
% if ~force_estm, normA = norm(A,1); end

if ~force_estm && normA <= 4*theta(m_max)*p_max*(p_max + 3)/m_max
    % Base choice of m on normA, not the alpha_p.
    unA = 1;
    c = normA;
    alpha = c*ones(p_max-1,1);
else
    unA = 0;
    eta = zeros(p_max,1); alpha = zeros(p_max-1,1);
    for p = 1:p_max
        %[c,k] = normAm(A,p+1);
        [c,k] = normAm_diffuse3D(A,Ap,n,p+1,len);
        c = c^(1/(p+1));
        c = c .* t; % A is not A.*t yet
        mv = mv + k;
        eta(p) = c;
    end
    for p = 1:p_max-1
        alpha(p) = max(eta(p),eta(p+1));
    end
end
M = zeros(m_max,p_max-1);
for p = 2:p_max
    for m = p*(p-1)-1 : m_max
        M(m,p-1) = alpha(p-1)/theta(m);
    end
end

end
