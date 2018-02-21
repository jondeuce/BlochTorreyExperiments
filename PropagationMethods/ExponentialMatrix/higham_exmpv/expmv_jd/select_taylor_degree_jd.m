function  [M,mv,alpha,unA] = ...
           select_taylor_degree_jd(A,b,m_max,p_max,prec,shift,bal,force_estm,force_no_estm)
%SELECT_TAYLOR_DEGREE_JD   Select degree of Taylor approximation.
%   [M,MV,alpha,unA] = SELECT_TAYLOR_DEGREE(A,b,m_max,p_max) forms a matrix M
%   for use in determining the truncated Taylor series degree in EXPMV
%   and EXPMV_TSPAN, based on parameters m_max and p_max.
%   MV is the number of matrix-vector products with A or A^* computed.

%   Reference: A. H. Al-Mohy and N. J. Higham, Computing the action of
%   the matrix exponential, with an application to exponential
%   integrators. MIMS EPrint 2010.30, The University of Manchester, 2010.

%   Awad H. Al-Mohy and Nicholas J. Higham, November 9, 2010.

if nargin < 9, force_no_estm = false; end %jd
if nargin < 8, force_estm = false; end
if nargin < 4 || isempty(p_max), p_max = 8; end
if nargin < 3 || isempty(m_max), m_max = 55; end

if p_max < 2 || m_max > 60 || m_max + 1 < p_max*(p_max - 1)
    error('>>> Invalid p_max or m_max.')
end

if force_estm && force_no_estm
    error('Flags ''force_estm'' and ''force_no_estm'' cannot both be set to true');
end

n = length(A);
if nargin < 7 || isempty(bal), bal = false; end
if bal
    [D B] = balance(A);
    if norm(B,1) < norm(A,1), A = B; end
end
if nargin < 5 || isempty(prec), prec = class(A); end
switch prec
    case 'double'
        load theta_taylor
    case 'single'
        load theta_taylor_single
    case 'half'
        load theta_taylor_half
end
if shift
    mu = trace(A)/n;
    mu = full(mu); % Much slower without the full! (for sparse matrices)
    A = A-mu*eye(size(A),'like',A);
end
mv = 0;
if ~force_estm, normA = norm(A,1); end

b_cols = round(numel(b)/size(A,2)); % jd: for the case where b is e.g. [512,512,512,N], but should be interpreted as [512^3,N]
alpha_estm_expensive = (normA <= 4*theta(m_max)*p_max*(p_max + 3)/(m_max*b_cols)); % jd: size(b,2) -> b_cols
if force_no_estm || (~force_estm && alpha_estm_expensive)
    % Base choice of m on normA, not the alpha_p.
    unA = 1;
    c = normA;
    alpha = c*ones(p_max-1,1);
else
    unA = 0;
    eta = zeros(p_max,1); alpha = zeros(p_max-1,1);
    for p = 1:p_max
        [c,k] = normAm(A,p+1); %call regular version; should be overloaded for custom powers A^m. jd
        c = c^(1/(p+1));
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
