function [c,mv] = normAm_diffuse3D(A,Ap,n,m,p)
%NORMAM_DIFFUSE3D   Estimate of 1-norm of power of matrix.
%   NORMAM(A,m) estimates norm(A^m,1).
%   If A has nonnegative elements the estimate is exact.
%   [C,MV] = NORMAM(A,m) returns the estimate C and the number MV of
%   matrix-vector products computed involving A or A^*.

%   Reference: A. H. Al-Mohy and N. J. Higham, A New Scaling and Squaring
%   Algorithm for the Matrix Exponential, SIAM J. Matrix Anal. Appl. 31(3):
%   970-989, 2009.

%   Awad H. Al-Mohy and Nicholas J. Higham, September 7, 2010.

t = 1; % Number of columns used by NORMEST1.
if nargin < 5 || isempty(p); p = 1; end

% A is a function handle for a tri-diagonal matrix with complex entries on
% the diagonal. A' = conj(A).
isPositive  =   false;
nn          =   [n,n,n,p]; % matrix is 3D or 4D
n           =   n^3*p; % matrix is 3D or 4D

if isPositive
    e = ones(n,1);
    for j=1:m         % for positive matrices only
        e = Ap(e);
        e = Ap(e);
    end
    c = norm(e,inf);
    mv = m;
else
    [c,v,w,it] = normest1(@afun_power,t);
    mv = it(2)*t*m;
end

  function Z = afun_power(flag,X)
       %AFUN_POWER  Function to evaluate matrix products needed by NORMEST1.

       if isequal(flag,'dim')
          Z = n;
       elseif isequal(flag,'real')
          %Z = isreal(A);
          Z = false;
       else

          [p,q] = size(X);
          if p ~= n, error('Dimension mismatch'), end

          if isequal(flag,'notransp')
             %for i = 1:m, X = A*X; end
             
             X = reshape(X,nn);
             for i = 1:m, X = A(X); end
             X = reshape(X,[],1);
          elseif isequal(flag,'transp')
             %for i = 1:m, X = A'*X; end
             
             X = reshape(X,nn);
             for i = 1:m, X = Ap(X); end
             X = reshape(X,[],1);
          end

          Z = X;

       end

  end
end
