function b = isSymmPosDef( A, tol )
%ISSYMMPOSDEF Checks if the input matrix 'A' is symmetric positive
%definite.

if nargin < 2
    tol     =   5 * eps(max(abs(A(:))));
end

% ensure matrix is valid and symmetric (chol does not check for symmetry)
isSquare	=   @(A) isequal( 0, diff(size(A)) );
maxSymmDiff	=   @(A) max(reshape(abs(A-A'),[],1));
isSymmMat	=   @(A) ( maxSymmDiff(A) <= tol );

b           =	isSquare(A) && isSymmMat(A);

% second output argument of chol is 0 if A is SPD, and a positive integer
% otherwise. see:
%   http://www.mathworks.com/matlabcentral/answers/...
%   101132-how-do-i-determine-if-a-matrix-is-positive-definite-using-matlab
if b
    [~,p]	=	chol(A);
    b       =   ~p;
end

end

