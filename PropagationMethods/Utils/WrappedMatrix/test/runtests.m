A = complex(randn(100),randn(100));
B = complex(randn(100),randn(100));
C = A*B;

f = WrappedMatrix(A);
g = WrappedMatrix(B);

% assert( norm( 