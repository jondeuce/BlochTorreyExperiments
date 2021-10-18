function [ R ] = nearestRotMat( A )
%NEARESTROTMAT Finds the nearest rotation matrix to A in the frobenius norm
%sense. R is the rotation matrix that minimizes ||R-A||_F^2.
% 
% For the proof of the algorithm, see:
%   https://arxiv.org/pdf/0904.1613.pdf

[U,~,V]	=   svd(A);
V       =   V';
C       =   diag( [ones(length(A)-1,1); det(U*V)] );
R       =   U*C*V;

end

