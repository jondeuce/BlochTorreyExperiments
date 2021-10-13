function [ C, IA, IB ] = intersectabstol( A, B, abstol )
%INTERSECTABSTOL Returns the indices IA such that A(IA) and B(IB) produce
%identical elements C, up to a tolerance ABSTOL. C is defined as A(IA), and
%is returned as a column vector.
%
% A, B will be treated as 1-dimensional arrays, regardless of their actual 
% size; calling INTERSECTABSTOL(A,B,ABSTOL) is identical to calling
% INTERSECTABSTOL(A(:),B(:),ABSTOL) in all cases.
% 
% NOTE: A(IA) and B(IA) are not guaranteed to be of the same size!
% Consider, for example, A = 1:5, B = [1.01, 1.02, 1.03], ABSTOL = 0.05;
% this results in IA = 1, IB = [1;2;3], and thus A(IA) = [1] and B(IB) = B.

absdiff = abs(bsxfun(@minus,A(:),B(:).'));

IA = find( any( absdiff <= abstol, 2 ) );
IB = find( any( absdiff <= abstol, 1 ) ).';
C  = A(IA);

end

% %Modified from stackoverflow:
% IA = reshape(find(any(squeeze(all(bsxfun(@le,abs(bsxfun(@minus,B(:),permute(A(:),[3 2 1]))),reltol*abs(B(:))),2)),1)),[],1);