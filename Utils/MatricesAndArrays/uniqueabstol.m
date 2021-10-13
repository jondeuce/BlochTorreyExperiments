function [ C, IA ] = uniqueabstol( A, abstol )
%UNIQUEABSTOL Returns the indices IA such that A(IA) contains only the
%unique elements of A, up to a tolerance ABSTOL.
%
% A will be treated as a 1-dimensional arrays, regardless of the actual
% size; calling UNIQUEABSTOL(A,ABSTOL) is identical to calling
% UNIQUEABSTOL(A(:),ABSTOL) in all cases. C is defined as A(IA).

absdiff = abs(bsxfun(@minus,A(:),A(:).'));

IA = true(numel(A),1);
for ii = 2:numel(A)
    IA(ii) = ~any( absdiff(ii,1:ii-1) <= abstol );
end

[C,ISort] = sort(A(IA));
IA = find(IA);
IA = IA(ISort);

end