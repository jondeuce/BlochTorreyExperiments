function [ C, IA ] = uniquereltol( A, reltol )
%UNIQUERELTOL Returns the indices IA such that A(IA) contains only the
%unique elements of A, up to a tolerance RELTOL.
%
% A will be treated as a 1-dimensional arrays, regardless of the actual
% size; calling UNIQUERELTOL(A,RELTOL) is identical to calling
% UNIQUERELTOL(A(:),RELTOL) in all cases. C is defined as A(IA).

absdiff = abs(bsxfun(@minus,A(:),A(:).'));

IA = true(numel(A),1);
for ii = 2:numel(A)
    IA(ii) = ~any( absdiff(ii,1:ii-1) <= reltol * abs(A(ii)) );
end

[C,ISort] = sort(A(IA));
IA = find(IA);
IA = IA(ISort);

end