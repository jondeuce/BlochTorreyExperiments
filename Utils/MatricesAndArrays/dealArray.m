function [ varargout ] = dealArray( A )
%DEALARRAY Deals each element of array A to the outputs. The number of
%outputs must be less than or equal to numel(A).

varargout	=   cell(1,numel(A));
for ii = 1:numel(A)
    varargout{ii}	=   A(ii);
end

end

