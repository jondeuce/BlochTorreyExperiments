function [b] = iscolumnvector(x)
%ISVECTOR Returns true if `x` is a vector, i.e. size(x) == [numel(x),1]
b = isequal(size(x), [numel(x),1]);
end

