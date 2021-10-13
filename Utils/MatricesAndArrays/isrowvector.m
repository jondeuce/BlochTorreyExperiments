function [b] = isrowvector(x)
%ISROWVECTOR Returns true if `x` is a row vector, i.e. size(x) == [1,numel(x)]
b = isequal(size(x), [1,numel(x)]);
end

