function y = vec( x )
%VEC y = vec( x ). Returns x(:).

% Faster than x(:) and correct shape for empty x.
y = reshape(x, numel(x), 1);

end
