function y = maximum(x)
%INFNORM Maximum value of input array, treated as a vector. That is,
%returns y = max(x(:).
y = max(vec(x));
end
