function y = extrema(x)
%INFNORM Extrema of input array, treated as a vector. That is, return
% y = [min(x(:)), max(x(:))].
y = [min(vec(x)), max(vec(x))];
end
