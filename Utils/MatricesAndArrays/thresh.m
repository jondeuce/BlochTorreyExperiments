function [ x ] = thresh( x, a, b )
%THRESH Thresholds x such that x(x<a)=a and x(x>b)=b.
if a >= b, error('a must be strictly less than b.'); end
x = max(min(x,b),a);
end

