function y = infnorm(x)
%INFNORM L-infinity norm of input array, treated as a vector. That is,
%returns y = max(abs(x(:)).
y = max(abs(vec(x)));
end
