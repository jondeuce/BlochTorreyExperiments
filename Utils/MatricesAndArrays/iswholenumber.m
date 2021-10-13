function [ b ] = iswholenumber( x )
%ISWHOLENUMBER Returns boolean array that is true if entry is a whole
% number (positive integer), and false otherwise.

b = (x>0 & x==round(x));

end

