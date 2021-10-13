function [ d ] = numdigits( x )
%NUMDIGITS Returns the number of digits in x
d	=   floor(log10(x))+1;
end

