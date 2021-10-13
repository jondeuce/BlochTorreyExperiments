function [ y ] = mysign( x )
%MYSIGN Same as sign(x), but with sign(0) := 1
y	=   sign(x) + (x==0);
end