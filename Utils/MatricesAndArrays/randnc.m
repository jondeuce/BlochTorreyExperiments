function [ A ] = randnc( varargin )
%RANDNC Generates array of complex numbers with their real and imaginary
% parts randomly normally distributed.
A	=   complex( randn(varargin{:}), randn(varargin{:}) );
end
