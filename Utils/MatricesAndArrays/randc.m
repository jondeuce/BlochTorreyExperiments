function [ A ] = randc( varargin )
%RANDC Generates array of complex numbers with their real and imaginary
% parts uniformly randomly distributed on [0,1].
A	=   complex( rand(varargin{:}), rand(varargin{:}) );
end
