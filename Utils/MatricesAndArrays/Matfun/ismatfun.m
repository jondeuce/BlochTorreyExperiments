function [ bool ] = ismatfun( F )
%ISMATFUN True if the input F is a matfun object, false otherwise
bool	=   isa(F,'matfun');
end

