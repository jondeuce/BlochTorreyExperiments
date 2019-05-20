function [ G ] = SetRmajor(G, Rmajor)
%SETRMAJOR Update the value of Rmajor in G (a scalar).
% 
% NOTE: You must recalculate the vasculature map, etc., manually if needed;
% this function does not do it by default:
%   G = Uncompress(G);

Rmajor = reshape(Rmajor, 1, []); % force to row vector
G.Rmajor = Rmajor;
G.RmajorFun = @(varargin) Rmajor .* ones(varargin{:});
G.r0 = Rmajor .* ones(1, G.Nmajor);

end

