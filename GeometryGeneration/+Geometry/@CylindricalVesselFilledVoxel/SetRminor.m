function [ G ] = SetRminor(G, Rminor)
%SETRMINOR Update the value of Rminor in G (a scalar).
% 
% NOTE: You must recalculate the vasculature map, etc., manually if needed;
% this function does not do it by default:
%   G = Uncompress(G);

Rminor = Rminor(:) .* ones(size(G.r)); % force to row vector
Rminor_mu = mean(G.Rminor); % mean of minor cylinder radii
Rminor_sig = std(G.Rminor); % std of minor cylinder radii
G.RminorFun = @(varargin) Rminor_mu + Rminor_sig .* randn(varargin{:});
G.r = Rminor .* ones(1, G.Nmajor);

end

