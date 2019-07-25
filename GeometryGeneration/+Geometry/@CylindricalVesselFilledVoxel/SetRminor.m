function [ G ] = SetRminor(G, Rminor)
%SETRMINOR Update the value of Rminor in G (a scalar).
% 
% NOTE: You must recalculate the vasculature map, etc., manually if needed;
% this function does not do it by default:
%   G = Uncompress(G);

if isscalar(Rminor)
    Rminor = Rminor * ones(size(G.r));
else
    Rminor = reshape(Rminor(:), size(G.r));
end
G.r = Rminor .* ones(size(G.r));

Rminor_mu = mean(G.r); % mean of minor cylinder radii
Rminor_sig = std(G.r); % std of minor cylinder radii
G.RminorFun = @(varargin) Rminor_mu + Rminor_sig .* randn(varargin{:});

end

