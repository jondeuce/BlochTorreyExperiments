function [ G ] = DownsampleMinor( G, Indices, ReEval )
%DOWNSAMPLEMINOR Keep a subset `Indices` of the minor vessels.

error('not impl');

% if nargin < 3; ReEval = true; end
% if nargin < 2; Indices = 1:G.Nminor; end
% 
% G.Nminor = numel(Indices);
% G.N = G.Nmajor + G.Nminor;
% 
% G.P = G.P(:,G.Nmajor+Indices);
% G.R = G.R(:,G.Nmajor+Indices);
% G.Vx = G.Vx(:,G.Nmajor+Indices);
% G.Vy = G.Vy(:,G.Nmajor+Indices);
% G.Vz = G.Vz(:,G.Nmajor+Indices);
% if ~isempty(G.Idx); G.Idx = G.Idx(:,G.Nmajor+Indices); end
% 
% [b, NewMinorArteries] = ismember(G.MinorArteries, Indices);
% G.MinorArteries = NewMinorArteries(b);
% G.NumMinorArteries = numel(G.MinorArteries);
% 
% if ReEval
%     G = Uncompress(G);
% end

end

