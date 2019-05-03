function [ G ] = SetMajorArteries(G, MajorArteries)
%SETRMAJOR Update the value of MajorArteries in G (a row vector).
% 
% NOTE: You must recalculate the vasculature map, etc., manually if needed;
% this function does not do it by default:
%   G = Uncompress(G);

G.MajorArteries = MajorArteries;

end

