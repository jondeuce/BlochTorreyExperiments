function [ G ] = Uncompress( G )
%UNCOMPRESS Repopulates properties compressed by COMPRESS for saving.

G = CalculateVasculatureMap(G);
G = AddArteries(G, G.MajorArteries, G.MinorArteries);

end

