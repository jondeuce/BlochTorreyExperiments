function [ G ] = Uncompress( G )
%UNCOMPRESS Repopulates properties compressed by COMPRESS for saving.

for ii = 1:numel(G)
    G(ii) = CalculateVasculatureMap(G(ii));
    G(ii) = AddArteries(G(ii), G(ii).MajorArteries, G(ii).MinorArteries);
end

end

