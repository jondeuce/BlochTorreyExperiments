function [ G ] = Uncompress( G )
%UNCOMPRESS Repopulates properties compressed by COMPRESS for saving.

for ii = 1:numel(G)
    G(ii) = CalculateVasculatureMap(G(ii));
    G(ii) = SetArteries(G(ii), G(ii).MajorArteries, G(ii).MinorArteries);
    G(ii) = SetMediumVessels(G(ii));
    G(ii) = SetVirchowRobinSpace(G(ii), G(ii).VRSRelativeRad);
end

end

