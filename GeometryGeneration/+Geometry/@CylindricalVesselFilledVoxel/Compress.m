function [ G ] = Compress( G )
%COMPRESS Compresses the geometry object G for storing by setting to empty
% properties that are large and can easily be recreated, such as
% VasculatureMap and Idx. UNCOMPRESS repopulates these properties.

for ii = 1:numel(G)
    G(ii).Idx = [];
    G(ii).VasculatureMap = [];
    G(ii).ArterialIndices = [];
end

end

