function [ G ] = Compress( G )
%COMPRESS Compresses the geometry object G for storing by setting to empty
% properties that are large and can easily be recreated, such as
% VasculatureMap and Idx. UNCOMPRESS repopulates these properties.

G.Idx = [];
G.VasculatureMap = [];
G.ArterialIndices = [];

end

