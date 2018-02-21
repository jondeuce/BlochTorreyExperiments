function [ Geom ] = RemoveArteries( Geom )
%REMOVEARTERIES Remove arteries from the geometry G

Geom.NumMajorArteries = 0;
Geom.NumMinorArteries = 0;
Geom.MinorArterialFrac = 0;
Geom.MajorArteries = [];
Geom.MinorArteries = [];
Geom.ArterialIndices = [];

end

