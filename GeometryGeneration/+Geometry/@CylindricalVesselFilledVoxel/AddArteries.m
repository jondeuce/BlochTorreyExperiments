function [ G ] = AddArteries( G, MajorArteries, MinorArteries )
%ADDARTERIES Add arteries to the geometry G

if nargin < 3; MinorArteries = []; end
if nargin < 2; MajorArteries = []; end

G.MajorArteries = MajorArteries;
G.MinorArteries = MinorArteries;

if G.NumMajorArteries > 0
    if isempty(G.MajorArteries)
        G.MajorArteries = randperm(G.Nmajor, G.NumMajorArteries);
    end
    G.ArterialIndices = cat(1,G.ArterialIndices,G.idx0{G.MajorArteries});
end

if G.MinorArterialFrac > 0
    if isempty(G.MinorArteries)
        G.NumMinorArteries = round(G.MinorArterialFrac * G.Nminor);
        G.MinorArteries = randperm(G.Nminor, G.NumMinorArteries);
    end
    G.ArterialIndices = cat(1,G.ArterialIndices,G.idx{G.MinorArteries});
end

if ~isempty(G.ArterialIndices)
    G.ArterialIndices = unique(G.ArterialIndices);
end

end

