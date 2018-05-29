function [ G ] = SetArteries( G, MajorArteries, MinorArteries )
%SETARTERIES Add arteries to the geometry G

if nargin < 3; MinorArteries = []; end
if nargin < 2; MajorArteries = []; end

G.MajorArteries = MajorArteries;
G.MinorArteries = MinorArteries;

if G.NumMajorArteries > 0
    if isempty(G.MajorArteries)
        G.MajorArteries = randperm(G.Nmajor, G.NumMajorArteries);
    end
    
    % Set arterial indices to be the (randomly selected) MajorArteries
    G.ArterialIndices = cat(1, G.idx0{G.MajorArteries});
end

if G.MinorArterialFrac > 0
    if isempty(G.MinorArteries)
        G.NumMinorArteries = round(G.MinorArterialFrac * G.Nminor);
        G.MinorArteries = randperm(G.Nminor, G.NumMinorArteries);
    end
    
    % Minor arterial indices should not include any major vessel indices
    MinorArterialIndices = cat(1, G.idx{G.MinorArteries});
    MinorArterialIndices = setdiff(MinorArterialIndices, cat(1,G.idx0{:}));
    
    % Add the minor arterial indices to the major arterial indices above
    G.ArterialIndices = cat(1, G.ArterialIndices, MinorArterialIndices);
end

if ~isempty(G.ArterialIndices)
    G.ArterialIndices = unique(G.ArterialIndices);
end

end

