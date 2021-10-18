function [ G ] = SetMediumVessels( G, RadiusThresh )
%SETMEDIUMVESSELS [ G ] = SetMediumVessels( G, RadiusThresh )
% Set medium minor vessels, where "medium" is defined as minor vessels with
% radii larger than threshold RadiusThresh.

if nargin < 2
    if ~isempty(G.MediumVesselRadiusThresh)
        RadiusThresh = G.MediumVesselRadiusThresh;
    else
        RadiusThresh = Inf;
    end
end

G.MediumVessels = G.r > RadiusThresh;

end

