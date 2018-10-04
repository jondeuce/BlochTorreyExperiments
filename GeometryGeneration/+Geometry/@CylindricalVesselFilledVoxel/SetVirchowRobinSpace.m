function [ G ] = SetVirchowRobinSpace( G, VRSRelativeRad )
%SETVIRCHOWROBINSPACE Add Virchow-Robin space indices to G

if nargin < 2; VRSRelativeRad = []; end

if ~isempty(VRSRelativeRad)
    G.VRSRelativeRad = VRSRelativeRad;
end

if ~isempty(G.VRSRelativeRad)
    isUnit = true;
    isCentered = true;
    prec = 'double';
    VRSRadii = G.VRSRelativeRad * G.r0;
    
    % Virchow-Robin space is the CSF-filled region surrounding the
    % anisotropic vessels; create boolean cylinder map with VRS radii
    [ VRSMap, ~ ] = getCylinderMask( ...
        G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
        G.p0, G.vz0, VRSRadii, G.vx0, G.vy0, ...
        isUnit, isCentered, prec );
    
    % Virchow-Robin space is the CSF-filled region surrounding the
    % anisotropic vessels; remove vessel containing regions from the VRSMap
    % (major and minor); remaining region is the VRS
    VRSMap = VRSMap & ~G.VasculatureMap;
    G.VRSIndices = find(VRSMap);
end

end

