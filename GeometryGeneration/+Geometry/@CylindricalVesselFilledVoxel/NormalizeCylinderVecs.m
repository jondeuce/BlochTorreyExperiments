function [ G ] = NormalizeCylinderVecs( G )
%NORMALIZECYLINDERVECS Normalizes the [3xN] array of main cylinder 
% directions G.Vz, as well as populating G.Vx and G.Vy

if ~isempty(G.Vz) > 0
    [G.Vx, G.Vy, G.Vz] = nullVectors3D( G.Vz );
end

end

