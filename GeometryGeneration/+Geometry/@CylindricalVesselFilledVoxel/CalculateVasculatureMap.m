function [ G ] = CalculateVasculatureMap( G )
%CALCULATEVASCULATUREMAP Calculates the vasculature map and stores it in G.
% If AddIndices is true, stores indices of vasculature map which
% correspond to each individual cylinder as well.
%
% N.B. AddIndices overrides G.opts.PopulateIdx, if given

isUnit = true;
isCentered = true;
prec = 'double';
is_2D = false;

G = NormalizeCylinderVecs(G);
G.Idx = cell(1,G.N);
G.VasculatureMap = [];

[ G.VasculatureMap, G.Idx ] = getCylinderMask( ...
    G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
    G.P, G.Vz, G.R, G.Vx, G.Vy, ...
    isUnit, isCentered, prec, G.VasculatureMap );

% ---- Calculate BVF ---- %
vec     = @(x) x(:);
BVF_Fun = @(x) sum(vec(sum(x,1)))/numel(x); % faster than sum(x(:)) for logical arrays

G.BVF = BVF_Fun(G.VasculatureMap);
G.aBVF = numel(unique(cat(1,G.idx0{:})))/prod(G.GridSize);
G.iBVF = G.BVF - G.aBVF;

end

function [ G ] = UpdateBloodVolumes( G )
%UPDATEBLOODVOLUMES Updates the blood volumes according to the cylinders.

G = NormalizeCylinderVecs(G);
isUnit = true;
isCentered = true;
prec = 'double';
is_2D = false;

if isempty(G.VasculatureMap)
    G.BVF = [];
else
    G.BVF = BVF_Fun(G.VasculatureMap);
end

if any(cellfun(@isempty,G.Idx))
    G.aBVF = BVF_Fun( getCylinderMask( ...
        G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
        G.p0, G.vz0, G.r0, G.vx0, G.vy0, ...
        isUnit, isCentered, prec, [], [], is_2D ) );
    
    G.iBVF = BVF_Fun( getCylinderMask( ...
        G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
        G.p, G.vz, G.r, G.vx, G.vy, ...
        isUnit, isCentered, prec, [], [], is_2D ) );
else
    G.aBVF = sum(cellfun(@numel,G.idx0)/prod(G.GridSize));
    G.iBVF = sum(cellfun(@numel,G.idx)/prod(G.GridSize));
end

G.BVF = G.aBVF + G.iBVF;
G.iRBVF = G.iBVF / G.BVF;
G.aRBVF = G.aBVF / G.BVF;

end
