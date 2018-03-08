function [ G ] = CalculateVasculatureMap( G, AddIndices )
%CALCULATEVASCULATUREMAP Calculates the vasculature map and stores it in G.
% If AddIndices is true, stores indices of vasculature map which
% correspond to each individual cylinder as well.
%
% N.B. add_indices overrides G.opts.PopulateIdx, if given

if nargin < 2 || isempty( AddIndices )
    AddIndices = G.opts.PopulateIdx;
end

G = NormalizeCylinderVecs(G);
isUnit = true;
isCentered = true;
prec = 'double';
is_2D = false;

G.Idx = cell(1,G.N);
G.VasculatureMap = [];
BVF_Fun = @(x) sum(x(:))/numel(x);

if AddIndices
    
    [ G.VasculatureMap, G.Idx ] = getCylinderMask( ...
        G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
        G.P, G.Vz, G.R, G.Vx, G.Vy, ...
        isUnit, isCentered, prec, G.VasculatureMap );
    
    G.BVF = BVF_Fun(G.VasculatureMap);
    G.aBVF = numel(unique(cat(1,G.idx0{:})))/prod(G.GridSize);
    G.iBVF = numel(unique(cat(1,G.idx{:})))/prod(G.GridSize);
    
else
    
    G.VasculatureMap  = getCylinderMask( ...
        G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
        G.P, G.Vz, G.R, G.Vx, G.Vy, ...
        isUnit, isCentered, prec, [], [], is_2D );
    
    G.BVF = BVF_Fun(G.VasculatureMap);
    
    G.aBVF = BVF_Fun( getCylinderMask( ...
        G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
        G.p0, G.vz0, G.r0, G.vx0, G.vy0, ...
        isUnit, isCentered, prec, [], [], is_2D ) );
    
    G.iBVF = BVF_Fun( getCylinderMask( ...
        G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
        G.p, G.vz, G.r, G.vx, G.vy, ...
        isUnit, isCentered, prec, [], [], is_2D ) );
    
end

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
