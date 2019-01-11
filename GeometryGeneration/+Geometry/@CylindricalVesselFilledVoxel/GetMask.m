function [ m ] = GetMask( G, masktype )
%GETMASK Get mask of type `masktype`.

if ~ischar(masktype)
    error('mask type must be a string');
end

switch upper(masktype)
    case {'VASC', 'VASCULATURE'}
        m = G.VasculatureMap; % true in blood, false elsewhere
        
    case {'PVS', 'VRS', 'PERIVASCULAR'}
        m = false(G.GridSize);
        m(G.VRSIndices) = true; % true in PVS, false elsewhere
        
    case {'PVSORVASC', 'VRSORVASC', 'PVSORVASCULATURE', 'VRSORVASCULATURE'}
        m = G.VasculatureMap;
        m(G.VRSIndices) = true; % true in blood or PVS, false elsewhere
        
    case {'PVSANDVASC', 'VRSANDVASC', 'PVSANDVASCULATURE', 'VRSANDVASCULATURE'}
        m = double(G.VasculatureMap);
        m(G.VRSIndices) = 2; % 0.0 in tissue, 1.0 in blood, 2.0 in PVS
        
    case ''
        m = logical([]);
        
    otherwise
        error('Unrecognized mask type "%s".', masktype);
end

end

