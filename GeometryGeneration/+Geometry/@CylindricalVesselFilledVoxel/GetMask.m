function [ m ] = GetMask( G, masktype )
%GETMASK Get mask of type `masktype`.

if ~ischar(masktype)
    error('mask type must be a string');
end

switch upper(masktype)
    case {'VASC', 'VASCULATURE'}
        m = G.VasculatureMap;
    case {'PVS', 'VRS', 'PERIVASCULAR'}
        m = false(G.GridSize);
        m(G.VRSIndices) = true;
    case ''
        m = logical([]);
    otherwise
        error('Unrecognized mask type "%s".', masktype);
end

end

