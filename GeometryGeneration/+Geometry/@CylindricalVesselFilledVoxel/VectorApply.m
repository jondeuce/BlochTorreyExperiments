function [ out ] = VectorApply( G, Function, varargin )
%VECTORAPPLY Apply `Function` to all elements of the vector or array of
%geometries `G` with arguments `varargin{:}`. Output a cell array of the
%results. As a special case, if all outputs are themselves
%`CylindricalVesselFilledVoxel`s, the output is converted to a vector.

out = cell(size(G));
isCylVesFilledVox = true;
for ii = 1:numel(G)
    out{ii} = Function(G(ii), varargin{:});
    isCylVesFilledVox = isCylVesFilledVox && isa(out{ii}, 'Geometry.CylindricalVesselFilledVoxel');
end

if isCylVesFilledVox
    out = cat(1, out{:});
    out = reshape(out, size(G));
end

end

