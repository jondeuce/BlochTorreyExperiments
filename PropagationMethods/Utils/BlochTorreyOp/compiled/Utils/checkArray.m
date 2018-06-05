function x = checkArray(x, gsize1D, gsize2D, gsize3D, gsize4D, forcecplx, forcedouble)

if nargin < 7; forcedouble = true; end
if nargin < 6; forcecplx = false; end

if isempty(x)
    x = 0;
elseif isscalar(x)
    x = double(x);
elseif ~( checkDims3D(x, gsize1D, gsize3D) || checkDims4D(x, gsize2D, gsize4D) )
    error('size(f) must be one of: scalar, (repeated-)grid size, (repeated-)flattened size');
end

if ~isa(x,'double') && forcedouble, x = double(x); end
if isreal(x) && forcecplx; x = complex(x); end

end