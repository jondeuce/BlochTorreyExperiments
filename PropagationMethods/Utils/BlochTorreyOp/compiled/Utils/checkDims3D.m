function bool = checkDims3D(x, gsize1D, gsize3D)
bool = ( isequal(size(x), gsize3D) || isequal(size(x), gsize1D) );
end