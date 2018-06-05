function bool = checkDims(x, ndim, gsize1D, gsize2D, gsize3D, gsize4D)
bool = ( ndim == 3 && checkDims3D(x, gsize1D, gsize3D) ) || ...
       ( ndim == 4 && checkDims4D(x, gsize2D, gsize4D) );
end