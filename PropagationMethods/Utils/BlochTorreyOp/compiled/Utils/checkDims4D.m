function bool = checkDims4D(x, gsize2D, gsize4D)
bool = ( isequal(size(x), gsize2D) || isequal(size(x), gsize4D) );
end