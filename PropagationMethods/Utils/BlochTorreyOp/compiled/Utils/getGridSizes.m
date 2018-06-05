function [ndim, ntime, gsize1D, gsize2D, gsize3D, gsize4D] = getGridSizes(x, gsize3D)
%GETGRIDSIZES

if ~(numel(gsize3D) == 3 && all(gsize3D == round(gsize3D)) && all(iswholenumber(gsize3D)))
    error('grid must be 3-dimensional; gsize3D must be a 3-element array of whole numbers');
end
gsize3D = gsize3D(:).'; 

nslice  = prod(gsize3D); %number of elems. on the 3D grid
ntotal  = numel(x); %total number of elements in x
ntime   = round(ntotal/nslice); %number of 'time' slices (i.e. reps of the 3D array)

if nslice * ntime ~= ntotal
    error('Incorrect dimensions: numel(x)/prod(gsize3D) must be an integer');
end

gsize1D = [nslice, 1]; % size of 3d-grid flattened to a column
gsize2D = [nslice, ntime]; % size of 3D array flattened to column with `ntime` repetitions along 2nd dimension
gsize4D = [gsize3D, ntime]; % size of 3D array with `ntime` repetitions along 4th dimension
ndim    = 3 + (ntime>1); %Indicates whether we have 3D (flattened 1D) or 4D (flattened 2D) data

if ~checkDims(x, ndim, gsize1D, gsize2D, gsize3D, gsize4D)
    % Allowable sizes are the repeated grid size in 4-D, or the flattened size in 2-D
    error('size(x) must be either the (repeated-)grid size or (repeated-)flattened size');
end

end

