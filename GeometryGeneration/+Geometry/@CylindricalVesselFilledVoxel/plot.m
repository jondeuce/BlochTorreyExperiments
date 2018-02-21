function [ h ] = plot( G, plotmajor, plotminor )
%PLOT Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2 || isempty(plotmajor); plotmajor = true; end
if nargin < 3 || isempty(plotminor); plotminor = true; end

H = [];
titlestr = '';

if plotminor
    col = 'b';
    alpha = 0.1;
    H = plot_cylinders_in_box( G.p, G.vz, G.r, ...
        G.VoxelSize, G.VoxelCenter, titlestr, H, col, alpha );
end

if plotmajor
    col = 'r';
    alpha = 0.5;
    H = plot_cylinders_in_box( G.p0, G.vz0, G.r0, ...
        G.VoxelSize, G.VoxelCenter, titlestr, H, col, alpha );
end

if nargout > 0; h = H; end

end

