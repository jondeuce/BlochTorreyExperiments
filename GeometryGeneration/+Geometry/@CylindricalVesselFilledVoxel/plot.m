function [ Fig ] = plot( G, plotmajor, plotminor, verb )
%PLOT Plots the CylindricalVesselFilledVoxel G.

if nargin < 4 || isempty(verb); verb = true; end
if nargin < 3 || isempty(plotminor); plotminor = true; end
if nargin < 2 || isempty(plotmajor); plotmajor = true; end

% set(0,'DefaultFigureVisible','off');

titlestr = '';
fig = figure;
hold on

if plotminor
    col = 'b';
    alpha = 0.1;
    fig = plot_cylinders_in_box( G.p, G.vz, G.r, ...
        G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, verb );
end

if plotmajor
    col = 'r';
    alpha = 0.5;
    fig = plot_cylinders_in_box( G.p0, G.vz0, G.r0, ...
        G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, false );
end

% Set oblique viewing angle
view([0.8, 0.5, 0.2]);

% set(0,'DefaultFigureVisible','on');
figure(fig);
drawnow;

if nargout > 0; Fig = fig; end

end

