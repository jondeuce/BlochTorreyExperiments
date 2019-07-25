function [ Fig ] = plot( G, plotmajor, plotminor, verb, newfig )
%PLOT Plots the CylindricalVesselFilledVoxel G.

if nargin < 5 || isempty(newfig); newfig = true; end
if nargin < 4 || isempty(verb); verb = true; end
if nargin < 3 || isempty(plotminor); plotminor = true; end
if nargin < 2 || isempty(plotmajor); plotmajor = true; end

if newfig; fig = figure; else; fig = gcf; end
hold on

for ii = 1:numel(G)
    fig = plot_single_geom( G(ii), plotmajor, plotminor, verb, fig );
end

VoxelSizes = cat(1, G.VoxelSize);
VoxelCenters = cat(1, G.VoxelCenter);
BoxBounds = [min(VoxelCenters - VoxelSizes/2, [], 1); max(VoxelCenters + VoxelSizes/2, [], 1)];
BoxCenter = repmat(mean(BoxBounds, 1), numel(G), 1);

DilatationFact = 1 + 1e-2;
DilatedBoxBounds = BoxCenter + DilatationFact * (BoxBounds - BoxCenter);
axis(DilatedBoxBounds(:)');

if nargout > 0; Fig = fig; end

end

function [ fig ] = plot_single_geom( G, plotmajor, plotminor, verb, fig )

% set(0,'DefaultFigureVisible','off');
titlestr = '';

if plotminor
    alpha = 0.1;
    
    if G.MinorArterialFrac > 0
        ArtInds = G.MinorArteries;
        VenInds = setdiff(1:G.Nminor, ArtInds);
        col = 'r';
        fig = plot_cylinders_in_box( G.p(:,ArtInds), G.vz(:,ArtInds), G.r(:,ArtInds), ...
            G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, verb );
        col = 'b';
        fig = plot_cylinders_in_box( G.p(:,VenInds), G.vz(:,VenInds), G.r(:,VenInds), ...
            G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, verb );
    else
        col = 'b';
        fig = plot_cylinders_in_box( G.p, G.vz, G.r, ...
            G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, verb );
    end
end

verb = false; % not necessary for major vessels; they're fast
if plotmajor
    alpha = 0.5;
    
    if G.NumMajorArteries > 0
        ArtInds = G.MajorArteries;
        VenInds = setdiff(1:G.Nmajor, ArtInds);
        col = 'r';
        fig = plot_cylinders_in_box( G.p0(:,ArtInds), G.vz0(:,ArtInds), G.r0(:,ArtInds), ...
            G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, false );
        col = 'b';
        fig = plot_cylinders_in_box( G.p0(:,VenInds), G.vz0(:,VenInds), G.r0(:,VenInds), ...
            G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, false );
    else
        col = 'b';
        fig = plot_cylinders_in_box( G.p0, G.vz0, G.r0, ...
            G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, false );
    end
    
    if G.VRSRelativeRad ~= 1
        col = 'g';
        [p, r, ~, ~, vz] = GetVRSCylinders(G);
        fig = plot_cylinders_in_box( p, vz, r, ...
            G.VoxelSize, G.VoxelCenter, titlestr, fig, col, alpha, false );
    end
end

% Set oblique viewing angle
view([0.8, 0.5, 0.2]);

% set(0,'DefaultFigureVisible','on');
figure(fig);
drawnow;

end

