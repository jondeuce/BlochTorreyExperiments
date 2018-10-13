function [ G ] = MajorMinorInitialGuess( G )
%CALCULATEINITIALGUESS

%-------------------------------------------------------------------------%
% Initial guess for major cylinders
%-------------------------------------------------------------------------%

% Initial points
[x,y] = regular_grid_2D(G.Nmajor,false);
G.p0  = roty(G.MajorAngle) * ... % rotate points by MajorAngle
    [ G.VoxelSize(1) * x(:)'
      G.VoxelSize(2) * y(:)'
      zeros(1,G.Nmajor,'double') ]; % Evenly spaced points in the plane
G.p0  = bsxfun( @plus, G.VoxelCenter(:), G.p0 ); % Translate to VoxelCenter

% Initial directions
vz0_Fun = @(th) [sind(th); 0; cosd(th)];
G.vz0 = repmat( vz0_Fun(G.MajorAngle), [1,G.Nmajor] ); % Points in the z-direction

G.r0  = G.RmajorFun(1,G.Nmajor,'double');
G.Rmajor = G.InitGuesses.Rmajor;

%-------------------------------------------------------------------------%
% Initial guess for minor cylinders
%-------------------------------------------------------------------------%

% Minor vessel radii
G.r = G.RminorFun(1,G.InitGuesses.Nminor,'double');

PlotCyls = false;
if G.opts.AllowMinorSelfIntersect
    if G.opts.AllowMinorMajorIntersect
        [p,vz,r] = addIntersectingCylinders( G.VoxelSize(:), G.VoxelCenter(:), G.r, ...
            G.InitGuesses.Nminor, G.opts.MinorOrientation, PlotCyls, [], [], [], G.Targets.BVF );
        [G.P,G.Vz,G.R] = deal([G.p0, p], [G.vz0, vz], [G.r0, r]);
    else
        [G.P,G.Vz,G.R] = addIntersectingCylinders( G.VoxelSize(:), G.VoxelCenter(:), G.r, ...
            G.InitGuesses.Nminor, G.opts.MinorOrientation, PlotCyls, G.p0, G.vz0, G.r0, G.Targets.BVF );
    end
else
    [G.P,G.Vz,G.R] = nonIntersectingCylinders( G.VoxelSize(:), G.VoxelCenter(:), G.r, ...
        G.InitGuesses.Nminor, G.opts.MinorOrientation, PlotCyls, G.p0, G.vz0, G.r0, G.Targets.BVF );
end

[G.Vx,G.Vy,G.Vz] = nullVectors3D( G.Vz );
G.N = numel(G.R);
G.Nminor = G.N - G.Nmajor;

end

