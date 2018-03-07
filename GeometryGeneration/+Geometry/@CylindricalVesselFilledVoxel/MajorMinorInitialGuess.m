function [ G ] = MajorMinorInitialGuess( G )
%CALCULATEINITIALGUESS 

%-------------------------------------------------------------------------%
% Initial guess for major cylinders
%-------------------------------------------------------------------------%

% Initial points
[x,y] = regular_grid_2D(G.Nmajor,false);
G.p0  = roty(G.MajorAngle) * ... % rotate points by MajorAngle
    [ min(G.VoxelSize) * x(:)'
      min(G.VoxelSize) * y(:)'
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

if G.opts.AllowMinorSelfIntersect
    [G.P,G.Vz,G.R] = addIntersectingCylinders( G.VoxelSize(:), G.VoxelCenter(:), G.r, ...
        G.InitGuesses.Nminor, G.opts.MinorOrientation, false, G.p0, G.vz0, G.r0, G.Targets.BVF );
else
    [G.P,G.Vz,G.R] = nonIntersectingCylinders( G.VoxelSize(:), G.VoxelCenter(:), G.r, ...
        G.InitGuesses.Nminor, G.opts.MinorOrientation, false, G.p0, G.vz0, G.r0, G.Targets.BVF );
end

[G.Vx,G.Vy,G.Vz] = nullVectors3D( G.Vz );
G.N = numel(G.R);
G.Nminor = G.N - G.Nmajor;

end

