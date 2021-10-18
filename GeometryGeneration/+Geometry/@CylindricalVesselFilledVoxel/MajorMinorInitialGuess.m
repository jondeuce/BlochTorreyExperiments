function [ G ] = MajorMinorInitialGuess( G )
%CALCULATEINITIALGUESS

%-------------------------------------------------------------------------%
% Initial guess for major cylinders
%-------------------------------------------------------------------------%

% Initial point distribution on the square [-1,1] x [-1,1]
if strcmpi(G.opts.MajorDistribution, 'Regular')
    [x,y] = regular_grid_2D(G.Nmajor,false);
else
    x = zeros(1, G.Nmajor);
    y = linspacePeriodic(-0.5, 0.5, G.Nmajor);
end

% Rotate major vessel points and scale
G.p0 = [ G.VoxelSize(1) * x(:)'
         G.VoxelSize(2) * y(:)'
         zeros(1, G.Nmajor)    ]; % Initial point distribution, scaled in xy-plane
G.p0  = roty(G.MajorAngle) * G.p0; % Rotate initial point distribution by MajorAngle
G.p0  = bsxfun( @plus, G.VoxelCenter(:), G.p0 ); % Translate to VoxelCenter

% Initial directions
vz0_Fun = @(th) [sind(th); 0; cosd(th)];
G.vz0 = repmat( vz0_Fun(G.MajorAngle), [1,G.Nmajor] ); % Points in the z-direction

G.r0  = G.RmajorFun(1,G.Nmajor,'double');
G.Rmajor = G.InitGuesses.Rmajor;

%-------------------------------------------------------------------------%
% Initial guess for minor cylinders
%-------------------------------------------------------------------------%

% Initial cylinder guess
SphereRadius = 0.5 * norm(G.VoxelSize(:)); % Sphere diameter should be at least voxel diagonal, preferably more
SphereCenter = G.VoxelCenter(:);
[p, vz, r] = randomCylindersInSphere( ...
    SphereRadius, SphereCenter, G.RminorFun, G.Targets.BVF, true, false );
[G.P, G.Vz, G.R] = deal([G.p0, p], [G.vz0, vz], [G.r0, r]);
G = CalculateVasculatureMap( G );

SpaceFactor = (G.iBVF / G.Targets.iBVF)^(1/2); % fractal dimension of infinite cylinders is two: iBVF_goal = iBVF_curr * SpaceFactor^(-2)
G = expandminorvessels(G, SpaceFactor);
G = CalculateVasculatureMap( G );

if G.opts.AllowInitialMinorPruning
    b = ~cellfun(@isempty, G.idx);
    [G.P, G.Vz, G.R] = deal([G.p0, G.p(:,b)], [G.vz0, G.vz(:,b)], [G.r0, G.r(:,b)]);
    G = CalculateVasculatureMap( G );
end    

[G.Vx,G.Vy,G.Vz] = nullVectors3D( G.Vz );
G.N = numel(G.R);
G.Nminor = G.N - G.Nmajor;

end

function G = expandminorvessels(G, SpaceFactor)
% Simply expand cylinder points away from the voxel center by `SpaceFactor`
dp = bsxfun(@minus, G.p, G.VoxelCenter(:));
G.p = bsxfun(@plus, G.VoxelCenter(:), SpaceFactor * dp);
end

