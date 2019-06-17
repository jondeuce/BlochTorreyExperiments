function [ G_out, SpaceFactor ] = ExpandMinorVesselsByVolumeFactor( G, VolumeFactor, TOLX, TOLF )
%EXPANDMINORVESSELSBYVOLUMEFACTOR This function expands the minor vessels
% away/towards eachother isotropically to achive an isotropic BVF which is
% 'VolumeFactor` times the current target BVF. The process is
% equivalent to expanding the space between the vessels uniformly, but
% leaving the vessels themselves the same size.
%
% See also DILATEMINORVESSELS, which dilates the minor vessel radii to
% achieve a desired isotropic BVF.

if nargin == 0; RunTests; return; end

if nargin < 4 || isempty(TOLF); TOLF = 1e-8; end
if nargin < 3 || isempty(TOLX); TOLX = 1e-8; end
if nargin < 2; VolumeFactor = 1; end

% initialize output
G_out = G;
G_out.Targets.iBVF = G.Targets.iBVF * VolumeFactor;

if abs(VolumeFactor - 1.0) < 1e-12
    return
elseif VolumeFactor < 0
    error('VolumeFactor must be positive');
end

fzero_opts = optimset('TolX', TOLX, 'TolFun', TOLF);
Bounds = getinitialbounds(G, VolumeFactor);
SpaceFactor = fzero(@BVF_Error, Bounds, fzero_opts );

    function BVF_err = BVF_Error(SpaceFactor)
        if SpaceFactor ~= 1
            G_out = expandminorvessels(G, SpaceFactor);
            G_out = CalculateVasculatureMap(G_out);
        end
        BVF_err = G.iBVF * VolumeFactor - G_out.iBVF;
    end

%Ensure geometry has all vascular map based properties set
G_out = Uncompress(G_out);

end

function G = expandminorvessels(G, SpaceFactor)
% Simply expand cylinder points away from the voxel center by `SpaceFactor`
dp = bsxfun(@minus, G.p, G.VoxelCenter(:));
G.p = bsxfun(@plus, G.VoxelCenter(:), SpaceFactor * dp);
end

function Bounds = getinitialbounds(G, VolumeFactor)
% Theoretically, scaling space by `SpaceFactor` along each dimension uniformly
% multiplies the resulting density of cylinders by SpaceFactor^3, and therefore
% the BVF by SpaceFactor^-3. So, the initial guess will be
%     BVFnew = SpaceFactor^-3 * BVFold <==> SpaceFactor = (VolumeFactor)^(-1/3)
% With infinitely long cylinders, discrete grids, etc., this heuristic turns out
% to really not be very good, so we simply loop until we have a valid bound.
GoalBVF = G.iBVF * VolumeFactor;
SpaceFactor = VolumeFactor^(-1/3);

G = expandminorvessels(G, SpaceFactor);
G = CalculateVasculatureMap(G);
CurrentBVF = G.iBVF;

ALPHA = 0.7; % Rate of contraction/expansion
if SpaceFactor < 1
    % Contract space; need to find BVF greater than the GoalBVF
    while CurrentBVF < GoalBVF
        SpaceFactor = SpaceFactor*ALPHA;
        G = expandminorvessels(G, ALPHA);
        G = CalculateVasculatureMap(G);
        CurrentBVF = G.iBVF;
    end
    Bounds = [SpaceFactor, 1];
else
    % Expand space; need to find BVF less than the GoalBVF
    while CurrentBVF > GoalBVF
        SpaceFactor = SpaceFactor/ALPHA;
        G = expandminorvessels(G, 1/ALPHA);
        G = CalculateVasculatureMap(G);
        CurrentBVF = G.iBVF;
    end
    Bounds = [1, SpaceFactor];
end

end

% ----------------------------------------------------------------------- %
% Tests
% ----------------------------------------------------------------------- %
function RunTests

GeomArgs = struct( 'iBVF', 2/100, 'aBVF', 2/100, ...
    'VoxelSize', [3000,3000,3000], 'GridSize', [256,256,256], 'VoxelCenter', [0,0,0], ...
    'Nmajor', 4, 'MajorAngle', 0, ......
    'NumMajorArteries', 0, 'MinorArterialFrac', 0, ...
    'Rminor_mu', 15, 'Rminor_sig', 0, ...
    'AllowMinorSelfIntersect', true, ...
    'AllowMinorMajorIntersect', true, ...
    'VRSRelativeRad', 1 );

NameValueArgs = struct2arglist(GeomArgs);
Geom = Geometry.CylindricalVesselFilledVoxel(NameValueArgs{:});

VFact = 0.5 + 0.5 * rand();
G = ExpandMinorVesselsByVolumeFactor( Geom, VFact );
ShowBVFResults(G);

end
