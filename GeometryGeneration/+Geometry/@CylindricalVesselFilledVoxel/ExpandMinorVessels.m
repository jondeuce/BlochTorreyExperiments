function [ G ] = ExpandMinorVessels( G, SpaceFactor )
%EXPANDMINORVESSELS This function expands space by 'SpaceFactor' in a way
% that is equivalent to expanding the space between the vessels uniformly,
% but leaves the vessels themselves the same size.
%
% NOTE: You must recalculate the vasculature map manually if needed; this
% function does not do it by default:
%   G = Uncompress(G);
% 
% See also EXPANDMINORVESSELSBYVOLUMEFACTOR, which expands space until a
% desired isotropic BVF is reached.

if nargin == 0; RunTests; return; end

if nargin < 4 || isempty(TOLF); TOLF = 1e-8; end
if nargin < 3 || isempty(TOLX); TOLX = 1e-8; end
if nargin < 2; VolumeFactor = 1; end

% Simply expand cylinder points away from the voxel center by `SpaceFactor`
dp = bsxfun(@minus, G.p, G.VoxelCenter(:));
G.p = bsxfun(@plus, G.VoxelCenter(:), SpaceFactor * dp);

end


% ----------------------------------------------------------------------- %
% Tests
% ----------------------------------------------------------------------- %
function RunTests

% GeomArgs = struct( 'iBVF', 2/100, 'aBVF', 2/100, ...
%     'VoxelSize', [3000,3000,3000], 'GridSize', [256,256,256], 'VoxelCenter', [0,0,0], ...
%     'Nmajor', 4, 'MajorAngle', 0, ......
%     'NumMajorArteries', 0, 'MinorArterialFrac', 0, ...
%     'Rminor_mu', 15, 'Rminor_sig', 0, ...
%     'AllowMinorSelfIntersect', true, ...
%     'AllowMinorMajorIntersect', true, ...
%     'VRSRelativeRad', 1 );
% 
% NameValueArgs = struct2arglist(GeomArgs);
% Geom = Geometry.CylindricalVesselFilledVoxel(NameValueArgs{:});
% 
% VFact = 0.5 + 0.5 * rand();
% G = ExpandMinorVesselsByVolumeFactor( Geom, VFact );
% ShowBVFResults(G);

end
