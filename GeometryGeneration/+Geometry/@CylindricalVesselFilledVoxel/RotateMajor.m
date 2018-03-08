function [ Geom ] = RotateMajor( Geom, type, Angle, units )
%ROTATEMAJOR Rotates the major vasculature of Geom according to the type of
%rotations and the Angle to rotate
%   Geom:    input geometry (Geometry.CylindricalVesselFilledVoxel)
%   type:    Type of rotation
%     'to':  rotate major such that the resulting angle is Angle with
%            the positive z-axis
%     'by':  rotate major by Angle degrees about the y axis
%   Angle:   Angle to be rotated to/by
%   units:   Units of Angle
%     'rad': Angle is in radians
%     'deg': Angle is in degrees (default)

if nargin < 4; units = 'deg'; end
if strcmpi(units,'rad'); Angle = rad2deg(Angle); end
if strcmpi(type,'to'); Angle = Angle - Geom.MajorAngle; end

% Compress geometry (i.e. get rid of vasc. map, etc.)
Geom = Compress(Geom);

% rotation matrix
Rot = roty(Angle);

% update cylinder vectors
Geom.vz0 = Rot * Geom.vz0; % rotate major
Geom = NormalizeCylinderVecs(Geom); % update vx, vy

% update cylinder points
Geom.p0 = bsxfun(@minus, Geom.p0, Geom.VoxelCenter(:));
Geom.p0 = Rot * Geom.p0; % rotate major
Geom.p0 = bsxfun(@plus, Geom.p0, Geom.VoxelCenter(:));

% update MajorAngle
Geom.MajorAngle = Geom.MajorAngle + Angle;

% Uncompress geometry (i.e. recompute vasc. map, etc.)
Geom = Uncompress(Geom);

end
