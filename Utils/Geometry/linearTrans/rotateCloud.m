function X = rotateCloud( X,theta_deg,rotation_axis,varargin )
% Parameters:
%   -'X': n x 3 point cloud in 3D
%   -'theta_deg': degrees in angle to rotate
%   -'rotation_axis': axis of rotation (accepts 'x', 'y', or 'z')
% 
% Optional:
%   -You can continue to add angle/axis pairs; the rotations will be
%   executed in the order specified
% 
% 	-e.g. Y=rotateCloud(X,30,'y',20,'x',10,'z') => Y=X*Ry(30)*Rx(20)*Rz(10)

if nargin==1; error('ERROR (rotateCloud): must provide an angle to rotate/axis about which to rotate'); end;
if nargin==2; error('ERROR (rotateCloud): must provide axis about which to rotate'); end;
if mod(length(varargin),2)~=0; error('ERROR (rotateCloud): each angle must have a corresponding axis about which to rotate'); end;

if ~ismember(3,size(X)); error('ERROR (rotateCloud): point clouds must have 3 coordinates'); end;
if size(X,2)~=3; X=X'; end;

angles_deg=[]; ax={};
angles_deg(1)=theta_deg; ax{1}=rotation_axis;
if nargin>3
    for i=1:2:length(varargin); angles_deg=[angles_deg;varargin{i}]; end; clear i;
    for i=2:2:length(varargin); ax=[ax;varargin{i}]; end; clear i;
end

if all(abs(angles_deg)<eps); return; end;

% Get rotation matrices
angles=angles_deg*pi/180; R=cell(size(ax));
for i=1:length(ax);
    if strcmp(ax{i},'x'); x=angles(i); R{i}=[1 0 0; 0 cos(x) sin(x); 0 -sin(x) cos(x)]; end;
    if strcmp(ax{i},'y'); y=angles(i); R{i}=[cos(y) 0 -sin(y); 0 1 0; sin(y) 0 cos(y)]; end;
    if strcmp(ax{i},'z'); z=angles(i); R{i}=[cos(z) sin(z) 0; -sin(z) cos(z) 0; 0 0 1]; end;
    clear x y z
end

% Get composite rotation matrix
Rot=R{1}; if length(R)>1; for i=2:length(R); Rot=Rot*R{i}; end; end;

% Get rotated points
X=X*Rot;

end

