function [ b, mx ] = getCylinderMask( MaskSize, VoxelSize, ...
    BoxCenter, BoxDims, p, vz, r, vx, vy, isUnit, isCentered, prec, b, mx, is_2D )
%GETCYLINDERMASK Returns boolean list which is true if the corresponding
% point in Points is in any of the cylinders defined by p, v, and r, and
% false otherwise
%
% INPUT ARGUMENTS
%   MaskSize:	[1x3]	Size of mask to create (scalar -> isotropic 3D)
%   VoxelSize:	[1x3]	Size of voxels (must be isotropic 3D)
%   BoxCenter:  [1x3]	Locations of centers of boxes
%   BoxDims:    [1x3]	Sidelengths of boxes
%   p:          [3xP]	Point on the cylinder
%   vz:         [3xP]	Direction of cylinder axis
%   r:          [3xP]	Radius of cylinder
%   vx:         [3xP]   (optional) First right-handed basis vector to vz
%   vy:         [3xP]   (optional) Second right-handed basis vector to vz
%   isUnit:     [T/F]   (optional) True indicates that the inputs vx (and
%                       vy, vz if given) are unit vectors and need not be
%                       normalized
%   isCentered: [T/F]   (optional) If true, gridpoints are interpreted as
%                       being in the centers of voxels. False indicates
%                       gridpoints are located on the edges. Default true.
%   prec:       [char]	Precision of data ('single' or 'double')
%   b:          [T/F]   Old cylinder mask to be added to
% 
% NOTE:     If provided, each V = [vx(i,:), vy(i,:), vz(i,:)] should form a
%           proper orthogonal transformation, i.e. det(V) = 1 and V*V' = I.
%           For example, if vz is [3x1] and vxy = null(vz'), then one of
%           V=[vxy,vz] or V=[fliplr(vxy),vz] will satisfy this condition.
%
% OUTPUT ARGUMENTS
%   b:      [MaskSize]	Boolean array; true if point is in any cylinder,
%                       otherwise false
%   mx:     {1xP}       Cell array of lists of indices of points that are
%                       each of the P cylinders

%==========================================================================
% Parse Inputs
%==========================================================================

if isscalar( MaskSize )
    MaskSize	=   [MaskSize, MaskSize, MaskSize];
end

if ~( isscalar(VoxelSize) || (max(abs(diff(VoxelSize)))<1e-14) )
    error( 'Voxels must be isotropic, else cylinders are deformed!' );
end
VoxelSize	=   VoxelSize(1);

if nargin < 9 || ( isempty( vx ) || isempty( vy ) )
    [vx,vy,vz]	=   nullVectors3D(vz);
end

if nargin < 10 || ~isUnit
    vx	=   unit(vx);
    vy	=   unit(vy);
    vz	=   unit(vz);
end

if nargin < 11 || isempty( isCentered )
    isCentered	=   true;
end

if nargin < 12 || isempty( prec )
    prec	=   'single';
end

if nargin < 13
    b       =   [];
end

if nargin < 14
    mx      =   {};
end

if nargin < 15 || isempty(is_2D)
    is_2D	=   false;
end

%==========================================================================
% Transform points to index-space
%==========================================================================

BoxCorner	=   BoxCenter(:) - BoxDims(:)/2;
p           =   bsxfun( @minus, p, BoxCorner ); % BoxCorner is the origin
r           =   r / VoxelSize;

if isCentered
    p           =	p / VoxelSize + 0.5; % 0.5 <= p <= MaskSize + 0.5
    BoxCorner	=   [0.5,0.5,0.5];
    BoxDims     =   MaskSize;
else
    p           =	p / VoxelSize + 1.0; % 1.0 <= p <= MaskSize
    BoxCorner	=   [1,1,1];
    BoxDims     =   MaskSize-1;
end
BoxCenter	=   BoxDims/2 + BoxCorner;

%==========================================================================
% Create cylinder mask
%==========================================================================

[tmin, tmax, Lmin, Lmax, Lmid, Pmin, Pmax]	=	...
    rayBoxIntersection( p, vz, BoxDims, BoxCenter );

if isempty(b)
    % Create new mask
    b	=   false( MaskSize );
end

for ii = 1:size(p,2)
    
    % Mask each cylinder
    if is_2D
        [ b, mx ]	=	flat_mask_iter(  b, mx, p(:,ii), vz(:,ii), r(ii), vx(:,ii), vy(:,ii),...
            tmin(ii), tmax(ii), Lmid(:,ii), Pmin(ii), Pmax(ii), MaskSize, prec );
    else
        [ b, mx ]	=	mask_iter(  b, mx, p(:,ii), vz(:,ii), r(ii), vx(:,ii), vy(:,ii),...
            tmin(ii), tmax(ii), Lmid(:,ii), Pmin(ii), Pmax(ii), MaskSize, prec );
    end
    
    if ~isequal(size(mx),[1,ii])
        [ b, mx ]	=	mask_iter(  b, mx, p(:,ii), vz(:,ii), r(ii), vx(:,ii), vy(:,ii),...
            tmin(ii), tmax(ii), Lmid(:,ii), Pmin(ii), Pmax(ii), MaskSize, prec );
    end

end

end

function [ b, mx ] = mask_iter(b,mx,p,vz,r,vx,vy,tmin,tmax,Lmid,Pmin,Pmax,MaskSize,prec)

if isnan(tmin) || isnan(tmax)
    mx = [mx, {[]}];
    return
end

D       =   2 * r; % cyl. diameter
%       =   2 * sqrt(sum(MaskSize.^2)); % Worst case cyl length: twice the longest diagonal
L       =   tmax - tmin; % cyl. length from center to center

% [N1,N2]	=   deal( zeros(3,1) );
% N1(Pmin)=	1;
% N2(Pmax)=	1;
% [t1,t2]	=   deal( minAngle3D(vz,N1), minAngle3D(vz,N2) );
% Lbot	=   D * tan(t1); % add length contributions from glancing incidence of cyl.
% Ltop	=   D * tan(t2); % at top and bottom ends
Lbot    =   0.25;
Ltop    =   0.25;

% Multiply by 1.6 instead of 1 to ensure no points are missed
nxy     =	ceil( 1.6 * D );
nz      =   ceil( 1.6 * (L+Lbot+Ltop+D) ); % L -> L+D for safety

% Divide by 1.90 instead of 2 to ensure no points are missed
xyi     =	linspace( -D/1.90, D/1.90, nxy );
zi      =	linspace( -(L+D+Lbot)/1.90, (L+D+Ltop)/1.90, nz );
if strcmpi( prec, 'single' ), xyi = single(xyi); zi = single(zi); end

[X,Y,Z]	=   ndgrid( xyi, xyi, zi );
idx     =   ( X.^2 + Y.^2 <= r^2 );
X       =   X(idx);
Y       =   Y(idx);
Z       =   Z(idx);
Cyl     =   [X,Y,Z];
clear X Y Z xyi zi idx

if      norm( vz - [0;0;1] ) < 10 * eps(prec) % No need to rotate; do nothing
elseif	norm( vz - [0;1;0] ) < 10 * eps(prec) % Z -> Y
    Cyl	=	circshift( Cyl, -1, 2 );
elseif	norm( vz - [1;0;0] ) < 10 * eps(prec) % Z -> X
    Cyl	=	circshift( Cyl,  1, 2 );
else
    V	=	[ vx, vy, vz ];
    Cyl	=	Cyl * (V');
end
Cyl     =	bsxfun( @plus, Cyl, Lmid' );

%{
    % Ensure that all points are inside the given cylinder
    try
        assert( all( isPointInCylinder( Cyl', p, vz, ...
            r*(1+1e-3), vx, vy, true ) ) );
    catch me
        keyboard
    end
%}

Cyl     =   round( Cyl );
idx     =   all(	bsxfun( @ge, Cyl, [1,1,1] )     &	...
                    bsxfun( @le, Cyl, MaskSize ),   2   );
Cyl     =   Cyl(idx, :);

%{
    % Ensure that all cylinders meet at least two faces
    try
        assert(	any( Cyl(:,1) == 1 ) + any( Cyl(:,1) == MaskSize(1) ) +	...
            	any( Cyl(:,2) == 1 ) + any( Cyl(:,2) == MaskSize(2) ) +	...
                any( Cyl(:,3) == 1 ) + any( Cyl(:,3) == MaskSize(3) )   ...
                >= 2 );
    catch me
        keyboard
    end
%}

if ~isempty( Cyl )
    Cyl    = unique(sub2ind( MaskSize, Cyl(:,1), Cyl(:,2), Cyl(:,3) ));
    b(Cyl) = true;
else
    Cyl    = []; % Cyl is 0x3 --> 0x0 for consistency
end

mx = [mx, {Cyl}];

end

function [ b, mx ] = flat_mask_iter(b,mx,p,vz,r,vx,vy,tmin,tmax,Lmid,Pmin,Pmax,MaskSize,prec)

% Decaying 1/r^2 induced dB outside of vasculature
[x,y,z]	=   ndgrid( 1:MaskSize(1), 1:MaskSize(2), 1:MaskSize(3) );
P       =   [x(:),y(:),z(:)];
clear x y z

RotPoints	=   @(p,p0,T)   bsxfun(@minus,p,p0) * T;

T	=   [vx, vy, vz];
T	=   nearestRotMat(T); % Should be orthogonal, but for safety...

P	=   RotPoints(P,p',T);
Cyl	=   ( P(:,1).^2 + P(:,2).^2 < r^2 );
clear P

if isempty(b)
    b	=   reshape(Cyl,MaskSize);
else
    b	=	b | reshape(Cyl,MaskSize);
end

idx	=	(1:numel(Cyl))';
mx	=	[mx, {idx(Cyl)}];

end
