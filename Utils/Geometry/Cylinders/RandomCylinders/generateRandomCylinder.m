function [ p, v, r ] = generateRandomCylinder( ...
    BoxDims, BoxCenter, Radii, NumCylinders, CylOrientation, PlotCylinders )
%GENERATERANDOMCYLINDER 
% 
% INPUT ARGUMENTS
%   -BoxDims:        (3x1)  Vector of box dimensions; BoxDims = [L,W,H] as
%                           pictured below
%   -BoxCenter:      (3x1)  Location of the center of the box
%   -Radii:          (1xn)  Radii for each cylinder
%   -NumCylinders:   (1x1)  Number of cylinders
%   -CylOrientation: (1x1)  Random = 0, Axis Aligned = 1, Random & Periodic = 2
%   -PlotCylinders:  (T/F)  Plot generated cylinders
% 
% OUTPUT ARGUMENTS
%   -p:              (3xn)  Point on the random cylinder
%   -v:              (3xn)  Axis direction of random cylinder
%   -r:              (3xn)  Radius of random cylinder
% 
%                    ___________________
%                   /                  /|      z
%                W /                  / |      |
%                 /_________L________/  |      |_____y
%                 |                  |  /     /
%               H |                  | /     /
%                 |__________________|/     x

% minDistPointBox = @(p,BoxDims,BoxCenter) ...
%     min( min( [ abs( p(:) - ( BoxCenter(:) + BoxDims(:)/2 ) )    ...
%                 abs( p(:) - ( BoxCenter(:) - BoxDims(:)/2 ) ) ], ...
%                 [], 2 ) );

%% Input Parsing

if nargin < 4 || isempty( NumCylinders )
    NumCylinders   =  numel( Radii );
end

if nargin < 5 || isempty( CylOrientation )
    CylOrientation =  0;
end

if nargin < 6 || isempty( PlotCylinders )
    PlotCylinders  =  false;
end

errstr    =   'CylOrientation must be 0 (or ''random''), 1 (or ''aligned''), or 2 (or ''periodic'').';
if isnumeric(CylOrientation)
    CylOrientation = round(CylOrientation);
    if ~( isscalar(CylOrientation) && any(CylOrientation == [0,1,2]) )
        warning('Invalid cylinder orientation %d; defaulting to random', CylOrientation);
        CylOrientation = 0;
    end
elseif isa(CylOrientation, 'char')
    switch upper(CylOrientation)
        case 'RANDOM'
            CylOrientation  =   0;
        case 'ALIGNED'
            CylOrientation    =   1;
        case 'PERIODIC'
            CylOrientation    =   2;
        otherwise
            warning('Invalid cylinder orientation %s; defaulting to random', CylOrientation);
            CylOrientation    =   0;
    end
else
    error(errstr);
end

%% Get random cylinders
n    =   numel( Radii );
r    =   Radii(:)';

if isscalar( r )
    n    =   NumCylinders;
    r    =   repmat( r, [1,n] );
end

if NumCylinders ~= n
    warning('Incorrect number of cylinders requested. Using numel(Radii)');
end

% This seems unnecessary:
% b    =   ( reshape( bsxfun(@minus, BoxDims(:)/2, r), [], 1 ) < 0    );
% if any( b )
%     inds    =   ( b(1:3:end) | b(2:3:end) | b(3:3:end) );
%     error( ['A cylinder of radius %0.2f will not fit in a box of size ' ...
%             '[%0.2f,%0.2f,%0.2f]\n'], ...
%             [ r(inds); repmat( BoxDims(:), [1,sum(inds)] ) ] );
% end


%% Generate Cylinders
switch CylOrientation
    case 0
        [ p, v ]  =  GenCyl_Random( BoxDims(:), BoxCenter(:), r, n );
    case 1
        [ p, v ]  =  GenCyl_AxisAligned( BoxDims(:), BoxCenter(:), r, n );
    case 2
        [ p, v ]  =  GenCyl_Periodic( BoxDims(:), BoxCenter(:), r, n );
end

%% Plot Cylinders
switch CylOrientation
    case 0, titlestr = sprintf('%d Random Cylinders: Randomly Oriented', n);
    case 1, titlestr = sprintf('%d Random Cylinders: Axis-Aligned', n);
    case 2, titlestr = sprintf('%d Random Cylinders: Periodic BC''s', n);
end
if PlotCylinders
    plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, titlestr );
end

end

function [p,v] = GenCyl_Random( BoxDims, BoxCenter, r, n )
%% Cylinders with random orientations

[p,v] = sampleRandomCylinders( BoxDims, BoxCenter, r, n );

end

function [p,v] = GenCyl_AxisAligned( BoxDims, BoxCenter, r, n )
%% Axis aligned cylinders

% random unit vectors in x, y, or z-directions
v           =   zeros(3,n);
idx         =   3*(0:n-1);
xyz         =   randi(3,[1,n]);
v(idx+xyz)  =   1;

% random points in the box
p           =   bsxfun( @plus,    BoxCenter, ...
                bsxfun( @times, BoxDims, (rand(3,n)-0.5) ) );
p(idx+xyz)  =   BoxCenter(xyz);

end

function [p,v] = GenCyl_Periodic( BoxDims, BoxCenter, r, n )

warning( 'Periodic random cylinders not implemented yet. Using random.');
[p,v] = GenCyl_Random( BoxDims(:), BoxCenter(:), r, n );

end

function fig = plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, titlestr )

if nargin < 6; titlestr = ''; end;

figure, hold on
fig    = get(gcf,'Number');

BoxCenter =  BoxCenter(:)';
BoxDims   =  BoxDims(:)';
BoxBounds =  [ BoxCenter - 0.5*BoxDims
               BoxCenter + 0.5*BoxDims ];
rectpatchPlot( BoxBounds, fig );

for ii = 1:size(p,2)
    cylinderPlot( p(:,ii), v(:,ii), r(ii), sqrt(3)*max( BoxDims ), fig );
end

axis image
AxisBounds =  [ BoxCenter - 0.5 * 1.001 * BoxDims
                BoxCenter + 0.5 * 1.001 * BoxDims ];
axis( AxisBounds(:)' );

xlabel('x'); ylabel('y'); zlabel('z');
if ~isempty( titlestr ); title( titlestr ); end;

end








