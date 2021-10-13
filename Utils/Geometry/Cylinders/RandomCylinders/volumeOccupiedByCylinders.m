function [ Volume, VolumeFrac ] = volumeOccupiedByCylinders( p, v, r, BoxDims, BoxCenter )
%VOLUMEOCCUPIEDBYCYLINDERS Find the volume occupied by the cylinders
%defined by p, v, r contained in the box defined by BoxDims, BoxCenter.
%
% NOTE: Assumes the cylinders are non-intersecting
%
% Example:
%{
	BoxDims         = [3000,4000,5000];
	BoxCenter       = [2000,10000,-5000];
	NumCylinders	= 50;
	Radii           = 13.7 + 2.1 * randn(1,NumCylinders);
	PlotCylinders	= false;
	[ p, v, r ]     = nonIntersectingCylinders( ...
        BoxDims, BoxCenter, Radii, NumCylinders, PlotCylinders );
    [ Volume, VolumeFrac ]	=   volumeOccupiedByCylinders( ...
        p, v, r, BoxDims, BoxCenter );
%}

%% Analytical solution
[Volume,VolumeFrac] = getVolume( p, v, r, BoxDims, BoxCenter );

%% Brute force solution
% [Volume,VolumeFrac] = getVolumeBrute( p, v, r, BoxDims, BoxCenter );

%% Testing
% [p, v, r, BoxDims, BoxCenter] = getMockCylinders;
% [Volume,VolumeFrac] = getVolumeTest( p, v, r, BoxDims, BoxCenter );

end

function [Volume,VolumeFrac] = getVolume( p, v, r, BoxDims, BoxCenter )

% Get the length of the cylinder axes within the box
[~, ~, Lmin, Lmax]	=	rayBoxIntersection( p, v, BoxDims, BoxCenter );
CylinderHeights     =   magn( Lmax - Lmin, 1 );

% Get the volumes occupied by each cylinder
Volumes     =   bsxfun( @times, pi * r.^2, CylinderHeights );

% Sum the volumes and return
Volume      =   sum( Volumes(:) );
VolumeFrac	=   Volume / prod( BoxDims(:) );

end

function [Volume,VolumeFrac] = getVolumeBrute( p, v, r, BoxDims, BoxCenter )
% Volume fraction is computing by simply counting the number of points that
% lie in any cylinder

BoxMin	=   BoxCenter(:) - BoxDims(:)/2;
BoxMax	=   BoxCenter(:) + BoxDims(:)/2;

[Nx,Ny,Nz]	=	deal( 200 );
vFlag       =   false;
if isequal( v(:), [0;0;1] )
    [Nx,Ny,Nz]	=   deal( ceil(Nx*sqrt(Nz)), ceil(Ny*sqrt(Nz)), 1 );
    vFlag       =	true;
end

[X,Y,Z]	=   ndgrid( linspace( BoxMin(1), BoxMax(1), Nx ),    ...
                    linspace( BoxMin(2), BoxMax(2), Ny ),    ...
                    linspace( BoxMin(3), BoxMax(3), Nz )	);

P	=   cat(4,X,Y,Z);
P	=   permute(P,[4,1,2,3]);
clear X Y Z

if vFlag
    Tz	=   unit( v );
    T	=   [null(Tz'), Tz];
end

b	=   false(Nx,Ny,Nz);
for ii = 1:size(p,2)
    if ~vFlag
        Tz	=   unit( v(:,ii) );
        T	=   [null(Tz'), Tz];
    end
    
    Q	=   mtimesx( T', bsxfun( @minus, P, p(:,ii) ) );
    b	=   b | squeeze( Q(1,:,:,:).^2 + Q(2,:,:,:).^2 <= r(ii)^2 );
end

VolumeFrac	=   sum(b(:)) / numel(b);
Volume      =   VolumeFrac * prod( BoxDims(:) );

end

function [Volume,VolumeFrac] = getVolumeTest( p, v, r, BoxDims, BoxCenter )


tic; [Volume,VolumeFrac]     =	getVolume( p, v, r, BoxDims, BoxCenter ); T(1) = toc;
tic; [VolumeB,VolumeFracB]	=	getVolumeBrute( p, v, r, BoxDims, BoxCenter ); T(2) = toc;

fprintf( '\nVolume   (direct):\t%0.6f\nVolume   (brute ):\t%0.6f', ...
    Volume, VolumeB );
fprintf( '\nFraction (direct):\t%0.6f\nFraction (brute ):\t%0.6f\n\n', ...
    VolumeFrac, VolumeFracB );

display_toc_time( T, {'direct','brute'} );
fprintf('\n');

end

function [p, v, r, BoxDims, BoxCenter] = getMockCylinders

BoxDims         = 5000 * rand(1,3) + [2000,2000,2000];
BoxCenter       = 20000 * (2*rand(1,3)-1);
% Radii           = 13.7 + 2.1 * randn(1,NumCylinders);
% NumCylinders	= 50;
NumCylinders	= min( 50, floor( 0.5 * min(BoxDims)/(2*(13.7+3*2.1)) ) );
Radii           = 13.7 + 2.1 * randn(1,NumCylinders);
PlotCylinders	= false;

% [ p, v, r ]     = nonIntersectingCylinders( ...
%     BoxDims, BoxCenter, Radii, NumCylinders, PlotCylinders );

x1	=   BoxCenter + [BoxDims(1:2)/2, 0];
x2	=   BoxCenter - [BoxDims(1:2)/2, 0];
[p,v,r]	= deal( linspaceVec(x1,x2,NumCylinders,true), [0;0;1], Radii );

end

