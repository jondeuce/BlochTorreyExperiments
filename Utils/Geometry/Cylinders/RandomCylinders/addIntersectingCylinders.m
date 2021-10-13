function [ p, v, r, h ] = addIntersectingCylinders( ...
    BoxDims, BoxCenter, Radii, NumCylinders, CylOrientation, PlotCylinders, ...
    p0, v0, r0, RelVolThresh )
%ADDINTERSECTINGCYLINDERS Adds random cylinders that are allowed to
% intersect with eachother but not with the original cylinders defined by
% the triples p0, v0, and r0

%% Input Parsing
n           =   numel( Radii );
r           =   Radii(:)';
BoxDims     =   BoxDims(:);
if isempty( BoxCenter ), BoxCenter = [0;0;0];
else BoxCenter	=   BoxCenter(:);
end

if isscalar( r ) && nargin >= 4
    if ~isempty( NumCylinders )
        n	=   NumCylinders;
        r	=   repmat( r, [1,n] );
    end
end

if nargin >= 4 && ~isempty( NumCylinders ) && ( NumCylinders ~= n )
    warning('Incorrect number of cylinders requested. Using numel(Radii)');
end

if nargin < 5 || isempty( CylOrientation )
    CylOrientation	=   'random';
end

if nargin < 6 || isempty( PlotCylinders )
    PlotCylinders	=   false;
end

if nargin < 7 || ( isempty(p0) || isempty(v0) || isempty(r0) )
    [p,v,~]	=   generateRandomCylinder( BoxDims, BoxCenter, r, [], CylOrientation, false );
    if PlotCylinders
        h	=	plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, ...
            sprintf('Cylinders: N = %d', n) );
        drawnow;
    else
        h	=   [];
    end
    return
end

if nargin < 10 || isempty(RelVolThresh)
    RelVolThresh    =   inf;
end

%% Initialize
n0         	=   size(p0,2);
p          	=   zeros(3,n);
v          	=   zeros(3,n);
count      	=   0;
drV         =   zeros(1,n);
[~,RelVol]	=   volumeOccupiedByCylinders( p0, v0, r0, BoxDims, BoxCenter );
iter       	=   1;

%% Generate cylinders that may intersect except with (p0,v0,r0)
loopcond	=   true;
while loopcond
    if count+1 > numel(r); r = [r,r(1:n)]; end
    [p1,v1,~]   =	generateRandomCylinder( ...
        BoxDims, BoxCenter, r(count+1), [], CylOrientation, false );
    
    if ~any( cylinderIntersection( p1, v1, r(count+1), p0, v0, r0 ) )
        count	=	count + 1;
        [p(:,count),v(:,count)]	=	deal( p1, v1 );
        [~,drV(count)]          =	volumeOccupiedByCylinders( ...
            p(:,count), v(:,count), r(count), BoxDims, BoxCenter );
        RelVol	=   RelVol + drV(count);
    end
    
    if isinf(RelVolThresh), loopcond = (count < n+n0);
    else                    loopcond = (RelVol < RelVolThresh);
    end
    iter = iter + 1;
end
n	=   count;
p	=   p(:,1:n);
v	=   v(:,1:n);
r	=   r(1:n);

errRelV	=   RelVol - RelVolThresh;
[~,idx]	=	min( abs( drV - errRelV ) );
if ~isinf(RelVolThresh) && ( abs(RelVol - RelVolThresh) > abs(RelVol-drV(idx) - RelVolThresh) )
    % Last set of cylinders was closer to threshold
    idx	=   [1:idx-1,idx+1:n];
    p	=   p(:,idx);
    v	=   v(:,idx);
    r	=   r(idx);
    n	=   n-1;
end

[p,v,r]	=   deal([p0,p],[v0,v],[r0,r]);

hh	=   [];
if PlotCylinders

    N	=	zeros( [ size(v), 2 ] );
    for ii = 1:n+n0
        N(:,ii,:)	=   null( v(:,ii)' );
    end
    
    hh	=	plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, ...
        sprintf('Non-intersecting Cylinders: N = %d', n+n0) );
    
    drawnow
end

if nargout > 3
    h	=   hh;
end

end

function b = isPointInBox(p,B,B0)

b	=	all(	bsxfun( @ge, B0(:)+B(:)/2, p )	&	...
                bsxfun( @le, B0(:)-B(:)/2, p ),     ...
                1	);

end

function h = plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, titlestr )

if nargin < 6; titlestr = ''; end;

h	=	figure; hold on
fig	=	get(gcf,'Number');

BoxBounds	=   [   BoxCenter(:)' - 0.5*BoxDims(:)'
                    BoxCenter(:)' + 0.5*BoxDims(:)' ];
rectpatchPlot( BoxBounds, fig );

for ii = 1:size(p,2)
    cylinderPlot( p(:,ii), v(:,ii), r(ii), sqrt(3)*max( BoxDims ), fig );
end

axis image
axis( BoxBounds(:)' )

xlabel('x'); ylabel('y'); zlabel('z');
if ~isempty( titlestr ); title( titlestr ); end;

end

