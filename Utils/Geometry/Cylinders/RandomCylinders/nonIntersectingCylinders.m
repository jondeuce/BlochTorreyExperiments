function [ p, v, r, h ] = nonIntersectingCylinders( ...
    BoxDims, BoxCenter, Radii, NumCylinders, CylOrientation, PlotCylinders, ...
    p0, v0, r0, RelVolThresh )
%NONINTERSECTINGCYLINDERS Generates a set of non-intersecting cylinders
%contained in the box with dimensions BoxDims = [L,W,H] where the center of
%the box is located at BoxCenter = [x0,y0,z0].

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
    PlotCylinders   =   false;
end

if nargin < 7
    [p0,v0,r0]      =   deal( [] );
end

if nargin < 10 || isempty(RelVolThresh)
    RelVolThresh    =   inf;
end

%% Initialize
if isempty(p0) || isempty(v0) || isempty(r0)
    n0              =   0;
    [p,v]           =	deal( zeros(3,n) );
    [p0,v0,~]       =	generateRandomCylinder( ...
        BoxDims, BoxCenter, r(1), [], CylOrientation, false );
    [p(:,1),v(:,1)]	=   deal( p0, v0 );
    [~,RelVol]      =   volumeOccupiedByCylinders( p(:,1), v(:,1), r(1), BoxDims, BoxCenter );
    count           =   1;
    iter            =   1;
else
    n0              =   size(p0,2);
    p               =   [ p0, zeros(3,n) ];
    v               =   [ v0, zeros(3,n) ];
    r               =   [ r0, r ];
    count           =   n0;
    [~,RelVol]      =   volumeOccupiedByCylinders( p0, v0, r0, BoxDims, BoxCenter );
    iter            =   1;
end

%% Generate non-intersecting cylinders
loopcond	=   true;
while loopcond
    if count+1 > numel(r); r = [r,r(1:n)]; end
    [p0,v0,~]   =	generateRandomCylinder( ...
        BoxDims, BoxCenter, r(count+1), [], CylOrientation, false );
    
    if ~any( cylinderIntersection( p0, v0, r(count+1), p(:,1:count), v(:,1:count), r(1:count) ) )
        count   =	count + 1;
        [p(:,count),v(:,count)]	=	deal( p0, v0 );
        [~,drV]	=	volumeOccupiedByCylinders( p(:,count), v(:,count), r(count), BoxDims, BoxCenter );
        RelVol	=   RelVol + drV;
    end
    
    if isinf(RelVolThresh), loopcond = (count < n+n0);
    else                    loopcond = (RelVol < RelVolThresh);
    end
    iter = iter + 1;
end
n	=   count - n0;
p	=   p(:,1:n+n0);
v	=   v(:,1:n+n0);
r	=   r(1:n+n0);

hh	=   [];
if PlotCylinders
    hh	=	plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, ...
        sprintf('Non-intersecting Cylinders: N = %d', n+n0) );
end

if nargout > 3
    h	=   hh;
end

%% Minimize potential energy of system
% Results = [];
% 
% N	=	zeros( [ size(v), 2 ] );
% for ii = 1:n
%     N(:,ii,:)	=   null( v(:,ii)' );
% end
% 
% % displacement functional
% % Mobility	=   0.5;
% % [x0,lb,ub]	=   deal( zeros(1,2*n) );
% % ub(:)	=   Mobility * max( BoxDims(:) );
% % lb	=   -ub;
% % E0	=   energy_functional( x0, n, p, v, r, N, BoxDims, BoxCenter );
% 
% % angular functional
% % PhiMax	=   10 * (pi/180);
% % [x0,lb,ub]	=   deal( zeros(1,2*n) );
% % x0(1:2:end) = 1e-3;
% % lb(1:2:end)	=   1e-3;
% % ub(1:2:end)	=   PhiMax;
% % ub(2:2:end)	=   2*pi;
% % E0	=   energy_functional( x0, n, p, v, r, N, BoxDims, BoxCenter );
% 
% % combined functional
% Mobility	=   0.5;
% PhiMax	=   45 * (pi/180);
% [x0,lb,ub]	=   deal( zeros(1,4*n) );
% 
% ub(1:4:end)	=   Mobility * max( BoxDims(:) );
% ub(2:4:end)	=   ub(1:4:end);
% lb          =	-ub;
% 
% x0(3:4:end) =	1e-3;
% lb(3:4:end)	=   1e-3;
% ub(3:4:end)	=   PhiMax;
% ub(4:4:end)	=   2*pi;
% 
% E0	=   energy_functional( x0, n, p, v, r, N, BoxDims, BoxCenter );
% 
% 
% if PlotCylinders
% %     close all force
%     plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, ...
%         sprintf( 'Before: N = %d, E = %0.2f', n, E0 ) );
%     drawnow
% end
% 
% options     =	optimoptions( 'fmincon', ...
%     'UseParallel', true, 'Display', 'iter', 'Algorithm', 'sqp', ...
%     'TolFun', 0.01, 'TolX', 1e-6, 'MaxFunEvals', 200*length(x0) );
% 
% [x,E,~,output]	=	fmincon( ...
%     @(x) energy_functional( x, n, p, v, r, N, BoxDims, BoxCenter ), ...
% 	x0, [], [], [], [], lb, ub, [], options );
% 
% % [p,v,r] = displacement_functional( x(:)', n, p, v, r, N );
% % [p,v,r] = angular_functional( x(:)', n, p, v, r, N );
% [p,v,r] = combined_functional( x(:)', n, p, v, r, N );
% 
% if PlotCylinders
%     plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, ...
%         sprintf( 'After: N = %d, E = %0.2f', n, E ) );
%     drawnow
% end
% 
% if nargout >= 4
%     [~,~,~,~,Lmid]	=	rayBoxIntersection( p, v, BoxDims, BoxCenter );
%     isValid         =   all( isPointInBox( Lmid, BoxDims, BoxCenter ) );
%     
%     Results     =   struct( ...
%         'isSolutionValid',  isValid,...
%         'init_xval',        x0(:)',	...
%         'init_fval',        E0,     ...
%         'xval',             x(:)',	...
%         'fval',             E,      ...
%         'fmincon_output',   output	...
%         );
% end

end

function [p,v,r] = displacement_functional( x, n, p, v, r, N )
% 'x' has length 2*n, where x(2*ii-1) is the displacement of the cylinder
% point p(:,ii) in the N(:,ii,1) direction, and x(2*ii) is the displacement
% in the N(:,ii,2) direction, where N(:,ii,:) span the nullspace of v(:,ii)
%	i.e.	p(:,ii)
%       ->	p(:,ii) + x(2*ii-1) * N(:,ii,1) + x(2*ii) * N(:,ii,2)

p	=   p + bsxfun( @times, x(1:2:end), N(:,:,1) ) +    ...
            bsxfun( @times, x(2:2:end), N(:,:,2) );

end

function [p,v,r] = angular_functional( x, n, p, v, r, N )
% 'x' has length 2*n, where x(2*ii-1) and x(2*ii) are the phi/theta angles
% respectively in the basis B = [v(:,ii), N(:,ii,1), N(:,ii,2)]. The new
% 'v' vector is the vector with spherical coordinates phi/theta in B
%	i.e.	v(:,ii)
%       ->	B * [ cos(th)*sin(phi); sin(phi)*sin(th); cos(phi) ]

for ii = 1:n
    phi	=   x(2*ii-1);
    th	=   x(2*ii);
    u	=   [ cos(th)*sin(phi); sin(phi)*sin(th); cos(phi) ];
    B	=   [ N(:,ii,1), N(:,ii,2), v(:,ii) ];
    v(:,ii)	=   B * u;
end

end

function [p,v,r] = combined_functional( x, n, p, v, r, N )
% combines displacement_functional and angular_functional, with
% x(1:4:end)/x(2:4:end) for displacement, x(3:4:end)/x(4:4:end) for angles

p	=   p + bsxfun( @times, x(1:4:end), N(:,:,1) ) +    ...
            bsxfun( @times, x(2:4:end), N(:,:,2) );

phi =	x(3:4:end);
th	=   x(4:4:end);
for ii = 1:n
    u	=   [ cos(th(ii))*sin(phi(ii)); sin(phi(ii))*sin(th(ii)); cos(phi(ii)) ];
    B	=   [ N(:,ii,1), N(:,ii,2), v(:,ii) ];
    v(:,ii)	=   B * u;
end

end

function E = energy_functional( x, n, p, v, r, N, BoxDims, BoxCenter )

x       =   x(:)';
[p,v,r]	=	displacement_functional( x, n, p, v, r, N );
% [p,v,r]	=	angular_functional( x, n, p, v, r, N );
% [p,v,r]	=	combined_functional( x, n, p, v, r, N );

% normalize distances
Scaling         =   n / max( BoxDims );

% push cylinders away from walls
[~,~,~,~,p]     =	rayBoxIntersection( p, v, BoxDims, BoxCenter );
Wall_Distances	=   Scaling * min( min( cat( 3, ...
    bsxfun( @plus, -(BoxCenter - BoxDims/2),  p ),	...
    bsxfun( @plus,  (BoxCenter + BoxDims/2), -p )	...
    ), [], 3 ), [], 1 );

Wall_Energy     =	sum(1./Wall_Distances.^2);

% push cylinders away from each other
n               =	size(p,2);
Cylinder_Energy	=   0;
for ii = 1:n-1
    Mutual_Distances	=	Scaling * ...
        skewLineDist(	repmat( p(:,ii), [1,n-ii] ), p(:,ii+1:end),	...
                        repmat( v(:,ii), [1,n-ii] ), v(:,ii+1:end) );
    
	Cylinder_Energy     =	Cylinder_Energy + sum(1./Mutual_Distances.^2);
end

E	=   Wall_Energy + Cylinder_Energy;

end

function b = isPointInBox(p,B,B0)

b	=	all(	bsxfun( @ge, B0(:)+B(:)/2, p )	&	...
                bsxfun( @le, B0(:)-B(:)/2, p ),     ...
                1	);

end

