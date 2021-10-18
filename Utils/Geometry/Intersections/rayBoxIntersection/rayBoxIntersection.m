function [tmin, tmax, Lmin, Lmax, Lmid, Pmin, Pmax] = rayBoxIntersection( Origins, Directions, BoxDims, BoxCenter )
%RAYBOXINTERSECTION 

%% Input parsing
dim	=   size( Origins, 1 );

if nargin < 4 || isempty( BoxCenter )
    BoxCenter	=	zeros( dim, 1 );
end

%% Calculate intersections
if nargout > 5
    [tmin, tmax, Lmin, Lmax, Pmin, Pmax]	=	...
        rayBoxIntersection_ND( Origins, Directions, BoxDims, BoxCenter );
else
    [tmin, tmax, Lmin, Lmax]	=	...
        rayBoxIntersection_ND( Origins, Directions, BoxDims, BoxCenter );
end

if nargout > 4
    Lmid	=   0.5 * ( Lmin + Lmax );
end

%% Testing against brute force
% m	=	3;
% n	=   14;
% N	=	1e4;
% [Origins, Directions, BoxDims, BoxCenter]	=   deal(	...
%     10000*(2*rand(m,n)-1),	unit(2*rand(m,n)-1,1),      ...
%     20000*rand(m,1),        1e5*(2*rand(m,1)-1)	);
% rayBoxIntersection_testing( Origins, Directions, BoxDims, BoxCenter, N );

end

function [tmin,tmax,Lmin,Lmax,Pmin,Pmax] = rayBoxIntersection_ND( p, v, B, B0 )

f_sign	=   @(x) sign(x) + (x==0); % sign(0) := 1
f_tmin	=	@(p,v,B,B0,ii)	bsxfun( @rdivide, ...
        B0(ii) - 0.5 * B(ii) * f_sign(v(ii,:)) - p(ii,:), v(ii,:) ); %f_sign(v(ii,:)) .* max(abs(v(ii,:)), eps) );
f_tmax	=	@(p,v,B,B0,ii)	bsxfun( @rdivide, ...
        B0(ii) + 0.5 * B(ii) * f_sign(v(ii,:)) - p(ii,:), v(ii,:) ); %, f_sign(v(ii,:)) .* max(abs(v(ii,:)), eps) );

tmin	=   f_tmin( p, v, B, B0, 1 );
tmax	=   f_tmax( p, v, B, B0, 1 );
b       =   false( size( tmin ) );

if nargout > 4
    Pmin	=   NaN( size( tmin ) );
    Pmax	=   NaN( size( tmin ) );
    Pmin(~isnan(tmin))	=   1;
    Pmax(~isnan(tmax))	=   1;
end

for ii	=  	2:size(p,1)
    
    t0      =   f_tmin( p, v, B, B0, ii );
    t1      =   f_tmax( p, v, B, B0, ii );
    
    t0(v(ii,:) == 0) = -Inf;
    t1(v(ii,:) == 0) = +Inf;
    
    bnew	=   ( tmin > t1 ) | ( tmax < t0 );
    b       =   b | bnew;
    
    if all( b )
        [tmin, tmax]	=	deal( NaN(size(b)) );
        break
    else
        tmin(bnew)	=   NaN;
        tmax(bnew)	=   NaN;
    end
    
    if nargout > 4
        Imin            =   t0 > tmin;
        tmin(Imin(~b))	=   t0(Imin(~b));
        Pmin(Imin(~b))	=   ii;
        
        Imax            =   t1 < tmax;
        tmax(Imax(~b))	=   t1(Imax(~b));
        Pmax(Imax(~b))	=   ii;
    else
        tmin(~b)	=   max( tmin(~b), t0(~b) );
        tmax(~b)	=   min( tmax(~b), t1(~b) );
    end
    
end

Lmin	=   bsxfun( @plus, p, bsxfun( @times, tmin, v ) );
Lmax	=   bsxfun( @plus, p, bsxfun( @times, tmax, v ) );

end

function [tmin,tmax,Lmin,Lmax] = rayBoxIntersection_ND_brute( p, v, B, B0, N )

% get initial guess
Corners	=   bsxfun( @plus,	B0(:), bsxfun(	@times,	...
                            0.5 * B(:), permuteVec([-1,1],3)'	) );
[~,t]       =	projPointLine( Corners, v, p, 1 );
[t0,t1,dt]	=	deal( min(t), max(t), max(t)-min(t) );
t       =   linspace( t0 - 0.1*dt, t1 + 0.1*dt, N );
Points	=   bsxfun( @plus, p, bsxfun( @times, v, t ) );

bools	=   isPointInBox( Points, B, B0 );
tmin	=   t( find( bools, true, 'first' ) );
tmax	=   t( find( bools, true, 'last' ) );

if isempty(tmin)
    [tmin,tmax,Lmin,Lmax] = deal( NaN, NaN, NaN*p, NaN*p );
    return
end

% fzero to find approx. tmin/tmax
f       =   @(t) sign( -1 + 2 * isPointInBox( p+t*v, B, B0 ) );
dt      =   tmax - tmin;
tmin	=   fzero( f, tmin + dt/4*[-1,1] );
tmax	=   fzero( f, tmax + dt/4*[-1,1] );

if isnan(tmin)
    disp(tmin);
end

if nargout > 2
    Lmin	=   p + tmin * v;
    Lmax	=   p + tmax * v;
end

end

function b = isPointInBox(p,B,B0)

b	=	all(	bsxfun( @ge, B0(:)+B(:)/2, p )	&	...
                bsxfun( @le, B0(:)-B(:)/2, p ),     ...
                1	);

end


function rayBoxIntersection_testing( Origins, Directions, BoxDims, BoxCenter, N )

% analytic
[tmin, tmax, Lmin, Lmax]	=	...
    rayBoxIntersection_ND( Origins, Directions, BoxDims, BoxCenter );

bmin = isnan(tmin); bmax = isnan(tmax);
if ~all(bmin)
    [b0,b1,b2] = deal(	...
        all( isPointInBox( bsxfun(@plus, Lmin(:,~bmin),1e-6*Directions(:,~bmin)), BoxDims, BoxCenter ) ),                ...
        all( isPointInBox( bsxfun(@minus,Lmax(:,~bmax),1e-6*Directions(:,~bmax)), BoxDims, BoxCenter ) ),                ...
        all( isPointInBox( 0.5*(Lmin(:,~bmin)+Lmax(:,~bmax)), BoxDims, BoxCenter ) ) );
    fprintf( 'Analytic solution:\nLmin:\t%d\nLmax:\t%d\nLmid:\t%d\n\n', b0, b1, b2 );
    if ~(b0 && b1 && b2)
        disp([b0;b1;b2]);
    end
end

% approximate
% [tmin_b, tmax_b]	=   deal( zeros(size(tmin)) );
% [Lmin_b, Lmax_b]	=   deal( zeros(size(Lmin)) );
% 
% for ii = 1:length( tmin_b )
%     [tmin_b(ii), tmax_b(ii), Lmin_b(:,ii), Lmax_b(:,ii)]	=	...
%         rayBoxIntersection_ND_brute( Origins(:,ii), Directions(:,ii), ...
%             BoxDims, BoxCenter, N );
% end
% 
% % comparing
% b01 = isnan( tmin ); b02 = isnan( tmin_b );
% b11 = isnan( tmax ); b12 = isnan( tmax_b );
% all_b0 = all( b01 == b02 );
% all_b1 = all( b11 == b12 );
% 
% fprintf( 'tmin NaN equality: %d\ntmax NaN equality: %d\n\n', all_b0, all_b1 );
% if ~all_b0 || ~all_b1
%     disp( [b01; b02; b11; b12] );
% end
% 
% fprintf( 'tmin comparison:\n' );
% if ~all_b0, b0 = ~(b01 | b02); else b0 = ~b01; end
% tmn     = tmin(b0);
% tmn_b	= tmin_b(b0);
% tmin_err = abs(tmn-tmn_b);
% if sum(b0) <= 20
%     disp( [ tmn; tmn_b; tmin_err ] );
% else
%     disp( [ tmn(1:20); tmn_b(1:20); tmin_err(1:20) ] );
% end
% 
% fprintf( 'tmax comparison:\n' );
% if ~all_b1, b1 = ~(b11 | b12); else b1 = ~b02; end
% tmx     = tmax(b1);
% tmx_b	= tmax_b(b1);
% tmax_err	=	abs(tmx-tmx_b);
% if length(tmx) <= 20
%     disp( [ tmx; tmx_b; tmax_err ] );
% else
%     disp( [ tmx(1:20); tmx_b(1:20); tmax_err(1:20) ] );
% end
% 
% fprintf( 'tmin/tmin_b maximum error: %0.16f\n', max(tmin_err) );
% fprintf( 'tmax/tmax_b maximum error: %0.16f\n', max(tmax_err) );
% fprintf( '\n\n' );

end













