function [ t, u, J ] = diffuse3D( D, f, u0, xb, yb, zb, Nx, Ny, Nz, T, prec, loc, solvertype, tol )
%DIFFUSE3D Solves the general 3D parabolic equation with periodic
% boundary conditions on u(x,y,z,t). The equation under consideration is:
% 
%   u_t = D*lap(u) - f*u
% 
% INPUT ARGUMENTS
%	D:          Diffusion coefficient (scalar constant)
%   f:          Decay term (3D array)
%   u0:         Value of u(x,y,z,0). u0 may be a function handle or an
%               array of values that is consistent with the domain size (up
%               to scalar expansion)
%   xb,yb,zb:   x, y, and z limits of the domain
%   Nx,Ny,Nz:	Number of gridpoints to simulated, including endpoints
%   T:          Vector of time points to simulate
%   prec:       Precision ('single' or 'double'; defaults to class(u0))
%   loc:        Location of gridpoints; include boundary points or not
%               ('endpoints' or 'interior'; default 'endpoints')
%   solvertype:	Solver type for time-independent problem ('expmv' or 'cn')
% 
% OUTPUT ARGUMENTS
%   u:          Solution function. size of u is [Nx,Ny,Nz,length(T)]

%==========================================================================
% Input handling
%==========================================================================
if ~all( [numel(xb),numel(yb),numel(zb)] == 2 ), error( 'Must provide exactly two endpoints' );
elseif ~all( [diff(xb),diff(yb),diff(zb)] > 0 ), error( 'Endpoints must be non-decreasing' );
end
if any([Nx,Ny,Nz] < 4), error( 'Must have at least 4 gridpoints' ); end
if any(T<0), error( 'T must be non-negative' ); end
if nargin < 11 || isempty(prec)
    if ~isa(u0,'function_handle');	prec = class(u0);
    else rnd = @(x) x(1)+rand*x(2);	prec = class(u0(rnd(xb),rnd(yb),rnd(zb)));
    end
end
if nargin < 12 || isempty(loc)
    loc         =   'endpoints';
end
if strcmpi( loc, 'interior' )
    isInterior	=   true;
    fxb         =   @(xb,Nx) xb + diff(xb)/Nx*[0.5,0.5];
    [xb,yb,zb]	=   deal( fxb(xb,Nx), fxb(yb,Ny), fxb(zb,Nz) );
    [Nx,Ny,Nz]	=   deal( Nx+1, Ny+1, Nz+1 );
else
    if ~strcmpi( loc, 'endpoints' )
        warning( [  'Location of points must be ''endpoints'' or ''interior''',	...
                    'using default value of ''endpoints''' ] );
    end
    isInterior	=   false;
end
if nargin < 13 || isempty(solvertype)
    solvertype  =   'expmv';
end
if nargin < 14 || isempty(tol)
    %tol     =   2^-24;
    %tol     =   2^-53;
    tol     =   1e-10;
end

x	=   linspace(xb(1),xb(2),Nx).';
y	=   linspace(yb(1),yb(2),Ny).';
z	=   linspace(zb(1),zb(2),Nz).';

fx	=   handle_coefficient( f, x, y, z );

u0          =   handle_u0( u0, x, y, z );
[t,tidx]	=	handle_T( T, true );

%==========================================================================
% Setup Problem
%==========================================================================

N	=	(Nx-1)*(Ny-1)*(Nz-1);
[hx,  hy,  hz  ]	=   deal( diff(xb)/(Nx-1), diff(yb)/(Ny-1), diff(zb)/(Nz-1) );
[ihx2,ihy2,ihz2]	=   deal( 1/hx^2, 1/hy^2, 1/hz^2 );

if ~( abs(hx-hy) < 1e-14 && abs(hy-hz) < 1e-14 )
    error('diffuse3D requires isotropic voxels.');
end

%==========================================================================
% Compute u(x,y,z,t)
%==========================================================================
%tic

u	=   zeros([Nx-1,Ny-1,Nz-1,length(t)],prec);
t	=   double(t);
u0	=   double(u0);

% Calculate jacobian matfun
%tic
[J,muJ,normJmu]	=	J_matfun;
%toc

switch upper(solvertype)
    case 'CN'
        % Crank-Nicolson based solver. Solution is computed through
        % adaptive time stepping of the ode system du/dt = A*u
        options	=   repmat(parabolicATSoptions,numel(t),1);
        for ii = 1:numel(t)
            options(ii)	=   parabolicATSoptions( u0, t(ii), varargin{:} );
        end
        %linsolver	=   @(A,b)  A\b;
        linsolver	=   @(A,b)  FMG(b,A);
        stepper     =   @(u,dt) cn(J,u,dt,linsolver);
        solver      =   @(ii)   parabolicATS(u0,t(ii),stepper,options(ii));
        
    case 'EXPMV'
        % Direct exponential matrix calculation based solver. Solution
        % is determined via computing the closed form directly:
        %	u(x,y,z,t) = expm(J*t)*u(x,y,z,0)
        solver	=   @(ii) expmv_solver(t(ii),u0,prec,muJ,normJmu);
        
    otherwise
        warning('Invalid solver type. Defaulting to ''expmv''.');
        solver	=   @(ii) expmv_solver(t(ii),u0,prec,muJ,normJmu);
        
end

% Compute solution
for ii = 1:length(t)
    %tic
    u(:,:,:,ii)	=	solver(ii);
    %toc
end

%toc

% Reshape u for output, adding back periodic endpoints (if necessary)
padones = @(x,n) [x,ones(1,n-length(x))];
if ~isequal(padones(size(u),4),[Nx-1,Ny-1,Nz-1,length(tidx)])
    u	=   reshape( u, Nx-1, Ny-1, Nz-1, length(tidx) );
end
if ~isInterior
    u	=   cat( 1, u, u(1,:,:,:) );
    u	=   cat( 2, u, u(:,1,:,:) );
    u	=   cat( 3, u, u(:,:,1,:) );
end

    %======================================================================
    % Derivative calculation
    %======================================================================
      
    function [J,muJ,normJmu] = J_matfun
        
        abc     =   (-6.*ihx2.*D) - fx(:);
        muJ     =   sum( abc )/N;
        normJmu	=   max( abs(abc-muJ) + abs(-6.*ihx2.*D) );
        clear abc
        
        dudt	=   @(u) fmg_diffuse(u,hx,D,fx);
        J       =   @(t) setop( shift( matfun( @(u) t.*dudt(u), [numel(u),numel(u)] ), t.*muJ ), ...
                    'class', @(~) class(u), 'trace', @(~) 0, 'norm', @(~,p) t.*normJmu, 'issparse', @(~) false );
        
    end
    
    function u = expmv_solver(t,u0,prec,muJ,normJmu)
        
        args	=   { [], 'double', false, false, false, true, false, tol };
        
        %u	=   expmv( [], J(t), u0, args{:} );
        u	=   expmv_diffuse3D( t, [], u0, args{:}, hx, D, fx, muJ, normJmu );
        u	=   exp(t*muJ) .* u;
        
        if ~isa(u,prec)
            u	=   cast(u,prec);
        end
            
    end
    
end

function [coeff, isTimeDep, isSpaceDep] = handle_coefficient( coeff, x, y, z )

% True if coefficient is time dependent
isTimeDep	=   false;
isSpaceDep	=   false;

if isa( coeff, 'function_handle')
    switch nargin( coeff )
        case 1 % t-dependent
            coeff       =   @(xx,yy,zz,tt) coeff(tt);
            isTimeDep	=   true;
        case 3 % x,y,z-dependent
            coeff       =   coeff( x(1:end-1), y(1:end-1), z(1:end-1) );
            coeff       =   checkInputArraySize( coeff, length(x), length(y), length(z) );
            isSpaceDep	=   true;
        case 4 % x,y,z,t-dependent
            isTimeDep	=   true;
            isSpaceDep	=   true;
        otherwise
            error( 'Coefficients must be functions f(t), f(x,y,z), or f(x,y,z,t) only' );
    end
else
    coeff	=	checkInputArraySize(coeff,length(x),length(y),length(z));
    if ~isscalar(coeff)
        isSpaceDep	=   true;
    end
end

end

function u0 = handle_u0( u0, x, y, z )

[Nx,Ny,Nz]	=   deal(length(x),length(y),length(z));

if isa( u0, 'function_handle' )
    u0	=   u0(x(1:Nx-1),y(1:Ny-1),z(1:Nz-1));
else
    u0	=	checkInputArraySize(u0,Nx,Ny,Nz);
end

% singleton expand u0 to size [Nx-1,Ny-1,Nz-1]
u0	=   bsxfun( @times, u0, ones(Nx-1,Ny-1,Nz-1,'like',u0) );

end

function A = checkInputArraySize(A,Nx,Ny,Nz)

Asize	=   size( A );
if length(Asize) == 2, Asize = [Asize, 1]; end

if Asize(1) == Nx
    A	=	A(1:end-1,:,:);
elseif ~( Asize(1) == 1 || Asize(1) == Nx-1 )
    error( 'size(array,1) must be 1, Nx-1, or Nx' );
end

if Asize(2) == Ny
    A	=	A(:,1:end-1,:);
elseif ~( Asize(2) == 1 || Asize(2) == Ny-1 )
    error( 'size(array,2) must be 1, Ny-1, or Ny' );
end

if Asize(3) == Nz
    A	=	A(:,:,1:end-1);
elseif ~( Asize(3) == 1 || Asize(3) == Nz-1 )
    error( 'size(array,3) must be 1, Nz-1, or Nz' );
end

end

function [t,tidx] = handle_T( T, closedForm )

T	=   T(:);
tidx=	[];

if any( T < 0 ) || any( diff(T) <= 0 )
    error( 'T must be non-negative and strictly increasing' );
end

if closedForm
    t	=   T;
    tidx=	1:length(T);
    return
end

if T(1) == 0
    T	=   T(2:end);
    tidx=	1;
end

switch length(T)
    case 1
        t	=	linspace(0,T,3);
        tidx=	[tidx, 3];
    otherwise
        t	=   [0,T.'];
        tidx=	[tidx, 2:length(T)+1];
end

end
