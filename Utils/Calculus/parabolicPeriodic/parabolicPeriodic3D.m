function [ t, u, J ] = parabolicPeriodic3D( ...
    a, b, c, d, u0, xb, yb, zb, Nx, Ny, Nz, T, prec, loc, J, solvertype, varargin )
%PARABOLICPERIODIC3D Solves the general 3D parabolic equation with periodic
% boundary conditions on u(x,y,z,t). The equation under consideration is:
% 
%   u_t = a(x,t)*u_xx + b(x,t)*u_yy + c(x,t)*u_zz + d(x)*u
% 
% INPUT ARGUMENTS
%	a,b,c,d:	Coefficients on u_xx, u_yy, u_zz, and u
%   u0:         Value of u(x,y,z,0). u0 may be a function handle or an
%               array of values that is consistent with the domain size (up
%               to scalar expansion)
%   xb,yb,zb:   x, y, and z limits of the domain
%   Nx,Ny,Nz:	Number of gridpoints to simulated, including endpoints
%   T:          Vector of time points to simulate
%   prec:       Precision ('single' or 'double'; defaults to class(u0))
%   loc:        Location of gridpoints; include boundary points or not
%               ('endpoints' or 'interior'; default 'endpoints')
%   J:          Precomputed Jacobian can be supplied (default empty).
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
if nargin < 13 || isempty(prec)
    if ~isa(u0,'function_handle');	prec = class(u0);
    else rnd = @(x) x(1)+rand*x(2);	prec = class(u0(rnd(xb),rnd(yb),rnd(zb)));
    end
end
if nargin < 14 || isempty(loc)
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
if nargin < 15; J = []; end
if nargin < 16; solvertype = 'expmv'; end

x	=   linspace(xb(1),xb(2),Nx).';
y	=   linspace(yb(1),yb(2),Ny).';
z	=   linspace(zb(1),zb(2),Nz).';

[ax, aTimeDep, aSpaceDep]	=   handle_coefficient( a, x, y, z );
[bx, bTimeDep, bSpaceDep]	=   handle_coefficient( b, x, y, z );
[cx, cTimeDep, cSpaceDep]	=   handle_coefficient( c, x, y, z );
[dx, dTimeDep, dSpaceDep]	=   handle_coefficient( d, x, y, z );
isTimeDep	=	( aTimeDep  || bTimeDep  || cTimeDep  || dTimeDep  );
isSpaceDep	=	( aSpaceDep || bSpaceDep || cSpaceDep || dSpaceDep );
closedForm	=  ~( isTimeDep );

u0          =   handle_u0( u0, x, y, z );
[t,tidx]	=	handle_T( T, closedForm );

%==========================================================================
% Setup Problem
%==========================================================================

N                   =	(Nx-1)*(Ny-1)*(Nz-1);
[iix,  iiy,  iiz ]	=	deal( 1:Nx-1, 1:Ny-1, 1:Nz-1 );
[iixf, iiyf, iizf]  =   deal( circshift(iix,-1,2), circshift(iiy,-1,2), circshift(iiz,-1,2) );
[iixb, iiyb, iizb]  =   deal( circshift(iix, 1,2), circshift(iiy, 1,2), circshift(iiz, 1,2) );

xii	=   x(iix); %remove endpoint (periodic BC's)
yii	=   y(iiy);
zii	=   z(iiz);

[hx,  hy,  hz  ]	=   deal( diff(xb)/(Nx-1), diff(yb)/(Ny-1), diff(zb)/(Nz-1) );
[ihx2,ihy2,ihz2]	=   deal( 1/hx^2, 1/hy^2, 1/hz^2 );

%==========================================================================
% Compute u(x,y,z,t)
%==========================================================================
if closedForm
    %tic
    
    u	=   zeros(N,length(t),prec);
    t	=   double(t);
    u0	=   double(u0);
    
    % Calculate jacobian if not provided
    if isempty(J)
        %tic
        [J,muJ]	=	deal(JFun,0);
        %[J,muJ]	=	J_matfun;
        %toc
    end
    
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
            solver	=   @(ii) expmv_solver(t(ii),J,u0,prec,muJ);
            
        otherwise
            warning('Invalid solver type. Defaulting to ''expmv''.');
            solver	=   @(ii) expmv_solver(t(ii),J,u0,prec,muJ);
            
    end
    
    % Compute solution
    for ii = 1:length(t)
        %tic
        u(:,ii)	=	reshape( solver(ii), [], 1 );
        %toc
    end
    
    %toc
else
    %tic
    options	=	odeset('abstol',1e-3,'reltol',1e-3);
    [~,u]	=	ode45(@dudt_fun,t,u0,options);
    u       =	u(tidx,:).';
    %toc
end

% Reshape u for output, adding back periodic endpoints (if necessary)
if ~isequal(size(u),[Nx-1,Ny-1,Nz-1,length(tidx)])
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
    function dudt = dudt_fun(t,u)
        
        usize	=   size(u);
        if ~isequal(size(u),[Nx-1,Ny-1,Nz-1]);
            u       =   reshape( u, Nx-1, Ny-1, Nz-1 );
        end
        
        if dTimeDep
            dudt	=   bsxfun( @times, dx(xii,yii,zii,t), u );
        else
            dudt	=   bsxfun( @times, dx, u );
        end
        
        um2     =   -2.0 .* u;
        
        uf      =   circshift(u,-1,1);
        ub      =   circshift(u, 1,1);
        uf      =   uf + um2 + ub;
        if aTimeDep
            dudt	=	dudt + bsxfun( @times, ihx2 * ax(xii,yii,zii,t), uf );
        else
            dudt	=	dudt + bsxfun( @times, ihx2 * ax, uf );
        end
        
        uf      =   circshift(u,-1,2);
        ub      =   circshift(u, 1,2);
        uf      =   uf + um2 + ub;
        if bTimeDep
            dudt	=   dudt + bsxfun( @times, ihy2 * bx(xii,yii,zii,t), uf );
        else
            dudt	=	dudt + bsxfun( @times, ihy2 * bx, uf );
        end
        
        uf      =   circshift(u,-1,3);
        ub      =   circshift(u, 1,3);
        uf      =   uf + um2 + ub;
        if cTimeDep
            dudt	=   dudt + bsxfun( @times, ihz2 * cx(xii,yii,zii,t), uf );
        else
            dudt	=	dudt + bsxfun( @times, ihz2 * cx, uf );
        end
        
        if ~isequal(size(dudt),usize)
            dudt	=   reshape( dudt, usize );
        end
        
    end
    
    %======================================================================
    % Jacobian calculation
    %======================================================================
    function J = JFun(~,~)
        
        if isTimeDep
            error( 'Jacobian implementation only for time-independent coefficients' );
        end
        
        % sparse matrices must have type 'double'
        [ex,ey,ez]	=   deal( ones(Nx-1,1), ones(Ny-1,1), ones(Nz-1,1) );
        E           =   ones( Nx-1, Ny-1, Nz-1 );
        
        Jx      =	spdiags( [ex, -2*ex, ex], [-1 0 1], Nx-1, Nx-1 );
        Jy      =	spdiags( [ey, -2*ey, ey], [-1 0 1], Ny-1, Ny-1 );
        Jz      =	spdiags( [ez, -2*ez, ez], [-1 0 1], Nz-1, Nz-1 );
        
        [	Jx(1,end), Jy(1,end), Jz(1,end),	...
            Jx(end,1), Jy(end,1), Jz(end,1)     ]	=   deal(1);    % periodic BC's
        
        [Jx,Jy,Jz]	=	deal( ihx2 * Jx, ihy2 * Jy, ihz2 * Jz );
        [Ix,Iy,Iz]	=	deal( speye(Nx-1), speye(Ny-1), speye(Nz-1) );
        
        coeff	=   bsxfun(@times, double(dx), E);
        J       =	spdiags(coeff(:),0,N,N);
        
        coeff	=   bsxfun(@times, double(ax), E);
        L       =   kron(Iz, kron(Iy, Jx));             % d^2/dx^2 matrix
        J       =   J + spdiags(coeff(:),0,N,N) * L;	% multiply by a(x,y,z)
        
        coeff	=   bsxfun(@times, double(bx), E);
        L       =   kron(Iz, kron(Jy, Ix));             % d^2/dy^2 matrix
        J       =   J + spdiags(coeff(:),0,N,N) * L;	% multiply by b(x,y,z)
        
        coeff	=   bsxfun(@times, double(cx), E);
        L       =   kron(kron(Jz, Iy), Ix);             % d^2/dz^2 matrix
        J       =   J + spdiags(coeff(:),0,N,N) * L;	% multiply by c(x,y,z)
        
    end
    
    function [J,muJ] = J_matfun(~,~)
        
        abc     =   (-2.*ihx2).*ax(:) + (-2.*ihy2).*bx(:) + (-2.*ihz2).*cx(:) + dx(:);
        muJ     =   sum( abc )/N;
        normJmu	=   max( abs(abc-muJ) + abs((-2.*ihx2).*ax(:)) + abs((-2.*ihy2).*bx(:)) + abs((-2.*ihz2).*cx(:)) );
        clear abc
        
        J       =   @(t) setop( shift( matfun( @(u) t.*dudt_fun([],u), [numel(u),numel(u)] ), t.*muJ ), ...
                    'class', @(~) class(u), 'trace', @(~) 0, 'norm', @(~,p) t.*normJmu );
    end
    
    function u = expmv_solver(t,J,u0,prec,muJ)
        
        tol	=   2^-23;
        if issparse(J)
            args	=   { [], 'double', true, false, false, false, false, tol };
        else
            args	=   { [], 'double', false, false, false, false, false, tol };
        end
        
        if issparse(J)
            u	=   expmv( t, J, u0(:), args{:} );
        else
            u	=   expmv( [], J(t), u0, args{:} );
            u	=   exp(t*muJ) .* u;
        end
        
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

function A = inputArray2func(A,Nx,Ny,Nz)

A	=	checkInputArraySize(A,Nx,Ny,Nz);
A	=   @(xx,yy,zz,tt) A;

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
