function [ t, u ] = parallel_diffuse3D( D, f, u0, xb, yb, zb, Nx, Ny, Nz, T, prec, loc, tol )
%PARALLEL_DIFFUSE3D Solves the general 3D parabolic equation with
% periodic boundary conditions on u. The equation under consideration is:
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
if ~isscalar(T) || any(T<0), error( 'T must be a scalar >= 0' ); end
if nargin < 12 || isempty(loc)
    loc         =   'endpoints';
end
if strcmpi( loc, 'interior' )
    isInterior	=   true;
    to_int      =   @(xb,Nx) xb + diff(xb)/Nx*[0.5,0.5];
    [xb,yb,zb]	=   deal( to_int(xb,Nx), to_int(yb,Ny), to_int(zb,Nz) );
    [Nx,Ny,Nz]	=   deal( Nx+1, Ny+1, Nz+1 );
else
    if ~strcmpi( loc, 'endpoints' )
        warning( [  'Location of points must be ''endpoints'' or ''interior''',	...
                    'using default value of ''endpoints''' ] );
    end
    isInterior	=   false;
end
if nargin < 13 || isempty(tol)
    %tol     =   2^-24;
    %tol     =   2^-53;
    tol     =   1e-10;
end

if ~isequal(size(u0),size(f)), error('u0 and f must have the same size'); end

%==========================================================================
% Setup Problem
%==========================================================================

M	=   size(f,4);
N	=	(Nx-1)*(Ny-1)*(Nz-1);
[hx, hy, hz]	=   deal( diff(xb)/(Nx-1), diff(yb)/(Ny-1), diff(zb)/(Nz-1) );

if ~( abs(hx-hy) < 1e-14 && abs(hy-hz) < 1e-14 )
    error('diffuse3D requires isotropic voxels.');
else
    [h,ih2]	=   deal(hx,1/hx^2);
end

%==========================================================================
% Compute u(x,y,z,t)
%==========================================================================
%tic

u	=   zeros(size(u0),prec);
t	=   double(T);
u0	=   double(u0);

% Compute solution
%tic
u	=	expmv_solver(t,u0,prec);
%toc

%toc

% Reshape u for output, adding back periodic endpoints (if necessary)
padones = @(x,n) [x,ones(1,n-length(x))];
if ~isequal(padones(size(u),4),[Nx-1,Ny-1,Nz-1,M])
    u	=   reshape( u, Nx-1, Ny-1, Nz-1, M );
end
if ~isInterior
    u	=   cat( 1, u, u(1,:,:,:) );
    u	=   cat( 2, u, u(:,1,:,:) );
    u	=   cat( 3, u, u(:,:,1,:) );
end

    %======================================================================
    % parallel expmv solver
    %======================================================================
    
    function u = expmv_solver(t,u0,prec)
        
        args	=   { [], 'double', false, false, false, false, false, tol };
        
        [muJ,normJ]	=   deal(zeros(M,1));
        for ii = 1:M
            d           =	reshape((-6.*ih2.*D) - f(:,:,:,ii), [], 1); % diagonal of sub-matrix
            muJ(ii)     =	sum( d )/N;
            normJ(ii)	=   max( abs(d-muJ(ii)) + abs(-6.*ih2.*D) );
            f(:,:,:,ii)	=   f(:,:,:,ii) + muJ(ii); %D*lap(u)-f-mu*I = D*lap(u)-(f+mu*I)
        end
        normJ	=   max(normJ);
        clear d
        
        % Absorb t into action of J: t*J = t*(D*lap(u)-f) = (t*d)*lap(u)-(t*f)
        u	=   parallel_expmv_diffuse3D( 1, [], u0, args{:}, h, t.*D, t.*f, [], t.*normJ );
        
        for ii = 1:M
            u(:,:,:,ii)	=   exp(t*muJ(ii)) .* u(:,:,:,ii);
        end
        
    end
    
end
