function [ u ] = parabolicPeriodic1D( a, b, c, u0, xb, N, T )
%PARABOLICPERIODIC1D Solves the general 1D parabolic equation with periodic
% boundary conditions on u(x,t) and u_x(x,t). The equation under
% consideration is:
% 
%   u_t = a(x,t)*u_xx + b(x,t)*u_x + c(x,t)*u
% 
% INPUT ARGUMENTS
%	a:  coefficient on u_xx
%   b:  coefficient on u_x
%   c:  coefficient on u
%   u0: value of u(x,0); may be function handle, vector of values or scalar
%   xb: domain boundary
%   N:  number of gridpoints to simulated, including endpoints
%   T:  vector of time points to simulate
% 
% OUTPUT ARGUMENTS
%   u:  solution function. size of u is [N x length(T)]

%==========================================================================
% Input handling
%==========================================================================
if numel(xb) ~= 2, error( 'Must provide exactly two endpoints' );
elseif diff(xb) < 0, error( 'Endpoints must be non-decreasing' );
end
if N < 5, error( 'Must have at least 5 gridpoints' ); end
if any(T<0), error( 'T must be non-negative' ); end

x	=   linspace(xb(1),xb(2),N).';
h	=   diff(xb)/(N-1);

ax	=   handle_coefficient( a, x );
bx	=   handle_coefficient( b, x );
cx	=   handle_coefficient( c, x );

N	=   N-1;
u0	=   handle_u0( u0, x );
x	=   x(1:end-1);

[t,tidx]	=	handle_T( T );

%==========================================================================
% Solve for u(x,t)
%==========================================================================
options	=	odeset('Jacobian',@JFun,'abstol',1e-3,'reltol',1e-3);
[~,u]	=	ode23s(@odefun,t,u0,options);
u       =   u(tidx,:).';
u       =   [u; u(1,:)];

    %======================================================================
    % Derivative calculation
    %======================================================================
    function dudt = odefun(t,u)
        dudt	=	ax(x,t) .* ( u([2:end,1]) - 2*u + u([end,1:end-1]) )/h^2 + ...
                  	bx(x,t) .* ( u([2:end,1]) - u([end,1:end-1]) )/(2*h) + ...
                  	cx(x,t) .* u;
    end
    
    %======================================================================
    % Jacobian calculation
    %======================================================================
    function J = JFun(t,u)
        e	=   ones(length(u),1); % for scalar expansion
        axx	=   ax(x,t) .* e;
        bxx	=   bx(x,t) .* e;
        cxx	=   cx(x,t) .* e;
        
        J	=   sparse(N,N);
        J	=   J + spdiags( [ [axx(2:end); 0], -2*axx, [0;axx(1:end-1)] ]/h^2, -1:1, N, N ) + ...
                    spdiags( [-[bxx(2:end-1);0;0], [0;0;bxx(2:end-1)] ]/(2*h), [-1,1], N, N ) + ...
                    spdiags( cxx, 0, N, N );
        
        J(1,1)	=  -1.5 * bxx(1  )/h + J(1,1);
        J(1,2)	=   2.0 * bxx(1  )/h + J(1,2);
        J(1,3)	=  -0.5 * bxx(1  )/h + J(1,3);
        J(N,N-2)=   0.5 * bxx(end)/h + J(N,N-2);
        J(N,N-1)=  -2.0 * bxx(end)/h + J(N,N-1);
        J(N,N)	=   1.5 * bxx(end)/h + J(N,N);
        J(1,N)	=   axx(1  )/h^2 + J(1,N);
        J(N,1)	=   axx(end)/h^2 + J(N,1);
    end

end

function coeff = handle_coefficient( coeff, x )

if isa( coeff, 'function_handle')
    
    switch nargin( coeff )
        case 1 % x-dependent
            coeff	=   @(y,s) coeff(x(1:end-1));
        case 2 % x,t-dependent
            % do nothing
        otherwise
            error( 'Coefficients must be functions of x and/or t only' );
    end
    
else
    
    coeff	=   coeff(:);
    switch length( coeff )
        case 1
            coeff	=   @(y,s) coeff;
        case length( x )
            coeff	=   @(y,s) coeff(1:end-1);
        case length( x ) - 1
            coeff	=   @(y,s) coeff;
        otherwise
            error( 'Coefficient length must be 1, N-1, or N' );
    end
    
end

end

function u0 = handle_u0( u0, x )

if isa( u0, 'function_handle' )
    u0	=   u0(x(:));
else
    u0	=   u0(:);
    switch length(u0)
        case 1
            u0	=   u0 * ones(length(x),1);
        case length(x)
            % do nothing
        case length(x)-1
            u0	=   [u0;u0(1)];
        otherwise
            error( 'u0 must have length 1, N-1, or N' );
    end
end

if abs(u0(1)-u0(end)) > 1e-12
    error( 'u0 must be periodic!' );
end

u0	=   u0(1:end-1);

end

function [t,tidx] = handle_T( T )

T       =   T(:);
tidx	=	[];
if T(1) == 0
    T       =   T(2:end);
    tidx	=	1;
end

switch length(T)
    case 1
        t       =	linspace(0,T,3);
        tidx	=	[tidx, 3];
    otherwise
        t       =   [0,T.'];
        tidx	=	[tidx, 2:length(T)+1];
end

end
