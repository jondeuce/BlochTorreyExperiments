function [ u ] = parabolicPeriodicDeriv1D( a, b, c, u0, xb, N, T )
%PARABOLICPERIODICDERIV1D Solves the general 1D parabolic equation with
% periodic boundary conditions on u(x,t) and u_x(x,t). The equation under
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
%   u:  solution function. size of u is [N x numel(T)]
% 

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
ii	=   2:N-1;
xii	=   x(ii);

ax	=   handle_coefficient( a, x );
bx	=   handle_coefficient( b, x );
cx	=   handle_coefficient( c, x );

u0	=   handle_u0( u0, x );

[t,tidx]	=	handle_T( T );

%==========================================================================
% Boundary conditions and Mass Matrix
%==========================================================================
BC1	=   sparse(1,N);
BC1([1,end])	=   [1, -1]; %periodic value bc

BC2	=   sparse(1,N);
BC2([1,2,end-1,end])	=   [-1,1,1,-1]; % periodic derivative bc

M       =	speye(N);
M(1,:)	=   BC1;
M(end,:)=	BC2;

%==========================================================================
% Solve for u(x,t)
%==========================================================================
options	=	odeset('Mass',M,'MassSingular',false,'Jacobian',@JFun);
[~,u]	=	ode23s(@dudt,t,u0,options);
u       =   u(tidx,:).';

    %======================================================================
    % Derivative calculation
    %======================================================================
    function dudt = dudt(t,u)
        dudt	=   [	0
                        ax(xii,t) .* ( u(ii+1) - 2*u(ii) + u(ii-1) )/h^2 + ...
                        bx(xii,t) .* ( u(ii+1) - u(ii-1) )/h + ...
                        cx(xii,t) .* u(ii)
                        0	];
    end
    
    %======================================================================
    % Jacobian calculation
    %======================================================================
    function J = JFun(t,u)
        e	=   ones(length(u)-2,1); % for scalar expansion
        axx	=   ax(xii,t) .* e;
        bxx	=   bx(xii,t) .* e;
        cxx	=   cx(xii,t) .* e;
        
        J	=   sparse(N,N);
        J	=   J + spdiags( [ [axx;0;0],[0;-2*axx;0],[0;0;axx] ]/h^2, -1:1, N, N ) + ...
                    spdiags( [-[bxx;0;0],[0;0;bxx] ]/(2*h), [-1,1], N, N ) + ...
                    spdiags( [ 0;cxx;0 ], 0, N, N );
    end

end

function coeff = handle_coefficient( coeff, x )

if isa( coeff, 'function_handle')
    
    switch nargin( coeff )
        case 1 % x-dependent
            coeff	=   @(y,s) coeff(x(2:end-1));
        case 2 % x,t-dependent
            % do nothing
        otherwise
            error( 'Coefficients must be functions of x and/or t only' );
    end
    
else
    
    if isscalar( coeff )
        coeff	=   @(y,s) coeff;
    else
        coeff	=   coeff(:);
        switch length( coeff )
            case length( x )
            coeff	=   @(y,s) coeff(2:end-1);
            case length( x ) - 2
            coeff	=   @(y,s) coeff;
            otherwise
                error( 'Coefficient length must be length(x) or length(x)-2' );
        end
    end
    
end

end

function u0 = handle_u0( u0, x )

if isa( u0, 'function_handle' )
    u0	=   u0(x);
else
    u0	=   u0(:);
    switch length(u0)
        case 1
            u0	=   u0 * ones(length(x),1);
        case length(x)
            % do nothing
        otherwise
            error( 'u0 must be scalar or have length N' );
    end
end

end

function [t,tidx] = handle_T( T )

T	=   T(:);
tidx=	[];
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
