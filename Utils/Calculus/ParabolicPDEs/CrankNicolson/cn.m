function [ x ] = cn( A, x, dt, solver )
%CN Performs an implicit forward time step on the linear system of ode's
% given by dx/dt = A*x using the Crank-Nicolson method:
% 
%	d/dt[x(t+dt/2)]	~	[x(t+dt) - x(t)]/dt + O(dt^2)
% 
%   =>	x(t+dt)     ~	x(t) + (dt/2)*(A*x(t+dt) + A*x(t))
%       x(0)        =	x0
%   
%	=>	x(t+dt)     =	((2/dt)*I - A)^-1 * ((2/dt)*I + A) * x(t)

%==========================================================================
% Parse Inputs
%==========================================================================
if nargin < 4
    solver	=	@mldivide;
end

%==========================================================================
% Solve for x(t+dt) via the Crank-Nicholson Method
%==========================================================================
if issparse(A)
    Idt	=   (2/dt) * speye(size(A));
else
    Idt	=   (2/dt) * eye(size(A));
end
x	=   (Idt+A)*x;
x	=   solver(Idt-A,x);

end

