function [ u ] = cn_DD( L, u0, dt, solver, varargin )
%CN_DD Performs an implicit forward time step on the linear system of ode's
% given by du/dt = L[u], where L is the Decay-Diffusion operator, using
% the Crank-Nicolson method:
% 
%       .
%       u(t+dt/2)	~	(u(t+dt) - u(t))/dt
%                   ~   1/2 * (L[u(t+dt)] + L[u(t)])
% 
%   =>  lhs	=   (2/dt)*u(t+dt) - L[u(t+dt)]
%       rhs	=   (2/dt)*u(t) + L[u(t)]
% 
% 
% The Decay-Diffusion operator is defined as:
% 
%       L[u]	=   div( D(x) * grad(u) ) - f(x)*u
% 
% Where D is the diffusion tensor, and f is the decay rate.
% 
% This method uses the user-supplied 'solver' to solve for u(t+dt). The
% specified solver must be iterative, depending only the action of the
% operator. In particular, L must be a function handle returning an array
% of the same size as u0; extra arguments will be passed to the function L.

%==========================================================================
% Parse Inputs
%==========================================================================
if nargin > 4
    L	=   @(u) L(u,varargin{:});
end

%==========================================================================
% Solve for u(t+dt) via the Crank-Nicholson Method
%==========================================================================
k	=   (2/dt);
b	=   k.*u0 + L(u0);
A	=   @(u) k.*u + L(u);

u	=   solver(A,b);

end
