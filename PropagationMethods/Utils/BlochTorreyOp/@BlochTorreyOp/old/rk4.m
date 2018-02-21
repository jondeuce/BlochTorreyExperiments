function [ y ] = rk4( A, tspan, y0, stepsize, varargin )
%RK4 Performs the "classical Runge–Kutta method" of order 4 for the linear
%ordinary differential equation:
% 
%	dy/dt = A*y
% 
% The RK4 algorithm for the case of a linear system of ODE's is simply
% given by:
% 
%   y_n+1 = y_n + h/6 * (k_1 + 2k_2 + 2k_3 + k_4)
%   t_n+1 = t_n + h
% 
% where:
% 
%   k_1 = A * (y_n            )
%   k_2 = A * (y_n + h/2 * k_1)
%   k_3 = A * (y_n + h/2 * k_2)
%   k_4 = A * (y_n + h   * k_3)

p = getInputParser;
parse(p,varargin{:})
opts = p.Results;

[h,h2,h6] = deal(stepsize, stepsize/2, stepsize/6);
[c1,c2,c3,c4] = deal(h, h^2/2, h^3/6, h^4/24);
nmax = round((tspan(2)-tspan(1))/stepsize);

t = tspan(1);
y = y0(:);

clockstart = tic;
for n = 1:nmax
%     k1 = A * (y          );
%     k2 = A * (y + h2 * k1);
%     k3 = A * (y + h2 * k2);
%     k4 = A * (y + h  * k3);
%     
%     y = y + h6 * (k1 + 2*k2 + 2*k3 + k4);
    
    tmp = A*y;
    y = y + c1*tmp;
    
    tmp2 = A*tmp; % A^2*y
    y = y + c2*tmp2;
    
    tmp = A*tmp2; % A^3*y
    y = y + c3*tmp;
    
    tmp2 = A*tmp; % A^4*y
    y = y + c4*tmp2;
    
    t = t + h;
    
    display_toc_time(toc(clockstart),sprintf('Time = %3.2fms',t));
end

end

function p = getInputParser

p = inputParser;
p.FunctionName = 'rk4';

addParameter( p,'observer',[], ...
    @(f) validateattributes(f,{'function_handle'}) && nargin(f)==2 );

end