function [tout, yout] = odeXX(A, tspan, y0, type, varargin)
%ODEXX

opts = getodeopts(varargin{:});
[tout, yout] = ode113(@(~,y)A*y,tspan,y0,opts);

end

function opts = getodeopts(varargin)

opts = odeset( ...
    'reltol',    1e-3, ...
    'abstol',    1e-3, ...
    'stats',     'on', ...
    'outputfcn', @myodeprint, ...
    'refine',    1 ...
    );

for ii = 1:2:length(varargin)
    opts.(varargin{ii}) = varargin{ii+1};
end

end

function status = myodeprint(t,y,flag,varargin)

if nargin < 3 || isempty(flag) % odeprint(t,y) [v5 syntax] or odeprint(t,y,'')
    S = sum(y(:));
    fprintf('t = %f ms\n', 1000*t(end));
    fprintf('S = %1.6e + %1.6ei\n', real(S(end)), imag(S(end)));
else
    switch(flag)
        case 'init'               % odeprint(tspan,y0,'init')
            S = sum(y(:));
            fprintf('Initial Step:\n');
            fprintf('t = %f ms\n', 1000*t(1));
            fprintf('S = %1.6e + %1.6ei\n\n', real(S(end)), imag(S(end)));
        case 'done'               % odeprint([],[],'done')
            fprintf('\n\n');
    end
end

status = 0;

end