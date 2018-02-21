function sol = sigodeXX(A, tspan, type, varargin)
%SIGODEXX 

M0 = 1i;
S0 = M0 * prod(A.gdims);
dt_max = 5e-3;

x_curr = M0 * ones(size(A,2),1);
x_best = x_curr;
t_curr = 0;
t_best = 0;

opts = getodeopts('OutputFcn',@cache_currentstep,varargin{:});
sol = ode45(@sigode,tspan,S0,opts);

    function dS = sigode(t,~)
        
        if t < t_best
            error('ruh roh');
        elseif t > t_best
            t_curr = t;
            npts = ceil((t_curr-t_best)/dt_max);
            tpts = linspace(0,t_curr-t_best,npts+1);
            [ x_curr, ~ ] = bt_expmv_tspan( tpts, A, x_best, type );
        end
        
        dS = sum(A*x_curr);
        
    end

    function status = cache_currentstep(t,S,flag,varargin)
        
        t_best = t_curr;
        x_best = x_curr;
        
        status = myodeprint(t,S,flag,varargin);
        
    end

end

function opts = getodeopts(varargin)

opts = odeset( ...
    'reltol',    1e-3, ...
    'abstol',    1e-3, ...
    'stats',     'on', ...
    'outputfcn', @myodeprint ...
    );

for ii = 1:2:length(varargin)
    if ~isfield(opts,varargin{ii})
        error('Unknown odeset field: %s',varargin{ii});
    end
    opts.(varargin{ii}) = varargin{ii+1};
end

end

function status = myodeprint(t,S,flag,varargin)

if nargin < 3 || isempty(flag) % odeprint(t,y) [v5 syntax] or odeprint(t,y,'')
    fprintf('t = %f ms\n', 1000*t(end));
    fprintf('S = %1.6e + %1.6ei\n', real(S(end)), imag(S(end)));
else
    switch(flag)
        case 'init'               % odeprint(tspan,y0,'init')
            fprintf('Initial Step:\n');
            fprintf('t = %f ms\n', 1000*t(1));
            fprintf('S = %1.6e + %1.6ei\n\n', real(S(end)), imag(S(end)));
        case 'done'               % odeprint([],[],'done')
            fprintf('\n\n');
    end
end

status = 0;

end