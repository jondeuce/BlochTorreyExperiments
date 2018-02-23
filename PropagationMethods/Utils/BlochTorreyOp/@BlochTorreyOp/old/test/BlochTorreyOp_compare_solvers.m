function [ SOLNS, ODESOLVER ] = BlochTorreyOp_compare_solvers
%BLOCHTORREYOP_COMPARE_SOLVERS Comparison of different ode solvers

N = 128;
% x = randnc(N,N,N);
x = randnc(N^3,1);
D = 3000; %um^2/s
G = randnc(size(x));
G = G - maxabs(G);
gsize = [N,N,N];
gdims = 3000*[1,1,1];
tspan = [0,0.125,0.25];

A = BlochTorreyOp(G,D,gsize,gdims);

% Solvers to test. Omit: ode15i (fully implicit), ode23s (needs jacobian)
% NB: Supplying jacobian to ode23s actually isn't a problem, but ode23s
%     also performs an LU factorization which I cannot overload

% ODESOLVER = {@ode113,@ode15s,@ode23,@ode23t,@ode23tb,@ode45};
% ODESOLVER = {@ode113,@ode23,@ode45}; %drop low error tolerance solvers
% ODESOLVER = {@ode113,@ode45}; %ode23 too slow
ODESOLVER = {@ode113}; %ode113 is the champion
OPTS      = odeset('abstol',1e-12,'reltol',1e-12);
SOLNS     = cell(size(ODESOLVER));

for ii = 1:numel(ODESOLVER)
    odeXX = ODESOLVER{ii};
    strXX = func2str(odeXX);
    
    tic
    SOLNS{ii} = odeXX(@odefun,tspan,x(:),OPTS);
    display_toc_time(toc,strXX);
end

    function dy = odefun(~,y)
        dy = A*y;
%         dy = BlochTorreyAction(y, A.h, A.D, A.Diag, A.gsize);
    end

% Compare with expmv
tic
ODESOLVER{end+1} = @expmv;
SOLNS{end+1} = expmv(tspan(end),A,x(:),[],'double',true,false);
display_toc_time(toc,'expmv');

% Display errors with expmv result (should be on the order of set tol's)
for ii = 1:numel(ODESOLVER)-1
    fprintf('Error (%s):\t%0.8e\n', func2str(ODESOLVER{ii}), ...
        maxabs(SOLNS{ii}.y(:,end) - SOLNS{end}));
end

end

