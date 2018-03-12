function [ Results ] = FminconFit( SimSettings, params0, lb, ub )
%FMINCONFIT Attempts to find the minizer of 'datafun' by using
% generalized constrained minimization via matlab's 'fmincon'.

%% Input Handling
params0	=   params0(:).';
lb      =   lb(:).';
ub      =   ub(:).';
SimSettings.RootPath	=   SimSettings.SavePath;

%% Minimization options
options                     =   optimoptions('fmincon');
options.Diagnostics         =   'on';
options.Display             =   'iter-detailed';
options.TolX                =   0.001;
options.TolFun              =   0.001;
options.MaxFunEvals         =   SimSettings.MinimizationMaxFn;
options.MaxIter             =   SimSettings.MinimizationMaxIt;
options.FinDiffRelStep      =   0.05;
options.PlotFcns            =   {   @optimplotx,            ...
                                    @optimplotfval,         ...
                                    @optimplotfunccount,    ...
                                    @optimplotstepsize      };

%% Record Minimization options
SimSettings.MinimizationOpts.InitGuess      =   params0;
SimSettings.MinimizationOpts.LowerBound     =   lb;
SimSettings.MinimizationOpts.UpperBound     =   ub;
SimSettings.MinimizationOpts.NormType       =   'TWONORM'; %see below
SimSettings.MinimizationOpts.options        =   options;

%% Data function
%   Note: Parameters are scaled and shifted so that minimization for all
%   parameters is in the range [0,1]. This way, plots are more clear and we
%   don't have to worry about numerical difficulties of having parameters
%   at differing magnitudes
switch upper(SimSettings.MinimizationOpts.NormType)
    case 'ONENORM'
        options.TypicalX                        =   0.5*ones(3,1); % Variables are scaled to [0,1]
        SimSettings.MinimizationOpts.options	=   options;
        
        x0          =	(params0 - lb)./(ub-lb);
        [xlb,xub]	=   deal([0,0,0],[1,1,1]);
        [A,b,Ae,be] =	deal([]);
        nonlcon     =   [];
        datafun     =   @(x) datafun_onenorm(x,lb,ub,SimSettings);
    case 'TWONORM'
        options.TypicalX                        =   0.5*ones(3,1); % Variables are scaled to [0,1]
        SimSettings.MinimizationOpts.options	=   options;
        
        x0          =	(params0 - lb)./(ub-lb);
        [xlb,xub]	=   deal([0,0,0],[1,1,1]);
        [A,b,Ae,be] =	deal([]);
        nonlcon     =   [];
        datafun     =   @(x) datafun_twonorm(x,lb,ub,SimSettings);
    case 'INFNORM'
        options.TypicalX                        =   [0.5*ones(3,1); 0.1];
        options.GradObj                         =   'on';
        SimSettings.MinimizationOpts.options	=   options;
        
        x0          =	[ (params0 - lb)./(ub-lb), 0.10 ];
        [xlb,xub]	=   deal([0,0,0,0],[1,1,1,0.20]);
        [A,b,Ae,be] =	deal([]);
        nonlcon     =   @(x) SimulationConstraint(x,lb,ub,SimSettings);
        datafun     =   @(x) datafun_infnorm(x,lb,ub,SimSettings);
    case 'INFNORM_TWOPEN'
        options.TypicalX                        =   [0.5*ones(3,1); 0.1];
        options.GradObj                         =   'on';
        SimSettings.MinimizationOpts.options	=   options;
        
        x0          =	[ (params0 - lb)./(ub-lb), 0.10 ];
        [xlb,xub]	=   deal([0,0,0,0],[1,1,1,0.20]);
        [A,b,Ae,be] =	deal([]);
        nonlcon     =   @(x) SimulationConstraintTwoNormPenalty(x,lb,ub,SimSettings);
        datafun     =   @(x) datafun_infnorm(x,lb,ub,SimSettings);
                
    otherwise
        error('NormType must be ''onenorm'', ''twonorm'', ''infnorm'', or ''infnorm_twopen''.');
end

%% Save SimSettings with optimzation options
if SimSettings.flags.SaveData
    try
        save( [SimSettings.RootSavePath, '/', 'SimSettings'], 'SimSettings', '-v7' );
    catch me
        warning(me.message);
    end
end

%% run optimization routine
[x,fval,exitFlag,output,Lambda,Grad,Hessian]	=	...
    fmincon(datafun,x0,A,b,Ae,be,xlb,xub,nonlcon,options);

Results	=   struct( ...
    'Params',       (ub-lb).*x(1:3) + lb,	...
    'RelParams',    x(1:3),     ...
    'fval',         fval,       ...
    'exitFlag',     exitFlag,	...
    'output',       output,     ...
    'Lambda',       Lambda,     ...
    'Grad',         Grad,       ...
    'Hessian',      Hessian     ...
    );


end

function FunVal = datafun_onenorm(x,lb,ub,SimSettings)

% Params have been scaled to [0,1]
x           =   x(:).';
params      =	(ub-lb).*x + lb;

% Run simulation and evaluate l1-norm of difference between sim/real data
[Angles,Sim_dR2,Real_dR2] = SimulationDataFun( params, SimSettings );
FunVal      =   mean(abs( Sim_dR2(:) - Real_dR2(:) ));

end

function FunVal = datafun_twonorm(x,lb,ub,SimSettings)

% Params have been scaled to [0,1]
x           =   x(:).';
params      =	(ub-lb).*x + lb;

% Run simulation and evaluate l2-norm of difference between sim/real data
[Angles,Sim_dR2,Real_dR2] = SimulationDataFun( params, SimSettings );
FunVal      =   rms( Sim_dR2(:) - Real_dR2(:) );

end

function [FunVal,Gradient] = datafun_infnorm(x,lb,ub,SimSettings)

% 4th paramater is inf-norm
FunVal      =   x(4);

% Calculate gradient
if nargout > 1 % gradient required
    Gradient	=	[0;0;0;1];
end

end

function [c,ceq] = SimulationConstraint(x,lb,ub,SimSettings)

% Params have been scaled to [0,1]
x       =   x(:).';
params	=	(ub-lb).*x(1:3) + lb;

% Last parameter is the infinity norm constraint
Linf	=	x(4);

% Run simulation to obtain simulated data
[Angles,Sim_dR2,Real_dR2] = SimulationDataFun( params, SimSettings );

% Constrain that abs(sim-real) <= t, where t = params(4) is the inf-norm
c       =   abs( Sim_dR2(:) - Real_dR2(:) ) - Linf;
ceq     =   [];

end

function [c,ceq] = SimulationConstraintTwoNormPenalty(x,lb,ub,SimSettings)

% Params have been scaled to [0,1]
x       =   x(:).';
params	=	(ub-lb).*x(1:3) + lb;

% Last parameter is the infinity norm constraint
Linf	=	x(4);

% Run simulation to obtain simulated data
[Angles,Sim_dR2,Real_dR2] = SimulationDataFun( params, SimSettings );

% Last parameter is the infinity norm constraint
Ltwo	=	rms( Sim_dR2(:) - Real_dR2(:) );

% Constrain that 1/2*(abs(sim-real)+rms(sim-real)) <= Linf
c       =   0.5 * ( abs(Sim_dR2(:)-Real_dR2(:)) + Ltwo ) - Linf;
ceq     =   [];

end
