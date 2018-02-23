function [ Results ] = SimulatedAnnealingFit( SimSettings, params0, lb, ub )
%SIMULATEDANNEALINGFIT Attempts to find the minizer of 'datafun' by
%using simulated annealing minimization.

%% Minimization options
options                     =   saoptimset('simulannealbnd');
options.PlotInterval        =   1;
options.PlotFcns            =   {   @saplotbestf,       ...
                                    @saplottemperature, ...
                                    @saplotf,           ...
                                    @saplotbestx    };
options.ReannealInterval	=   20;
options.ObjectiveLimit      =   1e-2;
options.MaxFunEvals         =   SimSettings.MinimizationMaxFn;
options.MaxIter             =   SimSettings.MinimizationMaxIt;
options.TolFun              =   1e-2;
options.TemperatureFcn      =   @temperaturefastwbnds;

%% Record Minimization options
SimSettings.MinimizationOpts.InitGuess      =   params0;
SimSettings.MinimizationOpts.LowerBound     =   lb;
SimSettings.MinimizationOpts.UpperBound     =   ub;
SimSettings.MinimizationOpts.CurrentGuess   =   params0;
SimSettings.MinimizationOpts.options        =   options;

%% Data function
%   Note: Parameters are scaled and shifted so that minimization for all
%   parameters is in the range [0,1]. This way, plots are more clear and we
%   don't have to worry about numerical difficulties of having parameters
%   at differing magnitudes
params0     =	(params0 - lb)./(ub-lb);
datafun     =   @(params) SimulationDataFun( (ub-lb).*params + lb, SimSettings );

%% run optimization routine
[x,fval,exitFlag,output]    =   simulannealbnd( datafun, params0, [0,0,0], [1,1,1], options );

Results	=   struct( ...
    'Params',       (ub-lb).*x + lb,    ...
    'RelParams',    x,                  ...
    'fval',         fval,               ...
    'exitFlag',     exitFlag,           ...
    'output',       output              ...
    );

end

