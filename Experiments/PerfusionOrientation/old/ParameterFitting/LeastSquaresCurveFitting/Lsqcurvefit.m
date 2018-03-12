function [ Results ] = Lsqcurvefit( SimSettings, params0, lb, ub )
%LSQCURVEFIT Attempts to find the minizer of 'datafun' by using
% generalized constrained minimization via matlab's 'lsqcurvefit'.

%% Input Handling
params0	=   params0(:).';
lb      =   lb(:).';
ub      =   ub(:).';
SimSettings.RootPath	=   SimSettings.SavePath;

%% Minimization options
options                     =   optimoptions('lsqcurvefit');
options.Algorithm           =   'trust-region-reflective';
options.Diagnostics         =   'on';
options.Display             =   'iter-detailed';
options.TolX                =   0.001;
options.TolFun              =   0.001;
options.TypicalX            =   0.5*ones(1,3); % Variables are scaled to [0,1]
options.MaxFunEvals         =   SimSettings.MinimizationMaxFn;
options.MaxIter             =   SimSettings.MinimizationMaxIt;
options.FinDiffRelStep      =   0.08;
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
x0          =	(params0 - lb)./(ub-lb);
[xlb,xub]	=   deal([0,0,0],[1,1,1]);
Angle_Data  =   SimSettings.Angles_Deg_Data;
dR2_Data    =   SimSettings.dR2_Data;
datafun     =   @(x,xdata) datafun_lsqcurvefit(x,xdata,lb,ub,SimSettings);

%% Save SimSettings with optimzation options
if SimSettings.flags.SaveData
    try
        save( [SimSettings.RootSavePath, '/', 'SimSettings'], 'SimSettings', '-v7' );
    catch me
        warning(me.message);
    end
end

%% run optimization routine
[x,resnorm,residual,exitFlag,output,Lambda,Jacobian] =  ...
    lsqcurvefit(datafun,x0,Angle_Data,dR2_Data,xlb,xub,options);

Results	=   struct( ...
    'Params',       (ub-lb).*x(1:3) + lb,	...
    'RelParams',    x(1:3),     ...
    'resnorm',      resnorm,    ...
    'residual',     residual,   ...
    'exitFlag',     exitFlag,   ...
    'output',       output,     ...
    'Lambda',       Lambda,     ...
    'Jacobian',     Jacobian    ...
    );

end

function FunVector = datafun_lsqcurvefit(x,xdata,lb,ub,SimSettings)

% Params have been scaled to [0,1]
x           =   x(:).';
params      =	(ub-lb).*x + lb;

% Function values requested by lsqcurvefit
intersect_tol   =   @(x,X,tol) reshape(any(abs(bsxfun(@minus,x(:),X(:).')) < tol,1),size(X));
inds            =   intersect_tol( xdata, SimSettings.Angles_Deg_Data, 1e-14 );
SimSettings.Angles_Deg_Data =   SimSettings.Angles_Deg_Data(inds);
SimSettings.Angles_Rad_Data =   SimSettings.Angles_Rad_Data(inds);
SimSettings.dR2_Data        =   SimSettings.dR2_Data(inds);

% Run simulation and return result
[Angles,Sim_dR2,Real_dR2] = SimulationDataFun( params, SimSettings );
FunVector      =   reshape(Sim_dR2, size(xdata));

end
