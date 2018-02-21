function [ Angles, Sim_dR2, Real_dR2 ] = SimulationDataFun( InputParams, SimSettings )
%SIMULATIONDATAFUN Data function for minimization

% Current minimization iteration (# of function calls)
persistent iteration
if isempty(iteration), iteration = 0; end
iteration	=	iteration + 1;

% Check if this is the first iteration or not
if SimSettings.flags.SaveData && ~exist( SimSettings.SavePath, 'dir' )
    iteration	=   1;
end

% Create savepath for this iteration
SimSettings.RootSavePath	=	get_savepath(SimSettings,iteration);
SimSettings.SavePath        =	SimSettings.RootSavePath;

% Save progress figure from last iteration
save_ProgressFigure(SimSettings,iteration-1);

% Update SimSettings
SimSettings.MinimizationIter                =   iteration;
SimSettings.MinimizationOpts.CurrentGuess   =   InputParams;

% Set iteration parameters
CA_Concentration	=   InputParams(1);
Total_BVF           =   InputParams(2);
MinorVessel_RelBVF	=   InputParams(3);
Params              =	table( CA_Concentration, Total_BVF, MinorVessel_RelBVF );
SimSettings.InitialParams	=   Params;

% Run simulation with supplied params
[ Results, AvgResults, ParamCombinations, SimSettings ] = ...
        SimulationMainSE( Params, SimSettings );

% Return simulation results and real data
Angles      =   SimSettings.Angles_Deg_Data;
Sim_dR2     =   AvgResults.dR2_TE;
Real_dR2	=   SimSettings.dR2_Data;

% Plot and save comparison with real data
plot_compare_dR2(Angles,Real_dR2,Sim_dR2,InputParams,SimSettings);

end

function rootsavepath = get_savepath(SimSettings,iter)
rootsavepath	=   [ SimSettings.RootPath, '/', ...
        SimSettings.MinimizationType, '_iter_', num2strpad(iter,3) ];
end

function save_ProgressFigure(SimSettings,iter)

switch upper(SimSettings.MinimizationType)
    case 'SIMULANNEALBND'
        h           =   findobj('type','figure','name','Simulated Annealing');
        filename	=   'SimulAnnealProgress';
    case 'FMINCON'
        h           =   findobj('type','figure','name','Optimization PlotFcns');
        filename	=   'FminconProgress';
    otherwise
        h           =   [];
end

if ~isempty(h)
    set(h,'visible','on','color','w');
    if SimSettings.flags.SaveData
        SimSettings.SavePath	=   get_savepath(SimSettings,iter);
        save_simulation_figure( filename, h, false, SimSettings );
    end
end

end

function h = plot_compare_dR2( Angles, Real_dR2, Sim_dR2, InputParams, SimSettings )

h	=	figure('visible',SimSettings.flags.PlotVisibility); hold on, grid on
hp	=	plot( Angles(:), [Real_dR2(:),Sim_dR2(:)], 'linewidth', 5 );

hl	=   legend( hp, '\DeltaR_2 Data', '\DeltaR_2 Simulated' );
set( hl, 'fontsize', 14, 'location', 'best');

xlabel( 'Angle [deg]', 'fontsize', 12 );
ylabel( '\DeltaR_2 [s^{-1}]', 'fontsize', 12 );
titlestr = sprintf( '\\DeltaR_2 vs. Angle: [CA, BVF, RelBVF] = [%0.4f, %0.4f, %0.4f]\nRMS = %0.6f, Infnorm = %0.6f', ...
        InputParams(1), InputParams(2), InputParams(3), rms(Real_dR2(:)-Sim_dR2(:)), max(abs(Real_dR2(:)-Sim_dR2(:))) );
title( titlestr, 'fontsize', 14, 'fontweight', 'bold' );
drawnow;

if SimSettings.flags.SaveData
    filename	=   'SimulationVsData';
    save_simulation_figure( filename, h, true, SimSettings );
else
    close(h);
end

end

