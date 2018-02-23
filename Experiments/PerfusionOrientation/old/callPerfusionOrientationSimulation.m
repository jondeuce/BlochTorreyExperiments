%% CALLPERFUSIONORIENTATIONSIMULATION
% Sets up parameters and initial values to call 
% PerfusionOrientationSimulation

%% add relevant paths
% % DataPath is where the subject data is stored
% DataPath    =    '/data/sashimi-backup02/jdoucette/data';
% addpath(genpath( DataPath ));

%% misc. useful
this_filename    =   mfilename;
current_date    =    datestr(clock,30);

%% Angles to Simulate
Angles_Deg  =   2.50:5.00:87.50;
Angles_Rad  =   (pi/180) * Angles_Deg;

% idx = 1:3:18;
% idx = 2:3:18;
% idx = 3:3:18;
% idx = [1, 3, 6, 10, 15, 18];
% idx = [1, 5, 7, 11, 17, 18];
% idx = [1, 2, 8, 12, 14, 18];
% idx = [1, 4, 9, 13, 16, 18];
% idx = 4:18;
% idx = 15;
idx = 18;
% idx = 1:3;
% idx = 1:18;
% nx  = 6;
% idx = round(linspace(1,18,nx));

%% Choose nonlinear minimization method
MinType    =   NaN; % no minimization
% MinType    =   'fmincon'; % general constrained minimization
% MinType    =   'lsqcurvefit'; % non-linear least squares constrained minimization
% MinType    =   'simulannealbnd'; %simulated annealing

%% Set variable parameters (e.g. for fitting)

%--------------------------------------------------------------------------
% GRE Parameters
%--------------------------------------------------------------------------
% CA_Concentration    =   linspace(3.0,6.0,5); % contrast agent concentration [mM]
% Total_BVF           =   linspace(0.01,0.03,4); % blood volume fraction of voxel
% MinorVessel_RelBVF  =   linspace(0.30,0.70,4); % fraction of Total_BVF contained in minor vasculature

% CA_Concentration    =   4.85; % contrast agent concentration [mM]
% Total_BVF           =   0.01 + (0.03-0.01)*rand; % blood volume fraction of voxel
% MinorVessel_RelBVF  =   0.30 + (0.70-0.30)*rand; % fraction of Total_BVF contained in minor vasculature

%--------------------------------------------------------------------------
% SE Parameters
%--------------------------------------------------------------------------

% % Current Best Params: SE results for angles 4:18, 4 major blood vessels, 512^3 grid size
% CA_Concentration    =   6.228679665259627; % contrast agent concentration [mM]
% Total_BVF           =   0.034513842500752; % blood volume fraction of voxel
% MinorVessel_RelBVF  =   0.658849701654930; % fraction of Total_BVF contained in minor vasculature

% CA_Concentration    =   linspace(5.300,5.900,4); % contrast agent concentration [mM]
% Total_BVF           =   linspace(0.038,0.043,3); % blood volume fraction of voxel
% MinorVessel_RelBVF  =   linspace(0.650,0.690,4); % fraction of Total_BVF contained in minor vasculature

% CA_Concentration    =   [5.9,6.2,6.5]; % contrast agent concentration [mM]
% Total_BVF           =   0.031:0.002:0.035; % blood volume fraction of voxel
% MinorVessel_RelBVF  =   0.640:0.020:0.680; % fraction of Total_BVF contained in minor vasculature

% [6.2285, 0.0349, 0.6588] - Fit Results for N = 5
CA_Concentration    =   0; % contrast agent concentration [mM]
% CA_Concentration    =   6.2285; % contrast agent concentration [mM]
Total_BVF           =   0.0349; % blood volume fraction of voxel
MinorVessel_RelBVF  =   0.6588; % fraction of Total_BVF contained in minor vasculature

Params    =    table( CA_Concentration, Total_BVF, MinorVessel_RelBVF );

%% Main simulation settings

% Flags for saving/plotting/etc. simulation results during and/or after sim
flags    =   struct(            ...
    'SaveData',         false,  ...     % save simulation results
    'PlotAnything',     true,   ...     % overrides subsequent plot flags
    'PlotVisibility',   'on',   ...     % visibility of figures
    'PlotGeometry',     false,  ...     % plot geometry
    'PlotAllReps',      true,   ...     % plot all repititions for repeated geometries
    'PlotSigvsAngles',  true,   ...     % plot resulting Signal vs. Angle
    'PlotSigvsTime',    false,  ...     % plot resulting Signal vs. Time
    'PlotR2vsAngles',   true,   ...     % plot resulting R2 vs. Angle
    'PlotR2vsTime',     true    ...     % plot resulting R2 vs. Time for each Angle
    );

SimSettings   =   struct(               ...
    'Date',             current_date,   ... % Date for saving data
    'RootPath',         [],             ... % Root folder for saving results (created below)
    'SavePath',         [],             ... % Path for saving data (created below)
    'RootSavePath',     [],             ... % Path for saving data from iteration of minimization (created in sim.)
    'DiaryPath',        [],             ... % Path for saving command window log
    'flags',            flags,          ... % flags for plotting/saving data etc.
    'ScanType',         'SE',           ... % Simulate 'SE' or 'GRE' scan
    'MinimizationType', MinType,        ... % Type of minimization algorithm to use (see above)
    'MinimizationOpts', NaN,            ... % Minimization options (will be set within sim.)
    'MinimizationIter', NaN,            ... % Minimization iteration number (will be set within sim.)
    'MinimizationMaxIt',5,              ... % Maximum allowable iterations for minimization
    'MinimizationMaxFn',100,            ... % Maximum allowable function evaluations for minimization
    'Dimension',        3,              ... % Dimension of simulation
    'NumRepetitions',   1,              ... % Number of random geometries simulated per set of parameters
    'NumMajorVessels',  5,              ... % Number of major vessels in sim. (integer <= 16)
    'InitialParams',    Params,         ... % Parameters for simulation(s)
    'VoxelSize',        [],             ... % Isotropic voxel dimensions [um]
    'VoxelCenter',      [],             ... % Location of voxel center [um]
    'EchoTime',         [],             ... % Total simulation time [s]
    'RepTime',          [],             ... % Repetition time (== TE so that sim ends after TE)
    'TimeStep',         2e-3,           ... % Simulation time step length (s)
    'dR2_Data',         [],             ... % Averaged delta R2 data to compare with simulation [s^-1]
    'Angles_Deg_Data',  Angles_Deg(idx),... % Angle data between major vessel and main magnetic field [deg]
    'Angles_Rad_Data',  Angles_Rad(idx),... % Same as above, but in radians [rad]
    'AddDiffusion',     true,           ... % Incorporate diffusion effects into sim [true/false]
    'DiffusionCoeff',   3.037,          ... % Diffusion constant of free water (default 1.0 if empty) [um^2/ms]
    'Smoothing',        0,              ... % Gauss-smoothing of R2-map; sigma = Smoothing * MinorVesselRadius [um/um]
    'GridSize',         [],             ... % Number of lattice points (must be isotropic); if empty, will use default
    'AllowClosedForms', false,          ... % Skips intermediate calculations where possible (e.g. diffusionless SE)
    'AllowParallel',    false,          ... % Allow parallel calculations where possible (WARNING: uses MUCH more memory!) [T/F]
    'UseConvDiffusion', true,           ... % Perform diffusion using simple gaussian convolution [T/F]
    'DiscreteMap',      true            ... % Use discrete boolean maps [true], or analytic [false]
    );

%% Clear unnecessary variables
clearvars( '-except', 'Params', 'SimSettings' );

%% Parse SimSettings for correct values and load data
[ SimSettings ] = Parse_SimSettings( SimSettings );

%% Save initial parameters/settings
if SimSettings.flags.SaveData
    try
        if exist( SimSettings.RootSavePath, 'dir' )
            error( 'Directory %s already exists!', SimSettings.RootSavePath );
        else
            mkdir( SimSettings.RootSavePath );
        end
        
        if isnan(SimSettings.MinimizationType)
            save( [SimSettings.RootSavePath, '/', 'Initial_Workspace'] );
        else
            SimSettings.RootSavePath    =   [ SimSettings.SavePath, '/', ...
                SimSettings.MinimizationType, '_Initial' ];
            mkdir( SimSettings.RootSavePath );
            
            % Save Initial Workspace
            save( [SimSettings.RootSavePath, '/', 'Initial_Workspace'] );
        end
    catch me
        warning('Couldn''t save initial workspace to specified folder; check your settings!');
        rethrow(me);
    end
end

%% Call simulation function, or run minimization

if isnan(SimSettings.MinimizationType)
    [ Results, AvgResults, ParamCombinations, SimSettings ] = ...
        PerfusionOrientationSimulation( Params, SimSettings );
else
    if strcmp(SimSettings.ScanType,'SE')
        %------------------------ SE Initial Guesses ---------------------%
        % Params Are: [CA_Concentration, Total_BVF, MinorVessel_RelBVF]
        lowerbound  =   [6.5000, 0.03000, 0.6200];
        upperbound  =   [6.0000, 0.03800, 0.7000];
        initguess   =   [6.1800, 0.03350, 0.6700];
    else
        %------------------------ GRE Initial Guesses --------------------%
        % Params Are: [CA_Concentration, Total_BVF, MinorVessel_RelBVF]
        lowerbound  =   [4.0000, 0.01800, 0.6200];
        upperbound  =   [5.5000, 0.02400, 0.7400];
        initguess   =   [4.8693, 0.02060, 0.6882]; %4 major, no diffusion
        %initguess  =   [4.8693, 0.02060, 0.6882]; %4 major, with diffusion
    end
    %-------------------------- Testing Guesses --------------------------%
    %lowerbound  =   [0.5, 0.5, 0.5]; %for test problem that fits a
    %upperbound  =   [1.5, 1.5, 1.5]; %polynomial to simulated data
    %initguess   =   0.5 + rand(1,3);
    
    FitResult   =   ParameterFitting( SimSettings, initguess, lowerbound, upperbound );
end

%% Clear unnecessary variables
clearvars( '-except', 'Results', 'AvgResults', 'ParamCombinations', 'SimSettings', 'FitResult' )

%% Save whole workspace
if SimSettings.flags.SaveData
    try
        if isnan(SimSettings.MinimizationType)
            save( [SimSettings.RootSavePath, '/', 'Final_Workspace'] );
        else
            SimSettings.RootSavePath    =   [ SimSettings.SavePath, '/', ...
                SimSettings.MinimizationType, '_Final' ];
            mkdir(SimSettings.RootSavePath);
            
            % Save minimization figure
            save_simulation_figure( 'MinimizationProgress', [], false, SimSettings );
            
            % Save Final Workspace
            save( [SimSettings.RootSavePath, '/', 'Final_Workspace'] );
        end
    catch me
        warning('Couldn''t save final workspace to specified folder; saving it to current directory.');
        save( [ 'Final_Workspace_', SimSettings.Date ] );
    end
end
