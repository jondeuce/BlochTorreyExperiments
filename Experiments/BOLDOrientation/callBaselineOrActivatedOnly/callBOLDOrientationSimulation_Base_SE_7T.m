%% CALLBOLDORIENTATIONSIMULATION
% Sets up parameters and initial values to call 
% BOLDORIENTATIONSIMULATION

p = mfilename('fullpath');

%% misc. useful
this_filename    =   mfilename;
current_date    =    datestr(clock,30);

%% Angles to Simulate (handled separately due to geometry)
Angles_Deg  =   0:5:90;
Angles_Rad  =   (pi/180) * Angles_Deg;

%% SPIN-ECHO: Set fixed parameters (could allow them to vary, but are fixed from previous simulations)

% % SE Simulation Parameters (from minimization simulations)
% Type = 'SE';
% 
% CA_Concentration = 0.0; % No contrast agent in this simulation [mM]
% Total_BVF = 0.0349; % Total blood volume fraction of voxel [fraction]
% MinorVessel_RelBVF = 0.6588; % Fraction of Total_BVF contained in minor vasculature [fraction]
% 
% NumMajor = 5; % Optimal number of major vessels for SE (from previous sim.) [unitless]
% 
% VoxDims = [3000,3000,3000]; % Voxel dimensions to match acquired data from previous sim. [um]
% VoxSize = [512,512,512]; % Voxel size to ensure isotropic subvoxels [unitless]
% 
% % Echotimes to simulate (handled separately for efficiency)
% EchoTimes = (0:10:120)/1000; % Echotimes in seconds to simulate [s]
% 
% % Time step for simulation
% dt = 2.5e-3; % 2.5ms time steps

%% GRADIENT-ECHO: Set fixed parameters (could allow them to vary, but are fixed from previous simulations)

% % GRE Simulation Parameters (from minimization simulations)
% Type = 'GRE';
% 
% CA_Concentration = 0.0; % No contrast agent in this simulation [mM]
% Total_BVF = 0.0207; % Total blood volume fraction of voxel [fraction]
% MinorVessel_RelBVF = 0.6909; % Fraction of Total_BVF contained in minor vasculature [fraction]
% 
% NumMajor = 5; % Optimal number of major vessels for GRE (from previous sim.) [unitless]
% 
% VoxDims = [1750,1750,4000]; % Voxel dimensions to match acquired data from previous sim. [um]
% VoxSize = [350,350,800]; % Voxel size to ensure isotropic subvoxels [unitless]
% 
% % Echotimes to simulate (handled separately for efficiency)
% EchoTimes = (0:5:120)/1000; % Echotimes in seconds to simulate [s]
% 
% % Time step for simulation
% dt = 5e-3; % GRE time step can be longer than SE; signal is more stable

%% BOLD: Same settings for SE and GRE

% SE Settings
Type = 'SE';
EchoTimes = (0:5:120)/1000; % Echotimes in seconds to simulate [s]
dt = 2.5e-3; % SE time step [s]

% % GRE Settings
% Type = 'GRE';
% EchoTimes = (0:5:120)/1000; % Echotimes in seconds to simulate [s]
% dt = 5e-3; % GRE time step can be longer than SE; signal is more stable

% Overall Settings
B0Field = -7.0;
CA_Concentration = 0.0; % No CA

% Pears, J.A., Francis, S.T., Butterworth, S.E., Bowtell, R.W., Gowland, P.A., 2003.
% Investigating the BOLD effect during infusion of Gd-DTPA using rapid T2* mapping.
% Magn. Reson. Med. 49, 61–70
%   - They found a 28% increase in venous vascular volume in activation case
isBase = false;
BVF_Multiplier = (isBase * 1.0) + (~isBase * 1.28); % 28% increase in non-baseline
MinorRadius_Multiplier = sqrt(BVF_Multiplier); % Minor volume is proportional to radius^2

Total_BVF = BVF_Multiplier * (3/4) * 0.0349; % Fraction of BVF found in SE experiment that is venous blood
MinorVessel_RelBVF = 0.6588; % Fraction of Total_BVF contained in minor vasculature [fraction]

NumMajor = 5; % Optimal number of major vessels for GRE (from SE sim.)

VoxDims = [2500,2500,2500]; % Typical isotropic voxel dimensions. [um]
VoxSize = [512,512,512]; % Voxel size to ensure isotropic subvoxels

%% Override some variables for testing

% EchoTimes = (0:10:20)/1000;
% Angles_Deg = [0,90];
% Angles_Rad = (pi/180) * Angles_Deg;
% 
% input('WARNING: Using testing variables! Don''t run full simulation with these parameters... [press enter to continue]');

%% Set variable parameters

% DeoxygenatedBloodLevel = 0.54; % Oxygen saturation fraction for deoxygenated blood, aka Y0 [fraction]
% OxygenatedBloodLevel = 0.65; % Oxygen saturation fractions for oxygenated blood to simulate, aka Y [fraction]
% Hct = 0.45; % Hematocrit = volume fraction of red blood cells

% Ref: Zhao et al., 2007, MRM, Oxygenation and hematocrit dependence of transverse relaxation rates of blood at 3T
DeoxygenatedBloodLevel = 0.61; % Yv_0, baseline venous oxygenated blood fraction [fraction]
OxygenatedBloodLevel = 0.73; % Yv, activated venous oxygenated blood fraction [fraction]
Hct = 0.44; % Hematocrit = volume fraction of red blood cells

Params = table( OxygenatedBloodLevel, DeoxygenatedBloodLevel, Hct, CA_Concentration, Total_BVF, MinorVessel_RelBVF );

%% Main simulation settings

% Flags for saving/plotting/etc. simulation results during and/or after sim
flags    =   struct(            ...
    'SaveData',         true,   ...     % save simulation results
    'PlotAnything',     true,   ...     % overrides subsequent plot flags
    'PlotVisibility',   'on',   ...     % visibility of figures
    'PlotGeometry',     false,  ...     % plot geometry
    'PlotAllReps',      true,   ...     % plot all repititions for repeated geometries
    'PlotSigvsAngles',  false,  ...     % plot resulting Signal vs. Angle
    'PlotSigvsTime',    false,  ...     % plot resulting Signal vs. Time
    'PlotR2vsAngles',   false,  ...     % plot resulting R2 vs. Angle
    'PlotR2vsTime',     false,  ...     % plot resulting R2 vs. Time for each Angle
    'RunBaseline',      isBase, ...     % Run baseline simulation for BOLD signal
    'RunActivated',    ~isBase  ...     % Run activated (non-baseline) sim. for BOLD
    );

SimSettings   =   struct(               ...
    'Date',             current_date,   ... % Date for saving data
    'ParentPath',       [],             ... % System-dependent path to all code (created below)
    'RootPath',         [],             ... % Root folder for saving results (created below)
    'SavePath',         [],             ... % Path for saving data (created below)
    'RootSavePath',     [],             ... % Path for saving data from iteration of minimization (created in sim.)
    'DiaryPath',        [],             ... % Path for saving command window log
    'flags',            flags,          ... % flags for plotting/saving data etc.
    'ScanType',         Type,           ... % Simulate 'SE' or 'GRE' scan
    'MinimizationType', NaN,            ... % Type of minimization algorithm to use (see above)
    'MinimizationOpts', NaN,            ... % Minimization options (will be set within sim.)
    'MinimizationIter', NaN,            ... % Minimization iteration number (will be set within sim.)
    'MinimizationMaxIt',5,              ... % Maximum allowable iterations for minimization
    'MinimizationMaxFn',100,            ... % Maximum allowable function evaluations for minimization
    'Dimension',        3,              ... % Dimension of simulation
    'NumRepetitions',   1,              ... % Number of random geometries simulated per set of parameters
    'NumMajorVessels',  NumMajor,       ... % Number of major vessels in sim. (integer <= 16)
    'InitialParams',    Params,         ... % Parameters for simulation(s)
    'VoxelSize',        VoxDims,        ... % Isotropic voxel dimensions [um]
    'VoxelCenter',      zeros(1,3),     ... % Location of voxel center (arbitrary) [um]
    'EchoTime',         EchoTimes,      ... % Total simulation time [s]
    'RepTime',          [],             ... % Repetition time (== TE so that sim ends after TE)
    'TimeStep',         dt,             ... % Simulation time step length (s)
    'Angles_Deg_Data',  Angles_Deg,     ... % Angle data between major vessel and main magnetic field [deg]
    'Angles_Rad_Data',  Angles_Rad,     ... % Same as above, but in radians [rad]
    'AddDiffusion',     true,           ... % Incorporate diffusion effects into sim [true/false]
    'DiffusionCoeff',   3037,           ... % Diffusion constant of free water (default 1000 if empty) [um^2/s]
    'GridSize',         VoxSize,        ... % Number of lattice points (must be isotropic); if empty, will use default
    'AllowClosedForms', false,          ... % Skips intermediate calculations where possible (e.g. diffusionless SE)
    'AllowParallel',    false,          ... % Allow parallel calculations where possible (WARNING: uses MUCH more memory!) [T/F]
    'UseConvDiffusion', true,           ... % Perform diffusion using simple gaussian convolution [T/F]
    'DiscreteMap',      true,           ... % Use discrete boolean maps [true], or analytic [false]
    ...
    'B0',               B0Field,        ... % External magnetic field [T]
	'GyroMagRatio',     2.67515255e8,   ... % Gyromagnetic ratio [s^-1*T^-1]
    'R2_CA_per_mM',     5.2,            ... % Relaxation constant of the CA [(s*mM)^-1]
    'dChi_CA_per_mM',   0.3393e-6,      ... % Susceptibility CA [T/T*(mM)^-1]
    'R_Minor_mu',       MinorRadius_Multiplier * 13.7, ... % Minor vessel mean radius [um]
    'R_Minor_sig',      MinorRadius_Multiplier * 2.1   ... % Minor vessel std radius [um]
    );

%% Clear unnecessary variables
clearvars( '-except', 'Params', 'SimSettings' );

%% Parse SimSettings for correct values and load data
[ SimSettings ] = Parse_BOLDSimSettings( SimSettings );

%% Save initial workspace and zip MagnetizationPropagation folder
if SimSettings.flags.SaveData
    try
        save( [SimSettings.RootSavePath, '/', 'Initial_Workspace'] );
        zip( [SimSettings.RootSavePath, '/', 'MagnetizationPropagation', '_', SimSettings.Date, '.zip'], MagnetizationPropagationPath() );
    catch me
        warning('Couldn''t save initial workspace to specified folder; check your settings!');
        rethrow(me);
    end
end

%% Call simulation function, or run minimization

[ Results, AvgResults, ParamCombinations, SimSettings ] = ...
    BOLDOrientationSimulation( Params, SimSettings );

%% Clear unnecessary variables
clearvars( '-except', 'Results', 'AvgResults', 'ParamCombinations', 'SimSettings', 'FitResult' )

%% Save whole workspace
if SimSettings.flags.SaveData
    try
        save( [SimSettings.RootSavePath, '/', 'Final_Workspace'] );
    catch me
        warning('Couldn''t save final workspace to specified folder; saving it to current directory.');
        save( [ 'Final_Workspace_', SimSettings.Date ] );
    end
end
