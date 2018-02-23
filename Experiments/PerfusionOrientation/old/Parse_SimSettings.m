function [ SimSettings ] = Parse_SimSettings( SimSettings )
%PARSE_SIMSETTINGS Parses SimSettings struct for correct
%fields/values. Upon being returned from this function, SimSettings may be
%used in SimulationMainSE for running a simulation.

Fields  = fieldnames(SimSettings);
errmsg  = cell(size(Fields));

isstring_f      = @(f) isa(SimSettings.(f),'char');
isinteger_f     = @(f) (abs(round(SimSettings.(f))-SimSettings.(f)) < 1e-14);
iswholenumber_f = @(f) isinteger_f(f) && (SimSettings.(f)>0);
isempty_f       = @(f) isempty(SimSettings.(f));
isstruct_f      = @(f) isstruct(SimSettings.(f));
isscalar_f      = @(f) isnumeric(SimSettings.(f)) && isscalar(SimSettings.(f));
islogical_f     = @(f) islogical(SimSettings.(f));
isscalarnan_f   = @(f) isscalar(SimSettings.(f)) && isa(SimSettings.(f),'double') && isnan(SimSettings.(f));

%==========================================================================
% Validate Individual Field Values
%==========================================================================
for ii = 1:numel(Fields)
    
    f = Fields{ii};
    switch f
        case 'Date'
            % Current Date (for saving data, etc.)
            if ~isstring_f(f), errmsg{ii} = 'Date must be a string.'; end
        case 'RootPath'
            % Root folder for saving results
            if ~isempty_f(f), errmsg{ii} = 'RootPath must be left empty. Change inside Parse_SimSettings.'; end
        case 'SavePath'
            % Path for saving data
            if ~isempty_f(f), errmsg{ii} = 'SavePath must be left empty. Change inside Parse_SimSettings.'; end
        case 'RootSavePath'
            % Path for saving data from iteration of minimization (created in sim.)
            if ~isempty_f(f), errmsg{ii} = 'RootSavePath must be left empty. Change inside Parse_SimSettings.'; end
        case 'DiaryPath'
            % Path for saving command window log
            if ~isempty_f(f), errmsg{ii} = 'DiaryPath must be left empty. Change inside Parse_SimSettings.'; end
        case 'flags'
            % flags for plotting/saving data etc.
            flagfields = {'SaveData','PlotAnything','PlotVisibility','PlotGeometry','PlotAllReps', ...
                          'PlotSigvsAngles','PlotSigvsTime','PlotR2vsAngles','PlotR2vsTime'};
            if ~isstruct_f(f) && hasfields(SimSettings.flags,flagfields{:})
                errmsg{ii} = ['flags be a struct with fields: ' genfieldlist(flagfields{:})];
            end
            for jj = 1:length(flagfields)
                flagfield = SimSettings.flags.(flagfields{ii});
                if ~( isa(flagfield,'logical') || (isoneofstrings(flagfields{jj},'PlotVisibility') && isoneofstrings(flagfield,'on','off')) )
                    errmsg{ii} = 'flag values must be logical, or ''on'' or ''off'' for ''PlotVisibility'' field.';
                    break;
                end
            end
        case 'ScanType'
            % Simulate 'SE' or 'GRE' scan
            if ~isoneofstrings(SimSettings.ScanType,'SE','GRE')
                errmsg{ii} = 'ScanType must be either ''SE'' or ''GRE''.';
            end
        case 'MinimizationType'
            % Type of minimization algorithm to use
            if ~(isscalarnan_f(f) || isoneofstrings(SimSettings.MinimizationType,'fmincon','lsqcurvefit','simulannealbnd'))
                errmsg{ii} = 'MinimizationType must be NaN, or one of: ''fmincon'', ''lsqcurvefit'', ''simulannealbnd''.';
            end
        case 'MinimizationOpts'
            % Minimization options (will be set within sim.)
            if ~isscalarnan_f(f)
                errmsg{ii} = 'MinimizationOpts must be NaN; they will be set within simulation.';
            end
        case 'MinimizationIter'
            % Minimization iteration number (will be set within sim.)
            if ~isscalarnan_f(f)
                errmsg{ii} = 'MinimizationIter must be NaN; they will be set within simulation.';
            end
        case 'MinimizationMaxIt'
            % Maximum allowable iterations for minimization
            if ~iswholenumber_f(f)
                errmsg{ii} = 'MinimizationMaxIt must be a positive integer.';
            end
        case 'MinimizationMaxFn'
            % Maximum allowable function evaluations for minimization
            if ~iswholenumber_f(f)
                errmsg{ii} = 'MinimizationMaxFn must be a positive integer.';
            end
        case 'Dimension'
            % Dimension of simulation
            if ~(isequal(SimSettings.Dimension,2) || isequal(SimSettings.Dimension,3))
                errmsg{ii} = 'Dimension must be 2 or 3.';
            end
        case 'NumRepetitions'
            % Number of random geometries simulated per set of parameters
            if ~iswholenumber_f(f)
                errmsg{ii} = 'NumRepetitions must be a positive integer.';
            end
        case 'NumMajorVessels'
            % Number of major vessels in sim. (square integer)
            if ~iswholenumber_f(f)
                errmsg{ii} = 'NumMajorVessels must be a positive integer.';
            end
        case 'InitialParams'
            % Parameters for simulation(s)
            if ~istable(SimSettings.InitialParams)
                errmsg{ii} = 'InitialParams must be a table.';
            end
            Params = SimSettings.InitialParams;
            variablenames = Params.Properties.VariableNames;
            if ~isequal(variablenames, {'CA_Concentration', 'Total_BVF', 'MinorVessel_RelBVF'});
                errmsg{ii} = 'Table must only contain columns: ''CA_Concentration'', ''Total_BVF'', ''MinorVessel_RelBVF''';
            end
            if ~(all(isnumeric(Params.CA_Concentration)) && all(Params.CA_Concentration >= 0))
                errmsg{ii} = '''CA_Concentration'' must be a non-negative real number.';
            end
            if ~(all(isnumeric(Params.Total_BVF)) && all(Params.Total_BVF >= 0 & Params.Total_BVF <= 1))
                errmsg{ii} = '''Total_BVF'' must be a real number in the range [0,1].';
            elseif any(Params.Total_BVF > 0.05)
                warning('''Total_BVF'' has unrealistically high values of %0.4f.', Params.MinorVessel_RelBVF);
            end
            if ~(all(isnumeric(Params.MinorVessel_RelBVF)) && all(Params.MinorVessel_RelBVF >= 0 & Params.MinorVessel_RelBVF <= 1))
                errmsg{ii} = '''MinorVessel_RelBVF'' must be a real number in the range [0,1].';
            end
        case 'VoxelSize'
            % Isotropic voxel dimensions [um]
            if ~isempty_f(f), errmsg{ii} = 'VoxelSize must be left empty; it will be set inside simulation.'; end
        case 'VoxelCenter'
            % Location of voxel center [um]
            if ~isempty_f(f), errmsg{ii} = 'VoxelCenter must be left empty; it will be set inside simulation.'; end
        case 'EchoTime'
            % Total simulation time [s]
            if ~isempty_f(f), errmsg{ii} = 'EchoTime must be left empty; it will be set inside simulation.'; end
        case 'RepTime'
            % Repetition time (== TE so that sim ends after TE)
            if ~isempty_f(f), errmsg{ii} = 'RepTime must be left empty; it will be set inside simulation.'; end
        case 'TimeStep'
            % Simulation time step length (s)
            dt = SimSettings.TimeStep;
            if ~(isnumeric(dt) && (0 < dt && dt <= 2/1000)), errmsg{ii} = 'TimeStep must be a positive number <= 0.002s'; end
            numsteps = (10/1000)/dt; % must divide into 10ms evenly
            if ~(abs(round(numsteps)-numsteps) < 1e-14), errmsg{ii} = 'TimeStep must divide evenly into 0.010s.'; end
        case 'dR2_Data'
            % Averaged delta R2 data to compare with simulation [s^-1]
            if ~isempty_f(f), errmsg{ii} = 'dR2_Data must be left empty; it will be set inside simulation.'; end
        case 'Angles_Deg_Data'
            % Angle data between major vessel and main magnetic field [deg]
            Angle_Deg_List = 2.50:5.00:87.50;
            Valid_Angles   = intersect(SimSettings.Angles_Deg_Data, Angle_Deg_List);
            if ~( isequal(size(Valid_Angles), size(SimSettings.Angles_Deg_Data)) && (max(abs(Valid_Angles-SimSettings.Angles_Deg_Data)) < 1e-14) )
                errmsg{ii} = 'Angles_Deg_Data must be a row vector containing angles in the set [2.5,7.5,12.5,...,82.5,87.5].';
            end
        case 'Angles_Rad_Data'
            % Same as above, but in radians [rad]
            Angle_Rad_List = (pi/180) * (2.50:5.00:87.50);
            Valid_Angles   = intersect(SimSettings.Angles_Rad_Data, Angle_Rad_List);
            if ~( isequal(size(Valid_Angles), size(SimSettings.Angles_Rad_Data)) && (max(abs(Valid_Angles-SimSettings.Angles_Rad_Data)) < 1e-14) )
                errmsg{ii} = 'Angles_Rad_Data must be a row vector containing angles in the set (pi/180)*[2.5,7.5,12.5,...,82.5,87.5].';
            end
        case 'AddDiffusion'
            % Incorporate diffusion effects into sim [true/false]
            if ~isa(SimSettings.AddDiffusion,'logical')
                errmsg{ii} = 'AddDiffusion must be a boolean value.';
            end
        case 'DiffusionCoeff'
            % Diffusion constant of free water (default 1.0 if empty) [um^2/ms]
            if ~(isnumeric(SimSettings.DiffusionCoeff) && isscalar(SimSettings.DiffusionCoeff) && (SimSettings.DiffusionCoeff >= 0))
                errmsg{ii} = 'Diffusion Coefficient must be a non-negative scalar value.';
            end
        case 'Smoothing'
            % Gauss-smoothing of R2-map; sigma = Smoothing * MinorVesselRadius [um/um]
            if ~isequal(SimSettings.Smoothing,0)
                errmsg{ii} = 'Smoothing is not implemented; it must be set to 0.';
            end
        case 'GridSize'
            % Number of lattice points (isotropic)
            if ~isempty_f(f)
                if isscalar_f(f), SimSettings.GridSize = SimSettings.GridSize * [1,1,1]; end
                if ~( isequal(size(SimSettings.GridSize),[1,3]) && all(SimSettings.GridSize >= 256) )
                    errmsg{ii} = 'GridSize must be a scalar value or [1x3] array with values >= 256.';
                end
            end
        case 'AllowClosedForms'
            % Skips intermediate calculations where possible (e.g. diffusionless SE)
            if ~islogical_f(f)
                errmsg{ii} = 'AllowClosedForms must be a logical value.';
            end
        case 'AllowParallel'
            % Allow parallel calculations where possible (WARNING: uses MUCH more memory!) [T/F]
            if ~islogical_f(f)
                errmsg{ii} = 'AllowParallel must be a logical value.';
            end
        case 'UseConvDiffusion'
            % Perform diffusion using simple gaussian convolution [T/F]
            if ~islogical_f(f)
                errmsg{ii} = 'UseConvDiffusion must be a logical value.';
            end
        case 'DiscreteMap'
            % Use discrete boolean maps [true], or analytic [false]
            if ~( islogical_f(f) && SimSettings.DiscreteMap )
                errmsg{ii} = 'Analytic maps are not implemented; DiscreteMap must be equal to logical true.';
            end
        otherwise
            errmsg{ii} = sprintf('Unknown field: ''%s''.', f);
    end %switch statement
    
end %for loop

%==========================================================================
% Throw errors, if any
%==========================================================================
throwerror = any(~cellfun(@isempty,errmsg));
if throwerror
    for ii = 1:numel(Fields)
        if ~isempty(errmsg{ii}), fprintf(['ERROR: ', errmsg{ii}, '\n']); end
    end
    error('SimSettings has incorrect fields/values! See error messages above.');
end

%==========================================================================
% Make paths for saving data
%==========================================================================
% RootPath is where all simulation results are stored
SimSettings.RootPath        =   '/data/ubcitar/jdoucette/code/Simulations/Results/PWI_Experiment';

% Specific Path for saving this simulation's results (SE or GRE folder, etc.)
SimSettings.RootSavePath    =   get_savepath( SimSettings );
SimSettings.SavePath        =   SimSettings.RootSavePath;

%==========================================================================
% Load Scan Data
%==========================================================================
switch SimSettings.ScanType
    case 'SE'
        SimSettings = load_SE_data(SimSettings);
    case 'GRE'
        SimSettings = load_GRE_data(SimSettings);
    otherwise
        error('ScanType must be one of ''SE'' or ''GRE''.');
end

%==========================================================================
% Enforce Relationships between SimSettings values
%==========================================================================
% Consistent angle data
try
    assert(max(abs(SimSettings.Angles_Deg_Data - (180/pi)*SimSettings.Angles_Rad_Data)) < 5*eps('double'));
catch me
    warning('Degrees/radians angle data is inconsistent; check your settings.');
    rethrow(me);
end

% Consistent data sizes
try
    assert(isequal(size(SimSettings.Angles_Deg_Data),size(SimSettings.dR2_Data)));
catch me
    warning('dR2 data and angle data are different sizes; check your settings.');
    rethrow(me);
end

% Isotropic subvoxels
try
    assert(isequal(size(SimSettings.VoxelSize),[1,3]));
    assert(isequal(size(SimSettings.VoxelCenter),[1,3]));
    assert(isequal(size(SimSettings.GridSize),[1,3]));
    assert(max(abs(diff( SimSettings.VoxelSize ./ SimSettings.GridSize ))) < 1e-14);
catch me
    warning('Subvoxels must be isotropic. That is, diff(VoxelSize./GridSize) must be all zeros.');
    rethrow(me);
end

% Parallel Processing
% if SimSettings.AllowParallel
%     if prod(SimSettings.GridSize) > 256^3
%         error('Parallel not allowed for grids with more than 256^3 elements.');
%     end
% end

% Compatible settings for closed-form solutions
if SimSettings.AddDiffusion
    if SimSettings.AllowClosedForms
        error('Closed-form solution not available with diffusion turned on.');
    end
else
    if ~SimSettings.AllowClosedForms
        warning(['Closed-form solutions are MUCH faster for diffusionless problems... ' ...
                 'Consider settings ''AllowClosedForms'' to true.']);
    end
end

%==========================================================================
% Warn about questionable SimSettings values
%==========================================================================
% Running simulation without saving
if ~SimSettings.flags.SaveData
    warning('Running simulation without saving results!');
end

end

function SimSettings = load_SE_data(SimSettings)

%--------------------------------------------------------------------------
% load SEPWI data directly
%--------------------------------------------------------------------------
%{
SEPWI_Results	=	load( [DataPath ...
    '/SEPWI/WhiteMatter_SEPWI_DTI_Data_Analysis_Results.mat']);
SEPWI_Data      =   struct(	...
    'type',         SEPWI_Results.type,           ...
    'TE',          	SEPWI_Results.TE,             ...
    'NumSubjects',  SEPWI_Results.subjNum,        ...
    'NumFrames',    SEPWI_Results.timeNum,        ...
    'PeakFrame',    SEPWI_Results.timePeakNum,    ...
    'Angles_Deg',   SEPWI_Results.Angles_Deg,     ...
    'Angles_Rad',	SEPWI_Results.Angles_Rad,     ...
    'time_frames',  SEPWI_Results.time_frames,    ...
    'time_s',       SEPWI_Results.time_s,         ...
    'dR2_Max',      SEPWI_Results.MAll,           ...
    'dR2_Min',      SEPWI_Results.mAll,           ...
    'S0',           SEPWI_Results.S0allSubj,      ...
    'SEPWI',        SEPWI_Results.SEPWIallSubj,   ...
    'BinCounts',    SEPWI_Results.NallSubj,       ...
    'dR2',          SEPWI_Results.meanR2allSubj	  ...
    );

SEPWI_Data	=   load('SEPWI_Data.mat');
SEPWI_Data	=   SEPWI_Data.SEPWI_Data;
%}

%--------------------------------------------------------------------------
% Get real data for matching with simulation
%--------------------------------------------------------------------------
%{
% Data is indexed as follows:
%   Dimension 1 is time, dim 2 is angle, and dim 3 is subject number
[TIME, ANGLE, SUBJECT]	=   deal(1,2,3);
WAvgData	=   @(x,KIND)	sum(x.*SEPWI_Data.BinCounts,KIND) ./	...
                            sum(SEPWI_Data.BinCounts,KIND);
AvgData     =   @(x,KIND)   mean(x,KIND);
MinData     =   @(x,KIND)	min(x,[],KIND);
MaxData     =   @(x,KIND)	max(x,[],KIND);

dR2_Avg     =   AvgData( SEPWI_Data.dR2, SUBJECT );
dR2_Peak	=   dR2_Avg( SEPWI_Data.PeakFrame, : );
Angles_Deg	=   double( SEPWI_Data.Angles_Deg );
Angles_Rad	=   double( SEPWI_Data.Angles_Rad );
EchoTime	=   double( SEPWI_Data.TE );
%}

%--------------------------------------------------------------------------
% Load data from local storage
%--------------------------------------------------------------------------
d = load('SEPWI_Data.mat');
d = d.SimParams;
Angles_Deg              = 2.50:5.00:87.50;
[~,~,Angle_Idx]         = intersect(SimSettings.Angles_Deg_Data,Angles_Deg);
SimSettings.EchoTime    = d.EchoTime;
SimSettings.RepTime     = SimSettings.EchoTime;
SimSettings.dR2_Data    = d.dR2_Peak(Angle_Idx);
SimSettings.VoxelSize   = 3000*[1,1,1]; % [um]
SimSettings.VoxelCenter = 1500*[1,1,1]; % [um]
% SimSettings.VoxelSize   = [3000,3000,3000]; % [um]
% SimSettings.VoxelCenter = SimSettings.VoxelSize/2; % [um]

%--------------------------------------------------------------------------
% Load default settings, if empty
%--------------------------------------------------------------------------
if isempty(SimSettings.GridSize), SimSettings.GridSize = [512,512,512]; end

end

function SimSettings = load_GRE_data(SimSettings)

%--------------------------------------------------------------------------
% Load data from local storage
%--------------------------------------------------------------------------

d = load('GREPWI_Data.mat');
d = d.GREPWI_Data;
Angles_Deg              = 2.50:5.00:87.50;
[~,~,Angle_Idx]         = intersect(SimSettings.Angles_Deg_Data,Angles_Deg);
SimSettings.EchoTime    = d.TE;
SimSettings.RepTime     = d.TE;
SimSettings.dR2_Data    = d.dR2_Peak(Angle_Idx);
SimSettings.VoxelSize   = [1750,1750,4000]; % [um]
SimSettings.VoxelCenter = SimSettings.VoxelSize/2; % [um]

%--------------------------------------------------------------------------
% Load default settings, if empty
%--------------------------------------------------------------------------
if isempty(SimSettings.GridSize), SimSettings.GridSize = [350,350,800]; end

end

function b = hasfields(s,varargin)
b = true;
for ii = 1:length(varargin)
    b = b && isfield(s.(varargin{ii}));
end
end

function str = genfieldlist(varargin)
str = sprintf('''%s'', ',varargin{:});
str = str(1:end-1);
str(end) = '.';
end

function b = isoneofstrings(s,varargin)
b = isa(s,'char');
if b
    for ii = 1:length(varargin)
        b = b || strcmp(s,varargin{ii});
    end
end
end

