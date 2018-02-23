function [ SimSettings ] = Parse_BOLDSimSettings( SimSettings )
%PARSE_BOLDSIMSETTINGS Parses SimSettings struct for correct
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
        case 'ParentPath'
            % Root folder for saving results
            if ~isempty_f(f), errmsg{ii} = 'ParentPath must be left empty. Change inside Parse_SimSettings.'; end
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
                'PlotSigvsAngles','PlotSigvsTime','PlotR2vsAngles','PlotR2vsTime','RunBaseline','RunActivated'};
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
            if ~isoneofstrings(SimSettings.ScanType,'SE')
                errmsg{ii} = 'ScanType must be ''SE''.';
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
            allowednames = { 'OxygenatedBloodLevel', 'DeoxygenatedBloodLevel', 'Hct', 'CA_Concentration', 'Total_BVF', 'MinorVessel_RelBVF' };
            allowedlist = strcat(allowednames,', ');
            allowedstr = [allowedlist{:}]; allowedstr = [allowedstr(1:end-1),'.'];
            if ~isequal(variablenames, allowednames);
                errmsg{ii} = ['Table must only contain columns: ', allowedstr];
            end
            if ~(all(isnumeric(Params.OxygenatedBloodLevel)) && all(Params.OxygenatedBloodLevel >= 0 & Params.OxygenatedBloodLevel <= 1))
                errmsg{ii} = '''Total_BVF'' must be a real number in the range [0,1].';
            end
            if ~(all(isnumeric(Params.DeoxygenatedBloodLevel)) && all(Params.DeoxygenatedBloodLevel >= 0 & Params.DeoxygenatedBloodLevel <= 1))
                errmsg{ii} = '''Total_BVF'' must be a real number in the range [0,1].';
            end
        case 'VoxelSize'
            % Isotropic voxel dimensions [um]
        case 'VoxelCenter'
            % Location of voxel center [um]
        case 'EchoTime'
            % Total simulation time [s]
            TE = SimSettings.EchoTime;
            if ~(all(isnumeric(TE) & TE >= 0)), errmsg{ii} = 'EchoTimes must be non-negative.'; end
        case 'RepTime'
            % Repetition time (== TE so that sim ends after TE)
            if ~isempty_f(f), errmsg{ii} = 'RepTime must be left empty; it will be set inside simulation.'; end
        case 'TimeStep'
            % Simulation time step length (s)
            dt = SimSettings.TimeStep;
            if ~(isnumeric(dt) && (0 < dt)), errmsg{ii} = 'TimeStep must be a positive number.'; end
        case 'Angles_Deg_Data'
            % Angle data between major vessel and main magnetic field [deg]
            alpha = SimSettings.Angles_Deg_Data;
            if ~(all(isnumeric(alpha) & alpha >= 0 & alpha <= 90)), errmsg{ii} = 'Angles_Deg_Data must be between 0 and 90 degrees.'; end
        case 'Angles_Rad_Data'
            % Same as above, but in radians [rad]
            alpha = SimSettings.Angles_Rad_Data;
            if ~(all(isnumeric(alpha) & alpha >= 0 & alpha <= pi/2)), errmsg{ii} = 'Angles_Rad_Data must be between 0 and pi/2 radians.'; end
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
        case 'GridSize'
            % Number of lattice points (isotropic)
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
        case 'B0'
            % Must be 3 or 7 Tesla currently
            if ~(isscalar(SimSettings.B0) && (SimSettings.B0 == -3.0 || SimSettings.B0 == -7.0))
                errmsg{ii} = 'B0 must be -3.0T or -7.0T currently.';
            end
        case 'GyroMagRatio'
        case 'R2_CA_per_mM'
        case 'dChi_CA_per_mM'
        case 'R_Minor_mu'
        case 'R_Minor_sig'
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
MagPropCommonFolder = what('MagPropCommon');
[MagPropCommonPath, ~, ~] = fileparts(MagPropCommonFolder.path);

SimSettings.RootPath        =   [MagPropCommonPath, '/MagPropCommon/Experiments/BOLDOrientation/Results'];

% Specific Path for saving this simulation's results (SE or GRE folder, etc.)
SimSettings.RootSavePath    =   get_BOLD_savepath( SimSettings );
SimSettings.SavePath        =   SimSettings.RootSavePath;

%==========================================================================
% Enforce Relationships between SimSettings values
%==========================================================================
% Consistent angle data
try
    assert(max(abs((pi/180)*SimSettings.Angles_Deg_Data - SimSettings.Angles_Rad_Data)) < 5*eps(pi/2));
catch me
    warning('Degrees/radians angle data is inconsistent; check your settings.');
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

% EchoTimes must each be twice a multiple of the TimeStep
switch upper(SimSettings.ScanType)
    case 'SE'
        try
            assert(all( abs(SimSettings.EchoTime - (2*SimSettings.TimeStep) * round(SimSettings.EchoTime/(2*SimSettings.TimeStep))) <= 1e-12 * SimSettings.EchoTime ));
        catch me
            warning('SE EchoTime''s must each be twice a multiple of the TimeStep dt.');
            rethrow(me);
        end
    case 'GRE'
        try
            assert(all( abs(SimSettings.EchoTime - (SimSettings.TimeStep) * round(SimSettings.EchoTime/SimSettings.TimeStep)) <= 1e-12 * SimSettings.EchoTime ));
        catch me
            warning('GRE EchoTime''s must each be a multiple of the TimeStep dt.');
            rethrow(me);
        end
end

% Parallel Processing
% if SimSettings.AllowParallel
%     if prod(SimSettings.GridSize) > 256^3
%         error('Parallel not allowed for grids with more than 256^3 elements.');
%     end
% end

% Compatible settings for closed-form solutions
if SimSettings.AllowClosedForms
    error('Closed-form solution not available.');
end

%==========================================================================
% Warn about questionable SimSettings values
%==========================================================================
% Running simulation without saving
if ~SimSettings.flags.SaveData
    warning('Running simulation without saving results!');
    user_continue = input('Do you want to continue? (y/n): ','s');
    if ~strcmpi(user_continue,'Y')
        error('Aborting simulation...');
    end
end

%==========================================================================
% Create new folders for saving data, if necessary
%==========================================================================
if SimSettings.flags.SaveData
    try
        if exist( SimSettings.RootSavePath, 'dir' )
            error( 'Directory %s already exists!', SimSettings.RootSavePath );
        else
            mkdir( SimSettings.RootSavePath );
        end
    catch me
        warning('Couldn''t create folder for saving results; check your settings!');
        rethrow(me);
    end
end

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

