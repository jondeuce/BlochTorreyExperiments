function [ AllResults, AvgResults, Params, SimSettings ] = ...
    PerfusionOrientationSimulation( Params, SimSettings )
%PERFUSIONORIENTATIONSIMULATION 

%==========================================================================
% Begin Simulation
%==========================================================================

% Check if running minimization or not
isMinimization      =   ~any(isnan( SimSettings.MinimizationType ));
if isMinimization, iteration = SimSettings.MinimizationIter; else iteration = []; end
if isMinimization, IterStr = sprintf('Iteration #%d',iteration); else IterStr = ''; end

% Create root folder for saving data
SavePath        =	SimSettings.SavePath;
RootSavePath	=   SimSettings.RootSavePath;
if SimSettings.flags.SaveData && ~exist( RootSavePath, 'dir' )
    mkdir( RootSavePath );
end

% Set default figure visibility
set(0, 'DefaultFigureVisible', SimSettings.flags.PlotVisibility);

% Set random number generator to default for inter-simulation consistency
rng('default');

% Begin diary
SimSettings.DiaryPath	=	[ RootSavePath, '/', 'Diary' ];
if SimSettings.flags.SaveData, diary( SimSettings.DiaryPath ); end

% Initial Simulation text
suffix	=   datestr(now); if ~isempty(IterStr), suffix = [IterStr ', ' suffix]; end
display_text( ['Starting Simulation: ' suffix], [], '%', true, [1,1] );

% Display Simulation Settings and Initial Parameters
fprintf('\nSimSettings:\n\n');
disp(SimSettings);
fprintf('\nParameters:\n\n');
disp(Params);

total_simulation_time = tic;

%==========================================================================
% Physical parameters for the problem
%==========================================================================

PhysicalParams	=   struct( ...
    'B0',              -3.0,            ... % External magnetic field [T]
	'gamma',            2.67515255e8,	... % Gyromagnetic ratio [s^-1*T^-1]
    'R2_Blood',         31.1,           ... % Relaxation constant of blood [s^-1]
    'R2_CA_per_mM',     5.2,            ... % Relaxation constant of the CA [(s*mM)^-1]
    'R2_Tissue',        14.5,           ... % Relaxation constant of tissue [s^-1]
    'R_Minor_mu',       13.7,           ... % Minor vessel mean radius [um]
    'R_Minor_sig',      2.1,            ... % Minor vessel std radius [um]
    'D',                1.0 * (1000),	... % Diffusion constant of water [um^2/s]
    'Hct',              0.45,           ... % Hematocrit = volume fraction of red blood cells
    'Y',                0.54,           ... % Oxygen saturation fraction
    'dChi_CA_per_mM',   0.3393e-6,      ... % Susceptibility CA [T/T*(mM)^-1]
    'dChi_Blood',       []              ... % Susceptibility of blood relative to tissue
    );

% Susceptibilty of Blood relative to tissue is given by:
%   deltaChi_Blood_Tissue  :=   Hct * (1-Y) * 2.26e-6 [T/T]
PhysicalParams.dChi_Blood	=   2.26e-6 * PhysicalParams.Hct * (1 - PhysicalParams.Y);

% Diffusion constant of water
if ~isempty(SimSettings.DiffusionCoeff)
    PhysicalParams.D	=   SimSettings.DiffusionCoeff * (1000); % convert [um^2/ms] -> [um^2/s]
end

%==========================================================================
% Get all combination of parameters
%==========================================================================

[ Params ]      =   getAllParamCombinations( Params );
NumParamSets	=   size(Params,1);

%==========================================================================
% Calculate derived simulation settings
%==========================================================================

[ Params, SimSettings ] = CalculateDerivedSettings( PhysicalParams, Params, SimSettings );

%==========================================================================
% Loop through all combinations of parameters
%==========================================================================

% Initialize AllResults and AvgResults structs
AllResults(NumParamSets,SimSettings.NumRepetitions)	=	struct;
AvgResults(NumParamSets,1)                          =	struct;

for ii = 1:NumParamSets
    
    % Make subfolder for saving this parameter set of data if necessary
    if SimSettings.flags.SaveData
        SimSettings.SavePath	=   sprintf( '%s/ParamSet_%s_%s', SavePath, ...
            num2strpad(ii,numdigits(NumParamSets)), num2str(NumParamSets) );
        mkdir(SimSettings.SavePath);
    end
    
    % Get a single set of parameters
    ParamSet        =	Params(ii,:);
    
    % Starting next ParamSet
    str	=   sprintf( 'Parameter Set %d/%d', ii, NumParamSets );
    display_text( str, [], '=', true, [1,1] );
    disp(ParamSet(1,1:size(SimSettings.InitialParams,2)));
    
    ParamSet_time	=   tic;
    
    for jj = 1:SimSettings.NumRepetitions
        
        % Starting next ParamSet
        str	=   sprintf( 'Parameter Set %d/%d, Repitition %d/%d', ...
            ii, NumParamSets, jj, SimSettings.NumRepetitions );
        display_text( str, [], '-', true, [1,1] );
    
        ParamSet.Rep        =   jj;
        ParamSetRep_time	=   tic;
        
        %==================================================================
        % Calculate and save geometry for a given set of parameters
        %==================================================================
        Geometry	=	CalculateGeometry( ...
            PhysicalParams, ParamSet, SimSettings );
        
        save_Geometry( Geometry, ParamSet, SimSettings );
        
        %==================================================================
        % Simulate time propogation of Trans. Mag. Field Inhomogeneities
        %==================================================================
        Results     =   PerfusionOrientation_PropTransMag( ...
            Geometry, PhysicalParams, ParamSet, SimSettings );
        
        %==================================================================
        % Save all results
        %==================================================================
        % Copy fields from Results to AllResults on the first iteration
        if ii == 1 && jj == 1
            AllResults = copy_structfields_blank(Results,AllResults);
        end
        
        ParamSetRep_time	=   toc(ParamSetRep_time);
        str                 =	...
            sprintf( 'ParamSet, Rep %d/%d', jj, SimSettings.NumRepetitions );
        display_toc_time( ParamSetRep_time, str, [1,1] );
    	
        %==================================================================
        % Plot delta R2 and save figures
        %==================================================================
        AllResults(ii,jj)	=   Results;
        
        if SimSettings.NumRepetitions > 1
            [ h_time, h_angle ]	=	plot_dR2( Results, [], ParamSet, SimSettings );
            save_dR2_plots( h_time, h_angle, ParamSet, SimSettings );
        end
        
        %==================================================================
        % Update Diary
        %==================================================================
        if SimSettings.flags.SaveData, diary( SimSettings.DiaryPath ); end
        
    end
    
    % Finish Timing Parameter Set
    ParamSet_time	=   toc(ParamSet_time);
    display_toc_time( ParamSet_time, 'ParamSet, All Reps' );
	
    % Save AllResults struct
    save_AllResults( AllResults(ii,:), SimSettings )
    
    % Copy fields from Results to AvgResults on the first iteration
    if ii == 1
        AvgResults = copy_structfields_blank(Results,AvgResults);
    end
    
    % Calculate and save average results
    AvgResults(ii,1)	=   get_avgResults( AllResults(ii,:), AvgResults(ii,1), SimSettings );
    save_AvgResults( AvgResults(ii,1), SimSettings )
    
    % Plot averaged dR2
    [ h_time, h_angle ]	=   ...
        plot_dR2( AllResults(ii,:), AvgResults(ii,1), ParamSet, SimSettings );
	save_AlldR2_plots( h_time, h_angle, SimSettings );
    
end

%==========================================================================
% Testing
%==========================================================================
% ParamInput = [Params.CA_Concentration, Params.Total_BVF, Params.MinorVessel_RelBVF];
% AvgResults.dR2_TE = test_datafun(ParamInput,SimSettings);
% AllResults = AvgResults;

%==========================================================================
% Finish simulation
%==========================================================================

% Set default figure visibility back to 'on'
set(0, 'DefaultFigureVisible', 'on');

% Save complete sets of results to RootSavePath
SimSettings.SavePath	=	RootSavePath;
save_FinalResults( AllResults, AvgResults, Params, SimSettings );

suffix	=   datestr(now); if ~isempty(IterStr), suffix = [IterStr ', ' suffix]; end
display_text( ['Simulation Finished: ' suffix], [], '%', true, [1,1] );

total_simulation_time	=	toc(total_simulation_time);
display_toc_time( total_simulation_time, 'Entire Simulation' );    

% Final Diary update
if SimSettings.flags.SaveData, diary( SimSettings.DiaryPath ); diary off; end

% Save SimSettings on each iter. for minimization
if isMinimization, save_ParamSetSimSettings( Params, SimSettings ); end

end

function [ParamCombinations] = getAllParamCombinations( Params )
%==========================================================================
% getAllParamCombinations
%==========================================================================
% Replicates the vectors in each variable of the table Params using ndgrid
% for simulating every parameter combination

ParamCombinations       =	table2cell(Params);
[ParamCombinations{:}]	=	ndgrid( ParamCombinations{:} );
for ii = 1:length(ParamCombinations)
    ParamCombinations{ii}	=	ParamCombinations{ii}(:);
end

ParamCombinations       =	table(	...
    ParamCombinations{:}, 'VariableNames', Params.Properties.VariableNames );

end

function [ Params, SimSettings ] = CalculateDerivedSettings( PhysicalParams, Params, SimSettings )
%==========================================================================
% CalculateDerivedSettings
%==========================================================================
% 
% This function calculates geometry independent settings for the simulation
% that are derived from the given simulation parameters 'Params' and
% settings 'SimSettings', as well as physical parameters 'PhysicalParams'

SubVoxSize	=	mean( SimSettings.VoxelSize ./ SimSettings.GridSize ); % [um]
Height      =	SimSettings.VoxelSize(3);
Area        =   prod(SimSettings.VoxelSize(1:2));

% Mean and std for the simulated minor vessel radii [um]
R_Minor_mu	=	PhysicalParams.R_Minor_mu;
R_Minor_sig	=   PhysicalParams.R_Minor_sig;

% Calculate blood volumes
Total_Volume	=   prod(SimSettings.VoxelSize); % total volume of voxel [um^3]
Total_BloodVol	=   Params.Total_BVF * Total_Volume; % total blood volume (main and minor vessels)
Minor_BloodVol	=   Params.MinorVessel_RelBVF .* Total_BloodVol; % blood volume for minor vessels
Major_BloodVol	=   Total_BloodVol - Minor_BloodVol; % blood volume for major vessels

% If the radius 'r' is normally distributed ~ N(mu,sig), then the
% expectation of r^2, E[r^2], is given by E[r^2] = mu^2 + sig^2
Minor_Area      =   pi * ( R_Minor_mu.^2 + R_Minor_sig.^2 );
NumMinorVessels	=   round( Minor_BloodVol ./ (Height * Minor_Area) ); % N*Area*Height = Volume (underestimated)

% Major blood vessel diameters: N*pi*r^2*h = V
R_Major         =   sqrt( Major_BloodVol./( SimSettings.NumMajorVessels * pi * Height ) );

% Relaxation constant difference in blood vs. tissue including contrast agent
R2_CA           =	PhysicalParams.R2_CA_per_mM * Params.CA_Concentration;
R2_Blood_Total	=	R2_CA + PhysicalParams.R2_Blood;

% Susceptibility difference in blood vs. tissue including contrast agent
deltaChi_CA     =   PhysicalParams.dChi_CA_per_mM * Params.CA_Concentration;
deltaChi_Total	=   deltaChi_CA + PhysicalParams.dChi_Blood;

% Update simulation settings
SimSettings.Total_Volume    =   Total_Volume;
SimSettings.SubVoxSize      =   SubVoxSize;

% Update variable parameters
NewParams	=	table(	Total_BloodVol, Minor_BloodVol, Major_BloodVol,	...
    NumMinorVessels, R_Major, deltaChi_Total, R2_Blood_Total	);
Params      =   [ Params, NewParams ];

end

function AvgResults = get_avgResults( Results, AvgResults, SimSettings )

% Results	=   struct( ...
%     'Angles_Rad',     Angles_Rad,                     ...
%     'Signal_noCA',	{cell(NumAngles,1)},	...
%     'Signal_CA',      {cell(NumAngles,1)},	...
%     'dR2_all',        {cell(NumAngles,1)},	...
%     'dR2_TE',         zeros(NumAngles,1)      ...
%     );

NumResults	=   numel( Results );
NumAngles	=   numel( Results(1).Angles_Rad );
NumTimes	=   numel( Results(1).Signal_noCA{1} );

% Average signal w/o CA
Signal_noCA	=   mean( reshape( cell2mat( [Results.Signal_noCA] ),	...
                            [NumAngles,NumTimes,NumResults]	), 3 );

% Average signal with CA
Signal_CA	=   mean( reshape( cell2mat( [Results.Signal_CA] ),	...
                            [NumAngles,NumTimes,NumResults]	), 3 );

% Averaged delta-R2
% Definition:
%   deltaR2	:=	-1/TE * log(|S|/|S0|)
dR2_func	=	@(S,S0) (-1/SimSettings.EchoTime) * log(abs(S)./abs(S0));
dR2_all     =	dR2_func(Signal_CA,Signal_noCA);
dR2_TE      =	dR2_all(:,end);

%{
dR2_all     =   reshape(    cell2mat( [Results.dR2_all] ),	...
                            [NumAngles,NumTimes,NumResults]	);
dR2_all     =   mean( dR2_all, 3 );
dR2_TE      =   dR2_all(:,end);
dR2_all     =   mat2cell(	dR2_all, ones(NumAngles,1), NumTimes     );
%}

% Convert back to cell arrays
Signal_noCA	=   mat2cell( Signal_noCA, ones(NumAngles,1), NumTimes );
Signal_CA	=   mat2cell( Signal_CA,   ones(NumAngles,1), NumTimes );
dR2_all     =   mat2cell( dR2_all,     ones(NumAngles,1), NumTimes );

% Update AvgResults
AvgResults.Time         =   Results(1).Time;
AvgResults.Angles_Rad	=   Results(1).Angles_Rad;
AvgResults.Signal_noCA	=   Signal_noCA;
AvgResults.Signal_CA	=   Signal_CA;
AvgResults.dR2_all      =   dR2_all;
AvgResults.dR2_TE       =   dR2_TE;

end

function save_Geometry( Geometry, Params, SimSettings )

if SimSettings.flags.SaveData
    % Don't save full vasculature map; can easily recreate
    Geometry.VasculatureMap	=	[];
    
    filename	=	[ SimSettings.SavePath, '/', ...
        sprintf( 'Geometry_Rep%d', Params.Rep ) ];
    
    try
        save( filename, 'Geometry' );
    catch me
        warning(me.identifier,me.message);
    end
    
end


end

function save_AllResults( AllResults, SimSettings )

if SimSettings.flags.SaveData
    filename	=	[ SimSettings.SavePath, '/', 'All_Results' ];
    try
        save( filename, 'AllResults' );
    catch me
        warning(me.identifier,me.message);
    end
end

end

function save_AvgResults( AvgResults, SimSettings )

if SimSettings.flags.SaveData
    filename	=	[ SimSettings.SavePath, '/', 'Avg_Results' ];
    try
        save( filename, 'AvgResults' );
    catch me
        warning(me.identifier,me.message);
    end
end

end

function save_FinalResults( AllResults, AvgResults, Params, SimSettings )

if SimSettings.flags.SaveData
    try
        if size(Params,1) > 1 || all(~isnan(SimSettings.MinimizationType))
            filename	=	[ SimSettings.SavePath, '/', 'All_Results' ];
            save( filename, 'AllResults' );
            
            filename	=	[ SimSettings.SavePath, '/', 'Avg_Results' ];
            save( filename, 'AvgResults' );
        end
    catch me
        warning(me.identifier,me.message);
    end
end

end

function save_ParamSetSimSettings( Params, SimSettings )

if SimSettings.flags.SaveData
    try
        ParamSet	=   Params(1,1:3);
        filename	=	[ SimSettings.SavePath, '/', 'ParamSet' ];
        save( filename, 'ParamSet' );
        
        filename	=	[ SimSettings.SavePath, '/', 'SimSettings' ];
        save( filename, 'SimSettings' );
    catch me
        warning(me.identifier,me.message);
    end
end

end

function save_dR2_plots( h_time, h_angle, Params, SimSettings )

if SimSettings.flags.SaveData
    
    TimePlot	=   ~isempty( h_time );
    AnglePlot	=   ~isempty( h_angle );
    
    filename	=   {};
    if TimePlot
        filename	=   [ filename, sprintf('dR2_vs_Time_Rep%d', Params.Rep) ];
    end
    if AnglePlot
        filename	=   [ filename, sprintf('dR2_vs_Angle_Rep%d', Params.Rep) ];
    end
    
    if TimePlot || AnglePlot
        save_simulation_figure( ...
            filename, [ h_time; h_angle ], true, SimSettings );
    end
    
end

end

function save_AlldR2_plots( h_time, h_angle, SimSettings )

if SimSettings.flags.SaveData
    
    TimePlot	=   ~isempty( h_time );
    AnglePlot	=   ~isempty( h_angle );
    
    filename	=   {};
    if TimePlot
        filename	=   [ filename, 'All_dR2_vs_Time' ];
    end
    if AnglePlot
        filename	=   [ filename, 'All_dR2_vs_Angle' ];
    end
    
    if TimePlot || AnglePlot
        save_simulation_figure( ...
            filename, [ h_time; h_angle ], true, SimSettings );
    end
    
end

end

%--------------------------------------------------------------------------
% Testing

function f = test_datafun(params,SimSettings)

% Approximate solution by nth order polynomial and let last 3 coeffs vary
% by multiplicative constants:
n	=   2;
x	=   SimSettings.Angles_Deg_Data;
y	=   SimSettings.dR2_Data;
p	=   polyfit(x,y,n);

p(end-2:end)	=   p(end-2:end) .* params(:).';
f               =   polyval(p,x);

% Solution under the L2-norm C = sum( (polyval(p,x)-y).^2 ):
%   min C: 	.00447898986832132
%   params: [0.001483621633878, -0.028752402650885, 3.270148884331981]

end

%{
function f = datafun(params)

[x,y,z]         =   dealArray(params);

gaussian_fun	=	@(x,y,z,x0,y0,z0,s) exp(-((x-x0).^2+(y-y0).^2+(z-z0).^2)./(2*s^2));
[xl,yl,zl]      =	meshgrid(linspacePeriodic(-1,1,5));
[xl,yl,zl]      =   deal(xl(:),yl(:),zl(:));
peaks           =   1 + 0.3 * sin(linspace(-pi,pi,numel(xl)));

f	=	0;
s	=   0.05;
for ii = 1:numel(xl)
    f	=   f + peaks(ii) * gaussian_fun(x,y,z,xl(ii),yl(ii),zl(ii),s);
end

f	=	max(peaks(:)) - f;

end
%}
