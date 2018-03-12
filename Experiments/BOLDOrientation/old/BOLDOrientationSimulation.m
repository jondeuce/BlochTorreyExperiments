function [ AllResults, AvgResults, Params, SimSettings ] = ...
    BOLDOrientationSimulation( Params, SimSettings )
%BOLDORIENTATIONSIMULATION 

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
SimSettings.DiaryPath	=    [ RootSavePath, '/', 'Diary' ];
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
% Initialize AllResults object and MetaData to be saved with it
%==========================================================================

AllResults = BOLDResults( ...
    SimSettings.EchoTime, ...
    SimSettings.Angles_Rad_Data, ...
    Params.DeoxygenatedBloodLevel, ...
    Params.OxygenatedBloodLevel, ...
    Params.Hct, ...
    1:SimSettings.NumRepetitions ...
    );

% Initialize MetaData that will be saved
[ DerivedParams, DerivedSimSettings ] = CalculateDerivedSettings( Params, SimSettings );
AllResults.MetaData = struct( ...
    'Params',         DerivedParams, ...
    'SimSettings',    DerivedSimSettings ...
    );

%==========================================================================
% Get all combination of parameters
%==========================================================================

[ Params ]      =   getAllParamCombinations( Params );
NumParamSets	=   size(Params,1);

%==========================================================================
% Calculate derived simulation settings
%==========================================================================

[ Params, SimSettings ] = CalculateDerivedSettings( Params, SimSettings );

%==========================================================================
% Loop through all combinations of parameters
%==========================================================================

for ii = 1:NumParamSets
    
    % Make subfolder for saving this parameter set of data if necessary
    if SimSettings.flags.SaveData
        SimSettings.SavePath	=   sprintf( '%s/ParamSet_%s_%s', SavePath, ...
            num2strpad(ii,numdigits(NumParamSets)), num2str(NumParamSets) );
        mkdir(SimSettings.SavePath);
    end
    
    % Get a single set of parameters
    ParamSet = Params(ii,:);
    
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
        Geometry	=	CalculateGeometry( ParamSet, SimSettings );
        
        %==================================================================
        % Simulate time propogation of Trans. Mag. Field Inhomogeneities
        %==================================================================
        Results     =   BOLDOrientation_PropTransMag( ...
            Geometry, [], ParamSet, SimSettings );
        
        %==================================================================
        % Save Results (includes compressed Geometry in Results.MetaData)
        %==================================================================
        save_ResultRep( Results, ParamSet, SimSettings )
        
        %==================================================================
        % Print elapsed time
        %==================================================================
        ParamSetRep_time  =  toc(ParamSetRep_time);
        str               =  sprintf( 'ParamSet, Rep %d/%d', jj, SimSettings.NumRepetitions );
        display_toc_time( ParamSetRep_time, str, [1,1] );
        
        %==================================================================
        % Plot delta R2 and save figures
        %==================================================================
        AllResults  = push( AllResults, Results );
        
        % TODO 
%         if SimSettings.NumRepetitions > 1
%             [ h_time, h_angle ]    =	plot_dR2( Results, [], ParamSet, SimSettings );
%             save_dR2_plots( h_time, h_angle, ParamSet, SimSettings );
%         end
        
        %==================================================================
        % Update Diary
        %==================================================================
        if SimSettings.flags.SaveData, diary( SimSettings.DiaryPath ); end
        
    end
    
    % Finish Timing Parameter Set
    ParamSet_time	=   toc(ParamSet_time);
    display_toc_time( ParamSet_time, 'ParamSet, All Reps' );
        
    %==========================================================================
    % Save simulation parameters, settings, and compressed geometry
    %==========================================================================
    
    % TODO
%     % Plot averaged dR2
%     [ h_time, h_angle ]    =   ...
%         plot_dR2( AllResults(ii,:), AvgResults(ii,1), ParamSet, SimSettings );
% 	save_AlldR2_plots( h_time, h_angle, SimSettings );
    
end

%==========================================================================
% Calculate Average Results, if necessary
%==========================================================================

if SimSettings.NumRepetitions > 1
    AvgResults  =  mean( AllResults ); %overloaded BOLDResults method
else
    AvgResults  =  AllResults;
end

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
[ParamCombinations{:}]    =	ndgrid( ParamCombinations{:} );
for ii = 1:length(ParamCombinations)
    ParamCombinations{ii}    =	ParamCombinations{ii}(:);
end

ParamCombinations       =	table(    ...
    ParamCombinations{:}, 'VariableNames', Params.Properties.VariableNames );

end

function [ Params, SimSettings ] = CalculateDerivedSettings( Params, SimSettings )
%==========================================================================
% CalculateDerivedSettings
%==========================================================================
% 
% This function calculates geometry independent settings for the simulation
% that are derived from the given simulation parameters 'Params' and
% settings 'SimSettings'

SubVoxSize	=	mean( SimSettings.VoxelSize ./ SimSettings.GridSize ); % [um]
Height      =	SimSettings.VoxelSize(3);
%Area       =   prod(SimSettings.VoxelSize(1:2));

% Mean and std for the simulated minor vessel radii [um]
R_Minor_mu	=	SimSettings.R_Minor_mu;
R_Minor_sig	=   SimSettings.R_Minor_sig;

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

% % Relaxation constant difference in blood vs. tissue including contrast agent
% R2_CA           =	SimSettings.R2_CA_per_mM * Params.CA_Concentration;
% R2_Blood_Total	=	R2_CA + Params.R2_Blood;

% Hematocrit fraction
Hct = Params.Hct;

BField = SimSettings.B0;
if BField == -3.0
    
    % Relaxation constant in blood vs. tissue as a function of Hct and Y:
    % 	Zhao et al., Oxygenation and Hematocrit Dependence of Transverse
    %   Relaxation Rates of Blood at 3T (2007)
    if ~all(Hct == 0.21 | Hct == 0.44 | Hct == 0.57)
        error('Hct must be one of 0.21, 0.44, or 0.57 for R2(Hct,Y) model. See MRM 2007 Zhao et al., Oxygenation and hematocrit dependence of transverse relaxation rates of blood at 3T');
    end
    
    [A,B,C] = deal(zeros(size(Params,1),1));
    if true
        % B coeff. == 0 Model (More physical)
        A(Hct == 0.21) = 8.2;  B(Hct == 0.21) = 0; C(Hct == 0.21) = 91.6;
        A(Hct == 0.44) = 11.0; B(Hct == 0.44) = 0; C(Hct == 0.44) = 125;
        A(Hct == 0.57) = 14.3; B(Hct == 0.57) = 0; C(Hct == 0.57) = 152;
    else
        % B coeff. ~= 0 Model (Less physical, better fit)
        A(Hct == 0.21) = 6.0;  B(Hct == 0.21) = 21.0; C(Hct == 0.21) = 94.3;
        A(Hct == 0.44) = 8.3;  B(Hct == 0.44) = 33.6; C(Hct == 0.44) = 71.9;
        A(Hct == 0.57) = 10.6; B(Hct == 0.57) = 39.3; C(Hct == 0.57) = 61.6;
    end
    
    R2_BloodModel = @(Y) A + B.*(1-Y) + C.*(1-Y).^2;
    
    % T2 Value in WM @ 3 Tesla:
    %    Deistung et al. Susceptibility Weighted Imaging at Ultra High,
    %    Magnetic Field Strengths: Theoretical Considerations and
    %    Experimental Results, MRM 2008
    T2_Tissue = 69; % +/- 3 [ms]
    R2_Tissue = 1000/T2_Tissue; % [ms] -> [1/s]
    
elseif BField == -7.0
    
    if ~all(abs(Hct-0.44) <= 0.01)
        error('Hct fraction must be within 0.01 of the nominal value of 0.44');
    end
    
    % Fit line through R2 vs. (1-Y)^2 data based on measurements from:
    %    Yacoub et al. Imaging Brain Function in Humans at 7 Tesla, MRM 2001
    Ydata = [0.38,0.39,0.59];
    T2data = [6.8,7.1,13.1];
    R2data = 1000./T2data;
    PP = polyfit((1-Ydata).^2,R2data,1);
    
    R2_BloodModel = @(Y) polyval(PP,(1-Y).^2);
    
    % T2 Value in WM @ 7 Tesla:
    %    Deistung et al. Susceptibility Weighted Imaging at Ultra High,
    %    Magnetic Field Strengths: Theoretical Considerations and
    %    Experimental Results, MRM 2008
    
    %T2_Tissue = 45.9; % +/-1.9 [ms]
    
    T2_Tissue = 69; % +/-3 [ms]
    input('WARNING: Running 7T simulation using T2 = 69 ms (3T value)!\n[press enter to continue...]\n');
    
    R2_Tissue = 1000/T2_Tissue; % [ms] -> [1/s]
    
else
    error('Only have R2 values for B0 = -3.0 or B0 = -7.0');
end

R2_DeoxyBlood   =   R2_BloodModel(Params.DeoxygenatedBloodLevel);
R2_OxyBlood     =   R2_BloodModel(Params.OxygenatedBloodLevel);

R2_Blood_Baseline  =  R2_DeoxyBlood;
R2_Blood_Total     =  R2_OxyBlood;

% Susceptibility difference in blood vs. tissue including contrast agent
% deltaChi_CA     =   SimSettings.dChi_CA_per_mM * Params.CA_Concentration;

% Susceptibilty of blood relative to tissue due to blood oxygenation is given by:
%   deltaChi_Blood_Tissue  :=   Hct * (1-Y) * 2.26e-6 [T/T]
dChi_DeoxyBlood =   2.26e-6 * Params.Hct .* (1 - Params.DeoxygenatedBloodLevel);
dChi_OxyBlood   =   2.26e-6 * Params.Hct .* (1 - Params.OxygenatedBloodLevel);

% Susceptibility difference in blood vs. tissue including contrast agent as well
deltaChi_Baseline  =   dChi_DeoxyBlood;
deltaChi_Total     =   dChi_OxyBlood;

% Update simulation settings
SimSettings.Total_Volume    =   Total_Volume;
SimSettings.SubVoxSize      =   SubVoxSize;

% Update variable parameters
NewParams	=	table( ...
    Total_BloodVol, Minor_BloodVol, Major_BloodVol, NumMinorVessels, R_Major, ...
    dChi_DeoxyBlood, dChi_OxyBlood, deltaChi_Baseline, deltaChi_Total, ...
    R2_Tissue, R2_DeoxyBlood, R2_OxyBlood, R2_Blood_Baseline, R2_Blood_Total ...
    );
Params      =   [ Params, NewParams ];

end

function save_ResultRep( Results, Params, SimSettings )

if SimSettings.flags.SaveData
    
    filename	=    [ SimSettings.SavePath, '/', ...
        sprintf( 'Results_Rep%d', Params.Rep ) ];
    
    try
        save( filename, 'Results', '-v7' );
    catch me
        warning(me.identifier,me.message);
    end
    
end


end

function save_FinalResults( AllResults, AvgResults, Params, SimSettings )

if SimSettings.flags.SaveData
    try
        if size(Params,1) > 1 || all(~isnan(SimSettings.MinimizationType))
            filename = [ SimSettings.SavePath, '/', 'All_Results' ];
            save( filename, 'AllResults', '-v7' );
            
            if SimSettings.NumRepetitions > 1
                filename = [ SimSettings.SavePath, '/', 'Avg_Results' ];
                save( filename, 'AvgResults', '-v7' );
            end
        end
    catch me
        warning(me.identifier,me.message);
    end
end

end

function save_ParamSetSimSettings( Params, SimSettings )

if SimSettings.flags.SaveData
    try
        ParamSet	=   Params(1,:);
        filename	=    [ SimSettings.SavePath, '/', 'ParamSet' ];
        save( filename, 'ParamSet', '-v7' );
        
        filename	=    [ SimSettings.SavePath, '/', 'SimSettings' ];
        save( filename, 'SimSettings', '-v7' );
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
%--------------------------------------------------------------------------

function f = test_datafun(params,SimSettings)

% Approximate solution by nth order polynomial and let last 3 coeffs vary
% by multiplicative constants:
n	=   2;
x	=   SimSettings.Angles_Deg_Data;
y	=   SimSettings.dR2_Data;
p	=   polyfit(x,y,n);

p(end-2:end)    =   p(end-2:end) .* params(:).';
f               =   polyval(p,x);

% Solution under the L2-norm C = sum( (polyval(p,x)-y).^2 ):
%   min C:     .00447898986832132
%   params: [0.001483621633878, -0.028752402650885, 3.270148884331981]

end

%{
function f = datafun(params)

[x,y,z]         =   dealArray(params);

gaussian_fun	=    @(x,y,z,x0,y0,z0,s) exp(-((x-x0).^2+(y-y0).^2+(z-z0).^2)./(2*s^2));
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
