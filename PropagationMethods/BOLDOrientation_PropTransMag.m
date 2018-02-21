function [ Results ] = BOLDOrientation_PropTransMag( ...
    Geometry, PhysicalParams, Params, SimSettings )
%BOLDORIENTATION_PROPTRANSMAG Propogates the transverse magnetization
%under the following time dependence:
%
%   dM(r,t)/dt = div( D * grad(M(r,t)) ) - (R2(r) + i*dw(r)) * M(r,t)
%
% Where:
%   -M(r,t) is the transverse magnetization
%   -D is the diffusion tensor (must be scalar; extensions to 3-vector or
%    3x3 tensor possible within framework)
%   -R2 is the tissue-specific transverse relaxation (R2*) constant
%   -dw(r) is the Larmor frequency of the spins at position r

switch SimSettings.Dimension
    case 2
        error('2D is not implemented for BOLD simulation.');
        Results  =  PropogateTransMagnetization_2D( Geometry, PhysicalParams, Params, SimSettings );
    case 3
        Results  =  PropogateTransMagnetization_3D( Geometry, PhysicalParams, Params, SimSettings );
end

%==========================================================================
% Save simulation parameters, settings, and compressed geometry
%==========================================================================

% Compress geometry for saving; Vasc. Map and mx's can be recreated if needed
Geo = Geometry;
Geo.VasculatureMap = [];
Geo.MainCylinders.mx = [];
Geo.MinorCylinders.mx = [];

Results.MetaData  =  struct( ...
    'Geometry',       Geo, ...
    'PhysicalParams', PhysicalParams, ...
    'Params',         Params, ...
    'SimSettings',    SimSettings ...
    );

end

function Results = PropogateTransMagnetization_2D( ...
    Geometry, PhysicalParams, Params, SimSettings )

if SimSettings.AddDiffusion
    PropogationMethod    =   ...
        @PropogateTransMagnetizationWithConvolutionDiffusion_Order2;
else
    PropogationMethod    =   ...
        @PropogateTransMagnetizationNoDiffusion;
end

% Extract parameters
Angles_Rad          =   SimSettings.Angles_Rad_Data;
NumAngles           =   numel(Angles_Rad);

% Initialize Results
Results    =   struct( ...
    'Time',          [],                      ...
    'Angles_Rad',    Angles_Rad,             ...
    'Signal_Base',   {cell(NumAngles,1)},    ...
    'Signal_noBase', {cell(NumAngles,1)},    ...
    'dR2_all',       {cell(NumAngles,1)},    ...
    'dR2_TE',        zeros(NumAngles,1)      ...
    );

if SimSettings.AllowParallel
    Results    =    Parfor_PropogateTransMag_2D( PropogationMethod, Results, ...
        Geometry, PhysicalParams, Params, SimSettings );
else
    Results    =    Linear_PropogateTransMag_2D( PropogationMethod, Results, ...
        Geometry, PhysicalParams, Params, SimSettings );
end

%==========================================================================
% Calculate dR2
%==========================================================================
Results     =    get_dR2( Results, PhysicalParams, Params, SimSettings );

end

function Results = PropogateTransMagnetization_3D( ...
    Geometry, PhysicalParams, Params, SimSettings )

if SimSettings.AddDiffusion
    if SimSettings.UseConvDiffusion
        %PropogationMethod   =   @PropogateTransMagnetizationWithConvolutionDiffusion_Order2;
        PropogationMethod   =   @PropogateTransMagnetizationWithConvolutionDiffusion_Order3;
    else
        error('only convolution diffusion is allowed.');
        %PropogationMethod   =   @PropogateTransMagnetizationWithDiffusion_3D;
    end
else
    PropogationMethod   =   @PropogateTransMagnetizationNoDiffusion;
end

% Extract parameters
Angles_Rad          =   SimSettings.Angles_Rad_Data;
EchoTimes           =   SimSettings.EchoTime;

% Initialize Results
Results = BOLDResults( ...
    EchoTimes, ...
    Angles_Rad, ...
    Params.DeoxygenatedBloodLevel, ...
    Params.OxygenatedBloodLevel, ...
    Params.Hct, ...
    Params.Rep );

% % For Testing
% Results = fillwithmockdata(Results);
% return

if SimSettings.AllowParallel
    %     Results    =    Parallel_PropogateTransMag_3D( PropogationMethod, Results, ...
    %         Geometry, PhysicalParams, Params, SimSettings );
    Results    =    Parfor_PropogateTransMag_3D( PropogationMethod, Results, ...
        Geometry, PhysicalParams, Params, SimSettings );
else
    Results    =    Linear_PropogateTransMag_3D( PropogationMethod, Results, ...
        Geometry, PhysicalParams, Params, SimSettings );
end

end

function Results = Linear_PropogateTransMag_2D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Angle data
Angles_Rad    =   SimSettings.Angles_Rad_Data;
NumAngles    =   numel(Angles_Rad);

% Geometry (Baseline)
Geometry.isBaseline    =    true;
Geometry    =    get_R2Map( Geometry, PhysicalParams, Params, SimSettings );

for ii = 1:NumAngles
    %======================================================================
    % Simulation: Baseline
    %======================================================================
    
    t_Base    =    tic;
    
    Geometry.alpha     =    Angles_Rad(ii);
    Geometry           =    ...
        get_dOmegaMap_analytic( Geometry, PhysicalParams, Params, SimSettings );
    [ Results.Signal_Base{ii}, Results.Time ]    =   ...
        PropMethod( Geometry, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_Base), sprintf( 'Angle %2d/%2d, %5.2f%s, Baseline', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
end

% Geometry (not Baseline)
Geometry.isBaseline    =    false;
Geometry    =    get_R2Map( Geometry, PhysicalParams, Params, SimSettings );

for ii = 1:NumAngles
    %======================================================================
    % Simulation: Not Baseline
    %======================================================================
    
    t_noBase    =    tic;
    
    Geometry.alpha    =    Angles_Rad(ii);
    Geometry           =   ...
        get_dOmegaMap_analytic( Geometry, PhysicalParams, Params, SimSettings );
    [ Results.Signal_noBase{ii}, Results.Time ]     =    ...
        PropMethod( Geometry, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_noBase), sprintf( 'Angle %2d/%2d, %5.2f%s, no Base', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
end

end

function Results = Parfor_PropogateTransMag_2D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Angle data
Angles_Rad    =   SimSettings.Angles_Rad_Data;
NumAngles    =   numel(Angles_Rad);

% Geometry (Baseline)
Geometry.isBaseline    =    true;
Geometry    =    get_R2Map( Geometry, PhysicalParams, Params, SimSettings );

Signal_Base    =   cell(NumAngles,1);
Time        =   cell(NumAngles,1);
NumWorkers    =   min(NumAngles,6);
parfor (ii = 1:NumAngles, NumWorkers)
    %======================================================================
    % Simulation: Baseline
    %======================================================================
    t_Base    =    tic;
    
    LoopGeo         =   Geometry
    LoopGeo.alpha     =    Angles_Rad(ii);
    LoopGeo           =    ...
        get_dOmegaMap_analytic( LoopGeo, PhysicalParams, Params, SimSettings );
    [ Signal_Base{ii}, Time{ii} ]    =   ...
        PropMethod( LoopGeo, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_Base), sprintf( 'Angle %2d/%2d, %5.2f%s, Baseline', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
end
Results.Signal_Base    =    Signal_Base;
Results.Time          =   Time{1};

% Geometry (not Baseline)
Geometry.isBaseline    =    false;
Geometry    =    get_R2Map( Geometry, PhysicalParams, Params, SimSettings );

Signal_noBase    =   cell(NumAngles,1);
parfor (ii = 1:NumAngles, NumWorkers)
    %======================================================================
    % Simulation: Not Baseline
    %======================================================================
    t_noBase    =    tic;
    
    LoopGeo         =   Geometry;
    LoopGeo.alpha     =    Angles_Rad(ii);
    LoopGeo           =    ...
        get_dOmegaMap_analytic( LoopGeo, PhysicalParams, Params, SimSettings );
    [ Signal_noBase{ii}, ~ ]     =    ...
        PropMethod( LoopGeo, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_noBase), sprintf( 'Angle %2d/%2d, %5.2f%s, no Base', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
end
Results.Signal_noBase    =    Signal_noBase;

end

function Results = Linear_PropogateTransMag_3D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Angle data
Angles_Rad  =   SimSettings.Angles_Rad_Data;
NumAngles   =   numel(Angles_Rad);
EchoTimes   =   SimSettings.EchoTime;
NumEchos    =   numel(EchoTimes);
ResultsArgs =   getargs(Results);

% Geometry (Baseline)
Geometry.isBaseline    =    true;

if SimSettings.flags.RunBaseline
    for ii = 1:NumAngles
        
        %======================================================================
        % Simulation: Baseline
        %======================================================================
        
        t_Base  =  tic;
        
        Geometry.alpha =  Angles_Rad(ii);
        Geometry       =  get_dOmegaMap( Geometry, PhysicalParams, Params, SimSettings );
        
        [ Signal_Base ]    =   PropMethod( Geometry, PhysicalParams, Params, SimSettings );
        Results = push( Results, Signal_Base, [], EchoTimes, Angles_Rad(ii), ResultsArgs{3:end} );
        
        display_toc_time( toc(t_Base), sprintf( 'Angle %2d/%2d, %5.2f%s, Baseline', ...
            ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
        
    end
end

% Geometry (not Baseline)
Geometry.isBaseline  =  false;

if SimSettings.flags.RunActivated
    for ii = 1:NumAngles
        
        %======================================================================
        % Simulation: Not Baseline
        %======================================================================
        
        t_noBase  =  tic;
        
        Geometry.alpha =  Angles_Rad(ii);
        Geometry       =  get_dOmegaMap( Geometry, PhysicalParams, Params, SimSettings );
        
        [ Signal_noBase ]  =  PropMethod( Geometry, PhysicalParams, Params, SimSettings );
        Results = push( Results, [], Signal_noBase, EchoTimes, Angles_Rad(ii), ResultsArgs{3:end} );
        
        display_toc_time( toc(t_noBase), sprintf( 'Angle %2d/%2d, %5.2f%s, no Base', ...
            ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
        
    end
end

end

function Results = Parfor_PropogateTransMag_3D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Angle data
Angles_Rad  =  SimSettings.Angles_Rad_Data;
NumAngles   =  numel(Angles_Rad);

% Geometry (Baseline)
Geometry.isBaseline  =    true;
Geometry    =    get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry    =    get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );

Signal_Base =   cell(NumAngles,1);
Time        =   cell(NumAngles,1);
parfor (ii = 1:NumAngles, NumAngles)
    %======================================================================
    % Simulation: Baseline
    %======================================================================
    t_Base    =    tic;
    
    LoopGeo         =   Geometry
    LoopGeo.alpha     =    Angles_Rad(ii);
    LoopGeo           =    ...
        get_dOmegaMap_step2( LoopGeo, PhysicalParams, Params, SimSettings );
    [ Signal_Base{ii}, Time{ii} ]    =   ...
        PropMethod( LoopGeo, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_Base), sprintf( 'Angle %2d/%2d, %5.2f%s, Baseline', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
    
end
Results.Signal_Base    =    Signal_Base;
Results.Time          =   Time{1};

% Geometry (not Baseline)
Geometry.isBaseline    =    false;
Geometry    =    get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry    =    get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );

Signal_noBase    =   cell(NumAngles,1);
parfor (ii = 1:NumAngles, NumAngles)
    %======================================================================
    % Simulation: Not Baseline
    %======================================================================
    t_noBase    =    tic;
    
    LoopGeo         =   Geometry
    LoopGeo.alpha    =    Angles_Rad(ii);
    LoopGeo           =   ...
        get_dOmegaMap_step2( LoopGeo, PhysicalParams, Params, SimSettings );
    [ Signal_noBase{ii}, ~ ]    =    ...
        PropMethod( LoopGeo, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_noBase), sprintf( 'Angle %2d/%2d, %5.2f%s, no Base', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
    
end
Results.Signal_noBase    =    Signal_noBase;

end

function Results = Parallel_PropogateTransMag_3D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Setup
toRowCell       =   @(x) mat2cell(x,ones(size(x,1),1),size(x,2));
Angles_Rad      =   SimSettings.Angles_Rad_Data;
Geometry.alpha    =    Angles_Rad; %all angles will be computed in parallel

%======================================================================
% Simulation: Baseline
%======================================================================

t_Base      =   tic;

% Step 1 (Baseline)
t_step1     =    tic;
Geometry.isBaseline    =    true;
Geometry    =    get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry    =    get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_step1), 'All Angles, Step 1 (Baseline)' );

% Step 2 (Baseline)
t_step2     =   tic;
Geometry    =    get_dOmegaMap_step2( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_step2), 'All Angles, Step 2 (Baseline)' );

% Propogation (Baseline)
t_prop      =   tic;
[ Signal_Base, Time ]    =   PropMethod( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_prop), 'All Angles, Prop   (Baseline)' );

Results.Signal_Base    =   toRowCell(Signal_Base);
Results.Time        =   Time(1,:);

display_toc_time( toc(t_Base), 'All Angles, Total  (Baseline)' );

%======================================================================
% Simulation: Not Baseline
%======================================================================

t_noBase        =   tic;

% Step 1 (not Baseline)
t_step1     =    tic;
Geometry.isBaseline    =    false;
Geometry    =    get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry    =    get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_step1), 'All Angles, Step 1 (not Baseline)' );

% Step 2 (not Baseline)
t_step2     =   tic;
Geometry    =   get_dOmegaMap_step2( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_step2), 'All Angles, Step 2 (not Baseline)' );

% Propogation (not Baseline)
t_prop      =   tic;
[ Signal_noBase, ~ ]    =    PropMethod( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_prop), 'All Angles, Prop   (not Baseline)' );

Results.Signal_noBase    =   toRowCell(Signal_noBase);

display_toc_time( toc(t_noBase), 'All Angles, Total  (not Baseline)' );

end

%==========================================================================
% Propogate 2D and 3D Magnetization WITHOUT Diffusion
%==========================================================================
function [Signal, Time] = PropogateTransMagnetizationNoDiffusion( ...
    Geometry, PhysicalParams, Params, SimSettings )

GridSize    =   SimSettings.GridSize;
ScaleSum    =   SimSettings.Total_Volume / prod( GridSize );
TE          =   double( SimSettings.EchoTime );
dt          =   double( SimSettings.TimeStep );
m           =   numel( Geometry.alpha );
n           =   round( TE/dt );

Time0       =   0;
M0          =   double(1i);
Signal0     =    M0 * SimSettings.Total_Volume * ones(m,1);

if SimSettings.AllowClosedForms
    
    Time1    =   TE;
    
    if strcmpi( SimSettings.ScanType, 'SE' )
        Signal1    =   (conj(M0) .* ScaleSum) .* sum( exp( (-TE) .* Geometry.R2Map(:) ) ) * ones(m,1);
    else
        if m > 1
            M       =   1i .* reshape( Geometry.dOmegaMap, prod(GridSize), m );
            M       =   bsxfun( @plus, Geometry.R2Map(:), M ); %R2+i*dw
            M       =   exp( (-TE).*M );
            Signal1    =   sum(M,1);
            Signal1    =   (M0 .* ScaleSum) .* (Signal1.');
        else
            Signal1    =   (M0 .* ScaleSum) .* sum( exp( (-TE) .* complex( Geometry.R2Map(:), Geometry.dOmegaMap(:) ) ) );
        end
    end
    
else
    
    Time1    =   repmat(dt*(1:n),m,1);
    Signal1    =   zeros(m,n);
    
    for ii = 1:m
        M       =   M0 * ones( GridSize, 'double' );
        E       =   exp( (-SimSettings.TimeStep) * ...
            complex( Geometry.R2Map, sliceND( Geometry.dOmegaMap, ii, SimSettings.Dimension+1 ) ) );
        for jj = 1:n
            M               =    M .* E;
            Signal1(ii,jj)    =   ScaleSum * sum(M(:));
            
            if ( jj == round(n/2) ) && strcmpi( SimSettings.ScanType, 'SE' )
                M    =   conj(M);
            end
        end
    end
    
    clear M E
    
end

Time    =   [ Time0, Time1 ];
Signal    =   [ Signal0, Signal1 ];

end

%==========================================================================
% Propogate 3D Magnetization WITH Gaussian Diffusion: O(dt^2) Method
%==========================================================================
function [Signal, Time] = PropogateTransMagnetizationWithConvolutionDiffusion_Order2( ...
    Geometry, PhysicalParams, Params, SimSettings )

Dim         =   SimSettings.Dimension;
GridSize    =   SimSettings.GridSize;
ScaleSum    =   SimSettings.Total_Volume / prod( GridSize );
dt          =   double( SimSettings.TimeStep );
m           =   numel( Geometry.alpha );

TE          =   double( SimSettings.EchoTime(:).' );

addStartPoint = false;
if TE(1) == 0.0
    TE = TE(2:end);
    addStartPoint = true;
end
n           =   round( TE(end)/dt );
p           =   numel( TE );

GRE_Steps          =   round( TE/dt );
GRE_Steps          =   [GRE_Steps(1), diff(GRE_Steps)];
SE_SecondHalfSteps =   round( (TE/dt)/2 );
SE_FirstHalfSteps  =   [SE_SecondHalfSteps(1), diff(SE_SecondHalfSteps)];

Time0       =   0;
M0          =   double(1i);
Signal0     =   M0 * SimSettings.Total_Volume;
Signal      =   cell(p,1);

sig         =   sqrt( 2 * SimSettings.DiffusionCoeff * dt );
vox         =   SimSettings.SubVoxSize;
min_width   =   8; % min width of gaussian (in standard deviations)
width       =   ceil( min_width / (vox/sig) );
unitsum     =   @(x) x/sum(x(:));
Gaussian1   =   unitsum( exp( -0.5 * ( (-width:width).' * (vox/sig) ).^2 ) );
Gaussian2   =   Gaussian1(:).';
Gaussian3   =   reshape( Gaussian1, 1, 1, [] );
Gaussian2D  =   unitsum( Gaussian1 * Gaussian2 );
Gaussian3D  =   unitsum( bsxfun(@times, Gaussian2D, Gaussian3) );

% IntegrateMagnetization = @(M) ScaleSum * sum(M(:));
IntegrateMagnetization = @(M) ScaleSum * sum_pw(M); % ScaleSum converts from units of voxels^3 to um^3

persistent Kernel3D Gaussian3D_last
if isempty(Gaussian3D_last) || ~isequal(Gaussian3D_last,Gaussian3D)
    Gaussian3D_last =  Gaussian3D;
    Kernel3D        =  padfastfft( Gaussian3D, GridSize - size(Gaussian3D), true, 0 );
    Kernel3D        =  ifftshift( Kernel3D );
    Kernel3D        =  fftn( Kernel3D );
    Kernel3D        =  real( Kernel3D );
end

if n > 0
    
    if size(Geometry.ComplexDecayMap,Dim+1) > 1
        E  =  exp( (-dt) * sliceND(Geometry.ComplexDecayMap,ii,Dim+1) );
    else
        E  =  exp( (-dt) * Geometry.ComplexDecayMap );
    end
    
    Mcurr  =  M0 * ones( GridSize, 'double' );
    
    looptime  =  tic;
    
    for ll = 1:p
        Signal{ll} = [Time0, Signal0];
    end
    
    for kk = 1:p
        
        switch upper(SimSettings.ScanType)
            
            case 'GRE'
                
                for jj = 1:GRE_Steps(kk)
                    
                    % Exponential decay step
                    Mcurr  =  Mcurr .* E;
                    
                    % Diffusion step
                    Mcurr  =  ifftn(fftn(Mcurr).*Kernel3D); %splitting into seperate steps doesn't save time or speed
                    
                    for ll = kk:p
                        Signal{ll}  =  [Signal{ll}; [Signal{ll}(end,1)+dt, IntegrateMagnetization(Mcurr)]];
                    end
                    
                end
                
            case 'SE'
                
                for jj = 1:SE_FirstHalfSteps(kk)
                    
                    % Exponential decay step
                    Mcurr  =  Mcurr .* E;
                    
                    % Diffusion step
                    Mcurr  =  ifftn(fftn(Mcurr).*Kernel3D); %splitting into seperate steps doesn't save time or speed
                    
                    for ll = kk:p
                        Signal{ll}  =  [Signal{ll}; [Signal{ll}(end,1)+dt, IntegrateMagnetization(Mcurr)]];
                    end
                    
                end
                
                M  =  Mcurr;
                M  =  conj(M);
                
                for jj = 1:SE_SecondHalfSteps(kk)
                    
                    % Exponential decay step
                    M  =  M .* E;
                    
                    % Diffusion step
                    M  =  ifftn(fftn(M).*Kernel3D); %splitting into seperate steps doesn't save time or speed
                    
                    Signal{kk}  =  [Signal{kk}; [Signal{kk}(end,1)+dt, IntegrateMagnetization(M)]];
                    
                end
                
        end
        
        str     =   sprintf( 'alpha = %5.2f°, TE = %5.1fms, Baseline = %d', ...
            (180/pi)*Geometry.alpha, 1000*Signal{kk}(end,1), Geometry.isBaseline );
        display_toc_time(toc(looptime),str);
        
    end
    
    clear M MCurr E
    
else
    str     =   sprintf( 'alpha = %5.2f°, t = %4.1fms, Baseline = %d', ...
        (180/pi)*Geometry.alpha, 0.0, Geometry.isBaseline );
    display_toc_time(0.0,str);
end

if addStartPoint
    Signal = [Signal{1}(1,:); Signal]; %Add first timepoint back onto front
end

end

%==========================================================================
% Propogate 3D Magnetization WITH Gaussian Diffusion: O(dt^3) Method
%==========================================================================
function [Signal, Time] = PropogateTransMagnetizationWithConvolutionDiffusion_Order3( ...
    Geometry, PhysicalParams, Params, SimSettings )

Dim         =   SimSettings.Dimension;
GridSize    =   SimSettings.GridSize;
ScaleSum    =   SimSettings.Total_Volume / prod( GridSize );
dt          =   double( SimSettings.TimeStep );
m           =   numel( Geometry.alpha );

TE          =   double( SimSettings.EchoTime(:).' );

addStartPoint = false;
if TE(1) == 0.0
    TE = TE(2:end);
    addStartPoint = true;
end
n           =   round( TE(end)/dt );
p           =   numel( TE );

GRE_Steps          =   round( TE/dt );
GRE_Steps          =   [GRE_Steps(1), diff(GRE_Steps)];
SE_SecondHalfSteps =   round( (TE/dt)/2 );
SE_FirstHalfSteps  =   [SE_SecondHalfSteps(1), diff(SE_SecondHalfSteps)];

Time0       =   0;
M0          =   double(1i);
Signal0     =   M0 * SimSettings.Total_Volume;
Signal      =   cell(p,1);

sig         =   sqrt( 2 * SimSettings.DiffusionCoeff * dt );
vox         =   SimSettings.SubVoxSize;
min_width   =   8; % min width of gaussian (in standard deviations)
width       =   ceil( min_width / (vox/sig) );
unitsum     =   @(x) x/sum(x(:));
Gaussian1   =   unitsum( exp( -0.5 * ( (-width:width).' * (vox/sig) ).^2 ) );
Gaussian2   =   Gaussian1(:).';
Gaussian3   =   reshape( Gaussian1, 1, 1, [] );
Gaussian2D  =   unitsum( Gaussian1 * Gaussian2 );
Gaussian3D  =   unitsum( bsxfun(@times, Gaussian2D, Gaussian3) );

% IntegrateMagnetization = @(M) ScaleSum * sum(M(:));
IntegrateMagnetization = @(M) ScaleSum * sum_pw(M); % ScaleSum converts from units of voxels^3 to um^3

persistent Kernel3D Gaussian3D_last
if isempty(Gaussian3D_last) || ~isequal(Gaussian3D_last,Gaussian3D)
    Gaussian3D_last =  Gaussian3D;
    Kernel3D        =  padfastfft( Gaussian3D, GridSize - size(Gaussian3D), true, 0 );
    Kernel3D        =  ifftshift( Kernel3D );
    Kernel3D        =  fftn( Kernel3D );
    Kernel3D        =  real( Kernel3D );
end

if n > 0
    
    if size(Geometry.ComplexDecayMap,Dim+1) > 1
        E2  =  exp( (-dt/2) * sliceND(Geometry.ComplexDecayMap,ii,Dim+1) );
    else
        E2  =  exp( (-dt/2) * Geometry.ComplexDecayMap );
    end
    
    Mcurr  =  M0 * ones( GridSize, 'double' );
    
    looptime  =  tic;
    
    for ll = 1:p
        Signal{ll} = [Time0, Signal0];
    end
    
    for kk = 1:p
        
        switch upper(SimSettings.ScanType)
            
            case 'GRE'
                
                for jj = 1:GRE_Steps(kk)
                    
                    % First Exponential decay half-step
                    Mcurr  =  Mcurr .* E2;
                    
                    % Diffusion step
                    Mcurr  =  ifftn(fftn(Mcurr).*Kernel3D); %splitting into seperate steps doesn't save time or speed
                    
                    % Second Exponential decay half-step
                    Mcurr  =  Mcurr .* E2;
                    
                    for ll = kk:p
                        Signal{ll}  =  [Signal{ll}; [Signal{ll}(end,1)+dt, IntegrateMagnetization(Mcurr)]];
                    end
                    
                end
                
            case 'SE'
                
                for jj = 1:SE_FirstHalfSteps(kk)
                    
                    % First Exponential decay half-step
                    Mcurr  =  Mcurr .* E2;
                    
                    % Diffusion step
                    Mcurr  =  ifftn(fftn(Mcurr).*Kernel3D); %splitting into seperate steps doesn't save time or speed
                    
                    % Second Exponential decay half-step
                    Mcurr  =  Mcurr .* E2;
                    
                    for ll = kk:p
                        Signal{ll}  =  [Signal{ll}; [Signal{ll}(end,1)+dt, IntegrateMagnetization(Mcurr)]];
                    end
                    
                end
                
                M  =  Mcurr;
                M  =  conj(M);
                
                for jj = 1:SE_SecondHalfSteps(kk)
                    
                    % First Exponential decay half-step
                    M  =  M .* E2;
                    
                    % Diffusion step
                    M  =  ifftn(fftn(M).*Kernel3D); %splitting into seperate steps doesn't save time or speed
                    
                    % Second Exponential decay half-step
                    M  =  M .* E2;
                    
                    Signal{kk}  =  [Signal{kk}; [Signal{kk}(end,1)+dt, IntegrateMagnetization(M)]];
                    
                end
                
        end
        
        str     =   sprintf( 'alpha = %5.2f°, TE = %5.1fms, Baseline = %d', ...
            (180/pi)*Geometry.alpha, 1000*Signal{kk}(end,1), Geometry.isBaseline );
        display_toc_time(toc(looptime),str);
        
    end
    
    clear M MCurr E
    
else
    str     =   sprintf( 'alpha = %5.2f°, t = %4.1fms, Baseline = %d', ...
        (180/pi)*Geometry.alpha, 0.0, Geometry.isBaseline );
    display_toc_time(0.0,str);
end

if addStartPoint
    Signal = [Signal{1}(1,:); Signal]; %Add first timepoint back onto front
end

end

%==========================================================================
% Propogate 3D Magnetization WITH Diffusion
%==========================================================================
function [Signal, Time] = PropogateTransMagnetizationWithDiffusion_3D( ...
    Geometry, PhysicalParams, Params, SimSettings )

GridSize    =   SimSettings.GridSize;
VoxelSize   =   SimSettings.VoxelSize;
ScaleSum    =   SimSettings.Total_Volume / prod( GridSize );
TE          =   double( SimSettings.EchoTime );
m           =   numel( Geometry.alpha );

Time0       =   zeros(m,1);
M0          =   double(1i);
Signal0     =    (M0 .* SimSettings.Total_Volume) .* ones(m,1);

[xb,yb,zb]  =   deal( [0,VoxelSize(1)], [0,VoxelSize(2)], [0,VoxelSize(3)] );
[Nx,Ny,Nz]  =   deal( GridSize(1), GridSize(2), GridSize(3) );
D       =   SimSettings.DiffusionCoeff;
if m == 1
    f    =   complex( Geometry.R2Map, Geometry.dOmegaMap );
else
    f    =   bsxfun(@plus, Geometry.R2Map, 1i .* Geometry.dOmegaMap);
end

if SimSettings.AllowClosedForms
    
    % Solve pde analytically using a spectral method to expand M(x,y,z,t)
    % as a sum of complex exponentials (summing over discrete wx, wy, wz):
    %    M(x,y,z,t)    :=    sum( T(t) * exp( i*(wx*x+wy*y+wz*z) ) )
    
    if strcmpi( SimSettings.ScanType, 'SE' )
        % RF pulse to flip signal applied at TE/2
        tflip    =   TE/2;
        Time1    =    repmat( [tflip, TE], m, 1 );
    else
        % No flipping pulse applied
        tflip    =   [];
        Time1    =    repmat( TE, m, 1 );
    end
    
    Signal1 =    zeros(m,1);
    for ii = 1:m
        % Get approximate analytic solutions
        [ S, u, b, T0, A ]    =    heatLikeEqn3D( D, D, D, f, M0, inf, ...
            xb, yb, zb, Nx-1, Ny-1, Nz-1, 'trap', tflip, 'interior' );
        
        % Evaluate solution at time t = TE
        Signal1(ii)    =   S(Time1(ii));
    end
    
else
    
    % Solve pde directly by discreting space (method of lines) and solving
    % the resulting system of ode's
    
    if strcmpi( SimSettings.ScanType, 'SE' )
        
        Time1    =   repmat( [TE/2, TE], m, 1 );
        tol     =    1e-10;
        
        % Simulate decay until TE/2
        if SimSettings.AllowParallel
            M0      =   M0 .* ones(size(f));
            [~,M]    =    parallel_diffuse3D( D, f, M0, xb, yb, zb, Nx, Ny, Nz, TE/2, 'double', 'interior', tol );
        else
            %[~,M,J]    =    parabolicPeriodic3D( D, D, D, -f, M0, xb, yb, zb, Nx, Ny, Nz, TE/2, 'double', 'interior', [] );
            [~,M,~] =   diffuse3D(D,f,M0,xb,yb,zb,Nx,Ny,Nz,TE/2,'double','interior','expmv',tol);
        end
        
        M       =   reshape( M, prod(GridSize), m );
        Signal1    =    ScaleSum * (sum(M,1).');
        M       =   reshape( M, [GridSize, m] );
        
        % RF pulse flip by conjugating M, and simulate the last TE/2
        M       =   conj(M);
        if SimSettings.AllowParallel
            [~,M]    =    parallel_diffuse3D( D, f, M, xb, yb, zb, Nx, Ny, Nz, TE/2, 'double', 'interior', tol );
        else
            %[~,M,~]    =    parabolicPeriodic3D( D, D, D, -f, M,  xb, yb, zb, Nx, Ny, Nz, TE/2, 'double', 'interior', J  );
            [~,M,~] =   diffuse3D(D,f,M,xb,yb,zb,Nx,Ny,Nz,TE/2,'double','interior','expmv',tol);
        end
        
        M       =   reshape( M, prod(GridSize), m );
        Signal1    =    [Signal1, ScaleSum * (sum(M,1).')];
        
        clear M M0 J
        
    else
        
        Time1    =   repmat( TE, m, 1 );
        tol     =    1e-10;
        
        % Simulate decay until TE
        if SimSettings.AllowParallel
            M0      =   M0 .* ones(size(f));
            [~,M]    =    parallel_diffuse3D( D, f, M0, xb, yb, zb, Nx, Ny, Nz, TE, 'double', 'interior', tol );
        else
            [~,M,~] =   diffuse3D(D,f,M0,xb,yb,zb,Nx,Ny,Nz,TE,'double','interior','expmv',tol);
            %[~,M,~]=    parabolicPeriodic3D( D, D, D, -f, M0, xb, yb, zb, Nx, Ny, Nz, TE,   'double', 'interior', [] );
        end
        
        M       =   reshape( M, prod(GridSize), m );
        Signal1    =    ScaleSum * (sum(M,1).');
        
        clear M M0
        
    end
    
end

Signal    =   [ Signal0, Signal1 ];
Time    =   [ Time0, Time1 ];

end

function Geometry = get_dOmegaMap(   ...
    Geometry, PhysicalParams, Params, SimSettings )

% Extract parameters, checking if it is the baseline simulation or not
if Geometry.isBaseline
    R2_Blood =  Params.R2_Blood_Baseline;
    dChi     =  Params.deltaChi_Baseline;
else
    R2_Blood =  Params.R2_Blood_Total;
    dChi     =  Params.deltaChi_Total;
end

% R2 of WM tissue is the same in both cases
R2_Tissue = Params.R2_Tissue;

%==========================================================================
% Calculate R2(*) map
%==========================================================================
% R2Map = { R2_Blood,   inside VasculatureMap
%         { R2_Tissue,  outside VasculatureMap

% Compute R2 map, checking for smoothing
a        =    double( R2_Blood );
b        =    double( R2_Tissue );
R2Map    =    (a-b) * Geometry.VasculatureMap + b;

% Check for diffusionless Spin-echo simulation (don't need imaginary part in this case)
if SimSettings.AllowClosedForms && ~SimSettings.AddDiffusion && strcmpi( SimSettings.ScanType, 'SE' )
    Geometry.ComplexDecayMap  =  R2Map;
    return
end

%==========================================================================
% Calculate deltaOmega map
%==========================================================================
% The change in frequency map, dOmegaMap, is obtained through the
% convolution of the dipole kernel with the susceptibility map and 
% multiplication by the gyromagnetic ratio:
%
%   dOmega    =   gyro * B0 * conv( dChi, Dipole )
%           =   F^-1[ F[gyro * B0 * dChi] .* F[Dipole] ]

%--------------------------------------------------------------------------
% Step 1: compute the magnetic field angle-independent fourier transform of
%         gyro * B0 * dChi
%--------------------------------------------------------------------------

% fft of susceptibility
fftVascMap  =  double(Geometry.VasculatureMap);
fftVascMap  =  fftn( fftVascMap );

%--------------------------------------------------------------------------
% Step 2: compute the angle-dependent unit-dipole, the multiplication
%         in k-space, and the inverse fft
%--------------------------------------------------------------------------

% Create dipole kernel(s)
GridSize      =  SimSettings.GridSize;
alpha         =  double(Geometry.alpha(:)); % main B-field angle
BDir          =  [sin(alpha), zeros(size(alpha)), cos(alpha)];
if numel(alpha) > 1
    DipoleKernel  =  zeros( [GridSize, numel(alpha)], 'double' );
    for ii = 1:numel(alpha)
        DipoleKernel(:,:,:,ii)  =  dipole( GridSize, 1, BDir(ii,:), 'double' );
    end
    % Convolve with dipole kernel
    dOmegaMap     =  bsxfun( @times, DipoleKernel, fftVascMap ); %convolution
else
    % Convolve with dipole kernel
    DipoleKernel  =  dipole( GridSize, 1, BDir, 'double' );
    dOmegaMap     =  DipoleKernel .* fftVascMap; %convolution
end
clear DipoleKernel fftVascMap

% Scale result to proper values (overall constant was omitted above)
dOmega_scale  =  double( SimSettings.GyroMagRatio * SimSettings.B0 * dChi );
dOmegaMap     =  dOmega_scale * dOmegaMap;

% ifft to get delta omega map
if numel(alpha) > 1
    for ii = 1:numel(alpha)
        % Note: parfor loop is not faster here as ifftn is already optimized for multi-core processing
        dOmegaMap(:,:,:,ii)  =  ifftn( dOmegaMap(:,:,:,ii), 'symmetric' );
    end
else
    dOmegaMap  =  ifftn( dOmegaMap, 'symmetric' );
end

%--------------------------------------------------------------------------
% Return complex decay map
%--------------------------------------------------------------------------
Geometry.ComplexDecayMap  =  complex( R2Map, dOmegaMap );

end

function Geometry = get_dOmegaMap_analytic(   ...
    Geometry, PhysicalParams, Params, SimSettings )

%{
% Check for diffusionless Spin-echo simulation
if SimSettings.AllowClosedForms && ~SimSettings.AddDiffusion && strcmpi( SimSettings.ScanType, 'SE' )
    Geometry.fftGyroDChi    =    [];
    return
end
    %}
    
    % Check if contrast agent has been added
    if Geometry.isBaseline
        dChi    =   Params.deltaChi_Baseline;
    else
        dChi    =   Params.deltaChi_Total;
    end
    B0      =   SimSettings.B0;
    alpha   =   Geometry.alpha;
    BDir    =    [sin(alpha), zeros(size(alpha)), cos(alpha)];
    gyro    =   SimSettings.GyroMagRatio;
    
    % Decaying 1/r^2 induced dB outside of vasculature
    if SimSettings.Dimension == 2
        [x,y]   =   ndgrid( SimSettings.VoxelSize(1) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(1)), ...
            SimSettings.VoxelSize(2) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(2)) );
        [x,y]   =   deal( x + SimSettings.VoxelCenter(1), y + SimSettings.VoxelCenter(2) );
        z       =   SimSettings.VoxelCenter(3) * ones(size(x));
    else
        [x,y,z] =   ndgrid( SimSettings.VoxelSize(1) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(1)), ...
            SimSettings.VoxelSize(2) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(2)), ...
            SimSettings.VoxelSize(3) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(3)) );
        [x,y,z] =   deal( x + SimSettings.VoxelCenter(1), y + SimSettings.VoxelCenter(2), z + SimSettings.VoxelCenter(3) );
    end
    
    [M,m]    =   deal( Geometry.MainCylinders, Geometry.MinorCylinders );
    [p,vz,r,~,~,mx,n]    =   deal([M.p,m.p]',[M.vz,m.vz]',[M.r,m.r]',[M.vx,m.vx]',[M.vy,m.vy]',[M.mx,m.mx]',M.n+m.n);
    clear M m
    
    Points    =   [x(:),y(:),z(:)];
    clear x y z
    
    if SimSettings.Dimension == 3
        ix    =   (1:size(Points,1))';
        mx    =   cell(1,n);
    end
    
    Geometry.dOmegaMap    =   zeros(size(Points,1),1);
    
    % RotVecs    =   @(v,T)      unit(v*T,2);
    RotPoints    =   @(p,p0,T)   bsxfun(@minus,p,p0) * T;
    
    %{
width    =   ceil(4*sqrt(2)*max(SimSettings.GridSize));
width    =   width + ~mod(width,2);
mid     =   floor(width/2) + 1;
[X,Y]    =    meshgrid( SimSettings.VoxelSize(1) * linspacePeriodic(-2*sqrt(2),2*sqrt(2),width), ...
                      SimSettings.VoxelSize(2) * linspacePeriodic(-2*sqrt(2),2*sqrt(2),width) );
X       =    X(:);
Y       =    Y(:);
iR2     =    1./(X.^2+Y.^2);
Phi     =   atan2(Y,X);
Scale    =   mean( SimSettings.GridSize ./ SimSettings.VoxelSize );
    %}
    
    for ii = 1:n
        
        vcyl    =   unit(vz(ii,:)') * mysign(vz(ii,3));
        
        if isParallel3D(BDir(:),vcyl)
            [vx,vy,~]    =    nullVectors3D(vcyl);
        else
            vy          =    unit(cross(BDir(:),vcyl));
            vx          =   unit(cross(vy,vcyl));
        end
        
        T    =   nearestRotMat([vx, vy, vcyl]);
        
        %{
    disp(T); disp(p0); disp(any( isnan(p) | isinf(p), 1 )); disp([norm(T),isRotationMatrix(T),norm(p0)<3000*sqrt(2)])
        %}
        
        P    =   RotPoints(Points,p(ii,:),T);
        
        % Need to compute cos(2*phi)/r^2, phi := atan2(y,x), r^2 := x^2 + y^2
        %   -Use identity: cos(2*phi) = (x^2-y^2)/(x^2+y^2)
        %   -Now: cos(2*phi)/r^2 = (x^2-y^2)/(x^2+y^2)^2
        
        ir2cos2phi  =   (P(:,1).^2-P(:,2).^2)./(P(:,1).^2+P(:,2).^2).^2;
        
        %{
    ir2    =   1./( P(:,1).^2 + P(:,2).^2 );
    phi    =   atan2( P(:,2), P(:,1) );
        %}
        %{
    ix  =    round( Scale * P(:,1) + mid);
    iy  =    round( Scale * P(:,2) + mid);
    iv  =    sub2ind([width,width],ix,iy);
    ir2    =   iR2(iv);
    phi    =   Phi(iv);
        %}
        
        if SimSettings.Dimension == 3
            %mx{ii}    =   ix(ir2 >= 1/r(ii)^2);
            mx{ii}    =   ix(P(:,1).^2 + P(:,2).^2 <= r(ii)^2);
        end
        
        %ir2(mx{ii})=   0;
        %phi(mx{ii})=   0;
        ir2cos2phi(mx{ii})  =   0;
        
        angle       =   angle3D( vcyl, BDir(:) );
        
        Geometry.dOmegaMap    =   Geometry.dOmegaMap + ...
            ( gyro * B0 * dChi/2 * sin(angle)^2 * r(ii)^2 ) * ir2cos2phi;
        
        Geometry.dOmegaMap(mx{ii})    =   Geometry.dOmegaMap(mx{ii}) + ...
            ( gyro * B0 * dChi/6 * (3*cos(angle)^2 - 1) );
        
        %{
    fprintf('%d\n',ii);
    
    if r(ii)^2 * max(iR2) > 1
        keyboard
    end
    
    f = reshape(Geometry.dOmegaMap,SimSettings.GridSize);
    h = figure; imagesc(f,[-1,1]), axis image; %surf(f,'edgecolor','none'); axis image, view([0,0,1]);
    close(h);
            %}
    end
    
    Geometry.dOmegaMap    =   reshape( Geometry.dOmegaMap, SimSettings.GridSize );
    
end

%==========================================================================
% Calculate resulting dR2
%==========================================================================
% Calculate the dR2 values from the results of the simulation
%
function Results = get_dR2( Results, PhysicalParams, Params, SimSettings )

% Definition:
%   deltaR2    :=    -1/TE * log(|S|/|S0|)
dR2_func        =    @(S,S0) (-1/SimSettings.EchoTime) * log(abs(S)./abs(S0));
NumAngles       =    numel(Results.Angles_Rad);
Results.dR2_TE    =   zeros(1,NumAngles);

for ii = 1:NumAngles
    
    Results.dR2_all{ii}    =    dR2_func( ...
        Results.Signal_noBase{ii}, Results.Signal_Base{ii} );
    
    Results.dR2_TE(ii)    =   Results.dR2_all{ii}(end);
    
end

end

