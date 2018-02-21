function [ Results ] = PerfusionOrientation_PropTransMag( ...
        Geometry, PhysicalParams, Params, SimSettings )
%PERFUSIONORIENTATION_PROPTRANSMAG Propogates the transverse magnetization
%under the following time dependence:
% 
%   dM(r,t)/dt = div( D * grad(M(r,t)) ) - (R2(r) + i*dw(r)) * M(r,t)
% 
% Where:
%   -M(r,t) is the transverse magnetization
%   -D is the diffusion tensor (may be 0, scalar, 3-vector, or 3x3 tensor)
%   -R2 is the tissue-specific transverse relaxation (R2*) constant
%   -dw(r) is the Larmor frequency of the spins at position r

switch SimSettings.Dimension
    case 2
        Results	=	PropogateTransMagnetization_2D( ...
                        Geometry, PhysicalParams, Params, SimSettings );
    case 3
        Results	=	PropogateTransMagnetization_3D( ...
                        Geometry, PhysicalParams, Params, SimSettings );
end

%==========================================================================
% Save simulation parameters and settings
%==========================================================================
Results.Settings	=   SimSettings;
Results.Params      =	Params;

end

function Results = PropogateTransMagnetization_2D( ...
	Geometry, PhysicalParams, Params, SimSettings )

if SimSettings.AddDiffusion
    PropogationMethod	=   ...
        @PropogateTransMagnetizationWithConvolutionDiffusion;
else
    PropogationMethod	=   ...
        @PropogateTransMagnetizationNoDiffusion;
end

% Extract parameters
Angles_Rad          =   SimSettings.Angles_Rad_Data;
NumAngles           =   numel(Angles_Rad);

% Initialize Results
Results	=   struct( ...
    'Time',         [],                  	...
    'Angles_Rad',   Angles_Rad,             ...
    'Signal_noCA',	{cell(NumAngles,1)},	...
    'Signal_CA',    {cell(NumAngles,1)},	...
    'dR2_all',      {cell(NumAngles,1)},	...
    'dR2_TE',       zeros(NumAngles,1)      ...
    );

if SimSettings.AllowParallel
    Results	=	Parfor_PropogateTransMag_2D( PropogationMethod, Results, ...
        Geometry, PhysicalParams, Params, SimSettings );
else
    Results	=	Linear_PropogateTransMag_2D( PropogationMethod, Results, ...
        Geometry, PhysicalParams, Params, SimSettings );
end

%==========================================================================
% Calculate dR2
%==========================================================================
Results     =	get_dR2( Results, PhysicalParams, Params, SimSettings );

end

function Results = PropogateTransMagnetization_3D( ...
	Geometry, PhysicalParams, Params, SimSettings )

if SimSettings.AddDiffusion
    if SimSettings.UseConvDiffusion
        PropogationMethod   =   @PropogateTransMagnetizationWithConvolutionDiffusion;
    else
        PropogationMethod   =   @PropogateTransMagnetizationWithDiffusion_3D;
    end
else
    PropogationMethod   =   @PropogateTransMagnetizationNoDiffusion;
end

% Extract parameters
Angles_Rad          =   SimSettings.Angles_Rad_Data;
NumAngles           =   numel(Angles_Rad);

% Initialize Results
Results	=   struct( ...
    'Time',         [],                  	...
    'Angles_Rad',   Angles_Rad,             ...
    'Signal_noCA',	{cell(NumAngles,1)},	...
    'Signal_CA',    {cell(NumAngles,1)},	...
    'dR2_all',      {cell(NumAngles,1)},	...
    'dR2_TE',       zeros(NumAngles,1)      ...
    );

if SimSettings.AllowParallel
%     Results	=	Parallel_PropogateTransMag_3D( PropogationMethod, Results, ...
%         Geometry, PhysicalParams, Params, SimSettings );
    Results	=	Parfor_PropogateTransMag_3D( PropogationMethod, Results, ...
        Geometry, PhysicalParams, Params, SimSettings );
else
    Results	=	Linear_PropogateTransMag_3D( PropogationMethod, Results, ...
        Geometry, PhysicalParams, Params, SimSettings );
end

%==========================================================================
% Calculate dR2
%==========================================================================
Results     =	get_dR2( Results, PhysicalParams, Params, SimSettings );

end

function Results = Linear_PropogateTransMag_2D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Angle data
Angles_Rad	=   SimSettings.Angles_Rad_Data;
NumAngles	=   numel(Angles_Rad);

% Geometry Without CA
Geometry.isContrastAgentAdded	=	false;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );

for ii = 1:NumAngles
    %======================================================================
    % Simulation: Without Contrast Agent
    %======================================================================
    
    t_noCA	=	tic;
    
    Geometry.alpha 	=	Angles_Rad(ii);
    Geometry       	=	...
        get_dOmegaMap_analytic( Geometry, PhysicalParams, Params, SimSettings );
    [ Results.Signal_noCA{ii}, Results.Time ]	=   ...
        PropMethod( Geometry, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_noCA), sprintf( 'Angle %2d/%2d, %5.2f%s, no CA', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
end

% Geometry With CA
Geometry.isContrastAgentAdded	=	true;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );

for ii = 1:NumAngles
    %======================================================================
    % Simulation: With Contrast Agent
    %======================================================================
    
    t_CA	=	tic;
    
    Geometry.alpha	=	Angles_Rad(ii);
    Geometry       	=   ...
        get_dOmegaMap_analytic( Geometry, PhysicalParams, Params, SimSettings );
    [ Results.Signal_CA{ii}, Results.Time ]     =	...
        PropMethod( Geometry, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_CA), sprintf( 'Angle %2d/%2d, %5.2f%s, with CA', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
end

end

function Results = Parfor_PropogateTransMag_2D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Angle data
Angles_Rad	=   SimSettings.Angles_Rad_Data;
NumAngles	=   numel(Angles_Rad);

% Geometry Without CA
Geometry.isContrastAgentAdded	=	false;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );

Signal_noCA	=   cell(NumAngles,1);
Time        =   cell(NumAngles,1);
NumWorkers	=   min(NumAngles,6);
parfor (ii = 1:NumAngles, NumWorkers)
    %======================================================================
    % Simulation: Without Contrast Agent
    %======================================================================
    t_noCA	=	tic;
    
    LoopGeo         =   Geometry
    LoopGeo.alpha 	=	Angles_Rad(ii);
    LoopGeo       	=	...
        get_dOmegaMap_analytic( LoopGeo, PhysicalParams, Params, SimSettings );
    [ Signal_noCA{ii}, Time{ii} ]	=   ...
        PropMethod( LoopGeo, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_noCA), sprintf( 'Angle %2d/%2d, %5.2f%s, no CA', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
end
Results.Signal_noCA	=	Signal_noCA;
Results.Time      	=   Time{1};

% Geometry With CA
Geometry.isContrastAgentAdded	=	true;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );

Signal_CA	=   cell(NumAngles,1);
parfor (ii = 1:NumAngles, NumWorkers)
    %======================================================================
    % Simulation: With Contrast Agent
    %======================================================================
    t_CA	=	tic;
    
    LoopGeo         =   Geometry;
    LoopGeo.alpha 	=	Angles_Rad(ii);
    LoopGeo       	=	...
        get_dOmegaMap_analytic( LoopGeo, PhysicalParams, Params, SimSettings );
    [ Signal_CA{ii}, ~ ]     =	...
        PropMethod( LoopGeo, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_CA), sprintf( 'Angle %2d/%2d, %5.2f%s, with CA', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
end
Results.Signal_CA	=	Signal_CA;

end

function Results = Linear_PropogateTransMag_3D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Angle data
Angles_Rad	=   SimSettings.Angles_Rad_Data;
NumAngles	=   numel(Angles_Rad);

% Geometry Without CA
Geometry.isContrastAgentAdded	=	false;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry	=	get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );

for ii = 1:NumAngles
    
    %======================================================================
    % Simulation: Without Contrast Agent
    %======================================================================
    
    t_noCA	=	tic;
    
    Geometry.alpha 	=	Angles_Rad(ii);
    Geometry       	=	...
        get_dOmegaMap_step2( Geometry, PhysicalParams, Params, SimSettings );
    [ Results.Signal_noCA{ii}, Results.Time ]	=   ...
        PropMethod( Geometry, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_noCA), sprintf( 'Angle %2d/%2d, %5.2f%s, no CA', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
    
end

% Geometry With CA
Geometry.isContrastAgentAdded	=	true;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry	=	get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );

for ii = 1:NumAngles
    
    %======================================================================
    % Simulation: With Contrast Agent
    %======================================================================
    
    t_CA	=	tic;
    
    Geometry.alpha	=	Angles_Rad(ii);
    Geometry       	=   ...
        get_dOmegaMap_step2( Geometry, PhysicalParams, Params, SimSettings );
    [ Results.Signal_CA{ii}, Results.Time ]     =	...
        PropMethod( Geometry, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_CA), sprintf( 'Angle %2d/%2d, %5.2f%s, with CA', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
    
end

end

function Results = Parfor_PropogateTransMag_3D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Angle data
Angles_Rad	=   SimSettings.Angles_Rad_Data;
NumAngles	=   numel(Angles_Rad);

% Geometry Without CA
Geometry.isContrastAgentAdded	=	false;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry	=	get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );

Signal_noCA	=   cell(NumAngles,1);
Time        =   cell(NumAngles,1);
parfor (ii = 1:NumAngles, NumAngles)
    %======================================================================
    % Simulation: Without Contrast Agent
    %======================================================================
    t_noCA	=	tic;
    
    LoopGeo         =   Geometry
    LoopGeo.alpha 	=	Angles_Rad(ii);
    LoopGeo       	=	...
        get_dOmegaMap_step2( LoopGeo, PhysicalParams, Params, SimSettings );
    [ Signal_noCA{ii}, Time{ii} ]	=   ...
        PropMethod( LoopGeo, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_noCA), sprintf( 'Angle %2d/%2d, %5.2f%s, no CA', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );

end
Results.Signal_noCA	=	Signal_noCA;
Results.Time      	=   Time{1};

% Geometry With CA
Geometry.isContrastAgentAdded	=	true;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry	=	get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );

Signal_CA	=   cell(NumAngles,1);
parfor (ii = 1:NumAngles, NumAngles)
    %======================================================================
    % Simulation: With Contrast Agent
    %======================================================================
    t_CA	=	tic;
    
    LoopGeo         =   Geometry
    LoopGeo.alpha	=	Angles_Rad(ii);
    LoopGeo       	=   ...
        get_dOmegaMap_step2( LoopGeo, PhysicalParams, Params, SimSettings );
    [ Signal_CA{ii}, ~ ]	=	...
        PropMethod( LoopGeo, PhysicalParams, Params, SimSettings );
    
    display_toc_time( toc(t_CA), sprintf( 'Angle %2d/%2d, %5.2f%s, with CA', ...
        ii, NumAngles, SimSettings.Angles_Deg_Data(ii), '°' ) );
    
end
Results.Signal_CA	=	Signal_CA;

end

function Results = Parallel_PropogateTransMag_3D( PropMethod, Results, Geometry, PhysicalParams, Params, SimSettings )

% Setup
toRowCell       =   @(x) mat2cell(x,ones(size(x,1),1),size(x,2));
Angles_Rad      =   SimSettings.Angles_Rad_Data;
Geometry.alpha	=	Angles_Rad; %all angles will be computed in parallel

%======================================================================
% Simulation: Without Contrast Agent
%======================================================================

t_noCA      =   tic;

% Step 1 (no CA)
t_step1     =	tic;
Geometry.isContrastAgentAdded	=	false;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry	=	get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_step1), 'All Angles, Step 1 (no CA)' );

% Step 2 (no CA)
t_step2     =   tic;
Geometry    =	get_dOmegaMap_step2( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_step2), 'All Angles, Step 2 (no CA)' );

% Propogation (no CA)
t_prop      =   tic;
[ Signal_noCA, Time ]	=   PropMethod( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_prop), 'All Angles, Prop   (no CA)' );

Results.Signal_noCA	=   toRowCell(Signal_noCA);
Results.Time        =   Time(1,:);

display_toc_time( toc(t_noCA), 'All Angles, Total  (no CA)' );

%======================================================================
% Simulation: With Contrast Agent
%======================================================================

t_CA        =   tic;

% Step 1 (w/ CA)
t_step1     =	tic;
Geometry.isContrastAgentAdded	=	true;
Geometry	=	get_R2Map( Geometry, PhysicalParams, Params, SimSettings );
Geometry	=	get_dOmegaMap_step1( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_step1), 'All Angles, Step 1 (w/ CA)' );

% Step 2 (w/ CA)
t_step2     =   tic;
Geometry	=   get_dOmegaMap_step2( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_step2), 'All Angles, Step 2 (w/ CA)' );

% Propogation (w/ CA)
t_prop      =   tic;
[ Signal_CA, ~ ]	=	PropMethod( Geometry, PhysicalParams, Params, SimSettings );
display_toc_time( toc(t_prop), 'All Angles, Prop   (w/ CA)' );

Results.Signal_CA	=   toRowCell(Signal_CA);

display_toc_time( toc(t_CA), 'All Angles, Total  (w/ CA)' );

end

%==========================================================================
% Propogate 2D and 3D Magnetization WITHOUT Diffusion
%==========================================================================
function [Signal, Time] = PropogateTransMagnetizationNoDiffusion( ...
	Geometry, PhysicalParams, Params, SimSettings )

GridSize	=   SimSettings.GridSize;
ScaleSum	=   SimSettings.Total_Volume / prod( GridSize );
TE          =   double( SimSettings.EchoTime );
dt          =   double( SimSettings.TimeStep );
m           =   numel( Geometry.alpha );
n           =   round( TE/dt );

Time0       =   0;
M0          =   double(1i);
Signal0     =	M0 * SimSettings.Total_Volume * ones(m,1);

if SimSettings.AllowClosedForms
    
    Time1	=   TE;
    
    if strcmpi( SimSettings.ScanType, 'SE' )
        Signal1	=   (conj(M0) .* ScaleSum) .* sum( exp( (-TE) .* Geometry.R2Map(:) ) ) * ones(m,1);
    else
        if m > 1
            M       =   1i .* reshape( Geometry.dOmegaMap, prod(GridSize), m );
            M       =   bsxfun( @plus, Geometry.R2Map(:), M ); %R2+i*dw
            M       =   exp( (-TE).*M );
            Signal1	=   sum(M,1);
            Signal1	=   (M0 .* ScaleSum) .* (Signal1.');
        else
            Signal1	=   (M0 .* ScaleSum) .* sum( exp( (-TE) .* complex( Geometry.R2Map(:), Geometry.dOmegaMap(:) ) ) );
        end
    end
    
else
    
    Time1	=   repmat(dt*(1:n),m,1);
    Signal1	=   zeros(m,n);
    
    for ii = 1:m
        M       =   M0 * ones( GridSize, 'double' );
        E       =   exp( (-SimSettings.TimeStep) * ...
            complex( Geometry.R2Map, sliceND( Geometry.dOmegaMap, ii, SimSettings.Dimension+1 ) ) );
        for jj = 1:n
            M               =	M .* E;
            Signal1(ii,jj)	=   ScaleSum * sum(M(:));
            
            if ( jj == round(n/2) ) && strcmpi( SimSettings.ScanType, 'SE' )
                M	=   conj(M);
            end
        end
    end
    
    clear M E
    
end

Time	=   [ Time0, Time1 ];
Signal	=   [ Signal0, Signal1 ];

end

%==========================================================================
% Propogate 2D or 3D Magnetization WITH Diffusion via Gaussian Convolution
%==========================================================================
function [Signal, Time] = PropogateTransMagnetizationWithConvolutionDiffusion( ...
	Geometry, PhysicalParams, Params, SimSettings )

Dim         =   SimSettings.Dimension;
GridSize	=   SimSettings.GridSize;
ScaleSum	=   SimSettings.Total_Volume / prod( GridSize );
TE          =   double( SimSettings.EchoTime );
dt          =   double( SimSettings.TimeStep );
m           =   numel( Geometry.alpha );
n           =   round( TE/dt );

Time0       =   0;
M0          =   double(1i);
Signal0     =	M0 * SimSettings.Total_Volume * ones(m,1);

Time1       =   repmat(dt*(1:n),m,1);
Signal1     =   zeros(m,n);

sig         =   sqrt( 2 * PhysicalParams.D * dt );
vox         =   SimSettings.SubVoxSize;
min_width	=   8; % min width of gaussian (in standard deviations)
width       =	ceil( min_width / (vox/sig) );
unitsum     =   @(x) x/sum(x(:));
Gaussian1	=	unitsum( exp( -0.5 * ( (-width:width).' * (vox/sig) ).^2 ) );
Gaussian2	=   Gaussian1(:).';
Gaussian3	=   reshape( Gaussian1, 1, 1, [] );
Gaussian2D	=   unitsum( Gaussian1 * Gaussian2 );
Gaussian3D	=   unitsum( bsxfun(@times, Gaussian2D, Gaussian3) );

persistent Kernel3D Gaussian3D_last

if Dim == 3
%     Kernel3D	=	padfastfft( Gaussian3D, GridSize - size(Gaussian3D), true, 0 );
%     Kernel3D	=	fftn( ifftshift( Kernel3D ) );
    if isempty(Gaussian3D_last) || ~isequal(Gaussian3D_last,Gaussian3D)
        Gaussian3D_last	=   Gaussian3D;
        Kernel3D        =	padfastfft( Gaussian3D, GridSize - size(Gaussian3D), true, 0 );
        Kernel3D        =	fftn( ifftshift( Kernel3D ) );
    end
end

for ii = 1:m
    M       =   M0 * ones( GridSize, 'double' );
    E       =   exp( (-dt) * complex( Geometry.R2Map, sliceND(Geometry.dOmegaMap,ii,Dim+1) ) );
    for jj = 1:n
        looptime	=   tic;
        M           =	M .* E;
        if Dim == 2
            M	=   imfilter(M,Gaussian2D,'circular','same');
            %M	=   imfilter(M,Gaussian1,'circular','same');
            %M	=   imfilter(M,Gaussian2,'circular','same');
        else
            %M	=   conv_even_per(M,Gaussian1,1);
            %M	=   conv_even_per(M,Gaussian1,2);
            %M	=   conv_even_per(M,Gaussian1,3);
            M	=	ifftn(fftn(M).*Kernel3D);
        end
        
        Signal1(ii,jj)	=   ScaleSum * sum(M(:));
        
        if ( jj == round(n/2) ) && strcmpi( SimSettings.ScanType, 'SE' )
            M	=   conj(M);
        end
        
        str     =   sprintf( 'alpha = %5.2f°, t = %4.1fms, CA = %d', ...
            (180/pi)*Geometry.alpha, 1000*dt*jj, Geometry.isContrastAgentAdded );
        display_toc_time(toc(looptime),str);
    end
end

clear M E

Time	=   [ Time0, Time1 ];
Signal	=   [ Signal0, Signal1 ];

% Simple testing for convolution speed:
%   -imfilter(...,'periodic') seems to be ~twice as fast as circular
%    padding/convolving/unpadding and using the faster convn(...,'same')
%{
x = randnc(1024,1024); n = 100;

t0 = tic;
for ii = 1:n
% y1 = imfilter(x,Gaussian1,'circular','same');
% y1 = imfilter(y1,Gaussian2,'circular','same');
y1 = imfilter(x,Gaussian1*Gaussian2,'circular','same');
end
t1 = toc(t0)/n; display_toc_time(t1,'imfilter');

t0 = tic;
% % x0 = padfastfft(x,2*width*[1,1],0);
% % for ii = 1:n
% % y2 = x0;
% % y2(:,1:width) = x0(:,end-2*width+1:end-width);
% % y2(:,end-width+1:end) = x0(:,width+(1:width));
% % y2(1:width,:) = x0(end-2*width+1:end-width,:);
% % y2(end-width+1:end,:) = x0(width+(1:width),:);
% % y2 = convn(y2,Gaussian1,'same');
% % end
% % y2 = unpadfastfft(y2,size(x));
for ii = 1:n
y2 = padfastfft(x,2*width*[1,1],true,'circular');
y2 = convn(y2,Gaussian1,'same');
y2 = convn(y2,Gaussian2,'same');
y2 = unpadfastfft(y2,size(x));
end
t2 = toc(t0)/n; display_toc_time(t2,'convn   ');

fprintf('err:  %0.16f\nmean: %0.16f\nstd:  %0.16f\n\n',maxabs(y2-y1), ...
mean(abs(y2(:)-y1(:))),std(abs(y2(:)-y1(:))));
%}

end

%==========================================================================
% Propogate 3D Magnetization WITH Diffusion
%==========================================================================
function [Signal, Time] = PropogateTransMagnetizationWithDiffusion_3D( ...
	Geometry, PhysicalParams, Params, SimSettings )

GridSize	=   SimSettings.GridSize;
VoxelSize   =   SimSettings.VoxelSize;
ScaleSum	=   SimSettings.Total_Volume / prod( GridSize );
TE          =   double( SimSettings.EchoTime );
m           =   numel( Geometry.alpha );

Time0       =   zeros(m,1);
M0          =   double(1i);
Signal0     =	(M0 .* SimSettings.Total_Volume) .* ones(m,1);

[xb,yb,zb]  =   deal( [0,VoxelSize(1)], [0,VoxelSize(2)], [0,VoxelSize(3)] );
[Nx,Ny,Nz]  =   deal( GridSize(1), GridSize(2), GridSize(3) );
D       =   PhysicalParams.D;
if m == 1
    f	=   complex( Geometry.R2Map, Geometry.dOmegaMap );
else
    f	=   bsxfun(@plus, Geometry.R2Map, 1i .* Geometry.dOmegaMap);
end

if SimSettings.AllowClosedForms
    
    % Solve pde analytically using a spectral method to expand M(x,y,z,t)
    % as a sum of complex exponentials (summing over discrete wx, wy, wz):
    %	M(x,y,z,t)	:=	sum( T(t) * exp( i*(wx*x+wy*y+wz*z) ) )
    
    if strcmpi( SimSettings.ScanType, 'SE' )
        % RF pulse to flip signal applied at TE/2
        tflip	=   TE/2;
        Time1	=	repmat( [tflip, TE], m, 1 );
    else
        % No flipping pulse applied
        tflip	=   [];
        Time1	=	repmat( TE, m, 1 );
    end
    
    Signal1 =	zeros(m,1);
    for ii = 1:m
        % Get approximate analytic solutions
        [ S, u, b, T0, A ]	=	heatLikeEqn3D( D, D, D, f, M0, inf, ...
            xb, yb, zb, Nx-1, Ny-1, Nz-1, 'trap', tflip, 'interior' );
        
        % Evaluate solution at time t = TE
        Signal1(ii)	=   S(Time1(ii));
    end
    
else
    
	% Solve pde directly by discreting space (method of lines) and solving
	% the resulting system of ode's
    
    if strcmpi( SimSettings.ScanType, 'SE' )
        
        Time1	=   repmat( [TE/2, TE], m, 1 );
        tol     =	1e-10;
        
        % Simulate decay until TE/2
        if SimSettings.AllowParallel
            M0      =   M0 .* ones(size(f));
            [~,M]	=	parallel_diffuse3D( D, f, M0, xb, yb, zb, Nx, Ny, Nz, TE/2, 'double', 'interior', tol );
        else
            %[~,M,J]	=	parabolicPeriodic3D( D, D, D, -f, M0, xb, yb, zb, Nx, Ny, Nz, TE/2, 'double', 'interior', [] );
            [~,M,~] =   diffuse3D(D,f,M0,xb,yb,zb,Nx,Ny,Nz,TE/2,'double','interior','expmv',tol);
        end
        
        M       =   reshape( M, prod(GridSize), m );
        Signal1	=	ScaleSum * (sum(M,1).');
        M       =   reshape( M, [GridSize, m] );
        
        % RF pulse flip by conjugating M, and simulate the last TE/2
        M       =   conj(M);
        if SimSettings.AllowParallel
            [~,M]	=	parallel_diffuse3D( D, f, M, xb, yb, zb, Nx, Ny, Nz, TE/2, 'double', 'interior', tol );
        else
            %[~,M,~]	=	parabolicPeriodic3D( D, D, D, -f, M,  xb, yb, zb, Nx, Ny, Nz, TE/2, 'double', 'interior', J  );
            [~,M,~] =   diffuse3D(D,f,M,xb,yb,zb,Nx,Ny,Nz,TE/2,'double','interior','expmv',tol);
        end
        
        M       =   reshape( M, prod(GridSize), m );
        Signal1	=	[Signal1, ScaleSum * (sum(M,1).')];
        
        clear M M0 J
        
    else
        
        Time1	=   repmat( TE, m, 1 );
        tol     =	1e-10;
        
        % Simulate decay until TE
        if SimSettings.AllowParallel
            M0      =   M0 .* ones(size(f));
            [~,M]	=	parallel_diffuse3D( D, f, M0, xb, yb, zb, Nx, Ny, Nz, TE, 'double', 'interior', tol );
        else
            [~,M,~] =   diffuse3D(D,f,M0,xb,yb,zb,Nx,Ny,Nz,TE,'double','interior','expmv',tol);
            %[~,M,~]=	parabolicPeriodic3D( D, D, D, -f, M0, xb, yb, zb, Nx, Ny, Nz, TE,   'double', 'interior', [] );
        end
        
        M       =   reshape( M, prod(GridSize), m );
        Signal1	=	ScaleSum * (sum(M,1).');
        
        clear M M0
        
    end
    
end

Signal	=   [ Signal0, Signal1 ];
Time	=   [ Time0, Time1 ];

end

%==========================================================================
% Calculate R2(*) map
%==========================================================================
% R2Map	=	{ R2_Blood,   inside VasculatureMap
%           { R2_Tissue,  outside VasculatureMap
function Geometry = get_R2Map(   ...
    Geometry, PhysicalParams, Params, SimSettings )

% Check if constrast agent is added or not
if Geometry.isContrastAgentAdded
    R2_Blood_Total	=   Params.R2_Blood_Total;
else
    R2_Blood_Total	=   PhysicalParams.R2_Blood;
end

% Compute R2 map, checking for smoothing
a               =	double( R2_Blood_Total );
b               =	double( PhysicalParams.R2_Tissue );
if SimSettings.Smoothing > 0.0
    Geometry.R2Map	=	make_Smooth_Map( a, b, ...
        Geometry, PhysicalParams, Params, SimSettings );
else
    Geometry.R2Map	=	(a-b) * Geometry.VasculatureMap + b;
end

end

%==========================================================================
% Calculate deltaOmega map
%==========================================================================
% The change in frequency map, dOmegaMap, is obtained through the convolution
% of the dipole kernel with the susceptibility map and multiplication by
% the gyromagnetic ratio:
% 
%   dOmega	=   gamma * B0 * conv( dChi, Dipole )
%           =   F^-1[ F[gamma * B0 * dChi] .* F[Dipole] ]
% 
% Step 1 is to compute the magnetic field angle-independent fourier 
% transform of gamma * B0 * dChi
% 
% Step 2 is to compute the angle-dependent unit-dipole, the multiplication
% in k-space, and the inverse fft
% 
function Geometry = get_dOmegaMap_step1(   ...
    Geometry, PhysicalParams, Params, SimSettings )

% Check for diffusionless Spin-echo simulation
if SimSettings.AllowClosedForms && ~SimSettings.AddDiffusion && strcmpi( SimSettings.ScanType, 'SE' )
    Geometry.fftGammaDChi	=	[];
    return
end

% Check if contrast agent has been added
if Geometry.isContrastAgentAdded
    dChi	=   Params.deltaChi_Total;
else
    dChi	=   PhysicalParams.dChi_Blood;
end

% Smoothing delta Chi map
Gamma_dChi	=   double( PhysicalParams.gamma * PhysicalParams.B0 * dChi );
if SimSettings.Smoothing > 0.0
    Geometry.fftGammaDChi	=	make_Smooth_Map( Gamma_dChi, 0.0, ...
        Geometry, PhysicalParams, Params, SimSettings );
else
    Geometry.fftGammaDChi	=	Gamma_dChi * Geometry.VasculatureMap;
end

% fft of susceptibility * gamma
Geometry.fftGammaDChi	=	fftn( Geometry.fftGammaDChi );

end

function Geometry = get_dOmegaMap_step2(   ...
    Geometry, PhysicalParams, Params, SimSettings )

% Check for diffusionless Spin-echo simulation
if SimSettings.AllowClosedForms && ~SimSettings.AddDiffusion && strcmpi( SimSettings.ScanType, 'SE' )
    Geometry.dOmegaMap	=	[];
    return
end

% create dipole kernel(s)
GridSize        =	SimSettings.GridSize;
alpha           =	double(Geometry.alpha(:)); % main B-field angle
BDir            =	[sin(alpha), zeros(size(alpha)), cos(alpha)];
DipoleKernel	=   zeros( [GridSize, numel(alpha)], 'double' );
for ii = 1:numel(alpha)
    DipoleKernel(:,:,:,ii)	=   dipole( GridSize, 1, BDir(ii,:), 'double' );
end

% convolution with dipole kernel
Geometry.dOmegaMap	=	bsxfun( @times, DipoleKernel, Geometry.fftGammaDChi ); %convolution
clear DipoleKernel

% ifft to get delta omega map
for ii = 1:numel(alpha)
    % Note: parfor loop is not faster here as ifftn is already optimized for multi-core processing
    Geometry.dOmegaMap(:,:,:,ii)	=   ifftn( Geometry.dOmegaMap(:,:,:,ii), 'symmetric' );
end

end

function Geometry = get_dOmegaMap_analytic(   ...
    Geometry, PhysicalParams, Params, SimSettings )

%{
% Check for diffusionless Spin-echo simulation
if SimSettings.AllowClosedForms && ~SimSettings.AddDiffusion && strcmpi( SimSettings.ScanType, 'SE' )
    Geometry.fftGammaDChi	=	[];
    return
end
%}

% Check if contrast agent has been added
if Geometry.isContrastAgentAdded
    dChi	=   Params.deltaChi_Total;
else
    dChi	=   PhysicalParams.dChi_Blood;
end
B0      =   PhysicalParams.B0;
alpha	=   Geometry.alpha;
BDir	=	[sin(alpha), zeros(size(alpha)), cos(alpha)];
gamma	=   PhysicalParams.gamma;

% Decaying 1/r^2 induced dB outside of vasculature
if SimSettings.Dimension == 2
    [x,y]	=   ndgrid( SimSettings.VoxelSize(1) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(1)), ...
                        SimSettings.VoxelSize(2) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(2)) );
    [x,y]	=   deal( x + SimSettings.VoxelCenter(1), y + SimSettings.VoxelCenter(2) );
    z       =   SimSettings.VoxelCenter(3) * ones(size(x));
else
    [x,y,z] =   ndgrid( SimSettings.VoxelSize(1) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(1)), ...
                        SimSettings.VoxelSize(2) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(2)), ...
                        SimSettings.VoxelSize(3) * linspacePeriodic(-0.5,0.5,SimSettings.GridSize(3)) );
    [x,y,z] =   deal( x + SimSettings.VoxelCenter(1), y + SimSettings.VoxelCenter(2), z + SimSettings.VoxelCenter(3) );
end

[M,m]	=   deal( Geometry.MainCylinders, Geometry.MinorCylinders );
[p,vz,r,~,~,mx,n]	=   deal([M.p,m.p]',[M.vz,m.vz]',[M.r,m.r]',[M.vx,m.vx]',[M.vy,m.vy]',[M.mx,m.mx]',M.n+m.n);
clear M m

Points	=   [x(:),y(:),z(:)];
clear x y z

if SimSettings.Dimension == 3
    ix	=   (1:size(Points,1))';
    mx	=   cell(1,n);
end

Geometry.dOmegaMap	=   zeros(size(Points,1),1);

% RotVecs	=   @(v,T)      unit(v*T,2);
RotPoints	=   @(p,p0,T)   bsxfun(@minus,p,p0) * T;

%{
width	=   ceil(4*sqrt(2)*max(SimSettings.GridSize));
width	=   width + ~mod(width,2);
mid     =   floor(width/2) + 1;
[X,Y]	=	meshgrid( SimSettings.VoxelSize(1) * linspacePeriodic(-2*sqrt(2),2*sqrt(2),width), ...
                      SimSettings.VoxelSize(2) * linspacePeriodic(-2*sqrt(2),2*sqrt(2),width) );
X       =	X(:);
Y       =	Y(:);
iR2     =	1./(X.^2+Y.^2);
Phi     =   atan2(Y,X);
Scale	=   mean( SimSettings.GridSize ./ SimSettings.VoxelSize );
%}

for ii = 1:n
    
    vcyl	=   unit(vz(ii,:)') * mysign(vz(ii,3));
    
    if isParallel3D(BDir(:),vcyl)
        [vx,vy,~]	=	nullVectors3D(vcyl);
    else
        vy          =	unit(cross(BDir(:),vcyl));
        vx          =   unit(cross(vy,vcyl));
    end
    
    T	=   nearestRotMat([vx, vy, vcyl]);
    
    %{
    disp(T); disp(p0); disp(any( isnan(p) | isinf(p), 1 )); disp([norm(T),isRotationMatrix(T),norm(p0)<3000*sqrt(2)])
    %}
    
    P	=   RotPoints(Points,p(ii,:),T);
    
    % Need to compute cos(2*phi)/r^2, phi := atan2(y,x), r^2 := x^2 + y^2
    %   -Use identity: cos(2*phi) = (x^2-y^2)/(x^2+y^2)
    %   -Now: cos(2*phi)/r^2 = (x^2-y^2)/(x^2+y^2)^2
    
    ir2cos2phi  =   (P(:,1).^2-P(:,2).^2)./(P(:,1).^2+P(:,2).^2).^2;
    
    %{
    ir2	=   1./( P(:,1).^2 + P(:,2).^2 );
    phi	=   atan2( P(:,2), P(:,1) );
	%}
    %{
    ix  =	round( Scale * P(:,1) + mid);
    iy  =	round( Scale * P(:,2) + mid);
    iv  =	sub2ind([width,width],ix,iy);
    ir2	=   iR2(iv);
    phi	=   Phi(iv);
    %}
    
    if SimSettings.Dimension == 3
        %mx{ii}	=   ix(ir2 >= 1/r(ii)^2);
        mx{ii}	=   ix(P(:,1).^2 + P(:,2).^2 <= r(ii)^2);
    end
    
    %ir2(mx{ii})=   0;
    %phi(mx{ii})=   0;
    ir2cos2phi(mx{ii})  =   0;
    
    angle       =   angle3D( vcyl, BDir(:) );
    
    Geometry.dOmegaMap	=   Geometry.dOmegaMap + ...
        ( gamma * B0 * dChi/2 * sin(angle)^2 * r(ii)^2 ) * ir2cos2phi;
    
    Geometry.dOmegaMap(mx{ii})	=   Geometry.dOmegaMap(mx{ii}) + ...
        ( gamma * B0 * dChi/6 * (3*cos(angle)^2 - 1) );
    
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

Geometry.dOmegaMap	=   reshape( Geometry.dOmegaMap, SimSettings.GridSize );

end

%==========================================================================
% Calculate resulting dR2
%==========================================================================
% Calculate the dR2 values from the results of the simulation
% 
function Results = get_dR2( Results, PhysicalParams, Params, SimSettings )

% Definition:
%   deltaR2	:=	-1/TE * log(|S|/|S0|)
dR2_func        =	@(S,S0) (-1/SimSettings.EchoTime) * log(abs(S)./abs(S0));
NumAngles       =	numel(Results.Angles_Rad);
Results.dR2_TE	=   zeros(1,NumAngles);

for ii = 1:NumAngles
	
    Results.dR2_all{ii}	=	dR2_func( ...
        Results.Signal_CA{ii}, Results.Signal_noCA{ii} );
    
    Results.dR2_TE(ii)	=   Results.dR2_all{ii}(end);
    
end

end

%==========================================================================
% Gaussian Smoothing of Discontinuous Maps
%==========================================================================
function Map = make_Smooth_Map( TrueVal, FalseVal, Geometry, PhysicalParams, Params, SimSettings )

% Initialize Map to boolean VasculatureMap
Map     =	double( Geometry.VasculatureMap );

% Check if smoothing is required
if SimSettings.Smoothing == 0
    Map	=   (TrueVal - FalseVal) * Map + FalseVal;
    return
end

% Standard deviation and subvoxel size
vsize	=   SimSettings.SubVoxSize;
sigma	=	SimSettings.Smoothing * PhysicalParams.R_Minor_mu;

% Check for underflow of small kernels
n       =   ceil( 5 * sigma / vsize );
N       =   2*n+1;
r       =   (-n:n) * ( vsize / sigma );
if ( n == 1 ) && ( r(end) >= 8.0 )
    % Kernel will be approx. [0,1,0] to double precision
    Map	=   (TrueVal - FalseVal) * Map + FalseVal;
    return
end

% Construct kernels
gy      =   double( exp( -0.5 * r.^2 ) );
gy      =   gy / sum(gy(:));
[gx,gz]	=	deal( gy(:), reshape(gy,1,1,[]) );

% Convolve with kernels
if N <= 25
    % Small kernel -> Direct convolution along each dimension
    Map	=	padarray(Map,[n,n,n],'circular','both');
    
    Map	=	convn(Map,gx,'same');
    Map	=	Map(n+1:end-n,:,:);
    
    Map	=	convn(Map,gy,'same');
    Map	=	Map(:,n+1:end-n,:);
    
    Map	=	convn(Map,gz,'same');
    Map	=	Map(:,:,n+1:end-n);
else
    % Large kernel -> 3D FFT convolution
    h	=	times3D(gx,gy,gz);
    h	=	h / sum(h(:));
    Map	=	fftconvn( Map, h, [N,N,N], false );
end

% Scale Map such that average value of map at true values is 1
% Map     =   Map ./ mean(reshape(Map(Geometry.VasculatureMap),[],1));

% Final Map values
Map     =   (TrueVal - FalseVal) * Map + FalseVal;

end
