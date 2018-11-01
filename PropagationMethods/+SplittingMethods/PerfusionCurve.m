function [ dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs, Geometries ] = PerfusionCurve( varargin )
%PERFUSIONCURVE [ dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs, Geometries ] = PerfusionCurve( varargin )
% See docs for example usage.
%{
AlphaRange = [0,90]; TE = 60e-3; Nsteps = 2; type = 'SE'; CA = 6; B0 = -3;
D_Tissue = 3037; D_Blood = []; D_VRS = [];
CADerivative = true;
rng('default');
GeomArgs = struct( 'iBVF', 1/100, 'aBVF', 1/100, ...
    'VoxelSize', [3000,3000,3000], 'GridSize', [256,256,256], 'VoxelCenter', [0,0,0], ...
    'Nmajor', 4, 'MajorAngle', 0, ...
    'NumMajorArteries', 1, 'MinorArterialFrac', 1/3, ...
    'Rminor_mu', 25, 'Rminor_sig', 0, ...
    'AllowMinorSelfIntersect', true, ...
    'AllowMinorMajorIntersect', true, ...
    'PopulateIdx', true, ...
    'seed', rng('default') );
% StepperArgs = struct('Stepper','BTSplitStepper','Order',2);
StepperArgs = struct('Stepper', 'ExpmvStepper', 'prec', 'half', 'full_term', false, 'prnt', false);

[dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs, Geometries] = SplittingMethods.PerfusionCurve(...
AlphaRange, TE, Nsteps, type, CA, B0, D_Tissue, ... % positional args
D_Blood, D_VRS, ... % optional positional args
'GeomArgs', GeomArgs, 'StepperArgs', StepperArgs, 'CADerivative', false ); %positionless args
%}

args = parseinputs(varargin{:});
[ dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs, Geometries ] = ...
    PerfusionCurveWithDiffusion( args );

end

% ---- Perfusion Curve Calculation ---- %
function [ dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs, Geometries ] = PerfusionCurveWithDiffusion( args )

% Get initial geometry
Geom = InitialGeometry(args);
Geometries = Compress(Geom);

% Deal out options for convenience
[AlphaRange, TE, Nsteps, type, D_Tissue, D_VRS, D_Blood] = deal(...
    args.AlphaRange, args.TE, args.Nsteps, args.type, args.D_Tissue, args.D_VRS, args.D_Blood);

dt = TE/Nsteps;
y0 = 1i;
Nalphas = numel(AlphaRange);

Gamma = []; % initialize stepper with empty Gamma (added later)
dGamma = {}; % same for dGamma
CAidx = 1; % index in cell array dGamma where CA deriv will go

Vox_Volume = prod(Geom.VoxelSize);
Num_Voxels = prod(Geom.GridSize);
um3_per_voxel = Vox_Volume/Num_Voxels;
IntegrateSignal = @(y) um3_per_voxel * sum(sum(sum(y,1),2),3); % more accurate than sum(y(:))

StepperArgs = args.StepperArgs;
switch upper(StepperArgs.Stepper)
    case 'BTSPLITSTEPPER'
        V = SplittingMethods.BTSplitStepper(...
            dt, D_Tissue, Gamma, dGamma, Geom.GridSize, Geom.VoxelSize, ...
            'NReps', 1, 'Order', StepperArgs.Order);
    case 'EXPMVSTEPPER'
        V = ExpmvStepper(dt, ...
            BlochTorreyOp(0, 0, Geom.GridSize, Geom.VoxelSize), ...
            Geom.GridSize, Geom.VoxelSize, ...
            'prec', StepperArgs.prec, ...
            'full_term', StepperArgs.full_term, ...
            'prnt', StepperArgs.prnt, ...
            'type', 'default', 'forcesparse', false, ...
            'shift', true, 'bal', false);
end

% Initialize outputs
[dR2, S0, S] = deal(zeros(Nsteps, Nalphas));
if args.CADerivative
    [dS_Derivs, dR2_Derivs] = deal( struct( 'CA', zeros(Nsteps, Nalphas) ) );
else
    [dS_Derivs, dR2_Derivs] = deal( [] );
end

for jj = 1:Nalphas
    
    alpha_loop_time = tic;
    alpha = AlphaRange(jj);
    
    % Update geometry
    Geom = UpdateGeometry(alpha, Geom, args);
    V = updateStepper(V, CalculateDiffusionMap( Geom, D_Tissue, D_Blood, D_VRS ), 'Diffusion', false);
    
    % Calculate complex decay settings
    [GammaSettingsNoCA, GammaSettingsCA] = GetGammaSettings(alpha, args);
    
    Gamma = CalculateComplexDecay( GammaSettingsNoCA, Geom );
    V = updateStepper(V, Gamma, 'Gamma');
    y = y0*ones(Geom.GridSize); %initial state
    
    for ii = 1:Nsteps
        no_CA_time = tic;
        
        [y,~,~,V] = step(V,y);
        S0(ii,jj) = IntegrateSignal(y);
        if strcmpi(type,'SE') && 2*ii == Nsteps
            y = conj(y);
        end
        
        str = sprintf('step %2d/%2d (no CA)',ii,Nsteps);
        display_toc_time(toc(no_CA_time), str);
    end
    
    % ---- Adjust Gamma to account for CA ---- %
    y = y0*ones(Geom.GridSize); %initial state
    
    if args.CADerivative
        Gamma_CA = AddContrastAgent(GammaSettingsNoCA, GammaSettingsCA, Geom, Gamma);
        V = updateStepper(V, ComplexDecayDerivative(GammaSettingsCA, Geom, Gamma_CA, 'CA', Gamma), 'dGamma');
        clear Gamma
        
        V = updateStepper(V, Gamma_CA, 'Gamma');
        clear Gamma_CA
        dy = {zeros(size(y),'like',y)};
    else
        V = updateStepper(V, AddContrastAgent(GammaSettingsNoCA, GammaSettingsCA, Geom, Gamma), 'Gamma');
        dy = {};
        clear Gamma
    end
    
    for ii = 1:Nsteps
        with_CA_time = tic;
        
        [y,dy,~,V] = step(V,y,dy);
        S(ii,jj) = IntegrateSignal(y);
        if args.CADerivative
            dS_Derivs.CA(ii,jj) = IntegrateSignal(dy{CAidx});
        end
        
        if strcmpi(type,'SE') && 2*ii == Nsteps
            y = conj(y);
            if args.CADerivative
                dy{CAidx} = conj(dy{CAidx});
            end
        end
        
        str = sprintf('step %2d/%2d (w/ CA)',ii,Nsteps);
        display_toc_time(toc(with_CA_time), str);
    end
    
    if args.CADerivative
        V = updateStepper(V, [], 'cleardgamma');
    end
    
    str = sprintf('alpha = %.2f',alpha);
    display_toc_time(toc(alpha_loop_time), str);
    
end

% Add t=0 signals
S0 = [y0*Vox_Volume*ones(1,Nalphas); S0];
S  = [y0*Vox_Volume*ones(1,Nalphas); S  ];
if args.CADerivative
    dS_Derivs.CA = [zeros(1,Nalphas); dS_Derivs.CA];
end

% Time points simulated
TimePts = linspace(0,TE,Nsteps+1).';

% Compute dR2 and derivatives
dR2 = (-1/TE) * log( abs(S) ./ abs(S0) );
if args.CADerivative
    dR2_Derivs = calc_dR2_Derivs(TE, S, dS_Derivs, dR2_Derivs, args);
end

end

function Geom = InitialGeometry(args)

% Check if initial geometry is supplied
if ~isempty(args.Geom)
    Geom = args.Geom;
    return
end

if isempty(args.GeomArgs)
    error('Neither initial geometry nor settings are supplied.');
else
    GivenNameValueArgs = struct2arglist(args.GeomArgs);
end

if args.RotateGeom
    % Geometry will be generated for zero degrees (vertical major
    % vessels) and rotated with this fixed radius
    GeomArgs = args.GeomArgs;
    GeomArgs.MajorAngle = 0.0;
    NameValueArgs = struct2arglist(GeomArgs);
    
    Geom = Geometry.CylindricalVesselFilledVoxel( NameValueArgs{:} );
else
    % Initial geometry generated will be used for all angles
    Geom = Geometry.CylindricalVesselFilledVoxel( GivenNameValueArgs{:} );
end

end

function Geom = UpdateGeometry(AngleDeg, Geom, args)

if args.RotateGeom
    % Geometry is rotated from previous position
    Geom = RotateMajor(Geom, 'to', AngleDeg);
else
    % Initial geometry generated is used for all angles; do nothing
end

end

function [GammaSettingsNoCA, GammaSettingsCA] = GetGammaSettings(AngleDeg, args)

if args.RotateGeom
    % Gamma settings (i.e. dipole orientation) are fixed, as B0 is vertical
    GammaSettingsNoCA = Geometry.ComplexDecaySettings('Angle_Deg', 0.0, 'B0', args.B0, 'CA', 0.0);
    GammaSettingsCA   = Geometry.ComplexDecaySettings('Angle_Deg', 0.0, 'B0', args.B0, 'CA', args.CA);
else
    % Gamma settings (i.e. dipole orientation) changes with angle
    GammaSettingsNoCA = Geometry.ComplexDecaySettings('Angle_Deg', AngleDeg, 'B0', args.B0, 'CA', 0.0);
    GammaSettingsCA   = Geometry.ComplexDecaySettings('Angle_Deg', AngleDeg, 'B0', args.B0, 'CA', args.CA);
end

end

function V = updateStepper(V, in, mode, precomp)

if nargin < 4; precomp = true; end

switch upper(class(V))
    case 'SPLITTINGMETHODS.BTSPLITSTEPPER'
        
        switch upper(mode)
            case 'GAMMA'
                if precomp; V = precomputeExpDecays(V, in); end
            case 'DGAMMA'
                if precomp; V = precomputeGammaDerivs(V, in); end
            case 'CLEARDGAMMA'
                V = clearGammaDerivs(V);
            case 'DIFFUSION'
                % Do nothing; diffusion is constant isotropic
        end
        
    case 'EXPMVSTEPPER'
        
        switch upper(mode)
            case 'GAMMA'
                A = BlochTorreyOp( in, V.A.D, V.A.gsize, V.A.gdims, false );
                A = setbuffer( A, BlochTorreyOp.DiagState );
                V = updateMatrix( V, A );
                clear A
                if precomp; V = precompute( V, in ); end
            case 'DGAMMA'
                error('Gamma derivatives are not implemented for ExpmvStepper''s');
            case 'CLEARDGAMMA'
                error('Gamma derivatives are not implemented for ExpmvStepper''s');
            case 'DIFFUSION'
                A = BlochTorreyOp( V.A.Gamma, in, V.A.gsize, V.A.gdims, false );
                A = setbuffer( A, BlochTorreyOp.DiagState );
                V = updateMatrix( V, A );
                clear A
                if precomp; V = precompute( V, in ); end
        end
        
end

end

% ---- delta R2(*) derivative calculation ---- %
function dR2_Derivs = calc_dR2_Derivs(TE, S, dS_Derivs, dR2_Derivs, args)
% For any parameter P, since we have that
%   dR2 = (-1/TE) * log(|S|/|S0|)
%
% It follows that
%   dR2_dP = (-1/TE) * (1/|S|) * d|S|_dP
%
% Now, abs(S) is not differentiable for complex variables P, but for
% real P we can simply consider
%   |S| = sqrt( R^2 + I^2 ), where S := R + iI,
% treating R and I independently.
%
% Then, we have that
%   d|S|_dP = (R*dR_dP + I*dI_dP) / |S|
%           = real( S * conj(dS_dP) ) / |S|

if args.CADerivative
    dR2_Derivs.CA = (-1/TE) * real( S .* conj(dS_Derivs.CA) ) ./ (abs(S).^2);
end

end

% ---- InputParsing ---- %
function args = parseinputs(varargin)

RequiredArgs = { 'AlphaRange', 'TE', 'Nsteps', 'type', 'CA', 'B0', 'D_Tissue' };
OptionalArgs = struct( 'D_Blood', [], 'D_VRS', [] );
DefaultArgs = struct(...
    'Geom', [], ...
    'GeomArgs', [], ...
    'StepperArgs', struct('Stepper','BTSplitStepper','Order',2), ...
    'CADerivative', false, ...
    'RotateGeom', false ...
    );

p = inputParser;

for f = RequiredArgs
    paramName = f{1};
    addRequired(p,paramName)
end

for f = fieldnames(OptionalArgs).'
    paramName = f{1};
    defaultVal = OptionalArgs.(f{1});
    addOptional(p,paramName,defaultVal)
end

for f = fieldnames(DefaultArgs).'
    paramName = f{1};
    defaultVal = DefaultArgs.(f{1});
    addParameter(p,paramName,defaultVal)
end

parse(p, varargin{:});
args = p.Results;

% ---- Inconsistent inputs checking ---- %
if strcmpi(args.type, 'SE') && mod(args.Nsteps, 2) ~= 0
    % Spin echo scans must have an even number of time steps for pi/2-pulse
    newNsteps = 2*floor(args.Nsteps/2) + 2;
    warning('Nsteps is not an even number and scan type is ''SE''; increasing from %d to %d.', ...
        args.Nsteps, newNsteps);
    args.Nsteps = newNsteps;
end

% Mean-squared diffusion length in n-dimensions is
%   d = sqrt(2*n*D*t)
% If this distance (where t is the entire simulation time TE) is less than
% half the minimum subvoxel dimension, we say that diffusion is negligible.
% if ~isempty(args.Geom)
%     SubVoxSize = args.Geom.SubVoxSize;
% else
%     if ~isempty(args.GeomArgs)
%         SubVoxSize = min( args.GeomArgs.VoxelSize ./ args.GeomArgs.GridSize );
%     else
%         SubVoxSize = 1;
%     end
% end
%
% isDiffusionNegligible = (sqrt( 6 * args.D_Tissue * args.TE ) <= 0.5 * SubVoxSize);
% if isDiffusionNegligible
%     args.D_Tissue = 0.0;
% end

end

% ---- Derivative testing ---- %
%{
% ---- initialize params ---- %
TE = 60e-3; Nsteps = 8; type = 'GRE'; D_Tissue = 4*3037; CA = 6; B0 = -3;
CADerivative = true; AlphaRange = [0,90];

% ---- function call with analytic derivative ---- %
[dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, D_Tissue, CA, B0, AlphaRange, Geom, type, ... % positional args
'Order', 2, 'CADerivative', true ); %optional positionless args

% ---- derivative test: forward difference ---- %
dCA = 1e-3 * CA;
[dR2_fwd, S0_fwd, S_fwd, ~, ~, ~] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, D_Tissue, CA + dCA, B0, AlphaRange, Geom, type, ... % positional args
'Order', 2, 'CADerivative', false ); %optional positionless args

dR2_Derivs.CA
dR2_dCA_fwd = (dR2_fwd - dR2)/dCA
dR2_dCA_fwd - dR2_Derivs.CA

% ---- derivative test: centered difference ---- %
dCA = 1e-6 * CA;
[dR2_fwd, S0_fwd, S_fwd, ~, ~, ~] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, D_Tissue, CA + dCA, B0, AlphaRange, Geom, type, ... % positional args
'Order', 2, 'CADerivative', false ); %optional positionless args

[dR2_bwd, S0_bwd, S_bwd, ~, ~, ~] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, D_Tissue, CA - dCA, B0, AlphaRange, Geom, type, ... % positional args
'Order', 2, 'CADerivative', false ); %optional positionless args

dR2_Derivs.CA
dR2_dCA_mid = (dR2_fwd - dR2_bwd)/(2*dCA)
dR2_dCA_mid - dR2_Derivs.CA

%}