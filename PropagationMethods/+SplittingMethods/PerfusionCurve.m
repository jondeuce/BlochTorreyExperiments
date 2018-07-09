function [ dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs, Geometries ] = PerfusionCurve( varargin )
%PERFUSIONCURVE [ dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs, Geometries ] = PerfusionCurve( varargin )
% See docs for example usage.
%{
TE = 60e-3; Nsteps = 8; type = 'SE'; Dcoeff = 3037; CA = 6; B0 = -3;
CADerivative = true; AlphaRange = [0,90];

[dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs, Geometries] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, Dcoeff, CA, B0, AlphaRange, type, ... % positional args
'Order', 2, 'CADerivative', true ); %optional positionless args
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
[AlphaRange, TE, Nsteps, type, Dcoeff] = deal(...
    args.AlphaRange, args.TE, args.Nsteps, args.type, args.Dcoeff);

dt = TE/Nsteps;
y0 = 1i;
Nalphas = numel(AlphaRange);

Gamma = []; % initialize stepper with empty Gamma (added later)
dGamma = {}; % same for dGamma
CAidx = 1; % index in cell array dGamma where CA deriv will go

Vox_Volume = prod(Geom.VoxelSize);
um3_per_voxel = Vox_Volume/prod(Geom.GridSize);
IntegrateSignal = @(y) um3_per_voxel * sum(sum(sum(y,1),2),3); % more accurate than sum(y(:))

switch upper(args.Stepper)
    case 'BTSPLITSTEPPER'
        V = SplittingMethods.BTSplitStepper(...
            dt, Dcoeff, Gamma, dGamma, Geom.GridSize, Geom.VoxelSize, ...
            'NReps', 1, 'Order', 2);
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
    
    % Calculate complex decay settings
    [GammaSettingsNoCA, GammaSettingsCA] = GetGammaSettings(alpha, args);
    
    Gamma = CalculateComplexDecay( GammaSettingsNoCA, Geom );
    V = precomputeExpDecays(V, Gamma);
    y = y0*ones(Geom.GridSize); %initial state
    
    for ii = 1:Nsteps
        no_CA_time = tic;
        
        y = step(V,y);
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
        V = precomputeGammaDerivs(V, ComplexDecayDerivative(GammaSettingsCA, Geom, Gamma_CA, 'CA', Gamma));
        clear Gamma
        
        V = precomputeExpDecays(V, Gamma_CA);
        clear Gamma_CA
        dy = {zeros(size(y),'like',y)};
    else
        V = precomputeExpDecays(V, AddContrastAgent(GammaSettingsNoCA, GammaSettingsCA, Geom, Gamma));
        dy = {};
        clear Gamma
    end
    
    for ii = 1:Nsteps
        with_CA_time = tic;
        
        [y,dy] = step(V,y,dy);
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
        V = clearGammaDerivs(V);
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

switch upper(args.MajorOrientation)
    case 'FIXEDPOSITION'
        % Initial geometry generated will be used for all angles
        Geom = Geometry.CylindricalVesselFilledVoxel( GivenNameValueArgs{:} );
        
    case 'FIXEDRADIUS'
        % Geometry will be generated for zero degrees (vertical major
        % vessels) and rotated with this fixed radius
        GeomArgs = args.GeomArgs;
        GeomArgs.MajorAngle = 0.0;
        NameValueArgs = struct2arglist(GeomArgs);
        
        Geom = Geometry.CylindricalVesselFilledVoxel( NameValueArgs{:} );
end

end

function Geom = UpdateGeometry(AngleDeg, Geom, args)

switch upper(args.MajorOrientation)
    case 'FIXEDPOSITION'
        % Initial geometry generated is used for all angles; do nothing
        
    case 'FIXEDRADIUS'
        % Geometry is rotated from previous position
        Geom = RotateMajor(Geom, 'to', AngleDeg);
end

end

function [GammaSettingsNoCA, GammaSettingsCA] = GetGammaSettings(AngleDeg, args)
    
switch upper(args.MajorOrientation)
    case 'FIXEDPOSITION'
        % Gamma settings change with angle
        GammaSettingsNoCA = Geometry.ComplexDecaySettings('Angle_Deg', AngleDeg, 'B0', args.B0, 'CA', 0.0);
        GammaSettingsCA   = Geometry.ComplexDecaySettings('Angle_Deg', AngleDeg, 'B0', args.B0, 'CA', args.CA);
        
    case 'FIXEDRADIUS'
        % Gamma settings are fixed, as B0 is vertical
        GammaSettingsNoCA = Geometry.ComplexDecaySettings('Angle_Deg', 0.0, 'B0', args.B0, 'CA', 0.0);
        GammaSettingsCA   = Geometry.ComplexDecaySettings('Angle_Deg', 0.0, 'B0', args.B0, 'CA', args.CA);
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

RequiredArgs = { 'TE', 'Nsteps', 'Dcoeff', 'CA', 'B0', 'AlphaRange', 'type' };
DefaultArgs = struct(...
    'Geom', [], ...
    'GeomArgs', [], ...
    'MajorOrientation', 'FixedPosition', ...
    'Order', 2, ...
    'Stepper', 'BTSplitStepper', ...
    'CADerivative', false, ...
    'RotateGeom', false ...
    );

p = inputParser;

for f = RequiredArgs
    paramName = f{1};
    addRequired(p,paramName)
end

for f = fieldnames(DefaultArgs).'
    paramName = f{1};
    defaultVal = DefaultArgs.(f{1});
    addParameter(p,paramName,defaultVal)
end

parse(p, varargin{:});
args = p.Results;

% Mean-squared diffusion length in n-dimensions is
%   d = sqrt(2*n*D*t)
% If this distance (where t is the entire simulation time TE) is less than
% half the minimum subvoxel dimension, we say that diffusion is negligible.
if ~isempty(args.Geom)
    SubVoxSize = args.Geom.SubVoxSize;
else
    SubVoxSize = min( args.GeomArgs.VoxelSize ./ args.GeomArgs.GridSize );
end

isDiffusionNegligible = (sqrt( 6 * args.Dcoeff * args.TE ) <= 0.5 * SubVoxSize);
if isDiffusionNegligible
    args.Dcoeff = 0.0;
end

end

% ---- Derivative testing ---- %
%{
% ---- initialize params ---- %
TE = 60e-3; Nsteps = 8; type = 'GRE'; Dcoeff = 4*3037; CA = 6; B0 = -3;
CADerivative = true; AlphaRange = [0,90];

% ---- function call with analytic derivative ---- %
[dR2, S0, S, TimePts, dR2_Derivs, dS_Derivs] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, Dcoeff, CA, B0, AlphaRange, Geom, type, ... % positional args
'Order', 2, 'CADerivative', true ); %optional positionless args

% ---- derivative test: forward difference ---- %
dCA = 1e-3 * CA;
[dR2_fwd, S0_fwd, S_fwd, ~, ~, ~] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, Dcoeff, CA + dCA, B0, AlphaRange, Geom, type, ... % positional args
'Order', 2, 'CADerivative', false ); %optional positionless args

dR2_Derivs.CA
dR2_dCA_fwd = (dR2_fwd - dR2)/dCA
dR2_dCA_fwd - dR2_Derivs.CA

% ---- derivative test: centered difference ---- %
dCA = 1e-6 * CA;
[dR2_fwd, S0_fwd, S_fwd, ~, ~, ~] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, Dcoeff, CA + dCA, B0, AlphaRange, Geom, type, ... % positional args
'Order', 2, 'CADerivative', false ); %optional positionless args

[dR2_bwd, S0_bwd, S_bwd, ~, ~, ~] = SplittingMethods.PerfusionCurve(...
TE, Nsteps, Dcoeff, CA - dCA, B0, AlphaRange, Geom, type, ... % positional args
'Order', 2, 'CADerivative', false ); %optional positionless args

dR2_Derivs.CA
dR2_dCA_mid = (dR2_fwd - dR2_bwd)/(2*dCA)
dR2_dCA_mid - dR2_Derivs.CA

%}