%% DATA LOADING

% alpha_range = 87.5;
% alpha_range = [2.5, 47.5, 87.5];
% alpha_range = [17.5, 32.5, 52.5, 67.5, 87.5];
% alpha_range = [2.5, 17.5, 27.5, 37.5, 47.5, 57.5, 67.5, 77.5, 82.5, 87.5];
alpha_range = 2.5:5.0:87.5;

type = 'GRE';
[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [150,150,150];
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [150,150,150];
TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [350,350,350];
% TE = 40e-3; VoxelSize = [1750,1750,4000]; VoxelCenter = [0,0,0]; GridSize = [350,350,800];
Weights = BinCounts / sum(BinCounts(:));

%% BLOCH-TORREY SETTINGS

CA   = 3.8418;
iBVF = 1.4920/100;
aBVF = 0.9306/100;

Nmajor = 1;
Rminor_mu = 7.0;
Rminor_sig = 0.0;
% Rminor_mu = 13.7;
% Rminor_sig = 2.1;

% ---- With Diffusion ---- %
% D_Tissue = 2000; %[um^2/s]
% D_Blood = []; %[um^2/s]
% D_VRS = []; %[um^2/s]
% MaskType = '';

% Nsteps = 8;
% StepperArgs = struct('Stepper', 'BTSplitStepper', 'Order', 2);

D_Tissue = 1500; %[um^2/s]
D_Blood = 3037; %[um^2/s]
D_VRS = 3037; %[um^2/s]
% MaskType = 'Vasculature'; % only true in Vasc.
% MaskType = 'PVS'; % only true in PVS
% MaskType = 'PVSOrVasculature'; % true in PVS or Vasc.
MaskType = 'PVSAndVasculature'; % 2 in PVS, 1 in Vasc., 0 else ("trinary" mask)

Nsteps = 1;
StepperArgs = struct('Stepper', 'ExpmvStepper', 'prec', 'half', 'full_term', false, 'prnt', false);

% % ---- Diffusionless ---- %
% D_Tissue = 0; %[um^2/s]
% D_Blood = []; %[um^2/s]
% D_VRS = []; %[um^2/s]
% MaskType = '';
% Nsteps = 1; % one exact step
% StepperArgs = struct('Stepper', 'BTSplitStepper', 'Order', 2);

B0 = -3.0; %[Tesla]
rng('default'); seed = rng; % for consistent geometries between sims.
Navgs = 1; % for now, if geom seed is 'default', there is no point doing > 1 averages
RotateGeom = false; % geometry is fixed; dipole rotates
MajorAngle = 0.0; % major vessel angle w.r.t z-axis [degrees]
NumMajorArteries = 0;
MinorArterialFrac = 0.0;

% Radius of Virchow-Robin space relative to major vessel radius [unitless];
% VRS space volume is approx (relrad^2-1)*BVF, so e.g. sqrt(2X radius) => 1X volume
% VRSRelativeRad = 1; % 1X => 0X volume
% VRSRelativeRad = sqrt(2); % sqrt(2X) => 1X volume
% VRSRelativeRad = sqrt(2.5); % sqrt(2.5X) => 1.5X volume
% VRSRelativeRad = sqrt(3); % sqrt(3X) => 2X volume
VRSRelativeRad = 2; % 2X => 3X volume


%% GEOMETRY

% Fixed Geometry Arguments
GeomArgs = struct( 'iBVF', iBVF, 'aBVF', aBVF, ...
    'VoxelSize', VoxelSize, 'GridSize', GridSize, 'VoxelCenter', VoxelCenter, ...
    'Nmajor', Nmajor, 'MajorAngle', MajorAngle, ......
    'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
    'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'VRSRelativeRad', VRSRelativeRad, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'ImproveMajorBVF', false, 'ImproveMinorBVF', true, ... % for speed
    'PopulateIdx', true, 'seed', seed );

GeomNameValueArgs = struct2arglist(GeomArgs);
Geom = Geometry.CylindricalVesselFilledVoxel( GeomNameValueArgs{:} );


%% Parameter Sweep

CA_sweep = [0.0, CA];
alpha_sweep = [0.0];
D_Tissue_sweep = [1000, 1500, 2000];
[CA_sweep, alpha_sweep, D_Tissue_sweep] = ndgrid(CA_sweep, alpha_sweep, D_Tissue_sweep);
params_sweep = [CA_sweep(:), alpha_sweep(:), D_Tissue_sweep(:)];

blank_u = struct('u', [], 'G', [], 'D', [], 'M', [], 'alpha', [], 'CA', [], 'D_Tissue', [], ...
    'MaskType', MaskType, 'VRSRelativeRad', VRSRelativeRad);
u0 = 1i * ones(Geom.GridSize);
u = repmat(blank_u, size(params_sweep, 1), 1);

for ii = 1:size(params_sweep, 1)
    tic;
    
    u(ii).CA = params_sweep(ii,1);
    u(ii).alpha = params_sweep(ii,2);
    u(ii).D_Tissue = params_sweep(ii,3);
    
    fprintf('Parameter sweep: ii = %d/%d\n', ii, size(params_sweep, 1));
    %     disp(u(ii));
    
    % Stepper
    dt = TE; % Propagation time
    GammaSettings = Geometry.ComplexDecaySettings('Angle_Deg', u(ii).alpha, 'B0', B0, 'CA', u(ii).CA);
    G = CalculateComplexDecay( GammaSettings, Geom );
    D = CalculateDiffusionMap( Geom, u(ii).D_Tissue, D_Blood, D_VRS );
    M = GetMask(Geom, MaskType);
    isGamma = true; % Input is Gamma = R2 + i*dw itself, not the operator diagonal
    A = BlochTorreyOp(G, D, Geom.GridSize, Geom.VoxelSize, ~isGamma, M);
    V = ExpmvStepper(dt, A, Geom.GridSize, Geom.VoxelSize, ...
        'prec', StepperArgs.prec, ...
        'full_term', StepperArgs.full_term, ...
        'prnt', StepperArgs.prnt, ...
        'type', 'default', 'forcesparse', false, ...
        'shift', true, 'bal', false);
    
    u(ii).u = step(V, u0);
    u(ii).G = G;
    u(ii).D = D;
    u(ii).M = M;
    phase = unwrapLap(angle(u(ii).u));
    
    title_str = sprintf('$\\alpha = %.1f$, $CA = %.4f$, $D_{Tissue} = %.0f$', u(ii).alpha, u(ii).CA, u(ii).D_Tissue);
    %     figure, imagesc(imag(u(ii).G(:,:,end/2))), axis image, title(['Gamma: ', title_str])
    %     figure, imagesc(u(ii).M(:,:,end/2)), axis image, title(['Mask: ', title_str])
    figure, imagesc(abs(u(ii).u(:,:,end/2)), [0.0,1.0]), axis image, title(['$|u|$: ', title_str])
    %     figure, imagesc(phase(:,:,end/2)), axis image, title(['$\phi$: ', title_str])
    drawnow;
    
    toc;
end

for ii = 2:2:length(u)
    S = sum(vec(u(ii).u));
    S0 = sum(vec(u(ii-1).u));
    dR2 = -1/TE * log(abs(S/S0));
    fprintf('Delta R2*: %.4f\n', dR2);
end

%% Voxel Stacking

REP = [1,1,2];
U0 = repmat(u0, REP);
U = repmat(blank_u, size(params_sweep, 1), 1);
for ii = 1:length(u)
    tic;
    
    U(ii).CA = u(ii).CA;
    U(ii).alpha = u(ii).alpha;
    U(ii).D_Tissue = u(ii).D_Tissue;
    
    fprintf('Voxel Stacking: ii = %d/%d\n', ii, size(params_sweep, 1));
    
    % Stepper
    dt = TE; % Propagation time
    G = repmat(u(ii).G, REP);
    D = repmat(u(ii).D, REP);
    M = repmat(u(ii).M, REP);
    isGamma = true; % Input is Gamma = R2 + i*dw itself, not the operator diagonal
    A = BlochTorreyOp(G, D, Geom.GridSize .* REP, Geom.VoxelSize .* REP, ~isGamma, M);
    V = ExpmvStepper(dt, A, Geom.GridSize .* REP, Geom.VoxelSize .* REP, ...
        'prec', StepperArgs.prec, ...
        'full_term', StepperArgs.full_term, ...
        'prnt', StepperArgs.prnt, ...
        'type', 'default', 'forcesparse', false, ...
        'shift', true, 'bal', false);
    
    U(ii).u = step(V, U0);
    U(ii).G = G;
    U(ii).D = D;
    U(ii).M = M;
    phase = unwrapLap(angle(U(ii).u));
    
    title_str = sprintf('$\\alpha = %.1f$, $CA = %.4f$, $D_{Tissue} = %.0f$', U(ii).alpha, U(ii).CA, U(ii).D_Tissue);
    %     figure, imagesc(imag(U(ii).G(:,:,end/4))), axis image, title(['Gamma: ', title_str])
    %     figure, imagesc(U(ii).M(:,:,end/4)), axis image, title(['Mask: ', title_str])
    figure, imagesc(abs(U(ii).u(:,:,end/4)), [0.0,1.0]), axis image, title(['$|U|$: ', title_str])
    %     figure, imagesc(phase(:,:,end/4)), axis image, title(['$\phi$: ', title_str])
    drawnow;
    
    toc;
end

for ii = 1:length(u)
    S = sum(vec(U(ii).u))/prod(REP);
    S0 = sum(vec(u(ii).u));
    fprintf('100*|S-S0|/|S0|: %.16f\n', 100*abs(S-S0)/abs(S0));
end