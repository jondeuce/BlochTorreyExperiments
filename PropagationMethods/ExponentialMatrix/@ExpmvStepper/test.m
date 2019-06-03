%% DATA LOADING

% alpha_range = 87.5;
% alpha_range = [2.5, 47.5, 87.5];
% alpha_range = [17.5, 32.5, 52.5, 67.5, 87.5];
% alpha_range = [2.5, 17.5, 27.5, 37.5, 47.5, 57.5, 67.5, 77.5, 82.5, 87.5];
alpha_range = 2.5:5.0:87.5;

type = 'GRE';
[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);

TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [150,150,150];
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [250,250,250];
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [350,350,350];
% TE = 40e-3; VoxelSize = [1750,1750,4000]; VoxelCenter = [0,0,0]; GridSize = [350,350,800];
Weights = BinCounts / sum(BinCounts(:));

REP = [1,1,2];
VoxelSizeREP = VoxelSize .* REP;
GridSizeREP = GridSize .* REP;


%% BLOCH-TORREY SETTINGS

CA   = 3.8418;
iBVF = 1.4920/100;
aBVF = 0.9306/100;

Nmajor = 4;
% Rminor_mu = 7.0;
% Rminor_sig = 0.0;
Rminor_mu = 13.7;
Rminor_sig = 2.1;
Rmedium_thresh = 13.7;

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
NumMajorArteries = 1;
MinorArterialFrac = 1/3;

% Radius of Virchow-Robin space relative to major vessel radius [unitless];
% VRS space volume is approx (relrad^2-1)*BVF, so e.g. sqrt(2X radius) => 1X volume
% VRSRelativeRad = 1; % 1X => 0X volume
% VRSRelativeRad = sqrt(2); % sqrt(2X) => 1X volume
% VRSRelativeRad = sqrt(2.5); % sqrt(2.5X) => 1.5X volume
% VRSRelativeRad = sqrt(3); % sqrt(3X) => 2X volume
VRSRelativeRad = 2; % 2X => 3X volume


%% Geometry

% Fixed Geometry Arguments
GeomArgs = struct( 'iBVF', iBVF, 'aBVF', aBVF, ...
    'VoxelSize', VoxelSize, 'GridSize', GridSize, 'VoxelCenter', VoxelCenter, ...
    'Nmajor', Nmajor, 'MajorAngle', MajorAngle, ......
    'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
    'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'VRSRelativeRad', VRSRelativeRad, ...
    'MediumVesselRadiusThresh', Rmedium_thresh, 'AllowInitialMinorPruning', true, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'ImproveMajorBVF', true, 'ImproveMinorBVF', true, ... % for speed
    'PopulateIdx', true, 'seed', seed );

GeomNameValueArgs = struct2arglist(GeomArgs);
Geom = Geometry.CylindricalVesselFilledVoxel( GeomNameValueArgs{:} );


%% "Repeated" geometries

GeomArgsREP = GeomArgs;
GeomArgsREP.VoxelSize = VoxelSizeREP;
GeomArgsREP.GridSize = GridSizeREP;
GeomNameValueArgsREP = struct2arglist(GeomArgsREP);
GeomREP = Geometry.CylindricalVesselFilledVoxel( GeomNameValueArgsREP{:} );

if mod(GeomREP.GridSize(3), 4) == 0
    GeomMID = Geom;
    GeomMID = SetCylinders(GeomMID, GeomREP);
end


%% Parameter Sweep

Geom_sweep = Geom;
% Geom_sweep = GeomREP;
% Geom_sweep = GeomMID;

CA_sweep = [0.0, CA];
alpha_sweep = [0.0];%, 90.0];
D_Tissue_sweep = [1000];%, 1500, 2000];
[CA_sweep, alpha_sweep, D_Tissue_sweep] = ndgrid(CA_sweep, alpha_sweep, D_Tissue_sweep);
params_sweep = [CA_sweep(:), alpha_sweep(:), D_Tissue_sweep(:)];

blank_u = struct('u', [], 'G', [], 'D', [], 'M', [], 'alpha', [], 'CA', [], 'D_Tissue', [], ...
    'MaskType', MaskType, 'VRSRelativeRad', VRSRelativeRad);
u0 = 1i * ones(Geom_sweep.GridSize);
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
    GammaSettings = Geometry.ComplexDecaySettings('Angle_Deg', u(ii).alpha, 'B0', B0, 'CA', u(ii).CA, 'isKspaceDipoleKernel', false);
    G = CalculateComplexDecay( GammaSettings, Geom_sweep );
    D = CalculateDiffusionMap( Geom_sweep, u(ii).D_Tissue, D_Blood, D_VRS );
    M = GetMask(Geom_sweep, MaskType);
    isGamma = true; % Input is Gamma = R2 + i*dw itself, not the operator diagonal
    A = BlochTorreyOp(G, D, Geom_sweep.GridSize, Geom_sweep.VoxelSize, ~isGamma, M);
    V = ExpmvStepper(dt, A, Geom_sweep.GridSize, Geom_sweep.VoxelSize, ...
        'prec', StepperArgs.prec, ...
        'full_term', StepperArgs.full_term, ...
        'prnt', StepperArgs.prnt, ...
        'type', 'default', 'forcesparse', false, ...
        'shift', true, 'bal', false);
    
    u(ii).u = step(V, u0);
    u(ii).G = G;
    u(ii).D = D;
    u(ii).M = M;
    
    title_str = sprintf('$\\alpha = %.1f$, $CA = %.4f$, $D_{Tissue} = %.0f$', u(ii).alpha, u(ii).CA, u(ii).D_Tissue);
    figure, imagesc(imag(u(ii).G(:,:,end/2))), axis image, title(['$\omega$: ', title_str]), colorbar
    figure, imagesc(u(ii).M(:,:,end/2)), axis image, title(['Mask: ', title_str]), colorbar
    figure, imagesc(abs(u(ii).u(:,:,end/2)), [0.0,1.0]), axis image, title(['$|u|$: ', title_str]), colorbar
    phase = unwrapLap(angle(u(ii).u));
    figure, imagesc(phase(:,:,end/2)), axis image, title(['$\phi$: ', title_str]), colorbar
    drawnow;
    
    toc;
end

for ii = 2:2:length(u)
    S = sum(vec(u(ii).u));
    S0 = sum(vec(u(ii-1).u));
    dR2 = -1/TE * log(abs(S/S0));
    fprintf('Delta R2*: %.4f\n', dR2);
end

u_Geom = u;
% u_GeomREP = u;
% u_GeomMID = u;
% u_GeomREP_Image = u;
% u_GeomMID_Image = u;


%% Plot side-by-side slices
% u1 = u_Geom;
% u1 = u_GeomMID;
u1 = u_GeomMID_Image;
% u2 = u_GeomREP;
% u2 = u_GeomMID_Image;
u2 = u_GeomREP_Image;

Geom1 = GeomMID;
Geom2 = GeomREP;

for ii = 1:2
    for num_slices = 1:2
        title_str = sprintf('$\\alpha = %.1f$, $CA = %.4f$, $D_{Tissue} = %.0f$', u(ii).alpha, u(ii).CA, u(ii).D_Tissue);
        idx1 = randi(Geom1.GridSize(3));
        if isequal(size(Geom1.GridSize), size(Geom2.GridSize))
            %idx2 = randi(Geom2.GridSize(3));
            idx2 = idx1;
        else
            idx2 = round(Geom2.GridSize(3)/4) + idx1;
        end
        
        Wslice1 = imag(u1(ii).G(:,:,idx1));
        Wslice2 = imag(u2(ii).G(:,:,idx2));
        SlabSize = max(1, ceil(Geom1.GridSize(2)/15));
        SlabMag = max(maximum(vec(Wslice1)), maximum(vec(Wslice2)));
        Wslice = [Wslice1, SlabMag * ones(size(Wslice1, 1), SlabSize), Wslice2];
        figure, imagesc(Wslice, SlabMag*[-1,1]), axis image, title(['$\omega$: ', title_str]), colorbar
        
        Uslice1 = abs(u1(ii).u(:,:,idx1));
        Uslice2 = abs(u2(ii).u(:,:,idx2));
        SlabSize = max(1, ceil(Geom1.GridSize(2)/15));
        SlabMag = 1; % Magnitude is normalized
        Uslice = [Uslice1, SlabMag * ones(size(Wslice1, 1), SlabSize), Uslice2];
        figure, imagesc(Uslice, SlabMag*[0,1]), axis image, title(['$|u|$: ', title_str]), colorbar
    end
end


%% Voxel Stacking

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