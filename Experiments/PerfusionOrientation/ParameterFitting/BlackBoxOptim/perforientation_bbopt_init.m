%PERFORIENTATION_BBOPT

%Save current repository information
% saverepostatus('BlochTorreyExperiments-active')

%Save a copy of this script in the directory of the caller
backupscript  = sprintf('%s__%s.m', datestr(now,30), mfilename);
currentscript = strcat(mfilename('fullpath'), '.m');
copyfile(currentscript, backupscript);

% ---- Angles to simulate ---- %
% alpha_range = [2.5, 47.5, 87.5];
% alpha_range = 2.5:5.0:87.5;
% alpha_range = 22.5:5.0:87.5;
% alpha_range = 7.5:10.0:87.5;
% alpha_range = 17.5:10.0:87.5;
% alpha_range = 7.5:20.0:87.5;
% alpha_range = [17.5, 32.5, 52.5, 67.5, 87.5];
% alpha_range = [37.5, 52.5, 72.5, 17.5, 87.5];
% alpha_range = [2.5, 17.5, 42.5, 67.5, 87.5];
% alpha_range = [2.5, 22.5, 42.5, 62.5, 87.5];
% alpha_range = [2.5, 12.5, 22.5, 37.5, 57.5, 77.5, 87.5];
alpha_range = [2.5, 17.5, 27.5, 42.5, 57.5, 77.5, 87.5];
% alpha_range = [2.5, 12.5, 22.5, 32.5, 47.5, 57.5, 67.5, 77.5, 87.5];
% alpha_range = [2.5, 17.5, 27.5, 37.5, 47.5, 57.5, 67.5, 77.5, 82.5, 87.5];

% =============================== DATA ================================== %

% % ---- GRE w/ Diffusion Initial Guess (large minor) ---- %
% lb  = [ 4.000,          1.0000/100,         0.8000/100 ];
% CA0 =   6.230;  iBVF0 = 2.2999/100; aBVF0 = 1.1601/100;
% ub  = [ 8.000,          3.5000/100,         1.5000/100 ];

% ---- GRE w/ Diffusion Initial Guess (small minor) ---- %
lb  = [ 3.0000,          0.8000/100,         0.5000/100 ];
CA0 =   3.8000;  iBVF0 = 1.5000/100; aBVF0 = 0.8000/100;
ub  = [ 6.0000,          2.0000/100,         1.5000/100 ];

% % ---- GRE w/ Diffusion Initial Guess (small minor) ---- %
% lb  = [ 3.5000,          1.0000/100,         0.6000/100 ];
% CA0 =   4.3226;  iBVF0 = 1.2279/100; aBVF0 = 0.7951/100;
% ub  = [ 5.5000,          1.5000/100,         1.1000/100 ];

x0 = [CA0, iBVF0, aBVF0];

type = 'GRE';
[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [150,150,150];
TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [350,350,350];
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [400,400,400];
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [500,500,500];
% TE = 40e-3; VoxelSize = [1750,1750,4000]; VoxelCenter = [0,0,0]; GridSize = [350,350,800];
Weights = sqrt(BinCounts); % BinCounts;
Weights = Weights / sum(vec(Weights));

% % ---- SE w/ Diffusion Initial Guess ---- %
% lb  = [ 1.0000,          0.5000/100,         0.5000/100 ];
% CA0 =   3.7152;  iBVF0 = 1.6334/100; aBVF0 = 1.0546/100;
% ub  = [ 8.0000,          2.5000/100,         2.5000/100 ];
% 
% x0  = [CA0, iBVF0, aBVF0];
% 
% type = 'SE';
% [alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize] = get_SE_data(alpha_range);
% TE = 60e-3; VoxelSize = [3000,3000,3000]; VoxelCenter = [0,0,0]; GridSize = [512,512,512];
% Weights = ones(size(alpha_range));
% Weights = Weights / sum(vec(Weights));

% ======================== BLOCH-TORREY SETTINGS ======================== %

Nmajor = 5;
Rminor_mu = 6.0; % Mean vessel size from Shen et al. (MRM 2012 https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.24258)
Rminor_sig = 0.0;
% Rminor_mu = 13.7; % Jochimsen et al. (Neuroimage 2010 https://www.sciencedirect.com/science/article/pii/S1053811910002053)
% Rminor_sig = 2.1;
Rmedium_thresh = 0.0; % Minor vessels with radii > Rmedium_thresh are surrounded by VRS

% ---- With Diffusion ---- %
% D_Tissue = 3000; %[um^2/s]
% D_Blood = []; %[um^2/s]
% D_VRS = []; %[um^2/s]
% MaskType = '';

D_Tissue = 1500; %[um^2/s]
D_Blood = 3037; %[um^2/s]
D_VRS = 3037; %[um^2/s]
% MaskType = 'Vasculature'; % only true in Vasc.
% MaskType = 'PVS'; % only true in PVS
% MaskType = 'PVSOrVasculature'; % true in PVS or Vasc. (don't think this is ever useful?)
MaskType = 'PVSAndVasculature'; % 2 in PVS, 1 in Vasc., 0 else ("trinary" mask)

% Nsteps = 8;
% StepperArgs = struct('Stepper', 'BTSplitStepper', 'Order', 2);
Nsteps = 1;
StepperArgs = struct('Stepper', 'ExpmvStepper', 'prec', 'half', 'full_term', false, 'prnt', true);

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
EffectiveVesselAngles = false; % use effective vessel angles instead of fibre angles directly
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

PlotFigs = true;
SaveFigs = true;
FigTypes = {'png'}; % outputs a lot of figures, so just 'png' is probably best
CloseFigs = true;
SaveResults = true;

% ============================= GEOMETRY ================================ %

% OptVariables = 'CA_iBVF_aBVF';
OptVariables = 'CA_Rmajor_MinorExpansion';

% Fixed Geometry Arguments
GeomArgs = struct( ...% 'iBVF', iBVF, 'aBVF', aBVF, ... % these are set below
    'VoxelSize', VoxelSize, 'GridSize', GridSize, 'VoxelCenter', VoxelCenter, ...
    'Nmajor', Nmajor, 'MajorAngle', MajorAngle, ......
    'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
    'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'VRSRelativeRad', VRSRelativeRad, ...
    'MediumVesselRadiusThresh', Rmedium_thresh, 'AllowInitialMinorPruning', true, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'ImproveMajorBVF', true, 'ImproveMinorBVF', true, ...
    'PopulateIdx', true, 'seed', seed );

% Generate initial guess and geometry
switch upper(OptVariables)
    case 'CA_IBVF_ABVF'
        Geom = []; % Geom is created inside perforientation_fun
    case 'CA_RMAJOR_MINOREXPANSION'
        GeomArgs.ImproveMajorBVF = true;
        GeomArgs.ImproveMinorBVF = true;
        
        % Generate geometry for contraction
        GeomArgs.iBVF = ub(2);
        GeomArgs.aBVF = ub(3);
        GeomNameValueArgs = struct2arglist(GeomArgs);
        Geom = Geometry.CylindricalVesselFilledVoxel( GeomNameValueArgs{:} );
        
        % Generate new initial guesses and bounds
        getRmajor0 = @(aBVF) sqrt( prod(VoxelSize) * aBVF / ( Nmajor * pi * rayBoxIntersectionLength( VoxelCenter(:), [sind(MajorAngle); 0; cosd(MajorAngle)], VoxelSize(:), VoxelCenter(:) ) ) );
        getSpaceFactor0 = @(iBVF) (Geom.iBVF/iBVF)^(1/2.3); % empirical model: iBVF = iBVF_max * SpaceFactor^(-2.3)
        
        lb_old = lb;
        ub_old = ub;
        lb = [lb_old(1), getRmajor0(lb_old(3)), getSpaceFactor0(ub_old(2))];
        x0 = [x0(1),     getRmajor0(x0(3)),     getSpaceFactor0(x0(2))];
        ub = [ub_old(1), getRmajor0(ub_old(3)), getSpaceFactor0(lb_old(2))];
        
    otherwise
        error('''OptVariables'' must be ''CA_iBVF_aBVF'' or ''CA_Rmajor_MinorExpansion''');
end

% =========================== OPTIMIZATION ============================== %

% Norm function type for the weighted residual (see PERFORIENTATION_OBJFUN)
Normfun = 'AICc';

% Save initial params
Params0 = struct('OptVariables', OptVariables,...
    'CA0', CA0, 'iBVF0', iBVF0, 'aBVF0', aBVF0,...
    'x0', x0, 'lb', lb, 'ub', ub);
save('Params0.mat', 'Params0', '-v7');

% Simulation function handle
ObjFunMaker = @(x,Geom) perforientation_objfun(x, alpha_range, dR2_Data, [], Weights, Normfun, ...
    TE, Nsteps, type, B0, D_Tissue, D_Blood, D_VRS, ...
    'OptVariables', OptVariables, ...
    'Navgs', Navgs, 'StepperArgs', StepperArgs, 'MaskType', MaskType, ...
    'Weights', Weights, 'Normfun', Normfun, ...
    'PlotFigs', PlotFigs, 'SaveFigs', SaveFigs, 'CloseFigs', CloseFigs, 'FigTypes', FigTypes, ...
    'SaveResults', SaveResults, 'DiaryFilename', '', ...
    'GeomArgs', GeomArgs, 'Geom', Geom, 'RotateGeom', RotateGeom);
save('ObjFunMaker.mat', 'ObjFunMaker', '-v7');

% Save resulting workspace
if ~isempty(Geom)
    % Clear anonymous functions which close over `Geom` for saving
    clear getRmajor0 getSpaceFactor0
    
    % Compress `Geom` for saving
    Geom = Compress(Geom);
    save('Geom.mat', 'Geom', '-v7');
end
save('BBOptWorkspace.mat', '-v7');

% ====== Code for regenerating figures/FminconIterationsOutput file ===== %

% fout = fopen([datestr(now,30),'__','FminconIterationsOutput.txt'], 'w');
% iter = 1;
% Norm_best = Inf;
% fprintf(fout, '%s', 'Timestamp       f-count            f(x)       Best f(x)');
% 
% for s = dir('*.mat')'
%     try
%         % Load results struct
%         Results = load(s.name);
%         Results = Results.Results;
%         
%         % % Set proper weights/normfun
%         % [~, ~, ~, ~, ~, ~, Results.args.Weights] = get_GRE_data(Results.alpha_range);
%         % Results.args.Weights = Results.args.Weights/sum(Results.args.Weights(:));
%         % Results.args.Normfun = 'L2w';
%         % save(s.name, 'Results');
%         
%         % replot and save fig
%         [ fig, ~ ] = perforientation_plot( Results.dR2, Results.dR2_all, Results.Geometries, Results.args );
%         
%         % recreate norm values file
%         f = perforientation_objfun(Results.params, Results.alpha_range, Results.dR2_Data, Results.dR2, Results.args.Weights, Results.args.Normfun);
%         Norm_best = min(f, Norm_best);
%         fprintf(fout, '\n%s%8d%16.8f%16.8f', s.name(1:15), iter, f, Norm_best);
%         iter = iter + 1;
%     catch me
%         warning(me.message);
%     end
% end
