%PERFORIENTATION_MISO

%Save current repository information
saverepostatus('BlochTorreyExperiments-temp1')

%Save a copy of this script in the directory of the caller
backupscript  = sprintf('%s__%s.m',datestr(now,30),mfilename);
currentscript = strcat(mfilename('fullpath'), '.m');
copyfile(currentscript,backupscript);

%Save diary of workspace
DiaryFilename = [datestr(now,30), '__', 'diary.txt'];

%Prompts to self
% display_text('WARNING: Set branch to temp1; geometry size to 350^3; initial guesses; minor vessel sizes', 75, '=', true, [true,true]);
% input('[Press enter to continue...]\n\n')

% ---- Angles to simulate ---- %
% alpha_range = [2.5, 47.5, 87.5];
% alpha_range = 2.5:5.0:87.5;
% alpha_range = 22.5:5.0:87.5;
% alpha_range = 7.5:10.0:87.5;
% alpha_range = 17.5:10.0:87.5;
% alpha_range = 7.5:20.0:87.5;
% alpha_range = [17.5, 32.5, 52.5, 67.5, 87.5];
% alpha_range = [37.5, 52.5, 72.5, 17.5, 87.5];
alpha_range = [2.5, 12.5, 22.5, 32.5, 47.5, 57.5, 67.5, 77.5, 87.5];
% alpha_range = [2.5, 17.5, 27.5, 37.5, 47.5, 57.5, 67.5, 77.5, 82.5, 87.5];

% =============================== DATA ================================== %

% ---- GRE w/ Diffusion Initial Guess (small minor) ---- %
lb  = [ 3.0000,          0.7000/100,         0.5000/100 ];
CA0 =   5.0000;  iBVF0 = 1.2000/100; aBVF0 = 0.8000/100;
ub  = [ 7.0000,          2.0000/100,         1.5000/100 ];

x0 = [CA0, iBVF0, aBVF0];

type = 'GRE';
[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [150,150,150];
TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [350,350,350];
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

Nmajor = 1:9;
Rminor_mu = 6.0; % Mean vessel size from Shen et al. (MRM 2012 https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.24258)
Rminor_sig = 0.0;
% Rminor_mu = 13.7; % Jochimsen et al. (Neuroimage 2010 https://www.sciencedirect.com/science/article/pii/S1053811910002053)
% Rminor_sig = 2.1;
Rmedium_thresh = Inf;

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
EffectiveVesselAngles = true; % use effective vessel angles instead of fibre angles directly
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

OptVariables = 'CA_Rmajor_MinorExpansion';

% Generate initial geometry
[X0, LB, UB] = deal({});
for ii = 1:numel(Nmajor)
    [ Geom(ii), GeomArgs(ii), X0{ii}, LB{ii}, UB{ii} ] = geom_initial( struct, OptVariables, x0, lb, ub, ...
        ... % 'iBVF', iBVF, 'aBVF', aBVF, ... % Set inside
        'VoxelSize', VoxelSize, 'GridSize', GridSize, 'VoxelCenter', VoxelCenter, ...
        'Nmajor', Nmajor(ii), 'MajorAngle', MajorAngle, ......
        'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
        'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
        'VRSRelativeRad', VRSRelativeRad, ...
        'MediumVesselRadiusThresh', Rmedium_thresh, 'AllowInitialMinorPruning', true, ...
        'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
        'ImproveMajorBVF', true, 'ImproveMinorBVF', true, ...
        'PopulateIdx', true, 'seed', seed );
end

% =========================== OPTIMIZATION ============================== %

% Call diary before the optimization starts
if ~isempty(DiaryFilename); diary(DiaryFilename); end

% Norm function type for the weighted residual (see PERFORIENTATION_OBJFUN)
Normfun = 'AICc';

% Objective function handle
NmajorMapMatrix = sparse(vec(Nmajor), 1, vec(1:numel(Nmajor)));
NmajorMap = @(x) full(NmajorMapMatrix(x)); % Nmajor => NmajorIndex

objfun = @(x) perforientation_objfun(x(1:3), alpha_range, dR2_Data, [], Weights, Normfun, ...
    TE, Nsteps, type, B0, D_Tissue, D_Blood, D_VRS, ...
    'OptVariables', OptVariables, ...
    'Navgs', Navgs, 'StepperArgs', StepperArgs, 'MaskType', MaskType, ...
    'Weights', Weights, 'Normfun', Normfun, ...
    'PlotFigs', PlotFigs, 'SaveFigs', SaveFigs, 'CloseFigs', CloseFigs, 'FigTypes', FigTypes, ...
    'SaveResults', SaveResults, 'DiaryFilename', DiaryFilename, ...
    'GeomArgs', GeomArgs(NmajorMap(x(4))), 'Geom', Geom(NmajorMap(x(4))), 'RotateGeom', RotateGeom, ...
    'EffectiveVesselAngles', EffectiveVesselAngles);

% Generate lower/upper bounds, initial x0 matrix, and if generate extra
% points are needed, make random points within the upper/lower bounds
lb = min(cat(1, LB{:}), [], 1);
ub = max(cat(1, UB{:}), [], 1);
x0_matrix = cat(1, X0{:});
while size(x0_matrix, 1) < size(x0_matrix, 2) + 1 % Num points < d + 1
    ix = randi(size(x0_matrix, 1));
    x0_matrix = [x0_matrix; x0_matrix(ix, :) + 0.05 .* (ub - lb) .* (2 .* rand(1, size(x0_matrix, 2)) - 1)];
end

if numel(Nmajor) > 1
    z0_vector = [vec(Nmajor); vec(Nmajor(randi(numel(Nmajor), size(x0_matrix, 1) - numel(Nmajor), 1)))];
    x0_matrix = [x0_matrix, z0_vector];
    lb = [lb, min(vec(Nmajor)) * ones(size(lb,1), 1)];
    ub = [ub, max(vec(Nmajor)) * ones(size(ub,1), 1)];
    integer_vars = 4;
else
    integer_vars = [];
end

% MISO optimization settings
miso_settings = {   ...
    500,            ... % Max iterations, integer (default: 50 * dimensions)
    'rbf_l',        ... % RBF surrogate type, string (default: 'rbf_c', cubic RBFs)
    [],             ... % Num. initial points, integer (default: 2 * (dimensions + 1))
    'own',          ... % Initial design, string (default: 'slhd')
    'cptv',         ... % Sample strategy, string (default: 'cptvl')
    x0_matrix,      ... % Partial or complete initial points, matrix (default: [])
    miso_filename   };  % Filename for saving miso results, string (default: '')

% MISO initialization function
miso_initfun = @() struct( ...,
    'xlow',         lb,                 ... % variable lower bounds
    'xup',          ub,                 ... % variable upper bounds
    'dim',          size(x0_matrix, 2),	... % problem dimesnion
    'integer',      integer_vars,       ... % indices of integer variables
    'continuous',   [1,2,3],            ... % indices of continuous variables
    'objfunction',  @(x) miso_call_fun(objfun, x) ); % wrapped objective function

% Call `miso` optimization routine
try
    [xbest, fbest, sol] = miso(miso_initfun, miso_settings{:});
catch e
    warning(e.message)
    sol = miso_remake_sol(miso_initfun, miso_settings);
    xbest = sol.xbest;
    fbest = sol.fbest;
end

% Plot resulting minimum
try
    sol.xfields = [sol.continuous, sol.integer];
    sol.xfieldnames = {'CA', 'Rminor', 'MinorExpansion', 'Nmajor'};
    sol.yfieldnames = {Normfun};
    miso_plot_surrogate(sol);
catch e
    warning(e.message);
end

% Save initial parameters
Params0 = struct('OptVariables', OptVariables, 'MISOSettings', miso_settings, ...
    'CA0', CA0, 'iBVF0', iBVF0, 'aBVF0', aBVF0, 'x0', x0, 'lb', lb, 'ub', ub);

% Generate text file of best simulation results
parse_iterations([datestr(now,30),'__','MISOIterationsOutput.txt']);

% Close diary
if ~isempty(DiaryFilename); diary(DiaryFilename); diary('off'); end

% ====================== Save resulting workspace ======================= %
% Clear anonymous functions which close over `Geom` and compress `Geom` for saving
Geom = Compress(Geom);
sol.objfunction = [];
clear objfun miso_initfun
save([datestr(now,30),'__','MISOResults'], '-v7');
