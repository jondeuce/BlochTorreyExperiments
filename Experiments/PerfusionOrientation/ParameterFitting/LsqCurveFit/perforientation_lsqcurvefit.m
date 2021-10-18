%perforientation_lsqcurvefit

%Save a copy of this script in the directory of the caller
backupscript  = sprintf('%s__%s.m',datestr(now,30),mfilename);
currentscript = strcat(mfilename('fullpath'), '.m');
copyfile(currentscript,backupscript);

%Save a copy of perforientation_fun in the directory of the caller
backupoptfun  = sprintf('%s__%s.m',datestr(now,30),'perforientation_fun');
currentoptfun = which('perforientation_fun');
copyfile(currentoptfun,backupoptfun);

%Save diary of workspace
DiaryFilename = [datestr(now,30), '__', 'diary.txt'];
% DiaryFilename = '';
if ~isempty(DiaryFilename)
    diary(DiaryFilename);
else
    display_text('WARNING: Not recording prompt output to diary', 75, '=', true, [true,true]);
    input('[Press enter to continue...]\n\n')
end

% ---- Angles to simulate ---- %
% alpha_range = [2.5, 47.5, 87.5];
alpha_range = 2.5:5.0:87.5;
% alpha_range = 22.5:5.0:87.5;
% alpha_range = 7.5:10.0:87.5;
% alpha_range = 17.5:10.0:87.5;
% alpha_range = 7.5:20.0:87.5;
% alpha_range = [17.5, 32.5, 52.5, 67.5, 87.5];
% alpha_range = [37.5, 52.5, 72.5, 17.5, 87.5];
% alpha_range = [2.5, 17.5, 27.5, 37.5, 47.5, 57.5, 67.5, 77.5, 82.5, 87.5];

% =============================== DATA ================================== %

% % ---- GRE w/ Diffusion Initial Guess (large minor) ---- %
% lb  = [ 4.000,          1.0000/100,         0.8000/100 ];
% CA0 =   6.230;  iBVF0 = 2.2999/100; aBVF0 = 1.1601/100;
% ub  = [ 8.000,          3.5000/100,         1.5000/100 ];

% ---- GRE w/ Diffusion Initial Guess (small minor) ---- %
lb  = [ 2.000,          0.3500/100,         0.3500/100 ];
CA0 =   4.520;  iBVF0 = 1.4200/100; aBVF0 = 0.6840/100;
ub  = [ 9.000,          2.5000/100,         2.5000/100 ];

x0 = [CA0, iBVF0, aBVF0];

type = 'GRE';
[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);
TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [350,350,350];
Weights = BinCounts / sum(BinCounts(:));

% % ---- SE w/ Diffusion Initial Guess ---- %
% lb  = [ 1.0000,          0.5000/100,         0.5000/100 ];
% CA0 =   3.7152;  iBVF0 = 1.6334/100; aBVF0 = 1.0546/100;
% ub  = [ 8.0000,          2.5000/100,         2.5000/100 ];
% 
% x0  = [CA0, iBVF0, aBVF0];
% 
% type = 'SE';
% [alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize] = get_SE_data(alpha_range);
% 
% % TE = 60e-3; VoxelSize = [3000,3000,3000]; VoxelCenter = [0,0,0]; GridSize = [512,512,512];

% ======================== BLOCH-TORREY SETTINGS ======================== %

Nmajor = 7;
Rminor_mu = 7.0;
Rminor_sig = 0.0;
% Rminor_mu = 13.7;
% Rminor_sig = 2.1;

% ---- With Diffusion ---- %
D_Tissue = 3037; %[um^2/s]
D_Blood = 3037; %[um^2/s]
D_VRS = 3037; %[um^2/s]
Nsteps = 8;
StepperArgs = struct('Stepper', 'BTSplitStepper', 'Order', 2);
% Nsteps = 2;
% StepperArgs = struct('Stepper', 'ExpmvStepper', 'prec', 'half', 'full_term', false, 'prnt', false);

% % ---- Diffusionless ---- %
% D_Tissue = 0; %[um^2/s]
% D_Blood = []; %[um^2/s]
% D_VRS = []; %[um^2/s]
% Nsteps = 1; % one exact step
% StepperArgs = struct('Stepper', 'BTSplitStepper', 'Order', 2);

B0 = -3.0; %[Tesla]
rng('default'); seed = rng; % for consistent geometries between sims.
Navgs = 1; % for now, if geom seed is 'default', there is no point doing > 1 averages
RotateGeom = false; % geometry is fixed; dipole rotates
MajorAngle = 0.0; % major vessel angle w.r.t z-axis [degrees]
NumMajorArteries = 0;
MinorArterialFrac = 0.0;
VRSRelativeRad = 2; % Radius of Virchow-Robin space relative to major vessel radius [unitless] => 3X volume (see below)
% VRSRelativeRad = sqrt(5/2); % VRS space volume is approx (relrad^2-1)*BVF, so sqrt(5/2) => 1.5X
% VRSRelativeRad = sqrt(3); % VRS space volume is approx (relrad^2-1)*BVF, so sqrt(3) => 2X

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
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'ImproveMajorBVF', true, 'ImproveMinorBVF', true, ...
    'PopulateIdx', true, 'seed', seed );

% Generate initial guess and geometry
switch upper(OptVariables)
    case 'CA_IBVF_ABVF'
        Geom = []; % Geom is created inside perforientation_fun
    case 'CA_RMAJOR_MINOREXPANSION'
        GeomArgs.ImproveMajorBVF = false;
        GeomArgs.ImproveMinorBVF = false;
        
        % Generate geometry for contraction
        GeomArgs.iBVF = ub(2);
        GeomArgs.aBVF = ub(3);
        GeomNameValueArgs = struct2arglist(GeomArgs);
        Geom = Geometry.CylindricalVesselFilledVoxel( GeomNameValueArgs{:} );
        
        % Generate new initial guesses and bounds
        getRmajor0 = @(aBVF) sqrt( prod(VoxelSize) * aBVF / ( Nmajor * pi * rayBoxIntersectionLength( VoxelCenter(:), [sind(MajorAngle); 0; cosd(MajorAngle)], VoxelSize(:), VoxelCenter(:) ) ) );
        getSpaceFactor0 = @(iBVF) (Geom.iBVF/iBVF)^(1/2); % fractal dimension of infinite cylinders is two: iBVF = iBVF_max * SpaceFactor^(-2)
        
        lb_old = lb;
        ub_old = ub;
        lb = [lb_old(1), getRmajor0(lb_old(3)), getSpaceFactor0(ub_old(2))];
        x0 = [x0(1),     getRmajor0(aBVF0),     getSpaceFactor0(iBVF0)];
        ub = [ub_old(1), getRmajor0(ub_old(3)), getSpaceFactor0(lb_old(2))];
        
    otherwise
        error('''OptVariables'' must be ''CA_iBVF_aBVF'' or ''CA_Rmajor_MinorExpansion''');
end

% =========================== OPTIMIZATION ============================== %

% Call lsqcurvefit or lsqnonlin (essentially the same function; lsqcurvefit
% is a special case of lsqnonlin, but it doesn't accept weight vectors).
% Also, lsqnonlin functions must output the residual, where lsqcurvefit is
% the simulated function values themselves.
OptFunction = 'lsqnonlin';
% OptFunction = 'lsqcurvefit';

% Limiting factor will always be MaxIter or MaxFunEvals, as due to
% simulation randomness, TolX/TolFun tend to not be reliable measures of 
% goodness of fit
LsqOpts = optimoptions(OptFunction, ...
    'MaxFunEvals', 500, ...
    'Algorithm', 'trust-region-reflective', ...
    'MaxIter', 15, ...
    'TolX', 1e-12, ...
    'TolFun', 1e-12, ...
    'TypicalX', x0, ...
    'FinDiffRelStep', [0.01, 0.01, 0.01], ...
    'Display', 'iter' ...
    );

% Call diary before the minimization starts
if ~isempty(DiaryFilename); diary(DiaryFilename); end

% Simulation function handle
optfun = @(x,xdata) perforientation_fun(x, xdata, dR2_Data, ...
    TE, Nsteps, type, B0, D_Tissue, D_Blood, D_VRS, ...
    'OptVariables', OptVariables, ...
    'Navgs', Navgs, 'StepperArgs', StepperArgs, ...
    'Weights', Weights, 'Normfun', 'L2w', ...
    'PlotFigs', PlotFigs, 'SaveFigs', SaveFigs, 'CloseFigs', CloseFigs, 'FigTypes', FigTypes, ...
    'SaveResults', SaveResults, 'DiaryFilename', DiaryFilename, ...
    'GeomArgs', GeomArgs, 'Geom', Geom);

% Weighted residual function handle
resfun = @(x) sqrt(Weights) .* (optfun(x, alpha_range) - dR2_Data);

% Call optimization routine
switch upper(OptFunction)
    case 'LSQCURVEFIT'
        [x,resnorm,residual,exitflag,output,lambda,jacobian] = ...
            lsqcurvefit(optfun, x0, alpha_range, dR2_Data, lb, ub, LsqOpts);
    case 'LSQNONLIN'
        [x,resnorm,residual,exitflag,output,lambda,jacobian] = ...
            lsqnonlin(resfun, x0, lb, ub, LsqOpts);
    otherwise
        error('OptFunction must be ''lsqcurvefit'' or ''lsqnonlin''');
end

Params0 = struct('OptVariables', OptVariables,...
    'CA0', CA0, 'iBVF0', iBVF0, 'aBVF0', aBVF0,...
    'x0', x0, 'lb', lb, 'ub', ub);

% Go back to original directory
% cd(currentpath);

if ~isempty(DiaryFilename); diary(DiaryFilename); diary('off'); end

% =========== Generate text file of best simulation results ============= %

fout = fopen([datestr(now,30),'__','LsqfitIterationsOutput.txt'], 'w');
iter = 1;
L2w_best = Inf;
fprintf(fout, '%s', 'Timestamp       f-count            f(x)       Best f(x)');
for s = dir('*.mat')'
    try
        Results = load(s.name);
        Results = Results.Results;
        f = perforientation_objfun(Results.params, Results.alpha_range, Results.dR2_Data, Results.dR2, Results.args.Weights);
        L2w_best = min(f, L2w_best);
        fprintf(fout, '\n%s%8d%16.8f%16.8f', s.name(1:15), iter, f, L2w_best);
        iter = iter + 1;
    catch me
        warning(me.message);
    end
end
fclose(fout);
clear fout iter Results f

% Save resulting workspace
if ~isempty(Geom)
    Geom = Compress(Geom);
    clear optfun resfun getRmajor0 getSpaceFactor0
end
save([datestr(now,30),'__','LsqcurvefitResults'], '-v7');
