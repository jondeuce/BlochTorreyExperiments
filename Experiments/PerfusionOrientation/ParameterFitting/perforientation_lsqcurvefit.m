%perforientation_lsqcurvefit

%Save a copy of this script in the directory of the caller
backupscript  = sprintf('%s__%s.m',datestr(now,30),mfilename);
currentscript = strcat(mfilename('fullpath'), '.m');
copyfile(currentscript,backupscript);

%Save a copy of perforientation_lsqoptfun in the directory of the caller
backupoptfun  = sprintf('%s__%s.m',datestr(now,30),'perforientation_lsqoptfun');
currentoptfun = which('perforientation_lsqoptfun');
copyfile(currentoptfun,backupoptfun);

%Save diary of workspace
DiaryFilename = [datestr(now,30), '__', 'diary.txt'];
diary(DiaryFilename);

%Save current directory and go to save path
% currentpath = cd;
% cd '/home/coopar7/Dropbox/Masters Year 1/UBCMRI/CurveFitting_Trial2_7umMinor'

% % ---- GRE w/ Diffusion Initial Guess ---- %
% lb  = [ 4.000,          1.0000/100,         0.8000/100 ];
% CA0 =   6.230;  iBVF0 = 2.2999/100; aBVF0 = 1.1601/100;
% ub  = [ 8.000,          3.5000/100,         1.5000/100 ];
% 
% x0 = [CA0, iBVF0, aBVF0];
% 
% % alpha_range = [17.5, 32.5, 52.5, 67.5, 87.5];
% alpha_range = 17.5:10.0:87.5;
% [alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);

% ---- SE w/ Diffusion Initial Guess ---- %
lb  = [ 4.0000,          0.8000/100,         0.8000/100 ];
CA0 =   6.4149;  iBVF0 = 1.1472/100; aBVF0 = 1.1748/100;
ub  = [ 8.0000,          2.0000/100,         2.0000/100 ];

x0  = [CA0, iBVF0, aBVF0];

% alpha_range = [37.5, 52.5, 72.5, 17.5, 87.5];
% alpha_range = 17.5:10.0:87.5;
alpha_range = 17.5:10.0:87.5;
[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize] = get_SE_data(alpha_range);

% Override some terms
% TE = 60e-3; VoxelSize = [3000,3000,3000]; VoxelCenter = [0,0,0]; GridSize = [512,512,512];
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [350,350,350];

VoxelSize = [3000,3000,3000];
GridSize = [512,512,512];
VoxelCenter = [0,0,0];
Nmajor = 9;
% Rminor_mu = 13.7;
% Rminor_sig = 2.1;
Rminor_mu = 7.0;
Rminor_sig = 0.5;

B0 = -3.0; %[Tesla]
Dcoeff = 3037; %[um^2/s]
order = 2;
Nsteps = 20/order;
rng('default'); seed = rng; % for consistent geometries between sims.
Navgs  = 1; % for now, if geom seed is 'default', there is no point doing > 1 averages
TE = 60e-3; %[s]
type = 'SE';
SaveFigs = true;
SaveIterResults = true;

% Limiting factor will always be MaxIter or MaxFunEvals, as due to
% simulation randomness, TolX/TolFun tend to not be reliable measures of 
% goodness of fit
opts = optimoptions('lsqcurvefit', ...
    'MaxFunEvals', 500, ...
    'Algorithm', 'trust-region-reflective', ...
    'MaxIter', 2, ...
    'TolX', 1e-12, ...
    'TolFun', 1e-12, ...
    'TypicalX', x0, ...
    'FinDiffRelStep', [0.03, 0.05, 0.05], ...
    'Display', 'iter' ...
    );

% Call diary before the minimization starts
diary(DiaryFilename);

optfun = @(x,xdata) perforientation_lsqoptfun(x, xdata, ...
    dR2_Data, TE, VoxelSize, VoxelCenter, GridSize,     ...
    B0, Dcoeff, Nsteps, Nmajor, Rminor_mu, Rminor_sig,  ...
    Navgs, type, order, SaveFigs, SaveIterResults, DiaryFilename, seed);

[x,resnorm,residual,exitflag,output] = lsqcurvefit(optfun, ...
    x0, alpha_range, dR2_Data, lb, ub, opts);

Params0 = struct('CA0',CA0,'iBVF0',iBVF0,'aBVF0',aBVF0,'lb',lb,'ub',ub);

CA = x(1);
iBVF = x(2);
aBVF = x(3);
BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
aRBVF = aBVF/BVF;

save([datestr(now,30),'__','LsqcurvefitResults']);

%Go back to original directory
% cd(currentpath);

diary(DiaryFilename);
diary off

