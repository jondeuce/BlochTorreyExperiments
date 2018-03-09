%perforientation_particleswarm

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
diary(DiaryFilename);

%Save current directory and go to save path
% currentpath = cd;
% cd '/home/coopar7/Dropbox/Masters Year 1/UBCMRI/CurveFitting_Trial2_7umMinor'

% ---- Angles to simulate ---- %
% alpha_range = 2.5:5.0:87.5;
alpha_range = 7.5:10.0:87.5;
% alpha_range = 17.5:10.0:87.5;
% alpha_range = 7.5:20.0:87.5;
% alpha_range = [17.5, 32.5, 52.5, 67.5, 87.5];
% alpha_range = [37.5, 52.5, 72.5, 17.5, 87.5];

% % ---- GRE w/ Diffusion Bounds (large minor) ---- %
% %       CA       iBVF          aBVF
% lb  = [ 3.000,   0.8000/100,   0.5000/100 ];
% ub  = [ 5.000,   1.5000/100,   1.2000/100 ];

% ---- GRE w/ Diffusion Bounds (small minor) ---- %
%       CA       iBVF          aBVF
lb  = [ 3.000,   0.8000/100,   0.5000/100 ];
ub  = [ 5.000,   1.8000/100,   1.5000/100 ];

[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);

% % ---- SE w/ Diffusion Initial Guess ---- %
% %       CA        iBVF          aBVF
% lb  = [ 4.0000,   0.8000/100,   0.8000/100 ];
% ub  = [ 8.0000,   2.0000/100,   2.0000/100 ];
% 
% [alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize] = get_SE_data(alpha_range);

% Override some terms
% TE = 60e-3; VoxelSize = [3000,3000,3000]; VoxelCenter = [0,0,0]; GridSize = [512,512,512];
TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [350,350,350];

VoxelCenter = [0,0,0];
Nmajor = 4;
% Rminor_mu = 13.7;
% Rminor_sig = 2.1;
Rminor_mu = 7.0;
Rminor_sig = 0.0;

B0 = -3.0; %[Tesla]
Dcoeff = 3037; %[um^2/s]
order = 2;
Nsteps = 8;
rng('default'); seed = rng; % for consistent geometries between sims.
Navgs  = 1; % for now, if geom seed is 'default', there is no point doing > 1 averages
% type = 'SE';
type = 'GRE';
MajorOrient = 'FixedPosition';
PlotFigs = true;
SaveFigs = true;
SaveResults = true;

% Initial Swarm
CA_init   = linspacePeriodic(lb(1), ub(1), 2);
iBVF_init = linspacePeriodic(lb(2), ub(2), 2);
aBVF_init = linspacePeriodic(lb(3), ub(3), 2);

[CA_mesh, iBVF_mesh, aBVF_mesh] = meshgrid(CA_init, iBVF_init, aBVF_init);

InitSwarm = [CA_mesh(:), iBVF_mesh(:), aBVF_mesh(:)];
SwarmSize = size(InitSwarm, 1);
NVars     = size(InitSwarm, 2);

Params_init = struct('CA_init',CA_init,'iBVF_init',iBVF_init,'aBVF_init',aBVF_init,'lb',lb,'ub',ub);

% Limiting factor should just be MaxTime
PSOpts = optimoptions(@particleswarm, ...
    'Display', 'iter', ...
    'SwarmSize', SwarmSize, ...
    'InitialSwarm', InitSwarm, ...
    'StallTimeLimit', 1*24*60*60, ... % one day
    'MaxTime', 1*24*60*60, ... % four days
    'OutputFcn', {@pswplotranges, @pswplotvalues} ...
    );

% Call diary before the minimization starts
diary(DiaryFilename);

% Objective function
weights = 'uniform';
normfun = 'default';

optfun = @(x) perforientation_objfun( ...
    x, alpha_range, dR2_Data, weights, normfun, ...
    TE, type, VoxelSize, VoxelCenter, GridSize, ...
    B0, Dcoeff, Nsteps, Nmajor, Rminor_mu, Rminor_sig, ...
    'Navgs', Navgs, 'order', order, ...
    'PlotFigs', PlotFigs, 'SaveFigs', SaveFigs, ...
    'SaveResults', SaveResults, 'DiaryFilename', DiaryFilename, ...
    'MajorOrientation', MajorOrient, 'geomseed', seed);

[x,objval,exitflag,output] = particleswarm(optfun,NVars,lb,ub,PSOpts);

CA = x(1);
iBVF = x(2);
aBVF = x(3);
BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
aRBVF = aBVF/BVF;

save([datestr(now,30),'__','ParticleSwarmFitResults']);

%Go back to original directory
% cd(currentpath);

diary(DiaryFilename);
diary off

