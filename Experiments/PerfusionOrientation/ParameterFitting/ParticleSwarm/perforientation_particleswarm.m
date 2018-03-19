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
% alpha_range = 7.5:10.0:87.5;
% alpha_range = 17.5:10.0:87.5;
% alpha_range = 7.5:20.0:87.5;
% alpha_range = [17.5, 32.5, 52.5, 67.5, 87.5];
% alpha_range = [37.5, 52.5, 72.5, 17.5, 87.5];
alpha_range = [2.5, 17.5, 27.5, 37.5, 47.5, 57.5, 67.5, 77.5, 82.5, 87.5];

% % ---- GRE w/ Diffusion Bounds (large minor) ---- %
% %       CA       iBVF          aBVF
% lb  = [ 3.000,   0.8000/100,   0.5000/100 ];
% ub  = [ 5.000,   1.5000/100,   1.2000/100 ];

% ---- GRE w/ Diffusion Bounds (small minor) ---- %
% %       CA       iBVF          aBVF
% lb  = [ 4.500,   1.1000/100,   0.4000/100 ];
% ub  = [ 5.750,   1.5000/100,   0.7000/100 ];

lb  = [ 3.000,   0.9000/100,   0.4000/100 ];
ub  = [ 6.000,   1.7500/100,   1.0000/100 ];

% lb  = [ 2.000,   0.4000/100,   0.4000/100 ];
% ub  = [ 8.000,   2.5000/100,   2.5000/100 ];

[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);

% % ---- SE w/ Diffusion Initial Guess ---- %
% %       CA        iBVF          aBVF
% lb  = [ 4.0000,   0.8000/100,   0.8000/100 ];
% ub  = [ 8.0000,   2.0000/100,   2.0000/100 ];
% 
% [alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize] = get_SE_data(alpha_range);

Nmajor = 3;
Rminor_mu = 7.0;
Rminor_sig = 0.0;
% Rminor_mu = 13.7;
% Rminor_sig = 2.1;

% type = 'SE';
% TE = 60e-3; VoxelSize = [3000,3000,3000]; VoxelCenter = [0,0,0]; GridSize = [512,512,512];
type = 'GRE';
% TE = 40e-3; VoxelSize = [1750,1750,1750]; VoxelCenter = [0,0,0]; GridSize = [350,350,350];
TE = 40e-3; VoxelSize = [3500,3500,3500]; VoxelCenter = [0,0,0]; GridSize = [700,700,700];

%with diffusion
Dcoeff = 3037; %[um^2/s]
order = 2;
Nsteps = 10;
NparticlesPerParam = 2;

%diffusionless
% Dcoeff = 0; %[um^2/s]
% order = 2;
% Nsteps = 1;
% NparticlesPerParam = 3;

B0 = -3.0; %[Tesla]
rng('default'); seed = rng; % for consistent geometries between sims.
Navgs  = 1; % for now, if geom seed is 'default', there is no point doing > 1 averages
MajorOrient = 'FixedPosition';

PlotFigs = true;
SaveFigs = true;
FigTypes = {'png'}; % outputs a lot of figures, so just 'png' is probably best
CloseFigs = true;
SaveResults = true;
StallTime_Days = 1.0; % max time without seeing an improvement in objective
MaxTime_Days = 4.0; % max time for full simulation

% Initial Swarm
linspace_fun = @linspacePeriodic; % lb < initial_param < ub
% linspace_fun = @linspace; % lb <= initial_param <= ub
CA_init   = linspace_fun(lb(1), ub(1), NparticlesPerParam);
iBVF_init = linspace_fun(lb(2), ub(2), NparticlesPerParam);
aBVF_init = linspace_fun(lb(3), ub(3), NparticlesPerParam);

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
    'StallTimeLimit', StallTime_Days*24*60*60, ... % StallTime_Days days [secs]
    'MaxTime', MaxTime_Days*24*60*60, ... % MaxTime_Days days [secs]
    'OutputFcn', {@pswplotranges, @pswplotvalues} ...
    );

% Call diary before the minimization starts
diary(DiaryFilename);

% Objective function
normfun = 'default';
% weights = 'uniform';
weights = BinCounts/sum(BinCounts);

optfun = @(x) perforientation_objfun( ...
    x, alpha_range, dR2_Data, [], weights, normfun, ...
    TE, type, VoxelSize, VoxelCenter, GridSize, ...
    B0, Dcoeff, Nsteps, Nmajor, Rminor_mu, Rminor_sig, ...
    'Navgs', Navgs, 'order', order, ...
    'PlotFigs', PlotFigs, 'SaveFigs', SaveFigs, 'CloseFigs', CloseFigs, 'FigTypes', FigTypes, ...
    'SaveResults', SaveResults, 'DiaryFilename', DiaryFilename, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'MajorOrientation', MajorOrient, 'geomseed', seed);

[x,objval,exitflag,output] = particleswarm(optfun,NVars,lb,ub,PSOpts);

CA = x(1);
iBVF = x(2);
aBVF = x(3);
BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
aRBVF = aBVF/BVF;

save([datestr(now,30),'__','ParticleSwarmFitResults'], '-v7');

%Go back to original directory
% cd(currentpath);

diary(DiaryFilename);
diary off

