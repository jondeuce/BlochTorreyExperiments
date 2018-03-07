function test()

%% Test parameters
% Glen = 256; % Grid side length [voxels]
% Vlen = 3000; % Voxel side length [um]
% Scale = Vlen/Glen;
% TypicalScale = 3000/512; % Typical scale [um/voxel]
% Gsize = Glen * [1,1,1];
% Vsize = Vlen * [1,1,1];

ScaleGsize = 2;
Gsize = [350,350,800] / ScaleGsize;
Vsize = [1750,1750,4000];
TypicalScale = 4000/800;
Scale = Vsize(3)/Gsize(3);

t = 60e-3;
% type = 'gre';
type = 'se';
Dcoeff = 3037 * (Scale/TypicalScale)^2; % Scale diffusion to mimic [3000 um]^3 512^3 grid

% Rminor_mu  = 25;
% Rminor_sig = 0;
% Rminor_mu  = 13.7;
% Rminor_sig = 2.1;
Rminor_mu  = 7;
Rminor_sig = 0;

iBVF = 1.1803/100;
aBVF = 1.3425/100;
Nmajor = 4; % Number of major vessels (optimal number is from SE perf. orientation. sim)
MajorAngle = 45; % Angle of major vessels
NumMajorArteries = 1; % Number of major arteries
MinorArterialFrac = 1/3; % Fraction of minor vessels which are arteries
rng('default'); seed = rng;

%% Calculate Geometry
GammaSettings = Geometry.ComplexDecaySettings('Angle_Deg', 90, 'B0', -3);
Geom = Geometry.CylindricalVesselFilledVoxel( ...
    'iBVF', iBVF, 'aBVF', aBVF, ...
    'VoxelSize', Vsize, 'GridSize', Gsize, 'VoxelCenter', [0,0,0], ...
    'Nmajor', Nmajor, 'MajorAngle', MajorAngle, ...
    'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
    'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', false, ...
    'PopulateIdx', true, 'seed', seed );

%% Calculate ComplexDecay
Gamma = CalculateComplexDecay(GammaSettings, Geom);
dGamma = {};

%% Initial data
% x0 = randnc(Gsize);
x0 = 1i*ones(Gsize);

%% Calculate exact solution via expmv
t_expmv = tic;

A = BlochTorreyOp(Gamma, Dcoeff, Gsize, Vsize);
x = bt_expmv( t, A, x0, 'prnt', false, 'type', type );
S = sum(sum(sum(x,1),2),3); %more accurate than sum(x(:))

t_expmv = toc(t_expmv);
display_toc_time(t_expmv,'expmv');

%% BTStepper Test Loop
xnorm = norm(vec(x));
xmax = maxabs(x);
blank_rho = struct('order',[],'nrep',[],'h',[],'relerr',[],'ratio',[],'rmaxerr',[],'rsumerr',[],'isumerr',[],'time',[]);
rho = [];
for steporder = [2,4]
    if strcmpi(type, 'gre')
        if steporder == 2; nreps = 2.^(0:6); end % for GRE
        if steporder == 4; nreps = 2.^(0:4); end % for GRE
    else
        if steporder == 2; nreps = 2.^(1:6); end % for SE
        if steporder == 4; nreps = 2.^(1:4); end % for SE
    end
    err_last = 0.0;
    for ii = 1:numel(nreps)
        nrep = nreps(ii);
        t_step = tic;
        
        dt = t/nrep;
        if strcmpi(type, 'gre')
            Vsub = SplittingMethods.BTSplitStepper( dt, Dcoeff, Gamma, dGamma, Gsize, Vsize, ...
                'Order', steporder, 'Nreps', nrep ); % for GRE
            xh = step(Vsub,x0);
        else
            Vsub = SplittingMethods.BTSplitStepper( dt, Dcoeff, Gamma, dGamma, Gsize, Vsize, ...
                'Order', steporder, 'Nreps', nrep/2 ); % for SE
            xh = step(Vsub,x0);
            xh = conj(xh);
            xh = step(Vsub,xh);
        end
        
        Sh = sum(sum(sum(xh,1),2),3); % get signal
        
        t_step = toc(t_step);
        
        relerr  = norm(vec(x-xh))/xnorm;
        ratio   = err_last/relerr;
        rmaxerr = maxabs(x-xh)/xmax;
        rsumerr = abs(real(S-Sh))/abs(real(S));
        isumerr = abs(imag(S-Sh))/abs(imag(S));
        
        rho = cat(1,rho,blank_rho);
        rho(end).order = steporder;
        rho(end).nrep = nrep;
        rho(end).h = dt;
        rho(end).relerr = relerr;
        rho(end).ratio = ratio;
        rho(end).rmaxerr = rmaxerr;
        rho(end).rsumerr = rsumerr;
        rho(end).isumerr = isumerr;
        rho(end).time = t_step;
        
        str = sprintf('order = %d, nrep = %2d, h = %1.3e, relerr = %1.3e, ratio = %1.3e, relmax = %1.3e, rsumerr = %1.3e, isumerr = %1.3e', ...
            rho(end).order, rho(end).nrep, rho(end).h, rho(end).relerr, rho(end).ratio, rho(end).rmaxerr, rho(end).rsumerr, rho(end).isumerr);
        display_toc_time(t_step,str);
        err_last = relerr;
    end
    fprintf('\n');
end

%% expmv Test Loop
xnorm = norm(vec(x));
xmax = maxabs(x);
blank_res = struct('prec',[],'nrep',[],'h',[],'relerr',[],'ratio',[],'rmaxerr',[],'rsumerr',[],'isumerr',[],'time',[]);
res = [];
for prec = {'half','single'} %,'double'}
    err_last = 0.0;
    if strcmpi(type,'gre')
        nreps = 2.^(0:3); % for gre
    else
        nreps = 2.^(1:3); % for SE
    end
    for ii = 1:numel(nreps)
        nrep = nreps(ii);
        t_step = tic;
        
        dt = t/nrep;
        xh = bt_expmv_nsteps( dt, A, x0, nrep, 'calcsignal', 'none', 'type', type, 'prec', prec{1}, 'prnt', false );
        Sh = sum(sum(sum(xh,1),2),3);
        
        t_step = toc(t_step);
        
        relerr  = norm(vec(x-xh))/xnorm;
        ratio   = err_last/relerr;
        rmaxerr = maxabs(x-xh)/xmax;
        rsumerr = abs(real(S-Sh))/abs(real(S));
        isumerr = abs(imag(S-Sh))/abs(imag(S));
        
        res = cat(1,res,blank_res);
        res(end).prec = prec{1};
        res(end).nrep = nrep;
        res(end).h = dt;
        res(end).relerr = relerr;
        res(end).ratio = ratio;
        res(end).rmaxerr = rmaxerr;
        res(end).rsumerr = rsumerr;
        res(end).isumerr = isumerr;
        res(end).time = t_step;
        
        str = sprintf('prec = %s, nrep = %2d, h = %1.3e, relerr = %1.3e, ratio = %1.3e, relmax = %1.3e, rsumerr = %1.3e, isumerr = %1.3e', ...
            res(end).prec, res(end).nrep, res(end).h, res(end).relerr, res(end).ratio, res(end).rmaxerr, res(end).rsumerr, res(end).isumerr);
        display_toc_time(t_step,str);
        err_last = relerr;
    end
    fprintf('\n');
end

end

function print_rho(rho)

for ii = 1:length(rho)
    str = sprintf('order = %d, nrep = %2d, h = %1.3e, relerr = %1.3e, ratio = %1.3e, relmax = %1.3e, rsumerr = %1.3e, isumerr = %1.3e', ...
        rho(ii).order, rho(ii).nrep, rho(ii).h, rho(ii).relerr, rho(ii).ratio, rho(ii).rmaxerr, rho(ii).rsumerr, rho(ii).isumerr);
    display_toc_time(rho(ii).time,str);
end

end

%% Curie: SE BTStepper times for 7um, 512^3 grid, TE = 60ms, Nmajor = 4, 1/3 arteries
% Elapsed time (order = 2, nrep =  2, h = 3.000e-02, relerr = 6.523e-02, ratio = 0.000e+00, relmax = 8.945e-01, rsumerr = 1.041e+00, isumerr = 1.268e-03):	09.601 secs
% Elapsed time (order = 2, nrep =  4, h = 1.500e-02, relerr = 2.150e-02, ratio = 3.034e+00, relmax = 3.263e-01, rsumerr = 6.241e-01, isumerr = 2.579e-03):	14.770 secs
% Elapsed time (order = 2, nrep =  8, h = 7.500e-03, relerr = 5.634e-03, ratio = 3.816e+00, relmax = 1.011e-01, rsumerr = 1.353e-01, isumerr = 8.538e-04):	24.805 secs
% Elapsed time (order = 2, nrep = 16, h = 3.750e-03, relerr = 2.614e-03, ratio = 2.156e+00, relmax = 4.082e-02, rsumerr = 3.952e-01, isumerr = 2.833e-04):	43.873 secs
% Elapsed time (order = 2, nrep = 32, h = 1.875e-03, relerr = 2.107e-03, ratio = 1.240e+00, relmax = 3.418e-02, rsumerr = 2.745e-01, isumerr = 4.807e-04):	01:23.145 mins
% Elapsed time (order = 2, nrep = 64, h = 9.375e-04, relerr = 2.340e-02, ratio = 9.007e-02, relmax = 2.195e-01, rsumerr = 6.121e-01, isumerr = 4.112e-04):	02:40.699 mins
% 
% Elapsed time (order = 4, nrep =  2, h = 3.000e-02, relerr = 1.216e-02, ratio = 0.000e+00, relmax = 2.156e-01, rsumerr = 1.023e+00, isumerr = 1.136e-03):	21.522 secs
% Elapsed time (order = 4, nrep =  4, h = 1.500e-02, relerr = 4.259e-03, ratio = 2.856e+00, relmax = 9.052e-02, rsumerr = 2.103e+00, isumerr = 3.600e-04):	37.086 secs
% Elapsed time (order = 4, nrep =  8, h = 7.500e-03, relerr = 4.642e-03, ratio = 9.175e-01, relmax = 6.858e-02, rsumerr = 9.699e-03, isumerr = 7.995e-04):	01:06.751 mins
% Elapsed time (order = 4, nrep = 16, h = 3.750e-03, relerr = 2.209e-03, ratio = 2.101e+00, relmax = 1.685e-02, rsumerr = 7.293e-01, isumerr = 2.089e-04):	02:05.645 mins

%% Tlaloc: GRE BTStepper times for 7um, 512^3 grid, TE = 60ms, Nmajor = 4, 1/3 arteries
% Elapsed time (total cylinder construction time):	02:03.588 mins
% Elapsed time (expmv):	05:46.225 mins
% Elapsed time (order = 2, nrep =  1, h = 6.000e-02, relerr = 1.308e-01, ratio = 0.000e+00, relmax = 1.458e+00, rsumerr = 6.506e-02, isumerr = 1.734e-02):	07.508 secs
% Elapsed time (order = 2, nrep =  2, h = 3.000e-02, relerr = 5.685e-02, ratio = 2.302e+00, relmax = 7.499e-01, rsumerr = 4.783e-02, isumerr = 1.128e-02):	12.755 secs
% Elapsed time (order = 2, nrep =  4, h = 1.500e-02, relerr = 2.184e-02, ratio = 2.603e+00, relmax = 3.067e-01, rsumerr = 1.425e-02, isumerr = 4.868e-03):	18.474 secs
% Elapsed time (order = 2, nrep =  8, h = 7.500e-03, relerr = 5.616e-03, ratio = 3.890e+00, relmax = 9.406e-02, rsumerr = 3.122e-03, isumerr = 9.615e-04):	33.385 secs
% Elapsed time (order = 2, nrep = 16, h = 3.750e-03, relerr = 3.483e-03, ratio = 1.612e+00, relmax = 4.114e-02, rsumerr = 3.252e-04, isumerr = 5.949e-04):	01:03.227 mins
% Elapsed time (order = 2, nrep = 32, h = 1.875e-03, relerr = 2.632e-03, ratio = 1.324e+00, relmax = 3.660e-02, rsumerr = 2.209e-04, isumerr = 3.921e-04):	02:00.433 mins
% Elapsed time (order = 2, nrep = 64, h = 9.375e-04, relerr = 4.841e-02, ratio = 5.437e-02, relmax = 3.085e-01, rsumerr = 9.749e-03, isumerr = 6.786e-03):	03:57.738 mins
% 
% Elapsed time (order = 4, nrep =  1, h = 6.000e-02, relerr = 3.721e-02, ratio = 0.000e+00, relmax = 5.670e-01, rsumerr = 5.760e-02, isumerr = 6.164e-03):	16.611 secs
% Elapsed time (order = 4, nrep =  2, h = 3.000e-02, relerr = 1.180e-02, ratio = 3.155e+00, relmax = 1.983e-01, rsumerr = 2.101e-02, isumerr = 1.202e-03):	32.609 secs
% Elapsed time (order = 4, nrep =  4, h = 1.500e-02, relerr = 4.922e-03, ratio = 2.396e+00, relmax = 9.030e-02, rsumerr = 5.918e-03, isumerr = 7.389e-04):	01:27.528 mins
% Elapsed time (order = 4, nrep =  8, h = 7.500e-03, relerr = 5.651e-03, ratio = 8.710e-01, relmax = 6.700e-02, rsumerr = 5.581e-03, isumerr = 1.260e-03):	02:55.230 mins

%% Curie: SE expmv times for 7um, 512^3 grid, TE = 60ms, Nmajor = 4, 1/3 arteries
% Elapsed time (prec = half, nrep =  2, h = 3.000e-02, relerr = 3.445e-04, ratio = 0.000e+00, relmax = 3.866e-04, rsumerr = 1.895e-01, isumerr = 3.398e-04):	01:59.704 mins
% Elapsed time (prec = half, nrep =  4, h = 1.500e-02, relerr = 3.445e-04, ratio = 1.000e+00, relmax = 3.866e-04, rsumerr = 1.895e-01, isumerr = 3.398e-04):	02:03.025 mins

%% Curie: GRE expmv times for 7um, 512^3 grid, TE = 60ms, Nmajor = 4, 1/3 arteries
% Elapsed time (pres = half,   nrep =  1, h = 6.000e-02, relerr = 3.490e-04, ratio = 0.000e+00, relmax = 3.867e-04, rsumerr = 5.059e-04, isumerr = 3.339e-04):	02:01.454 mins
% Elapsed time (pres = half,   nrep =  2, h = 3.000e-02, relerr = 3.490e-04, ratio = 1.000e+00, relmax = 3.867e-04, rsumerr = 5.059e-04, isumerr = 3.339e-04):	02:03.940 mins
% Elapsed time (pres = half,   nrep =  4, h = 1.500e-02, relerr = 3.490e-04, ratio = 1.000e+00, relmax = 3.867e-04, rsumerr = 5.059e-04, isumerr = 3.339e-04):	02:04.582 mins
% Elapsed time (pres = half,   nrep =  8, h = 7.500e-03, relerr = 5.235e-04, ratio = 6.667e-01, relmax = 5.663e-04, rsumerr = 7.128e-04, isumerr = 5.063e-04):	02:45.233 mins
% Elapsed time (pres = half,   nrep = 16, h = 3.750e-03, relerr = 1.264e-04, ratio = 4.142e+00, relmax = 1.355e-04, rsumerr = 1.675e-04, isumerr = 1.227e-04):	04:13.569 mins
% 
% Elapsed time (pres = single, nrep =  1, h = 6.000e-02, relerr = 7.437e-09, ratio = 0.000e+00, relmax = 8.815e-09, rsumerr = 1.223e-08, isumerr = 6.906e-09):	02:56.249 mins
% Elapsed time (pres = single, nrep =  2, h = 3.000e-02, relerr = 7.437e-09, ratio = 1.000e+00, relmax = 8.815e-09, rsumerr = 1.223e-08, isumerr = 6.906e-09):	02:55.727 mins
% Elapsed time (pres = single, nrep =  4, h = 1.500e-02, relerr = 7.437e-09, ratio = 1.000e+00, relmax = 8.815e-09, rsumerr = 1.223e-08, isumerr = 6.906e-09):	02:56.413 mins
% Elapsed time (pres = single, nrep =  8, h = 7.500e-03, relerr = 1.399e-08, ratio = 5.314e-01, relmax = 1.595e-08, rsumerr = 2.151e-08, isumerr = 1.323e-08):	04:06.856 mins
% Elapsed time (pres = single, nrep = 16, h = 3.750e-03, relerr = 7.105e-09, ratio = 1.970e+00, relmax = 7.911e-09, rsumerr = 1.039e-08, isumerr = 6.788e-09):	06:12.311 mins
% 
% Elapsed time (pres = double, nrep =  1, h = 6.000e-02, relerr = 0.000e+00, ratio = NaN, relmax = 0.000e+00, rsumerr = 0.000e+00, isumerr = 0.000e+00):	04:46.923 mins
