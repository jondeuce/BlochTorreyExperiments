function [ U ] = testblochtorrey2D(varargin)
%TESTBLOCHTORREY2D [ U ] = testblochtorrey2D(varargin)
% 
% Example usage:
%   U = testblochtorrey2D('D_Axon', 1, 'D_Tissue', 1, 'D_Sheath', 1, 'Plot', true);

opts = parseinputs(varargin{:});
assert(opts.D_Axon == opts.D_Tissue && opts.D_Tissue == opts.D_Sheath);

D = opts.D_Tissue;
N = opts.Npts;
a = opts.Domain(1);
b = opts.Domain(2);
h = (b-a)/(N-1);
pts = linspace(a,b,N);
[X,Y] = meshgrid(pts);

Gamma = vec(complex(r2decay(X,Y,opts), omega(X,Y,opts)));
[~,~,negL] = laplacianSparseOp([N,N],{'NN','NN'});
A = -D/h^2 * negL - spdiags(Gamma, 0, N^2, N^2);

T = opts.Time;
u0 = 1i * ones(N^2,1);
u = expmv(T,A,u0);
U = reshape(u,N,N);

unwrap = @(x) sliceND(unwrapLap(repmat(x,1,1,11)),6,3);
figure, imagesc(rot90(unwrap(angle(U)))); axis image; title phase; colorbar
figure, imagesc(rot90(abs(U))); axis image; title magnitude; colorbar

end

function [ w ] = omega(x,y,opts)
%OMEGA [ w ] = omega(x,y,opts)

B0    = opts.B0;    % -3.0;         % External magnetic field (z-direction) [T]
gamma = opts.gamma; % 2.67515255e8; % Gyromagnetic ratio [rad/s/T]
w0    = gamma * B0; % Resonance frequency [rad/s]
th    = opts.theta; % pi/2;   % Main magnetic field angle w.r.t B0 [rad]
c2    = cos(th)^2;
s2    = sin(th)^2;
ChiI  = opts.ChiI; % -60e-9;  % Isotropic susceptibility of myelin [ppb] (check how to get it) (Xu et al. 2017)
ChiA  = opts.ChiA; % -120e-9; % Anisotropic Susceptibility of myelin [ppb] (Xu et al. 2017)
E     = opts.E;    %  10e-9;  % Exchange component to resonance freqeuency [ppb] (Wharton and Bowtell 2012)

g  = opts.g_ratio; % 0.8;
ro = opts.R_outer_rel * opts.R_mu; %0.5;
ri = g*ro;
ri2 = ri^2;
ro2 = ro^2;

r2 = x.^2 + y.^2;
r = sqrt(r2);
t = atan2(y,x);
w = zeros(size(x));

b = (r < ri);
w(b) = w0 * ChiA * 3*s2/4 * log(ro/ri);

b = (ri <= r & r <= ro);
w(b) = ...
    w0 * ChiI * (1/2) * (c2 - 1/3 - s2 * cos(2*t(b)) .* (ri2./r2(b))) + ...
    w0 * E + ...
    w0 * ChiA * (s2 * (-5/12 - cos(2*t(b))/8 .* (1+ri2./r2(b)) + (3/4) * log(ro./r(b))) - c2/6);

b = (ro < r);
w(b) = ...
    w0 * ChiI * (s2/2) * cos(2*t(b)) .* (ro2 - ri2) ./ r2(b) + ...
    w0 * ChiA * (s2/8) * cos(2*t(b)) .* (ro2 - ri2) ./ r2(b);

end

function [ R2 ] = r2decay(x,y,opts)
%R2DECAY [ R2 ] = r2decay(x,y,opts)

R2_sp = opts.R2_sp; % 1/15e-3; % Relaxation rate of small pool [s^-1] (Myelin) (Xu et al. 2017) (15e-3s)
R2_lp = opts.R2_lp; % 1/63e-3; % Relaxation rate of large pool [s^-1] (Intra/Extra-cellular)

g  = opts.g_ratio; % 0.8;
ro = opts.R_outer_rel * opts.R_mu; %0.5;
ri = g*ro;
ri2 = ri^2;
ro2 = ro^2;

r2 = x.^2 + y.^2;
R2 = zeros(size(x));

b = (ri2 <= r2 & r2 <= ro2);
R2( b) = R2_sp;
R2(~b) = R2_lp;

end

function opts = parseinputs(varargin)

DefaultArgs = struct( ...
    'B0',            -3.0, ...          % External magnetic field (z-direction) [T]
    'gamma',         2.67515255e8, ...  % Gyromagnetic ratio [rad/s/T]
    'theta',         pi/2, ...          % Main magnetic field angle w.r.t B0 [rad]
    'g_ratio',       0.8370, ...        % g-ratio (original 0.71), 0.84658 for healthy, 0.8595 for MS.
    'R2_sp',         1/15e-3, ...       % %TODO (play with these?) Relaxation rate of small pool [s^-1] (Myelin) (Xu et al. 2017) (15e-3s)
    'R2_lp',         1/63e-3, ...       % %TODO (play with these?) 1st attempt was 63E-3. 2nd attempt 76 ms
    'R2_Tissue',     1/63e-3, ...       % %TODO (was 14.5Hz; changed to match R2_lp) Relaxation rate of tissue [s^-1]
    'R2_water',      1/2.2, ...         % Relaxation rate of pure water
    'D_Tissue',      1500.0, ...        % %TODO (reference?) Diffusion coefficient in tissue [um^2/s]
    'D_Sheath',      250.0, ...         % %TODO (reference?) Diffusion coefficient in myelin sheath [um^2/s]
    'D_Axon',        2000.0, ...        % %TODO (reference?) Diffusion coefficient in axon interior [um^2/s]
    'D_Blood',       3037.0, ...        % Diffusion coefficient in blood [um^2/s]
    'D_Water',       3037.0, ...        % Diffusion coefficient in water [um^2/s]
    'K_perm',        1.0e-3, ...        % %TODO (reference?) Interface permeability constant [um/s]
    'R_mu',          0.46, ...          % Axon mean radius [um] ; this is taken to be outer radius.
    'R_shape',       5.7, ...           % Axon shape parameter for Gamma distribution (Xu et al. 2017)
    'R_scale',       0.46/5.7, ...      % Axon scale parameter for Gamma distribution (Xu et al. 2017)
    'AxonPDensity',  0.83, ...          % Axon packing density based region in white matter. (Xu et al. 2017) (originally 0.83)
    'AxonPDActual',  0.64, ...          % The actual axon packing density you're aiming for.
    'PD_sp',         0.5, ...           % Relative proton density (Myelin)
    'PD_lp',         1.0, ...           % Relative proton density (Intra Extra)
    'PD_Fe',         1.0, ...           % Relative proton density (Ferritin)
    'ChiI',          -60e-9, ...        % Isotropic susceptibility of myelin [ppb] (check how to get it) (Xu et al. 2017)
    'ChiA',          -120e-9, ...       % Anisotropic Susceptibility of myelin [ppb] (Xu et al. 2017)
    'E',             10e-9, ...         % Exchange component to resonance freqeuency [ppb] (Wharton and Bowtell 2012)
    'R2_Fe',         1/1e-6, ...        % Relaxation rate of iron in ferritin. Assumed to be really high.
    'R2_WM',         1/70e-3, ...       % Relaxation rate of frontal WM. This is empirical;taken from literature. (original 58.403e-3) (patient 58.4717281111171e-3)
    'R_Ferritin',    4.0e-3, ...        % Ferritin mean radius [um].
    'R_conc',        0.0, ...           % Conntration of iron in the frontal white matter. [mg/g] (0.0424 in frontal WM) (0.2130 in globus pallidus; deep grey matter)
    'Rho_tissue',    1.073, ...         % White matter tissue density [g/ml]
    'ChiTissue',     -9.05e-6, ...      % Isotropic susceptibility of tissue
    'ChiFeUnit',     1.4e-9, ...        % Susceptibility of iron per ppm/ (ug/g) weight fraction of iron.
    'ChiFeFull',     520.0e-6, ...      % Susceptibility of iron for ferritin particle FULLY loaded with 4500 iron atoms. (use volume of FULL spheres) (from Contributions to magnetic susceptibility)
    'Rho_Iron',      7.874, ...         % Iron density [g/cm^3]
    'Npts',          101, ...           % Number of points per dimension
    'Domain',        [-1,1], ...        % Bounds for side of square domain
    'R_outer_rel',   1, ...             % Outer radius relative to R_mu
    'Time',          60e-3, ...         % Simulation time
    'Plot',          true ...           % Plot resulting magnitude and phase
);

p = getParser(DefaultArgs);
parse(p,varargin{:});
opts = p.Results;

end

function p = getParser(DefaultArgs)

p = inputParser;
for f = fields(DefaultArgs).'
    paramName = f{1};
    defaultVal = DefaultArgs.(f{1});
    addParameter(p,paramName,defaultVal)
end

end

dom = [-pi, pi, 0.1, 1];
tspan = [0, 1];
S = spinop2(dom, tspan);

f = @(t,r) sin(21*pi*(1+cos(pi*r)).*(r.^2-2*r.^5.*cos(5*(t-0.11))));
f = @(t,r) cos(t) .* r;
F = diskfun(f, 'polar', dom);
% F = chebfun2(f, dom);

S.lin = @(u) lap(u);
S.nonlin = @(u) -u;
S.init = F;

U = spin2(S, 100, 0.01);

% dom = [-1 1, -1 1];
% tspan = [0, 60e-3];
% w = @(x,y) testomega(x,y);
% W = chebfun2(w, 64, dom, 'splitting', 'on');
% 
% dom = [-1 1, -1 1];
% tspan = [0, 60e-3];
% w = @(x,y) testomega(x,y);
% W = chebfun2(w, 64, dom, 'splitting', 'on');
% 
% S = spinop2(dom, tspan);
% S.lin = @(u) lap(u);
% S.nonlin = @(u) -u;
% S.init = randnfun2(4, dom, 'trig');
% S.init = S.init/norm(S.init, inf);
% U = spin2(S, 128, 1e-3);
