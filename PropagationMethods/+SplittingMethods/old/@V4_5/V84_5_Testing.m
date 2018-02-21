%This method chains together VTV terms as follows:
%   b1 a1 b2 a2 b3 a3 b3 a2 b2 a1 b1
% where
%   b_n <--> exp(-b_n * V * t)
%   a_n <--> exp(-a_n * T * t)
% V,T are the potential/kinetic terms, respectively

load('/home/coopar7/Documents/code/magprop_master/MagPropCommon/Experiments/BOLDOrientation/Results/SE_Experiment/20170803T153101/Final_Workspace.mat')
G = Geometry.CylindricalVesselFilledVoxel(SimSettings,ParamCombinations);
alpha = 60*(pi/180);
d = dipole( G.GridSize, G.VoxelSize, [sin(alpha),0,cos(alpha)], 'double' );
dw = real(ifftn(fftn(G.VasculatureMap).*d));

TE = pi/std(dw(:)); % dephase for ~pi
R2b = 5/TE; R2t = 1/TE; % T2blood is shorter
r2 = (R2b - R2t)*G.VasculatureMap + R2t;

% rms diffusion distance for whole scan: sqrt(6*D*TE) = n_sub * subvoxsize
n_sub = 10;
Dcoeff = (n_sub * G.SubVoxSize)^2 / (6*TE);

Gamma = complex(r2,dw);
A = BlochTorreyOp(Gamma, Dcoeff, G.GridSize, G.VoxelSize);
clear d dw r2 Gamma

x0 = 1i*ones(G.GridSize);

% Time step
dt = TE/10;

b1 = 0.052472525516129026 - 0.010958940842458138i;
a1 = 0.175962140656732362 - 0.054483056228160557i;
b2 = 0.246023563332753880 - 0.125228547924834352i;
a2 = 0.181259898687454283 - 0.034864508232090522i;
b3 = 1/2 - (b1 + b2 );
a3 = 1 - 2*(a1 + a2 );

D1 = a1 * A.D;
D2 = a2 * A.D;
D3 = a3 * A.D;
K1 = Geometry.GaussianKernel(sqrt(2*D1*dt),G.GridSize,G.VoxelSize);
K2 = Geometry.GaussianKernel(sqrt(2*D2*dt),G.GridSize,G.VoxelSize);
K3 = Geometry.GaussianKernel(sqrt(2*D3*dt),G.GridSize,G.VoxelSize);

y = conv(K1, exp(-b1*A.Gamma*dt).*x0); % a1 b1
y = conv(K2, exp(-b2*A.Gamma*dt).*y); % a2 b2
y = conv(K3, exp(-b3*A.Gamma*dt).*y); % a3 b3
y = conv(K2, exp(-b3*A.Gamma*dt).*y); % a2 b3
y = conv(K1, exp(-b2*A.Gamma*dt).*y); % a1 b2
y = exp(-b1*A.Gamma*dt).*y; % b1

K = Geometry.GaussianKernel(sqrt(2*A.D*dt),G.GridSize,G.VoxelSize);
z = exp(-A.Gamma*dt/2).*x0; % b1
z = conv(K, z); % a1
z = exp(-A.Gamma*dt/2).*z; % b1

