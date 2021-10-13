% Set path
[btroot, ~, ~] = fileparts(mfilename('fullpath'));
btroot = fullfile(btroot, '../..');
addpath(btroot);
setbtpath;

% Generate voxel geometry object
GeomArgs = struct( ...
    'VoxelSize', [2000,2000,2000], ... # Voxel dimensions [um]
    'VoxelCenter', [0,0,0], ... # Voxel origin [um]
    'GridSize', [512,512,512], ... # Grid size [Int]
    'Nmajor', 3, ... # Number of large anisotropic vessels [Int]
    'MajorAngle', 0.0, ... # Angle of large vessels w.r.t. main magnetic field [deg]
    'NumMajorArteries', 1, ... # Number of large anisotropic vessels which are arteries; roughly one third of 'Nmajor' [Int]
    'MinorArterialFrac', 1/3, ... # Fraction of isotropic bed of minor vessels which are arteries; typically 1/3 [Fraction]
    'BVF',    2.5/100, ... % Total blood volume fraction [Fraction]
    'iRBVF',  60/100, ... % Amount of blood contained in isotropic minor vasculature relative to 'BVF' [Fraction]
    'Rminor_mu', 10.0, ... % Minor vessel mean radius [um]
    'Rminor_sig', 0.0, ... % Minor vessel std radius [um]
    'VRSRelativeRad', 1.5, ... % Radius of Virchow-Robin space (aka perivascular space) relative to the diameter of the major anisotropic vessels [Float]
    'Verbose', true, ... # Verbose printing [Bool]
    'seed', 1234 ... # Random seed [Int]
);
GeomArgList = struct2arglist(GeomArgs);
Geom = Geometry.CylindricalVesselFilledVoxel( GeomArgList{:} );
% plot(Geom); % Visualize vessels in 3D
% Plotter({GetMask(Geom, 'PVSAndVasculature')}, 'color', 'plasma'); % Visualize vasculature/tissue/PVS mask

% Generate field map
GammaSettings = Geometry.ComplexDecaySettings();
Gamma = CalculateComplexDecay(GammaSettings, Geom);
% Plotter({real(Gamma), imag(Gamma)}, 'color', 'plasma') % Visualize complex decay Gamma = R2 + i * dOmega
