%% Set path
currdir = pwd;
[scriptdir, scriptname, scriptext] = fileparts(mfilename('fullpath'));
scriptname = [scriptname, scriptext];
btrootdir = fullfile(scriptdir, '../..');
addpath(btrootdir);
setbtpath;

%% Save zipped source code folder
btrootdir = absfolderpath(btrootdir);
zip('src.zip', btrootdir);

%% Generate voxel geometry object:
%   -Can set any two of 'Rmajor', 'iRBVF', 'BVF' depending on which two should you would like fixed and which should be determined automatically
%   -Subvoxel dimensions must be isotropic, i.e. 'VoxelSize ./ GridSize' should be a constant
%   -Subvoxel sidelength should be on the order of 'Rminor_mu', but this will require a lot of memory for typical voxel dimensions so can test with more coarse grained resolution and/or shorter voxels in z-dimension
GeomArgs = struct( ...
    'VoxelSize', [2500,2500,2500/4], ... # Voxel dimensions [um]
    'VoxelCenter', [0,0,0], ... # Voxel origin [um]
    'GridSize', [512,512,512/4], ... # Grid size [Int]
    'Nmajor', 3, ... # Number of large anisotropic vessels [Int]
    'MajorAngle', 0.0, ... # Angle of large vessels w.r.t. main magnetic field; typically should be zero, set orientation using GammaArgs below [deg]
    'NumMajorArteries', 1, ... # Number of large anisotropic vessels which are arteries; roughly one third of 'Nmajor' [Int]
    'MinorArterialFrac', 1/3, ... # Fraction of isotropic bed of minor vessels which are arteries; typically 1/3 [Fraction]
    'Rmajor', 85, ... % Major vessel radius [um]
    'iRBVF', 60/100, ... % Amount of blood contained in isotropic minor vasculature relative to 'BVF'; amont of blood in anisotropic major vessels is aRBVF = 1-iRBVF [Fraction]
    ... % 'BVF', 2.5/100, ... % Total blood volume fraction [Fraction]
    'ImproveMajorBVF', false, ... % Iteratively increase/decrease Rmajor to improve anisotropic blood volume toward target; should set to false if Rmajor is set explicitly [Bool]
    'ImproveMinorBVF', true, ... % Iteratively add/remove minor vessels to improve isotropic blood volume toward target; should typically be set to true, but can be set to false for speed [Bool]
    'Rminor_mu', 5.0, ... % Minor vessel mean radius [um]
    'Rminor_sig', 0.0, ... % Minor vessel std radius [um]
    'VRSRelativeRad', sqrt(3), ... % Radius of Virchow-Robin space (aka perivascular space) relative to the diameter of the major anisotropic vessels; VRS space volume is approx (r^2-1)*aBVF, so r = sqrt(3) => VRS vol = 2*aBVF [Float]
    'Verbose', true, ... # Verbose printing [Bool]
    'seed', 1234 ... # Random seed [Int]
);
Geom = Geometry.CylindricalVesselFilledVoxel( GeomArgs );

%% Generate default settings for complex field map:
% Gamma(x) = R2(x) + i * dOmega(x), where
%   - dOmega(x) = gamma * dB(x)
%   - dB(x) is the local field
GammaArgs = struct( ...
    'Angle_Deg', 90.0, ... % Angle of external field w.r.t. z-axis [deg]
    'B0', -3.0, ... % External magnetic field [Tesla]
    'Y', 0.61, ... % Venous Blood Oxygenation [Fraction]
    'Ya', 0.98 ... % Arterial Blood Oxygenation [Fraction]
);

%% Optional: inspect field, geometry, etc.
% Gamma = CalculateComplexDecay(Geometry.ComplexDecaySettings(GammaArgs), Geom);
% Plotter({real(Gamma), imag(Gamma)}, 'color', 'plasma') % Visualize complex decay Gamma = R2 + i * dOmega
% Plotter({GetMask(Geom, 'PVSAndVasculature')}, 'color', 'plasma'); % Visualize tissue/blood/PVS mask; tissue = 0, blood = 1, PVS = 2
% plot(Geom); % Visualize vessels in 3D (NOTE: can take a long time to plot/be very slow to interactive for large numbers of vessels)

%% Run simulation
for nTE = [8,16,32,64]
    % The Bloch-Torrey equation
    %   - dM(x,t)/dt = div(grad(D(x)*M(x,t))) - Gamma(x) * M(x,t)
    % where
    %   - D(x) is the diffusion map; generally it can be a tensor, but here 
    %     it is isotropic, i.e. equivalent to the tensor D(x)*I
    %   - Gamma(x) = R2(x) + i * dOmega(x) is the complex field (see above)
    % The BT equation is solved exactly using matrix exponential methods.
    % Generally, lowering the time step 'dt' slows the solving down, so
    % unless you specifically want to sample the simulated signal more
    % densely, 'dt' should be left equal to 'TE/2'.
    TotalTime = 360e-3;
    TE = TotalTime / nTE;
    dt = TE/2;
    CPMGArgs = struct( ...
        'TE', TE, ... % echo time [s]
        'nTE', nTE, ... % number of echoes [Int]
        'dt', dt, ... % time step; must divide evenly into TE/2
        'D_Tissue', 2000, ... % Diffusion constant in tissue [um^2/s]
        'D_Blood', 3037, ... % Diffusion constant in blood [um^2/s]
        'D_VRS', 3037, ... % Diffusion constant in perivascular space [um^2/s]
        'VRSRelativeRad', sqrt(3), ... % VRS space volume is approx (relrad^2-1)*BVF, so sqrt(3) => 2X [um/um]
        'SavePlots', true, ... % Save and close plots as they appear; if false, leave them open [Bool]
        'PlotRate', [] ... % Create progress plots every 'PlotRate' steps; defaults to TE/dt [Int]
    );

    Time = {};
    Signal = {};
    FieldAngles = [0,90];

    for ii = 1:numel(FieldAngles)
        logdir = fullfile(currdir, sprintf('TE_%05.2f_ms_FieldAngle_%05.2f_deg', 1000 * CPMGArgs.TE, FieldAngles(ii)));
        [~,~,~] = mkdir(logdir);
        cd(logdir);
        set(0,'DefaultFigureVisible','off');

        % Update the field angle and simulate a CPMG signal
        GammaArgs_ii = GammaArgs;
        GammaArgs_ii.Angle_Deg = FieldAngles(ii);
        [Time{ii}, Signal{ii}] = cpmg_simulation_fun(Geom, GammaArgs_ii, CPMGArgs);

        Results = struct( ...
            'Time', Time{ii}, ...
            'Signal', Signal{ii}, ...
            'GeomArgs', GeomArgs, ...
            'GammaArgs', GammaArgs_ii, ...
            'CPMGArgs', CPMGArgs, ...
            'Geom', Compress(Geom), ...
            'script', scriptname ...
        );
        save results.mat -struct Results -v7.3

        cd(currdir)
        set(0,'DefaultFigureVisible','on');
    end
end
