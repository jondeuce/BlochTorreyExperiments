function [t, S] = cpmg_simulation_fun(Geom, GammaArgs, varargin)

% Parse keyword inputs
args = parseinputs(varargin{:});

% Calculate complex decay 
GammaSettings = Geometry.ComplexDecaySettings(GammaArgs);
Gamma = CalculateComplexDecay(GammaSettings, Geom);

% Calculate diffusion coefficient map
DiffusionMap = CalculateDiffusionMap( Geom, args.D_Tissue, args.D_Blood, args.D_VRS );

% Calculate trinary mask taking value 0/1/2 at locations of tissue/blood/PVS
GeomMask = GetMask(Geom, 'PVSAndVasculature');

% Create matrix-free Bloch-Torrey operator object
A = BlochTorreyOp( ...
    Gamma, ...
    DiffusionMap, ...
    Geom.GridSize, ...
    Geom.VoxelSize, ...
    false, ...
    GeomMask ...
);

% Create exponential matrix-vector product stepper object
V = ExpmvStepper( ...
    args.dt, ...
    A, ...
    Geom.GridSize, ...
    Geom.VoxelSize, ...
    'prec', 'single', ...
    'full_term', false, ...
    'prnt', false, ...
    'type', 'default', ...
    'forcesparse', false, ...
    'shift', true, ...
    'bal', false ...
);

% Misc. utilities
Vox_Volume = prod(Geom.VoxelSize);
Num_Voxels = prod(Geom.GridSize);
um3_per_voxel = Vox_Volume/Num_Voxels;
IntegrateSignal = @(M) um3_per_voxel * sum(sum(sum(M,1),2),3); % more accurate than sum(M(:))

% Initialize outputs
t  = 0.0;
m0 = 1i; % must be on imaginary axis in order for conj(M) to correspond to a 180 degree pulse
S  = m0 * Vox_Volume; % signal at time t = 0
M  = m0 * ones(Geom.GridSize); % initial magnetization

if ~isempty(args.PlotRate)
    fprintf('Logging: t = %.2f ms, step %2d/%2d\n', 1000 * t(end), 0, args.TotalSteps);
    cpmg_simulation_logger('SavePlots', args.SavePlots, 'Gamma', Gamma, 'DiffusionMap', DiffusionMap);
end

total_time = tic;
for ii = 1:args.TotalSteps
    loop_time = tic;
    
    [M,~,~,V] = step(V,M);
    t = [t; ii * args.dt];
    S = [S; IntegrateSignal(M)];

    if ~isempty(args.PlotRate) && mod(ii, args.PlotRate) == 0
        fprintf('Logging: t = %.2f ms, step %2d/%2d\n', 1000 * t(end), ii, args.TotalSteps);
        cpmg_simulation_logger('SavePlots', args.SavePlots, 'Time', t, 'Signal', S, 'Magnetization', M);
    end

    if mod(ii - args.StepsPerTE/2, args.StepsPerTE) == 0
        fprintf('Flipping: t = %.2f ms, step %2d/%2d\n', 1000 * t(end), ii, args.TotalSteps);
        M = conj(M);
        S(end) = conj(S(end));
    end
    
    str = sprintf('Step complete: t = %.2f ms, step %2d/%2d', 1000 * t(end), ii, args.TotalSteps);
    display_toc_time(toc(loop_time), str);
end
str = sprintf('Finished: t = %.2f ms, step %2d/%2d', 1000 * t(end), ii, args.TotalSteps);
display_toc_time(toc(total_time), str);

end

function args = parseinputs(varargin)

p = inputParser;
p.FunctionName = 'cpmg_simulation_fun';
addParameter(p, 'TE', 10e-3); % echo time [s]
addParameter(p, 'nTE', 32); % number of echoes [Int]
addParameter(p, 'dt', 1e-3); % time step; must divide evenly into TE/2
addParameter(p, 'D_Tissue', 2000); % Diffusion constant in tissue [um^2/s]
addParameter(p, 'D_Blood', 3037); % Diffusion constant in blood [um^2/s]
addParameter(p, 'D_VRS', 3037); % Diffusion constant in perivascular space [um^2/s]
addParameter(p, 'VRSRelativeRad', sqrt(3)); % VRS space volume is approx (relrad^2-1)*BVF, so sqrt(3) => 2X [um/um]
addParameter(p, 'PlotRate', []); % Create progress plots every 'PlotRate' steps [Int]
addParameter(p, 'SavePlots', true); % Save and close plots as they appear; if false, leave them open [Bool]

parse(p, varargin{:});
args = p.Results;

assert(rem(args.TE/2, args.dt) == 0)
args.TotalSteps = round(args.nTE * args.TE / args.dt);
args.StepsPerTE = round(args.TE / args.dt);

if isempty(args.PlotRate)
    args.PlotRate = args.StepsPerTE;
end

end
