%BOLDORIENTATION

%% Setup simulation for saving prompt output + current script
DiaryFilename = [datestr(now,30), '__', 'diary.txt']; %Save diary of console outputs
diary(DiaryFilename);

%Save a copy of this script (and others) in the directory of the caller
BOLDscripts = {'BOLDOrientation','SplittingMethods.BOLDCurve'};
for boldscript = BOLDscripts
    backupscpt  = sprintf('%s__%s.m',datestr(now,30),strrep(boldscript{1},'.',''));
    currentscpt = which(boldscript{1});
    copyfile(currentscpt,backupscpt);
end
clear BOLDscripts boldscript backupscpt currentscpt

%% Oxygenation parameters (w/ references)
% Y0 = 0.54; % Oxygen saturation fraction for deoxygenated blood, aka Y0 [fraction]
% Y  = 0.65; % Oxygen saturation fractions for oxygenated blood to simulate, aka Y [fraction]
% Hct = 0.45; % Hematocrit = volume fraction of red blood cells

% Ref: Zhao et al., 2007, MRM, Oxygenation and hematocrit dependence of transverse relaxation rates of blood at 3T
Y0  = 0.61; % Yv_0, baseline venous oxygenated blood fraction [fraction]
Y   = 0.73; % Yv, activated venous oxygenated blood fraction [fraction]
Hct = 0.44; % Hematocrit = volume fraction of red blood cells

%% BOLD Common Settings
type = 'SE';
dt = 2.5e-3;
% type = 'GRE';
% dt = 2.5e-3;

% EchoTimes = 0:dt:120e-3; % Echotimes in seconds to simulate [s]
% alpha_range = [0, 45, 90]; % degrees
% EchoTimes   = 1e-3 * [0:10:20, 25:5:45, 50:15:80, 100:20:120]; % Echotimes in seconds to simulate [s]
% alpha_range = [0, 15, 25:5:60, 70:10:90]; % degrees
EchoTimes   = [0     5    10    15    20    30    35    50    65    80   105   120] * 1e-3; % [s]
alpha_range = [0     5    15    25    30    35    40    45    60    70    85    90]; % [deg]
% EchoTimes = 0:5e-3:120e-3; % Echotimes in seconds to simulate [s]
% alpha_range = 0:5:90; % degrees

B0 = -3.0; %[Tesla]
% D_Tissue = 3037; %[um^2/s]
% D_Blood = []; %[um^2/s]
% D_VRS = []; %[um^2/s]
% MaskType = '';

D_Tissue = 1000; %[um^2/s]
D_Blood = 3037; %[um^2/s]
D_VRS = 3037; %[um^2/s]
% MaskType = 'PVS';
% MaskType = 'PVSOrVasculature';
MaskType = 'PVSAndVasculature';
% MaskType = 'Vasculature';

%% Geometry Settings
% Results from SE perfusion orientation simulations, N = 3
% NOTE: For calculating the BOLD curve, it is important to consider that
%       approximately 1/3 of the vasculature is arterial
iBVF = 1.5775668712247/100;
aBVF = 1.2390732542589/100;

BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
aRBVF = aBVF/BVF;

Nmajor = 3; % Number of major vessels (optimal number is from SE perf. orientation. sim)
MajorAngle = 90.0; % Major vessel angles compared to B0 [degrees]
NumMajorArteries = 1; % Number of major arteries
MinorArterialFrac = 1/3; % Fraction of minor vessels which are arteries
VRSRelativeRad = 1; % Radius of Virchow-Robin space relative to major vessel radius [unitless]

Navgs = 1; % Number of geometries to simulate
VoxelSize = [2500,2500,2500]; % Typical isotropic voxel dimensions. [um]
GridSize = [256,256,256]; % Voxel size to ensure isotropic subvoxels
VoxelCenter = [0,0,0];

Rminor_mu = 13.7;
Rminor_sig = 2.1;
% Rminor_mu = 7.0;
% Rminor_sig = 0.5;
rng('default'); seed = rng; % for consistent geometries between sims.

%% Mock Common/Geometry Settings (for testing)
% type = 'SE';
% % type = 'GRE';
% dt = 2.5e-3;
% 
% % alpha_range = [0, 45, 90];
% % EchoTimes = (0:5:60)/1000; % Echotimes in seconds to simulate [s]
% % EchoTimes = [0:5:20, 30:30:60]/1000; % Echotimes in seconds to simulate [s]
% alpha_range = [0     5    15    25    30    35    40    45    60    70    85    90]; % [deg]
% EchoTimes   = [0     5    10    15    20    30    35    50    65    80   105   120] * 1e-3; % [s]
% 
% % Results from SE perfusion orientation simulations
% % NOTE: Even though only approx. 2/3 of vasculature is veinous and therefore
% %       contributes to the BOLD affect, we use the full simulated BVF and
% %       "black out" 1/3 of the blood by setting them to be arteries below
% iBVF = 1.5/100;
% aBVF = 1.5/100;
% BVF  = iBVF + aBVF; iRBVF = iBVF/BVF; aRBVF = aBVF/BVF;
% 
% Nmajor = 3; % Number of major vessels (optimal number is from SE perf. orientation. sim)
% Navgs = 1; % Number of geometries to simulate
% MajorAngle = 0.0; % Major vessel angles compared to B0 [degrees]
% NumMajorArteries = 1; % Number of major arteries
% MinorArterialFrac = 1/3; % Fraction of minor vessels which are arteries
% VRSRelativeRad = 2; % Radius of Virchow-Robin space relative to major vessel radius [unitless]
% 
% VoxelSize = [2500,2500,2500]; % Typical isotropic voxel dimensions. [um]
% GridSize = [128,128,128]; % Voxel size to ensure isotropic subvoxels
% % GridSize = [256,256,256]; % Voxel size to ensure isotropic subvoxels
% VoxelCenter = [0,0,0];
% 
% % Rminor_mu = 13.7;
% % Rminor_sig = 2.1;
% Rminor_mu = 25.0;
% Rminor_sig = 0.5;
% rng('default'); seed = rng; % for consistent geometries between sims.

%% Geometry generator
NewGeometry = @() Geometry.CylindricalVesselFilledVoxel( ...
    'iBVF', iBVF, 'aBVF', aBVF, ...
    'VoxelSize', VoxelSize, 'GridSize', GridSize, 'VoxelCenter', VoxelCenter, ...
    'Nmajor', Nmajor, 'MajorAngle', MajorAngle, ...
    'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
    'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'ImproveMajorBVF', true, 'ImproveMinorBVF', true, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'VRSRelativeRad', VRSRelativeRad, ...
    'MajorDistribution', 'Line', 'MajorDilation', 1.2, 'MinorDilation', 1.2, ...
    'PopulateIdx', true, 'seed', seed );

%% Bloch-Torrey propagation stepper
% stepper = 'BTSplitStepper'; % Splitting method
stepper = 'ExpmvStepper'; % Expmv-based method

%% Update Diary
diary(DiaryFilename);

%% BOLD Simulation

% BOLDResults object for storing results
AllResults = BOLDResults( EchoTimes, deg2rad(alpha_range), Y0, Y, Hct, 1:Navgs );

% Anon. func. for computing the BOLD Curve
ComputeBOLDCurve = @(R,G) SplittingMethods.BOLDCurve(R, EchoTimes, dt, Y0, Y, Hct, ...
    CalculateDiffusionMap( G, D_Tissue, D_Blood, D_VRS ), ...
    B0, alpha_range, G, MaskType, type, stepper);

Geometries = [];
for ii = 1:Navgs
    Geom = NewGeometry();
    
    Results = BOLDResults( EchoTimes, deg2rad(alpha_range), Y0, Y, Hct, ii );
    Results.MetaData.Geom = Compress(Geom);
    Results = ComputeBOLDCurve(Results, Geom);
    
    Geom = Compress(Geom);
    Geometries = [Geometries; Geom]; %#ok<AGROW>
    AllResults = push(AllResults, Results);
    
    try
        title_lines = { sprintf('$Nmajor = %d, iBVF = %.4f%%, aBVF = %.4f%%, BVF = %.4f%%, iRBVF = %.2f%%, PVSVol = %.4fX$', Nmajor, iBVF*100, aBVF*100, BVF*100, iRBVF*100, mean(vec([Geometries.VRSRelativeRad]))^2 - 1), ...
                        sprintf('$B0 = %1.1fT, Y0 = %.2f, Y = %.2f, Hct = %.2f, Rmajor = %.2fum, Rminor = %.2fum$', -B0, Y0, Y, Hct, mean(Geom.Rmajor(:)), Geom.Rminor_mu) };
        title_lines = cellfun(@(s)strrep(s,'%','\%'),title_lines,'uniformoutput',false);
        
        title_str = title_lines{1}; for ll = 2:numel(title_lines); title_str = [title_str, newline(), title_lines{ll}]; end
        fig = plot(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','BOLD Signal [\%]'], 'legendlocation', 'eastoutside', 'interp', true);
        
        fname = [title_lines{1},', ',title_lines{2}];
        fname = strrep( fname, ' ', '' ); fname = strrep( fname, '$', '' ); fname = strrep( fname, '\%', '' ); fname = strrep( fname, '=', '-' ); fname = strrep( fname, ',', '__' );   fname = strrep( fname, '.', 'p' ); fname = [datestr(now,30), '__', fname];
        
        save(fname,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(fig, fname);
        export_fig(fname, '-pdf', '-transparent');
    catch me
        warning('Error occured while plotting figure and saving LoopResults.\nError message: %s', me.message);
    end
end

%% Save workspace in current directory
clear fig
save([datestr(now,30),'__','BOLDOrientationResults'],'-v7');

diary(DiaryFilename);
diary off
