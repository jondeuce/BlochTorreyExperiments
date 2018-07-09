%BOLDORIENTATION

%% Setup simulation for saving prompt output + current script

%Save diary of workspace
DiaryFilename = [datestr(now,30), '__', 'diary.txt'];
diary(DiaryFilename);

%Save a copy of this script (and others) in the directory of the caller
BOLDscripts = {'BOLDOrientation','SplittingMethods.BOLDCurve'};
for boldscript = BOLDscripts
    backupscpt  = sprintf('%s__%s.m',datestr(now,30),strrep(boldscript{1},'.',''));
    currentscpt = which(boldscript{1});
    copyfile(currentscpt,backupscpt);
end
clear BOLDscripts backupscpt currentscpt

%% Oxygenation parameters (w/ references)

% Y0 = 0.54; % Oxygen saturation fraction for deoxygenated blood, aka Y0 [fraction]
% Y  = 0.65; % Oxygen saturation fractions for oxygenated blood to simulate, aka Y [fraction]
% Hct = 0.45; % Hematocrit = volume fraction of red blood cells

% Ref: Zhao et al., 2007, MRM, Oxygenation and hematocrit dependence of transverse relaxation rates of blood at 3T
Y0  = 0.61; % Yv_0, baseline venous oxygenated blood fraction [fraction]
Y   = 0.73; % Yv, activated venous oxygenated blood fraction [fraction]
Hct = 0.44; % Hematocrit = volume fraction of red blood cells

%% BOLD Common Settings

% type = 'SE';
% dt = 2.5e-3;
type = 'GRE';
dt = 2.5e-3;

% EchoTimes = 0:dt:120e-3; % Echotimes in seconds to simulate [s]
% alpha_range = [0, 45, 90];
% EchoTimes = 0:5e-3:120e-3; % Echotimes in seconds to simulate [s]
% alpha_range = 0:10:90; % angles in degrees
% EchoTimes = 0:2.5e-3:120e-3; % Echotimes in seconds to simulate [s]
EchoTimes = 0:2.5e-3:20e-3; % Echotimes in seconds to simulate [s]
%alpha_range = [0:5:90]; % angles in degrees
alpha_range = [0,54.7]; % angles in degrees

B0 = -7.0; %[Tesla]
D_Tissue = 2000; %[um^2/s] TODO: check literature?
D_Blood = 3037; %[um^2/s]
D_VRS = 3037; %[um^2/s]
% D_Tissue = 0; %[um^2/s] TODO: check literature?
% D_Blood = []; %[um^2/s]
% D_VRS = []; %[um^2/s]

%% Geometry Settings
% Results from SE perfusion orientation simulations
% NOTE: For calculating the BOLD curve, it is important to consider that
%       approximately 1/3 of the vasculature is arterial
iBVF = 1.1803/100;
% iBVF = 0/100;
aBVF = 1.3425/100;
% aBVF = 0/100;
BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
aRBVF = aBVF/BVF;

Nmajor = 4; % Number of major vessels (optimal number is from SE perf. orientation. sim)
MajorAngle = 0.0; % Major vessel angles compared to B0 [degrees]
% NumMajorArteries = 1; % Number of major arteries
NumMajorArteries = 0; % Number of major arteries
MinorArterialFrac = 1/3; % Fraction of minor vessels which are arteries
VRSRelativeRad = 2; % Radius of Virchow-Robin space relative to major vessel radius [unitless]

Navgs = 1; % Number of geometries to simulate
VoxelSize = [2500,2500,2500]; % Typical isotropic voxel dimensions. [um]
GridSize = [512,512,512]; % Voxel size to ensure isotropic subvoxels
VoxelCenter = [0,0,0];

% Rminor_mu = 13.7;
% Rminor_sig = 2.1;
Rminor_mu = 7.0;
Rminor_sig = 0.5;
rng('default'); seed = rng; % for consistent geometries between sims.

%% Geometry generator
NewGeometry = @() Geometry.CylindricalVesselFilledVoxel( ...
    'iBVF', iBVF, 'aBVF', aBVF, ...
    'VoxelSize', VoxelSize, 'GridSize', GridSize, 'VoxelCenter', VoxelCenter, ...
    'Nmajor', Nmajor, 'MajorAngle', MajorAngle, ...
    'NumMajorArteries', NumMajorArteries, 'MinorArterialFrac', MinorArterialFrac, ...
    'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', true, ...
    'VRSRelativeRad', VRSRelativeRad, ...
    'PopulateIdx', true, 'seed', seed );

%% Bloch-Torrey propagation stepper
% stepper = 'BTSplitStepper';
stepper = 'ExpmvStepper';

%% Update Diary
diary(DiaryFilename);

%% BOLD Simulation

% BOLDResults object for storing results
AllResults = BOLDResults( EchoTimes, deg2rad(alpha_range), Y0, Y, Hct, 1:Navgs );

% Anon. func. for computing the BOLD Curve
ComputeBOLDCurve = @(R,G) SplittingMethods.BOLDCurve(R, EchoTimes, dt, Y0, Y, Hct, ...
    CalculateDiffusionMap( G, D_Tissue, D_Blood, D_VRS ), ...
    B0, alpha_range, G, type, stepper);

Geometries = [];
for ii = 1:Navgs
    Geom = NewGeometry();
    Results = BOLDResults( EchoTimes, deg2rad(alpha_range), Y0, Y, Hct, ii );
    Results = ComputeBOLDCurve(Results, Geom);
    
    Geom = Compress(Geom);
    Geometries = [Geometries; Geom]; %#ok<AGROW>
    AllResults = push(AllResults, Results);
    
    try
        title_lines = { sprintf('Nmajor = %d, iBVF = %.4f%%, aBVF = %.4f%%, BVF = %.4f%%, iRBVF = %.2f%%', Nmajor, iBVF*100, aBVF*100, BVF*100, iRBVF*100), ...
                        sprintf('B0 = %1.1fT, Y0 = %.2f, Y = %.2f, Hct = %.2f, Rmajor = %.2fum, Rminor = %.2fum', -B0, Y0, Y, Hct, mean(Geom.Rmajor(:)), Geom.Rminor_mu) };
        title_lines = cellfun(@(s)strrep(s,'%','\%'),title_lines,'uniformoutput',false);
        
        title_str = title_lines{1}; for ll = 2:numel(title_lines); title_str = [title_str,char(10),title_lines{ll}]; end
        
        fname = [title_lines{1},', ',title_lines{2}];
        fname = strrep( fname, ' ', '' ); fname = strrep( fname, '\%', '' ); fname = strrep( fname, '=', '-' );
        fname = strrep( fname, ',', '__' );   fname = strrep( fname, '.', 'p' ); 
        
        figTotal = plotBOLD(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','BOLD Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameTotal = [datestr(now,30), '__Total__', fname];
       
        save(fnameTotal,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figTotal, fnameTotal);
        export_fig(fnameTotal, '-pdf', '-transparent');
        
        %oxy Total
        figOxyTotal = plotOxyTotal(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','Oxygenated Total Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameOxyTotal = [datestr(now,30), '__OxyTotal__', fname];
        save(fnameOxyTotal,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figOxyTotal, fnameOxyTotal);
        export_fig(fnameOxyTotal, '-pdf', '-transparent');
        
         %Deoxy Total
        figDeoxyTotal = plotDeoxyTotal(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','Deoxygenated Total Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameDeoxyTotal = [datestr(now,30), '__DeoxyTotal__', fname];
        save(fnameDeoxyTotal,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figDeoxyTotal, fnameDeoxyTotal);
        export_fig(fnameDeoxyTotal, '-pdf', '-transparent');
                
        %Intra
        figIntra = plotBOLDIntra(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','BOLD Signal Intra [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameIntra = [datestr(now,30), '__Intra__', fname];
        save(fnameIntra,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figIntra, fnameIntra);
        export_fig(fnameIntra, '-pdf', '-transparent');
        
          %Deoxy Intra
        figDeoxyIntra = plotDeoxyIntra(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','Deoxygenated Intra Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameDeoxyIntra = [datestr(now,30), '__DeoxyIntra__', fname];
        save(fnameDeoxyIntra,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figDeoxyIntra, fnameDeoxyIntra);
        export_fig(fnameDeoxyIntra, '-pdf', '-transparent');
        
        %oxy Intra
        figOxyIntra = plotOxyIntra(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','Oxygenated Intra Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameOxyIntra = [datestr(now,30), '__OxyIntra__', fname];
        save(fnameOxyIntra,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figOxyIntra, fnameOxyIntra);
        export_fig(fnameOxyIntra, '-pdf', '-transparent');
        
        %Extra
        figExtra = plotBOLDExtra(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','BOLD Signal Extra [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameExtra = [datestr(now,30), '__Extra__', fname];
        save(fnameExtra,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figExtra, fnameExtra);
        export_fig(fnameExtra, '-pdf', '-transparent');
        
         %Deoxy Extra
        figDeoxyExtra = plotDeoxyExtra(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','Deoxygenated Extra Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameDeoxyExtra = [datestr(now,30), '__DeoxyExtra__', fname];
        save(fnameDeoxyExtra,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figDeoxyExtra, fnameDeoxyExtra);
        export_fig(fnameDeoxyExtra, '-pdf', '-transparent');
        
        %oxy Extra
        figOxyExtra = plotOxyExtra(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','Oxygenated Extra Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameOxyExtra = [datestr(now,30), '__OxyExtra__', fname];
        save(fnameOxyExtra,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figOxyExtra, fnameOxyExtra);
        export_fig(fnameOxyExtra, '-pdf', '-transparent');
        
        %VRS
        figVRS = plotBOLDVRS(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','BOLD Signal VRS [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameVRS = [datestr(now,30), '__VRS__', fname];
        save(fnameVRS,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figVRS, fnameVRS);
        export_fig(fnameVRS, '-pdf', '-transparent');
        
         %oxy VRS
        figOxyVRS = plotOxyVRS(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','Oxygenated VRS Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameOxyVRS = [datestr(now,30), '__OxyVRS__', fname];
        save(fnameOxyVRS,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figOxyVRS, fnameOxyVRS);
        export_fig(fnameOxyVRS, '-pdf', '-transparent');
        
        %Deoxy VRS
        figDeoxyVRS = plotDeoxyVRS(Results, 'title', title_str, 'scalefactor', 100/prod(VoxelSize), 'ylabel', [upper(type),' ','Deoxygenated VRS Signal [\%]'],'legendlocation', 'eastoutside', 'interp', true);
        fnameDeoxyVRS = [datestr(now,30), '__DeoxyVRS__', fname];
        save(fnameDeoxyVRS,'Results','Geom','-v7');
        diary(DiaryFilename);
        savefig(figDeoxyVRS, fnameDeoxyVRS);
        export_fig(fnameDeoxyVRS, '-pdf', '-transparent');
        
    catch me
        warning('Error occured while plotting figure and saving LoopResults.\nError message: %s', me.message);
    end
end

%% Save workspace in current directory
save([datestr(now,30),'__','BOLDOrientationResults'],'-v7');

diary(DiaryFilename);
diary off
