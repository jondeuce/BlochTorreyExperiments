function [dR2, ResultsStruct] = perforientation_fun(params, xdata, dR2_Data, varargin)
%[dR2, ResultsStruct] = PERFORIENTATION_FUN(params, xdata, dR2_Data, varargin)
% Calculates the perfusion curve dR2(*) vs. angle. dR2 is [1xNumAngles].

args = parseinputs(params, xdata, dR2_Data, varargin{:});

CA = args.params(1);
iBVF = args.params(2);
aBVF = args.params(3);
BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
alpha_range = args.xdata;

GeomArgs = struct( 'iBVF', iBVF, 'aBVF', aBVF, ...
    'VoxelSize', args.VoxelSize, 'GridSize', args.GridSize, 'VoxelCenter', args.VoxelCenter, ...
    'Nmajor', args.Nmajor, 'MajorAngle', args.MajorAngle, ...
    'NumMajorArteries', args.NumMajorArteries, 'MinorArterialFrac', args.MinorArterialFrac, ...
    'Rminor_mu', args.Rminor_mu, 'Rminor_sig', args.Rminor_sig, ...
    'AllowMinorSelfIntersect', args.AllowMinorSelfIntersect, ...
    'AllowMinorMajorIntersect', args.AllowMinorMajorIntersect, ...
    'PopulateIdx', args.PopulateIdx, ...
    'seed', args.geomseed );

solver = @() SplittingMethods.PerfusionCurve( ...
    args.TE, args.Nsteps, args.Dcoeff, CA, args.B0, alpha_range, args.type, ...
    'Order', args.order, 'GeomArgs', GeomArgs, 'MajorOrientation', args.MajorOrientation );

[~, S_noCA, S_CA, TimePts, ~, ~, Geoms] = solver();
AllGeoms = Compress(Geoms); % Geoms should be already compressed

for ii = 2:args.Navgs
    [~, S_noCA__, S_CA__, TimePts__, ~, ~, Geoms] = solver();
    AllGeoms = [AllGeoms; Geoms];
    
    TimePts = cat(3, TimePts, TimePts__);
    S_noCA  = cat(3, S_noCA, S_noCA__);
    S_CA    = cat(3, S_CA, S_CA__);
end

S_noCA_avg = mean(S_noCA(end,:,:),3); %row vector: [1 x Nalphas]
S_CA_avg   = mean(S_CA(end,:,:),3); %row vector: [1 x Nalphas]

%Need to squeeze after shift dim to handle cases of Navgs = 1
dR2_all = -1/args.TE .* squeeze( shiftdim( log(abs(S_CA(end,:,:)./S_noCA(end,:,:))) ) ).'; %array: [Navgs x Nalphas]
dR2 = -1/args.TE .* log( abs(S_CA_avg)./abs(S_noCA_avg) ); %row vector: [1 x Nalphas]

Results = struct( ...
    'params', args.params, ...
    'xdata', args.xdata, ...
    'dR2_Data', args.dR2_Data, ...
    'CA', CA, ...
    'iBVF', iBVF, ...
    'aBVF', aBVF, ...
    'alpha_range', alpha_range, ...
    'TimePts', TimePts, ...
    'dR2', dR2, ...
    'dR2_all', dR2_all, ...
    'S_CA', S_CA, ...
    'S_noCA', S_noCA, ...
    'Geometries', AllGeoms, ...
    'args', args ...
    );

% ----------------------------------------------------------------------- %
% Cleanup: plotting and saving simulation
% ----------------------------------------------------------------------- %

% ---- Plotting ---- %
try
    if args.PlotFigs
        [fig, FileName] = perforientation_plot( dR2, dR2_all, AllGeoms, args );
    end
catch me
    warning('Unable to draw figures.\nError message: %s', me.message);
end

% ---- Save Figure ---- %
try
    if args.PlotFigs && args.SaveFigs
        savefig(fig, FileName);
        export_fig(FileName, '-pdf', '-transparent');
    end
catch me
    warning('Unable to save figure.\nError message: %s', me.message);
end

% ---- Save Results ---- %
try
    if args.SaveResults
        save(FileName,'Results')
    end
catch me
    warning('Unable to save results.\nError message: %s', me.message);
    save(datestr(now,30),'Results');
end

% ---- Save Diary ---- %
try
    if ~isempty(args.DiaryFilename)
        diary(args.DiaryFilename);
    end
catch me
    warning('Unable to save diary.\nError message: %s', me.message);
end

if nargout > 1
    ResultsStruct = Results;
end

end

function args = parseinputs(varargin)

RequiredArgs = { ...
    'params', 'xdata', 'dR2_Data', 'TE', 'type', ...
    'VoxelSize', 'VoxelCenter', 'GridSize', ...
    'B0', 'Dcoeff', 'Nsteps', 'Nmajor', ...
    'Rminor_mu', 'Rminor_sig' ...
    };
DefaultArgs = struct(...
    'Navgs', 1, ...
    'order', 2, ...
    'MajorOrientation', 'FixedPosition', ...
    'MajorAngle', 0.0, ...
    'NumMajorArteries', 0, ...
    'MinorArterialFrac', 0.0, ...
    'AllowMinorSelfIntersect', true, ...
    'AllowMinorMajorIntersect', false, ...
    'PopulateIdx', true, ...
    'PlotFigs', true, ...
    'SaveFigs', true, ...
    'SaveResults', true, ...
    'DiaryFilename', [datestr(now,30),'__','diary.txt'], ...
    'geomseed', rng ...
    );

p = inputParser;

for f = RequiredArgs
    paramName = f{1};
    addRequired(p,paramName)
end

for f = fieldnames(DefaultArgs).'
    paramName = f{1};
    defaultVal = DefaultArgs.(f{1});
    addParameter(p,paramName,defaultVal)
end

parse(p, varargin{:});
args = p.Results;

end

