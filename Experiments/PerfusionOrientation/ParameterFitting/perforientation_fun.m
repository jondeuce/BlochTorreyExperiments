function [ dR2, ResultsStruct ] = perforientation_fun( params, xdata, dR2_Data, varargin )
%[ dR2, ResultsStruct ] = PERFORIENTATION_FUN(params, xdata, dR2_Data, varargin)
% Calculates the perfusion curve dR2(*) vs. angle. dR2 is [1 x NumAngles].

if nargin == 1
    % all arguments are given in a single struct
    args = parseinputs(params);
else
    args = parseinputs(params, xdata, dR2_Data, varargin{:});
end

% Get opt variables, and set geometry if necessary
alpha_range = args.xdata;

switch upper(args.OptVariables)
    case 'CA_IBVF_ABVF'
        CA = args.params(1);
        iBVF = args.params(2);
        aBVF = args.params(3);
    case 'CA_RMAJOR_MINOREXPANSION'
        CA = args.params(1);
        Rmajor = args.params(2);
        SpaceFactor = args.params(3);
        args.Geom = SetRmajor(args.Geom, Rmajor); % Update Rmajor in input Geom
        args.Geom = ExpandMinorVessels(args.Geom, SpaceFactor); % Expand minor vessels in input Geom
        args.Geom = Uncompress(args.Geom); % Re-calculate vasculature map, add arteries, etc.
        iBVF = args.Geom.iBVF; % extract resulting iBVF
        aBVF = args.Geom.aBVF; % extract resulting aBVF
    otherwise
        error('''OptVariables'' must be ''CA_iBVF_aBVF'' or ''CA_Rmajor_MinorExpansion''');
end

% Set useful fields of args for plotting
if ~isempty(args.Geom)
    args.Nmajor = args.Geom.Nmajor;
else
    args.Nmajor = args.GeomArgs.Nmajor;
end
args.CA = CA;
args.iBVF = iBVF;
args.aBVF = aBVF;
args.BVF = iBVF + aBVF;
args.iRBVF = iBVF/(iBVF + aBVF);
args.params = [CA, iBVF, aBVF];

msg = { 'Calling perforientation_fun:', '', ...
    ['CA     = ', sprintf('%8.6f', CA), ' [mM]'], ...
    ['iBVF   = ', sprintf('%8.6f', 100*iBVF), ' [percent]'], ...
    ['aBVF   = ', sprintf('%8.6f', 100*aBVF), ' [percent]'], ...
    ['Nmajor = ', sprintf('%8d', args.Geom.Nmajor), ' [unitless]'] };
display_text(msg, 30, '-', false, [true, false]);

solver = @() SplittingMethods.PerfusionCurve( ...
    alpha_range, args.TE, args.Nsteps, args.type, CA, args.B0, ...
    args.D_Tissue, args.D_Blood, args.D_VRS, ...
    'GeomArgs', args.GeomArgs, 'Geom', args.Geom, ...
    'MaskType', args.MaskType, ...
    'StepperArgs', args.StepperArgs, ...
    'RotateGeom', args.RotateGeom, ...
    'EffectiveVesselAngles', args.EffectiveVesselAngles, ...
    'CADerivative', false );

% Finished with input args.Geom; clear it (Geoms will already be saved)
if ~isempty(args.Geom); args.Geom = []; end

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

% Save results
Results = struct( ...
    'OptVariables', args.OptVariables, ...
    'params', params, ...
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

% ---- Close Figure ---- %
try
    if args.PlotFigs && args.CloseFigs
        close(fig);
    end
catch me
    warning('Unable to close figure.\nError message: %s', me.message);
end

% ---- Save Results ---- %
try
    if args.SaveResults
        if ~args.PlotFigs
            FileName = datestr(now,30); % FileName above not created
        end
        save(FileName, 'Results', '-v7')
    end
catch me
    warning('Unable to save results.\nError message: %s', me.message);
    save(datestr(now,30), 'Results', '-v7');
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
    'params', 'xdata', 'dR2_Data', 'TE', 'Nsteps', 'type', 'B0', 'D_Tissue', ...
    };
OptionalArgs = struct( ...
    'D_Blood', [], ... %[um^2/s]
    'D_VRS', [] ... %[um^2/s]
    );
DefaultArgs = struct(...
    'OptVariables', 'CA_iBVF_aBVF', ...
    'Geom', [], ...
    'GeomArgs', [], ...
    'RotateGeom', false, ...
    'EffectiveVesselAngles', false, ...
    'Navgs', 1, ...
    'StepperArgs', struct('Stepper','BTSplitStepper','Order',2), ...
    'MaskType', '', ...
    'Weights', 'uniform', ...
    'Normfun', 'AICc', ...
    'TitleLinesFun', [], ...
    'PlotFigs', true, ...
    'SaveFigs', true, ...
    'FigTypes', {'fig','pdf'}, ...
    'CloseFigs', true, ...
    'SaveResults', true, ...
    'DiaryFilename', [datestr(now,30),'__','diary.txt'] ...
    );

p = inputParser;

for f = RequiredArgs
    paramName = f{1};
    addRequired(p,paramName)
end

for f = fieldnames(OptionalArgs).'
    paramName = f{1};
    defaultVal = OptionalArgs.(f{1});
    addOptional(p,paramName,defaultVal)
end

for f = fieldnames(DefaultArgs).'
    paramName = f{1};
    defaultVal = DefaultArgs.(f{1});
    addParameter(p,paramName,defaultVal)
end

if nargin == 1
    % all arguments are given in a single struct
    inputargs = varargin{1};
    requiredargs = cell(1,numel(RequiredArgs));
    optionalargs = {};
    for ii = 1:numel(RequiredArgs)
        requiredargs{ii} = inputargs.(RequiredArgs{ii});
    end
    for f = fieldnames(OptionalArgs).'
        if isfield(inputargs, f{1})
            optionalargs{end+1} = inputargs.(f{1});
        end
    end
    inputarglist = struct2arglist(rmfield(inputargs, RequiredArgs));
    parse(p, requiredargs{:}, optionalargs{:}, inputarglist{:});
else
    parse(p, varargin{:});
end
args = p.Results;

end

