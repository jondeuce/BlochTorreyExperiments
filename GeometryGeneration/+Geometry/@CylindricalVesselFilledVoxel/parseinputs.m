function G = parseinputs( G, varargin )
%PARSEKEYWORDARGS Parses keyword and required arguments for class
% CylindricalVesselFilledVoxel.

if length(varargin) >= 2 && isstruct(varargin{1}) && isstruct(varargin{2})
    % OLD VERSION
    SimSettings = varargin{1};
    Params = varargin{2};
    G = parse_SimSettings_Params( G, SimSettings, Params, varargin );
    
    % Get parser and parse name-value args
    p = getSimSettingsParamsInputParser();
    parse(p,varargin{:})
    G.opts = p.Results;
    
    G.MinorDilation = G.opts.MinorDilation;
    G.MinorRadiusFactor = sqrt(G.opts.MinorDilation);
    G.isMinorDilated = false;
else
    % Get parser and parse name-value args
    p = getInputParser();
    parse(p,varargin{:});
    G = setParsedArgs(G,p);
    G = getDerivedArgs(G);
end

end

function p = getInputParser()

p = inputParser;
p.FunctionName = 'CylindricalVesselFilledVoxel';

VA = @(varargin) validateattributes(varargin{:});
VS = @(varargin) validatestring(varargin{:}); %note: case-insensitive

%-------------------------------------------------------------------------%
% Default parameters
%-------------------------------------------------------------------------%
addParameter(p,'VoxelSize', [3000,3000,3000]);
addParameter(p,'VoxelCenter', [0,0,0]);
addParameter(p,'GridSize', [512,512,512]);
addParameter(p,'Nmajor', 5);
addParameter(p,'MajorAngle', 0.0);
addParameter(p,'NumMajorArteries', 0);
addParameter(p,'MinorArterialFrac', 0.0);
addParameter(p,'Verbose', true);
addParameter(p,'seed', rng);

addParameter(p,'Rminor_mu', 13.7); % Minor vessel mean radius [um]
addParameter(p,'Rminor_sig', 2.1);  % Minor vessel std radius [um]

addParameter(p,'BVF',   []);
addParameter(p,'iRBVF', []);
addParameter(p,'aRBVF', []);
addParameter(p,'iBVF',  []);
addParameter(p,'aBVF',  []);

addParameter(p,'Rmajor',[]);
addParameter(p,'VRSRelativeRad',[]);
        
%-------------------------------------------------------------------------%
% Optional parameters
%-------------------------------------------------------------------------%
% Allow intersecting cylinders or not
addParameter(p,'AllowMinorSelfIntersect',true,@(x)VA(x,{'logical'},{'scalar'}));
addParameter(p,'AllowMinorMajorIntersect',false,@(x)VA(x,{'logical'},{'scalar'}));

% Minor vessel orientation
expectedinterptype = {'RANDOM','ALIGNED','PERIODIC'};
addParameter(p,'MinorOrientation','RANDOM',@(x) any(VS(x,expectedinterptype)));

% Minor/major dilation factors
addParameter(p,'MinorDilation',1.0,@(x)VA(x,{'scalar'},{'positive'}));

% Populate Idx field, or leave blank
addParameter(p,'PopulateIdx',true,@(x)VA(x,{'logical'},{'scalar'}));

end

function G = setParsedArgs(G,p)

NamedArgs = {'VoxelSize','VoxelCenter','GridSize','Nmajor','MajorAngle',...
    'NumMajorArteries','MinorArterialFrac','MinorDilation','Rmajor',...
    'Rminor_mu','Rminor_sig','VRSRelativeRad','Verbose','seed'};
TargetArgs = {'BVF','iRBVF','aRBVF','iBVF','aBVF'};

for f = NamedArgs; G.(f{1}) = p.Results.(f{1}); end
for f = TargetArgs; G.Targets.(f{1}) = p.Results.(f{1}); end

G.opts = p.Results;
G.opts = rmfield(G.opts,NamedArgs);
G.opts = rmfield(G.opts,TargetArgs);

if isempty(p.Results.Rmajor)
    G.opts.MajorVesselMode = 'aBVF_Fixed';
else
    G.opts.MajorVesselMode = 'Rmajor_Fixed';
end

if G.Verbose
    display_text('CylindricalVesselFilledVoxel: using the following default values:',75,'-',true,[1,1]);
    for f = p.UsingDefaults
        val = p.Results.(f{1});
        if isnumeric(val) || islogical(val)
            str = mat2str(val);
        elseif ischar(val)
            str = val;
        else
            str = strcat('\n',evalc('disp(val)'));
        end
        fprintf(['*** Using default: %s = ', str, '\n'], f{1});
    end
    fprintf('\n');
end

G.opts.parser = p;

end

function G = getDerivedArgs(G)

%==============================================================
% Derived Quantities
%==============================================================

switch upper(G.opts.MajorVesselMode)
    case 'ABVF_FIXED'
        [G] = CalculateBVFValues(G);
    case 'RMAJOR_FIXED'
        % do nothing
    otherwise
        error('Unknown option "MajorVesselMode = %s"', G.opts.MajorVesselMode)
end

G.SubVoxSize  = mean(G.VoxelSize./G.GridSize);
G.RminorFun  = @(varargin) G.Rminor_mu + G.Rminor_sig .* randn(varargin{:});

% Calculate blood volumes
Total_Volume   = prod(G.VoxelSize); % total volume of voxel [um^3]
Total_BloodVol = G.Targets.BVF * Total_Volume; % total blood volume (main and minor vessels)
Minor_BloodVol = G.Targets.iRBVF .* Total_BloodVol; % blood volume for minor vessels
Major_BloodVol = Total_BloodVol - Minor_BloodVol; % blood volume for major vessels

% If the radius 'r' is normally distributed ~ N(mu,sig), then the
% expectation of r^2, E[r^2], is given by E[r^2] = mu^2 + sig^2
Minor_Area = pi * ( G.Rminor_mu.^2 + G.Rminor_sig.^2 );

% Minor Volume ~ N*Area*Height (underestimated)
VoxHeight = G.VoxelSize(3);
% NumMinorVesselsGuess = round( Minor_BloodVol ./ (VoxHeight * Minor_Area) );

% Empirical model for average cylinder length:
%     see: GeometryGeneration/old/test/AvgLineLength.m
[xc,yc,zc] = deal(1, 2/3, 0);
[xr,yr,zr] = deal(1, 4/3, 1/2);
avgCylLengthFun = @(relX, relY) zc + zr * sqrt(1 - min((relX-xc).^2 / xr^2 + (relY-yc).^2 / yr^2, 1));

% avg length should be greater than the smallest dimension; if not,
% something is likely wrong with the empirical estimate, so should default
% to the smallLength
sVSize = sort(G.VoxelSize);
smallLength = min(G.VoxelSize);
avgCylLength = norm(G.VoxelSize) * avgCylLengthFun(sVSize(1)/sVSize(3), sVSize(2)/sVSize(3));
avgCylLength = max(avgCylLength, smallLength);

NumMinorVesselsGuess = round( Minor_BloodVol ./ (avgCylLength * Minor_Area) );

% Major blood vessel diameters: N*pi*r^2*len = V
majAngleRad = deg2rad(G.MajorAngle);
majorDir = [sin(majAngleRad), 0, cos(majAngleRad)];
[tmin, tmax, ~, ~] = rayBoxIntersection( G.VoxelCenter(:), majorDir(:), G.VoxelSize(:), G.VoxelCenter(:) );

MajorLength = tmax - tmin;
R_MajorGuess = sqrt( Major_BloodVol./( G.Nmajor * pi * MajorLength ) );

G.InitGuesses = struct( ...
    'N',      G.Nmajor + NumMinorVesselsGuess, ...
    'Nminor', NumMinorVesselsGuess, ...
    'Rmajor', R_MajorGuess ...
    );

[G.P,G.Vx,G.Vy,G.Vz] = deal(zeros(3,G.InitGuesses.N));

G.RmajorFun = @(varargin) G.InitGuesses.Rmajor * ones(varargin{:});
G.R = G.RmajorFun(1,G.InitGuesses.N);

G.MinorRadiusFactor = sqrt(G.MinorDilation);
G.isMinorDilated = false;

end

function [G] = CalculateBVFValues(G)

p = G.opts.parser;
BVFfields = {'BVF','iRBVF','aRBVF','iBVF','aBVF'};
BVFvalues = {[],[],[],[],[]};

NumSpecified = 0;
for ii = 1:numel(BVFfields)
    b = BVFfields{ii};
    if ~any(ismember(p.UsingDefaults,b))
        NumSpecified = NumSpecified + 1;
        BVFvalues{ii} = G.opts.parser.Results.(b);
    end
end

if ~( NumSpecified == 0 || NumSpecified == 2 )
    bvffieldstrings = strcat(BVFfields,',');
    bvffieldstrings{end} = [bvffieldstrings{end}(1:end-1),'.'];
    bvffieldstrings = strrep(bvffieldstrings,',',', ');
    error(['Must specify either 0 (use defaults) or 2 of: ' [bvffieldstrings{:}] ]);
end

if NumSpecified == 2 && (~isempty(BVFvalues{2}) && ~isempty(BVFvalues{3}))
    error('Cannot specify only iRBVF and aRBVF, as 1 == iRBVF + aRBVF');
end

%-------------------------------------------------------------%
% Convert inputs to BVF and iRBVF and set everything from there
%-------------------------------------------------------------%
if ~isempty(BVFvalues{1}) && ~isempty(BVFvalues{2})
    %BVF and iRBVF
    BVF = BVFvalues{1}; iRBVF = BVFvalues{2};
    G.Targets.BVF   = BVF;
    G.Targets.iRBVF = iRBVF;
    
elseif ~isempty(BVFvalues{1}) && ~isempty(BVFvalues{3})
    %BVF and aRBVF
    BVF = BVFvalues{1}; aRBVF = BVFvalues{3};
    G.Targets.BVF   = BVF;
    G.Targets.iRBVF = (1-aRBVF);
    
elseif ~isempty(BVFvalues{1}) && ~isempty(BVFvalues{4})
    %BVF and iBVF
    BVF = BVFvalues{1}; iBVF = BVFvalues{4};
    G.Targets.BVF   = BVF;
    G.Targets.iRBVF = iBVF/BVF;
    
elseif ~isempty(BVFvalues{1}) && ~isempty(BVFvalues{5})
    %BVF and aBVF
    BVF = BVFvalues{1}; aBVF = BVFvalues{5};
    G.Targets.BVF   = BVF;
    G.Targets.iRBVF = (1-aBVF/BVF);
    
    %--------- iRBVF and aRBVF is NOT enough information ---------%
    % elseif ~isempty(BVFvalues{2}) && ~isempty(BVFvalues{3})
    %-------------------------------------------------------------%
    
elseif ~isempty(BVFvalues{2}) && ~isempty(BVFvalues{4})
    %iRBVF and iBVF
    iRBVF = BVFvalues{2}; iBVF = BVFvalues{4};
    G.Targets.BVF   = iBVF/iRBVF;
    G.Targets.iRBVF = iRBVF;
    
elseif ~isempty(BVFvalues{2}) && ~isempty(BVFvalues{5})
    %iRBVF and aBVF
    iRBVF = BVFvalues{2}; aBVF = BVFvalues{5};
    G.Targets.BVF   = aBVF/(1-iRBVF);
    G.Targets.iRBVF = iRBVF;
    
elseif ~isempty(BVFvalues{3}) && ~isempty(BVFvalues{4})
    %aRBVF and iBVF
    aRBVF = BVFvalues{3}; iBVF = BVFvalues{4};
    G.Targets.BVF   = iBVF/(1-aRBVF);
    G.Targets.iRBVF = (1-aRBVF);
    
elseif ~isempty(BVFvalues{3}) && ~isempty(BVFvalues{5})
    %aRBVF and aBVF
    aRBVF = BVFvalues{3}; aBVF = BVFvalues{5};
    G.Targets.BVF   = aBVF/aRBVF;
    G.Targets.iRBVF = (1-aRBVF);
    
elseif ~isempty(BVFvalues{4}) && ~isempty(BVFvalues{5})
    %iBVF and aBVF
    iBVF = BVFvalues{4}; aBVF = BVFvalues{5};
    G.Targets.BVF   = (iBVF+aBVF);
    G.Targets.iRBVF = iBVF/(iBVF+aBVF);
end

G.Targets.iBVF  = G.Targets.iRBVF * G.Targets.BVF;
G.Targets.aBVF  = G.Targets.BVF - G.Targets.iBVF;
G.Targets.aRBVF = 1 - G.Targets.iRBVF;

if G.Verbose; disp(G.Targets); end

end

function G = parse_SimSettings_Params( G, SimSettings, Params, varargin )

%==============================================================
% Store SimSettings and Params
%==============================================================
G.SimSettings = SimSettings;
G.Params = Params;

%==============================================================
% Extract immutable parameters from SimSettings and Params
%==============================================================
G.VoxelSize   = SimSettings.VoxelSize;
G.SubVoxSize  = SimSettings.SubVoxSize;
G.VoxelCenter = SimSettings.VoxelCenter;
G.GridSize    = SimSettings.GridSize;
G.Nmajor      = SimSettings.NumMajorVessels;
G.seed        = rng;

G.Rminor_mu  = SimSettings.R_Minor_mu;
G.Rminor_sig = SimSettings.R_Minor_sig;
G.RminorFun  = @(varargin) G.Rminor_mu + G.Rminor_sig .* randn(varargin{:});

%==============================================================
% Goal BVF, etc.
%==============================================================
G.Targets = struct( ...
    'BVF',   Params.Total_BVF, ...
    'iRBVF', Params.MinorVessel_RelBVF ...
    );
G.Targets.iBVF  = G.Targets.iRBVF * G.Targets.BVF;
G.Targets.aBVF  = G.Targets.BVF - G.Targets.iBVF;
G.Targets.aRBVF = 1 - G.Targets.iRBVF;

%==============================================================
% Initial guesses
%==============================================================
G.InitGuesses = struct( ...
    'N',      G.Nmajor + Params.NumMinorVessels, ...
    'Nminor', Params.NumMinorVessels, ...
    'Rmajor', Params.R_Major ...
    );

[G.P,G.Vx,G.Vy,G.Vz] = deal(zeros(3,G.InitGuesses.N));

G.RmajorFun = @(varargin) G.InitGuesses.Rmajor * ones(varargin{:});
G.R = G.RmajorFun(1,G.InitGuesses.N);

end

function p = getSimSettingsParamsInputParser()

p = inputParser;
p.FunctionName = 'CylindricalVesselFilledVoxel';

VA = @(varargin) validateattributes(varargin{:});
VS = @(varargin) validatestring(varargin{:}); %note: case-insensitive

%-------------------------------------------------------------------------%
% Optional parameters
%-------------------------------------------------------------------------%
% Allow intersecting cylinders or not
addParameter(p,'AllowMinorSelfIntersect',true,@(x)VA(x,{'logical'},{'scalar'}));
addParameter(p,'AllowMinorMajorIntersect',false,@(x)VA(x,{'logical'},{'scalar'}));

% Minor vessel orientation
expectedinterptype = {'RANDOM','ALIGNED','PERIODIC'};
addParameter(p,'MinorOrientation','RANDOM',@(x) any(VS(x,expectedinterptype)));

% Minor/major dilation factors
addParameter(p,'MinorDilation',1.0,@(x)VA(x,{'scalar'},{'positive'}));

% Populate Idx field, or leave blank
addParameter(p,'PopulateIdx',true,@(x)VA(x,{'logical'},{'scalar'}));

end
