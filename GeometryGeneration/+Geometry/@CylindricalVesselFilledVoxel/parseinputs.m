function G = parseinputs( G, varargin )
%PARSEKEYWORDARGS Parses keyword and required arguments for class
% CylindricalVesselFilledVoxel.

% Get parser and parse name-value args
p = getInputParser();
parse(p,varargin{:});
G = setParsedArgs(G,p);
G = getDerivedArgs(G);

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

addParameter(p,'MediumVessels', []);
addParameter(p,'MediumVesselRadiusThresh', Inf);

%-------------------------------------------------------------------------%
% Optional parameters
%-------------------------------------------------------------------------%
% Allow intersecting cylinders or not
addParameter(p,'AllowMinorSelfIntersect',true,@(x)VA(x,{'logical'},{'scalar'}));
addParameter(p,'AllowMinorMajorIntersect',true,@(x)VA(x,{'logical'},{'scalar'}));
addParameter(p,'ImproveMajorBVF',true,@(x)VA(x,{'logical'},{'scalar'}));
addParameter(p,'ImproveMinorBVF',true,@(x)VA(x,{'logical'},{'scalar'}));

% Allow pruning of initial minor cylinders, i.e. if minor cylinders don't
% intersect with the voxel, remove them
addParameter(p,'AllowInitialMinorPruning',true,@(x)VA(x,{'logical'},{'scalar'}));

% Major vessel distribution
majordisttypes = {'REGULAR', 'LINE'};
addParameter(p,'MajorDistribution','REGULAR',@(x) any(VS(x,majordisttypes)));

% Minor vessel orientation
minororientationtypes = {'RANDOM', 'ALIGNED', 'PERIODIC'};
addParameter(p,'MinorOrientation','RANDOM',@(x) any(VS(x,minororientationtypes)));

% Minor/major dilation factors
addParameter(p,'MinorDilation',1.0);%,@(x)VA(x,{'scalar'},{'positive'}));
addParameter(p,'MajorDilation',1.0);%,@(x)VA(x,{'scalar'},{'positive'}));

% Populate Idx field, or leave blank
addParameter(p,'PopulateIdx',true,@(x)VA(x,{'logical'},{'scalar'}));

end

function G = setParsedArgs(G,p)

NamedArgs = {'VoxelSize','VoxelCenter','GridSize','Nmajor','MajorAngle',...
    'NumMajorArteries','MinorArterialFrac',...
    'MediumVessels','MediumVesselRadiusThresh',...
    'Rmajor','Rminor_mu','Rminor_sig',...
    'VRSRelativeRad','MinorDilation','MajorDilation',...
    'Verbose','seed'};
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

% Compute target BVF values
[G] = CalculateTargetBVFValues(G);

% Unpack so that RminorFun doesn't close over G
[Rminor_mu, Rminor_sig] = deal(G.Rminor_mu, G.Rminor_sig);
G.RminorFun  = @(varargin) Rminor_mu + Rminor_sig .* randn(varargin{:});

% Calculate blood volumes
G.SubVoxSize  = mean(G.VoxelSize./G.GridSize);
TotalVolume   = prod(G.VoxelSize); % total volume of voxel [um^3]
TotalBloodVol = G.Targets.BVF * TotalVolume; % total blood volume (main and minor vessels)
MinorBloodVol = G.Targets.iRBVF .* TotalBloodVol; % blood volume for minor vessels
MajorBloodVol = TotalBloodVol - MinorBloodVol; % blood volume for major vessels

% If the radius 'r' is normally distributed ~ N(mu,sig), then the
% expectation of r^2, E[r^2], is given by E[r^2] = mu^2 + sig^2
Minor_Area = pi * ( G.Rminor_mu.^2 + G.Rminor_sig.^2 );

% Generate N random cylinders and use their average length to compute the initial guesses
N = 100000;
[Origins, Directions, ~] = sampleRandomCylinders( G.VoxelSize, G.VoxelCenter, [], N );
[tmin, tmax] = rayBoxIntersection( Origins, Directions, G.VoxelSize, G.VoxelCenter );
avgCylLength = mean(tmax - tmin);

% Expected number of simply minor blood vol divided by expected vessel volume
NumMinorVesselsGuess = round( MinorBloodVol ./ (avgCylLength * Minor_Area) );

switch upper(G.opts.MajorVesselMode)
    case 'ABVF_FIXED'
        % Major blood vessel diameters: N*pi*r^2*len = V
        MajorDir = [sind(G.MajorAngle), 0, cosd(G.MajorAngle)];
        [tmin, tmax, ~, ~] = rayBoxIntersection( G.VoxelCenter(:), MajorDir(:), G.VoxelSize(:), G.VoxelCenter(:) );
        
        MajorLength = tmax - tmin;
        RMajorInit  = sqrt( MajorBloodVol./( G.Nmajor * pi * MajorLength ) );
    case 'RMAJOR_FIXED'
        RMajorInit  = G.Rmajor;
    otherwise
        error('Unknown option "MajorVesselMode = %s"', G.opts.MajorVesselMode)
end

G.InitGuesses = struct( ...
    'N',      G.Nmajor + NumMinorVesselsGuess, ...
    'Nminor', NumMinorVesselsGuess, ...
    'Rmajor', RMajorInit ...
    );

[G.P,G.Vx,G.Vy,G.Vz] = deal(zeros(3,G.InitGuesses.N));

Rmajor = G.InitGuesses.Rmajor; % Unpack so RmajorFun doesn't close over G
G.RmajorFun = @(varargin) Rmajor .* ones(varargin{:});
G.R = G.RmajorFun(1,G.InitGuesses.N);

G.MinorRadiusFactor = sqrt(G.MinorDilation);
G.MajorRadiusFactor = sqrt(G.MajorDilation);
G.isMinorDilated = false;
G.isMajorDilated = false;

end

function [G] = CalculateTargetBVFValues(G)

p = G.opts.parser;

% Any two of these parameters determines target values
BVF    = p.Results.BVF;
iRBVF  = p.Results.iRBVF;
aRBVF  = p.Results.aRBVF;
iBVF   = p.Results.iBVF;
aBVF   = p.Results.aBVF;
Rmajor = p.Results.Rmajor;

% Rmajor determines aBVF
if ~isempty(Rmajor)
    % Major blood vessel diameters: N*pi*r^2*len = V (ignoring pathological cases of collision with corners/edges)
    MajorDir = [sind(G.MajorAngle), 0, cosd(G.MajorAngle)];
    [tmin, tmax, ~, ~] = rayBoxIntersection( G.VoxelCenter(:), MajorDir(:), G.VoxelSize(:), G.VoxelCenter(:) );
    MajorLength = tmax - tmin;
    TotalVolume = prod(G.VoxelSize);
    aBVF = Rmajor^2 * ( G.Nmajor * pi * MajorLength ) / TotalVolume;
end

numset = ~isempty(BVF) + ~isempty(iRBVF) + ~isempty(aRBVF) + ~isempty(iBVF) + ~isempty(aBVF);
if numset == 1
    error('Must specify either 0 (use defaults) or 2 of: BVF, iRBVF, aRBVF, iBVF, aBVF, Rmajor');
elseif numset == 2 && ~isempty(iRBVF) && ~isempty(aRBVF)
    error('Cannot specify only iRBVF and aRBVF, as 1 == iRBVF + aRBVF');
end

%-------------------------------------------------------------%
% Convert inputs to BVF and iRBVF and set everything from there
%-------------------------------------------------------------%
if ~isempty(BVF) && ~isempty(iRBVF)
    BVF   = BVF;
    iRBVF = iRBVF;
    
elseif ~isempty(BVF) && ~isempty(aRBVF)
    BVF   = BVF;
    iRBVF = (1-aRBVF);
    
elseif ~isempty(BVF) && ~isempty(iBVF)
    BVF   = BVF;
    iRBVF = iBVF/BVF;
    
elseif ~isempty(BVF) && ~isempty(aBVF)
    BVF   = BVF;
    iRBVF = (1-aBVF/BVF);
    
% elseif ~isempty(iRBVF) && ~isempty(aRBVF) % iRBVF and aRBVF is not enough information
    
elseif ~isempty(iRBVF) && ~isempty(iBVF)
    BVF   = iBVF/iRBVF;
    iRBVF = iRBVF;
    
elseif ~isempty(iRBVF) && ~isempty(aBVF)
    BVF   = aBVF/(1-iRBVF);
    iRBVF = iRBVF;
    
elseif ~isempty(aRBVF) && ~isempty(iBVF)
    BVF   = iBVF/(1-aRBVF);
    iRBVF = (1-aRBVF);
    
elseif ~isempty(aRBVF) && ~isempty(aBVF)
    BVF   = aBVF/aRBVF;
    iRBVF = (1-aRBVF);
    
elseif ~isempty(iBVF) && ~isempty(aBVF)
    BVF   = (iBVF+aBVF);
    iRBVF = iBVF/(iBVF+aBVF);
end

G.Targets.BVF   = BVF;
G.Targets.iRBVF = iRBVF;
G.Targets.iBVF  = iRBVF * BVF;
G.Targets.aBVF  = BVF * (1 - iRBVF);
G.Targets.aRBVF = 1 - iRBVF;

end
