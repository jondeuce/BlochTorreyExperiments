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
addParameter(p,'MediumVesselRadiusThresh', 0.0);
        
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

switch upper(G.opts.MajorVesselMode)
    case 'ABVF_FIXED'
        [G] = CalculateBVFValues(G);
    case 'RMAJOR_FIXED'
        % do nothing
    otherwise
        error('Unknown option "MajorVesselMode = %s"', G.opts.MajorVesselMode)
end

% unpack so that RminorFun doesn't close over G
[Rminor_mu, Rminor_sig] = deal(G.Rminor_mu, G.Rminor_sig);
G.RminorFun  = @(varargin) Rminor_mu + Rminor_sig .* randn(varargin{:});

% Calculate blood volumes
G.SubVoxSize  = mean(G.VoxelSize./G.GridSize);
Total_Volume   = prod(G.VoxelSize); % total volume of voxel [um^3]
Total_BloodVol = G.Targets.BVF * Total_Volume; % total blood volume (main and minor vessels)
Minor_BloodVol = G.Targets.iRBVF .* Total_BloodVol; % blood volume for minor vessels
Major_BloodVol = Total_BloodVol - Minor_BloodVol; % blood volume for major vessels

% If the radius 'r' is normally distributed ~ N(mu,sig), then the
% expectation of r^2, E[r^2], is given by E[r^2] = mu^2 + sig^2
Minor_Area = pi * ( G.Rminor_mu.^2 + G.Rminor_sig.^2 );

% Minor Volume ~ N*Area*Height (underestimated)
% VoxHeight = G.VoxelSize(3);
% NumMinorVesselsGuess = round( Minor_BloodVol ./ (VoxHeight * Minor_Area) );

% ----------------- %
% ------ OLD ------ %

% % Empirical model for average cylinder length:
% %     see: GeometryGeneration/old/test/AvgLineLength.m
% [xc,yc,zc] = deal(1, 2/3, 0);
% [xr,yr,zr] = deal(1, 4/3, 1/2);
% avgCylLengthFun = @(relX, relY) zc + zr * sqrt(1 - min((relX-xc).^2 / xr^2 + (relY-yc).^2 / yr^2, 1));
% 
% % avg length should be greater than the smallest dimension; if not,
% % something is likely wrong with the empirical estimate, so should default
% % to the smallLength
% sVSize = sort(G.VoxelSize);
% smallLength = min(G.VoxelSize);
% avgCylLength = norm(G.VoxelSize) * avgCylLengthFun(sVSize(1)/sVSize(3), sVSize(2)/sVSize(3));
% avgCylLength = max(avgCylLength, smallLength);

% % If you model two random vectors X = (x,y,z) and X0 = (x0,y0,z0) as being
% % drawn uniformly randomly in a domain [0,a]x[0,b]x[0,c], then the
% % expectation E(|X-X0|^2) = (a^2+b^2+c^2)/6.
% % An (over-)estimation of the average length of each cylinder, then, is
% % sqrt((a^2+b^2+c^2)/6). Simulating this empirically, the over-estimation
% % is never more than ~18%, and never less than ~6%. Since we would over-
% % estimating the number of cylinders, the expected length is reduced by 25%
% avgCylLength = norm(G.VoxelSize)/6; % * 0.75;

% f = @(X) sqrt(((X(:,1)-X(:,2)).^2 + X(:,1).^2 + X(:,2).^2)) + ...
%          sqrt(((X(:,3)-X(:,4)).^2 + X(:,3).^2 + X(:,4).^2)) + ...
%          sqrt(((X(:,5)-X(:,6)).^2 + X(:,5).^2 + X(:,6).^2));
% a = G.VoxelSize(1); b = G.VoxelSize(2); c = G.VoxelSize(3); 
% bd = [0 a; 0 a; 0 b; 0 b; 0 c; 0 c];
% I = integralN_mc(f, bd, 'k', 1, 'reltol', 1e-12, 'abstol', 1e-8);
% avgCylLength = I/(3*(a*b*c)^2);

% ---- END OLD ---- %
% ----------------- %

% Just simulate it! Generate N random cylinder intersections, take the
% average, and use this for the initial guess.
N = 100000;
% a = G.VoxelSize(1); b = G.VoxelSize(2); c = G.VoxelSize(3); 
% Origins = [a*rand(1,N); b*rand(1,N); c*rand(1,N)];
% Directions = randn(3,N);
% Directions = bsxfun(@rdivide, Directions, sqrt(sum(Directions.^2, 1)));
[Origins, Directions, ~] = sampleRandomCylinders( G.VoxelSize, G.VoxelCenter, [], N );
[tmin, tmax] = rayBoxIntersection( Origins, Directions, G.VoxelSize, G.VoxelCenter );
avgCylLength = mean(tmax - tmin);

% Expected number of simply minor blood vol divided by expected vessel volume
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

Rmajor = G.InitGuesses.Rmajor; % Unpack so RmajorFun doesn't close over G
G.RmajorFun = @(varargin) Rmajor .* ones(varargin{:});
G.R = G.RmajorFun(1,G.InitGuesses.N);

G.MinorRadiusFactor = sqrt(G.MinorDilation);
G.MajorRadiusFactor = sqrt(G.MajorDilation);
G.isMinorDilated = false;
G.isMajorDilated = false;

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
