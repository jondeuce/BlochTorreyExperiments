function G = FillWithRandomCylinders(G)

t_StartTime = tic;

%==========================================================================
% Set the random number generator to the given seed
%==========================================================================
prevstate = rng(G.seed);

%==========================================================================
% Calculate Initial Guess for Major/Minor cylinders
%==========================================================================

t_InitialGuess = tic;
G = MajorMinorInitialGuess(G);
t_InitialGuess = toc(t_InitialGuess);
if G.Verbose; display_toc_time( t_InitialGuess, 'major/minor cylinders initial guess' ); end

%==========================================================================
% Improve Major Cylinders
%==========================================================================

t_ImproveMajor = tic;
G = ImproveMajorBVF(G);
t_ImproveMajor = toc(t_ImproveMajor);
if G.Verbose; display_toc_time( t_ImproveMajor, 'impoving major cylinders BVF', 0 ); end

%==========================================================================
% Improve Minor Cylinders
%==========================================================================

t_ImproveMinor = tic;
G = ImproveMinorBVF(G);
t_ImproveMinor = toc(t_ImproveMinor);
if G.Verbose; display_toc_time( t_ImproveMinor, 'impoving minor cylinders BVF', 0 ); end

%==========================================================================
% Calculate Vascular Map from Major/Minor Cylinders
%==========================================================================

G = CalculateVasculatureMap( G );

%==========================================================================
% Add arteries
%==========================================================================

t_AddArteries = tic;
G = AddArteries(G);
t_AddArteries = toc(t_AddArteries);
if G.Verbose; display_toc_time( t_AddArteries, 'adding arteries', 0 ); end

%==========================================================================
% Show Resulting Accuracy Info
%==========================================================================

if G.Verbose; ShowBVFResults(G); end

%==========================================================================
% Return the random number generator to the previous state
%==========================================================================
rng(prevstate);

%==========================================================================
% Return Geometry object, recording timing info
%==========================================================================

t_TotalTime = toc(t_StartTime);
if G.Verbose; display_toc_time( t_TotalTime, 'total cylinder construction time', 0 ); end

G.MetaData.Timings = struct( ...
    'GenerateInitGuess', t_InitialGuess, ...
    'ImproveMajorCyls',  t_ImproveMajor, ...
    'ImproveMinorCyls',  t_ImproveMinor, ...
    'TotalCylTime',      t_TotalTime     ...
    );

end
