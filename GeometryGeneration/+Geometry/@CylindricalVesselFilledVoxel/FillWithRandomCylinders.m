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
display_toc_time( t_InitialGuess, 'major/minor cylinders initial guess' );

%==========================================================================
% Improve Major Cylinders
%==========================================================================

t_ImproveMajor = tic;
G = ImproveMajorBVF(G);
t_ImproveMajor = toc(t_ImproveMajor);
display_toc_time( t_ImproveMajor, 'impoving major cylinders BVF', 0 );

%==========================================================================
% Improve Minor Cylinders
%==========================================================================

t_ImproveMinor = tic;
G = ImproveMinorBVF(G);
t_ImproveMinor = toc(t_ImproveMinor);
display_toc_time( t_ImproveMinor, 'impoving minor cylinders BVF', 0 );

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
display_toc_time( t_AddArteries, 'adding arteries', 0 );

%==========================================================================
% Show Resulting Accuracy Info
%==========================================================================

ShowBVFResults(G);

%==========================================================================
% Return the random number generator to the previous state
%==========================================================================
rng(prevstate);

%==========================================================================
% Return Geometry object, recording timing info
%==========================================================================

t_TotalTime = toc(t_StartTime);
display_toc_time( t_TotalTime, 'total cylinder construction time', 0 );

G.MetaData.Timings = struct( ...
    'GenerateInitGuess', t_InitialGuess, ...
    'ImproveMajorCyls',  t_ImproveMajor, ...
    'ImproveMinorCyls',  t_ImproveMinor, ...
    'TotalCylTime',      t_TotalTime     ...
    );

end
