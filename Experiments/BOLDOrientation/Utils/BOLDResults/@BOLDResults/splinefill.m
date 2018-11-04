function [InterpResults] = splinefill(Results, AllTEs, AllAngles, plotresult)
%SPLINEFILL Fills in `Results` with times `TEs` and angles `Angles`.
% Plots results if `plot` is `true`.

if nargin < 4; plotresult = false; end

[TEs, Angles] = deal(Results.EchoTimes, Results.Alphas);
[TEsGrid, AnglesGrid] = meshgrid(TEs(:), Angles(:));
[AllTEsGrid, AllAnglesGrid] = meshgrid(AllTEs(:), AllAngles(:));

NumTEs = numel(AllTEs);
NumAngles = numel(AllAngles);
ResultsArgs = getargs(Results);
InterpResults = BOLDResults(AllTEs, AllAngles, ResultsArgs{3:end});
InterpResults.MetaData = Results.MetaData;

for ii = 1:NumAngles
    
    S0 = repmat(Results.DeoxySignal(1), NumTEs, NumAngles); % initial signal is always the same
    S = cellfun(@(x) x(end,2), Results.DeoxySignal).'; % second column is the complex signal
    S = interp2(TEsGrid, AnglesGrid, S, AllTEsGrid, AllAnglesGrid, 'spline'); % interpolate S onto new grid
    S = cellfun(@(t,s) [t,s], num2cell(AllTEsGrid.'), num2cell(S.'), 'uniformoutput', false); % cell array of 1x2 row vectors [t, Scplx]
    InterpDeoxySignal = cellfun(@(s0,s) [s0;s], S0, S, 'uniformoutput', false); % cell array of 2x2 matrices [t0, Scplx0; t, Scplx]
    
    S0 = repmat(Results.OxySignal(1), NumTEs, NumAngles);
    S = cellfun(@(x) x(end,2), Results.OxySignal).';
    S = interp2(TEsGrid, AnglesGrid, S, AllTEsGrid, AllAnglesGrid, 'spline');
    S = cellfun(@(t,s) [t,s], num2cell(AllTEsGrid.'), num2cell(S.'), 'uniformoutput', false);
    InterpOxySignal = cellfun(@(s0,s) [s0;s], S0, S, 'uniformoutput', false);
    
    InterpResults = push(InterpResults, InterpDeoxySignal, InterpOxySignal, AllTEs, AllAngles, ResultsArgs{3:end});
    
end

if plotresult
    plot(Results, 'scalefactor', 'none')
    plot(InterpResults, 'scalefactor', 'none')

    % figure, surf(1000*TEsGrid, 180/pi*AnglesGrid, BOLDs/max(BOLDs(:))); xlabel('time [ms]'); ylabel('angles [deg]'); zlabel('original'); zlim([0,1]);
    % figure, surf(1000*AllTEsGrid, 180/pi*AllAnglesGrid, InterpBOLDs/max(InterpBOLDs(:))); xlabel('time [ms]'); ylabel('angles [deg]'); zlabel('interp'); zlim([0,1]);
end

end