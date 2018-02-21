function [ h_time, h_angle ] = plot_Signal( Results, AvgResults, Params, SimSettings )
%PLOT_SIGNAL Plots the resulting Signal values before and after contrast
% agent arrival from the simulation.

PlotAnything	=   SimSettings.flags.PlotAnything;
PlotAnything	=   PlotAnything && ...
                    ~( isempty(AvgResults) && ~SimSettings.flags.PlotAllReps );
PlotAnything	=   PlotAnything && ...
                    ~( isempty(Results) && isempty(AvgResults) );

PlotTimeSeries	=   PlotAnything && SimSettings.flags.PlotSigvsTime;
PlotAngleSeries	=	PlotAnything && SimSettings.flags.PlotSigvsAngles;

h_time	=   [];
h_angle	=   [];

if PlotTimeSeries
    
    if isempty(Results)
        for ii = 1:size(AvgResults,1)
            h	=   plot_Signal_TimeSeries( [], AvgResults(ii,:), Params(ii,:), SimSettings );
            h_time	=   [ h_time; h ];
        end
    elseif isempty(AvgResults)
        for ii = 1:size(Results,1)
            h	=   plot_Signal_TimeSeries( Results(ii,:), [], Params(ii,:), SimSettings );
            h_time	=   [ h_time; h ];
        end
    else
        for ii = 1:size(Results,1)
            h	=   plot_Signal_TimeSeries( Results(ii,:), AvgResults(ii,:), Params(ii,:), SimSettings );
            h_time	=   [ h_time; h ];
        end
    end
    
end

if PlotAngleSeries

    if isempty(Results)
        for ii = 1:size(AvgResults,1)
            h	=   plot_Signal_AngleSeries( [], AvgResults(ii,:), Params(ii,:), SimSettings );
            h_angle	=   [ h_angle; h ];
        end
    elseif isempty(AvgResults)
        for ii = 1:size(Results,1)
            h	=   plot_Signal_AngleSeries( Results(ii,:), [], Params(ii,:), SimSettings );
            h_angle	=   [ h_angle; h ];
        end
    else
        for ii = 1:size(Results,1)
            h	=   plot_Signal_AngleSeries( Results(ii,:), AvgResults(ii,:), Params(ii,:), SimSettings );
            h_angle	=   [ h_angle; h ];
        end
    end
    
end

end

function H = plot_Signal_TimeSeries( Results, AvgResults, Params, SimSettings )

Angles_Deg	=   SimSettings.Angles_Deg_Data(:); % [deg]
if ~isempty(Results),   Time	=	1000 * Results(1).Time; % [ms]
else                    Time	=	1000 * AvgResults(1).Time; % [ms]
end

NumResults	=	numel(Results);
NumAngles	=   numel(Angles_Deg);
NumTimes	=   numel(Time);

% LS = linestyle, LW = linewidth, MS = markersize
if isempty(AvgResults)
    LS_Res	=   '-';    LW_Res	=	4;	MS_Res	=   10;
    LS_Avg	=   [];     LW_Avg	=   [];	MS_Avg	=   [];
else
    LS_Res	=   '--';	LW_Res	=	2;  MS_Res	=   5;	
    LS_Avg	=   '-';    LW_Avg	=   4;	MS_Avg	=   10;
end

% Colours
Colours	=	distinguishable_colors( numel( Angles_Deg ) );

% legend labels
labels	=   cell(1,NumAngles);
for ii = 1:length(labels)
    labels{ii}	=   sprintf('%5.2f deg',Angles_Deg(ii));
end

%--------------------------------------------------------------------------
% Plot initial signal S0
%--------------------------------------------------------------------------
H = figure;
hold on, grid on

% Plot all results
if ~isempty(Results)
    for ii = 1:NumResults
        for jj = 1:NumAngles
            S0	=   Results(ii).Signal_noCA{jj}(:);
            h(jj)	=   plot( Time(:), abs(S0), LS_Res, 'linewidth', LW_Res, ...
                'markersize', MS_Res, 'color', Colours(jj,:) );
            S	=   Results(ii).Signal_CA{jj}(:);
            h(jj)	=   plot( Time(:), abs(S), LS_Res, 'linewidth', LW_Res, ...
                'markersize', MS_Res, 'color', Colours(jj,:) );
        end
    end
end

% Overlay average results on plot
if ~isempty(AvgResults)
    for jj = 1:NumAngles
        S0	=   AvgResults.Signal_noCA{jj}(:);
        h(jj)	=   plot( Time(:), abs(S0), LS_Avg, 'linewidth', LW_Avg, ...
            'markersize', MS_Avg, 'color', Colours(jj,:) );
        S	=   AvgResults.Signal_CA{jj}(:);
        h(jj)	=   plot( Time(:), abs(S), LS_Avg, 'linewidth', LW_Avg, ...
            'markersize', MS_Avg, 'color', Colours(jj,:) );
    end
end

% Create Legend
legend( h, labels{:}, 'location', 'northwest' );

% Add titles
xlabel( 'Time [ms]' ); ylabel( 'Signal [J/T]' );
titlestr	=	sprintf( ...
    'Signal vs. Time: CA = %0.4f mM, Num Major Vessels = %d, BVF = %0.4f%%, Minor Vessel Rel-BVF = %0.4f%%',     ...
    Params.CA_Concentration, SimSettings.NumMajorVessels, 100*Params.Total_BVF, 100*Params.MinorVessel_RelBVF );

title( titlestr );

% Draw plot
drawnow

end

function H = plot_Signal_AngleSeries( Results, AvgResults, Params, SimSettings )

Angles_Deg	=   SimSettings.Angles_Deg_Data(:); % [deg]

if ~isempty(Results),   Time	=	1000 * Results(1).Time; % [ms]
else                    Time	=	1000 * AvgResults(1).Time; % [ms]
end
TotNumTimes	=   numel(Time);
TimeIdx     =   round(linspace(1,TotNumTimes,min(TotNumTimes,3)));
Time        =   Time(TimeIdx);

NumAngles	=   numel(Angles_Deg);
NumTimes	=   numel(Time);
NumResults	=   numel(Results);

if ~isempty( Results )
    Signal_CA	=   reshape( cell2mat( [Results.Signal_CA] ),      ...
                            [NumAngles,TotNumTimes,NumResults]	);
    Signal_CA	=   Signal_CA(:,TimeIdx,:);
    Signal_noCA	=   reshape( cell2mat( [Results.Signal_noCA] ),      ...
                            [NumAngles,TotNumTimes,NumResults]	);
    Signal_noCA	=   Signal_noCA(:,TimeIdx,:);
end

if ~isempty( AvgResults )
    S_avg	=	cell2mat( AvgResults.Signal_CA );
    S_avg	=   S_avg(:,TimeIdx,:);
    S0_avg	=	cell2mat( AvgResults.Signal_noCA );
    S0_avg	=   S0_avg(:,TimeIdx,:);
end

% LS = linestyle, LW = linewidth, MS = markersize
if isempty(AvgResults)
    LS_Res	=   '-';    LW_Res	=	4;	MS_Res	=   10;
    LS_Avg	=   [];     LW_Avg	=   [];	MS_Avg	=   [];
else
    LS_Res	=   '--';	LW_Res	=	2;  MS_Res	=   5;	
    LS_Avg	=   '-';    LW_Avg	=   4;	MS_Avg	=   10;
end

Colours	=	distinguishable_colors( NumTimes );
labels	=   cell(1,NumTimes);
for ii = 1:length(labels)
    labels{ii}	=   sprintf('%5.2f ms',Time(ii));
end

%--------------------------------------------------------------------------
% Plot initial signal S0
%--------------------------------------------------------------------------
H = figure;
hold on, grid on

% Plot all results
if ~isempty(Results)
    for ii = 1:NumResults
        for jj = 1:NumTimes
            S0	=   Signal_noCA(:,jj,ii);
            h(jj)	=   plot( Angles_Deg(:), abs(S0), LS_Res, 'linewidth', LW_Res, ...
                'markersize', MS_Res, 'color', Colours(jj,:) );
        end
    end
end

% Overlay average results on plot
if ~isempty(AvgResults)
    for jj = 1:NumTimes
        S0	=   S0_avg(:,jj);
        h(jj)	=   plot( Angles_Deg(:), abs(S0), LS_Avg, 'linewidth', LW_Avg, ...
            'markersize', MS_Avg, 'color', Colours(jj,:) );
    end
end

% Create Legend
legend( h, labels{:}, 'location', 'northwest' );

% Add titles
xlabel( 'Angle [deg]' ); ylabel( 'Signal []' );
titlestr	=	sprintf( ...
    'Initial Signal vs. Angle: CA = %0.4f mM, Num Major Vessels = %d, BVF = %0.4f%%, Minor Vessel Rel-BVF = %0.4f%%',     ...
    Params.CA_Concentration, SimSettings.NumMajorVessels, 100*Params.Total_BVF, 100*Params.MinorVessel_RelBVF );

title( titlestr );

% Draw plot
drawnow

%--------------------------------------------------------------------------
% Plot final signal S
%--------------------------------------------------------------------------
H = figure;
hold on, grid on

% Plot all results
if ~isempty(Results)
    for ii = 1:NumResults
        for jj = 1:NumTimes
            S	=   Signal_CA(:,jj,ii);
            h(jj)	=   plot( Angles_Deg(:), abs(S), LS_Res, 'linewidth', LW_Res, ...
                'markersize', MS_Res, 'color', Colours(jj,:) );
        end
    end
end

% Overlay average results on plot
if ~isempty(AvgResults)
    for jj = 1:NumTimes
        S	=   S_avg(:,jj);
        h(jj)	=   plot( Angles_Deg(:), abs(S), LS_Avg, 'linewidth', LW_Avg, ...
            'markersize', MS_Avg, 'color', Colours(jj,:) );
    end
end

% Create Legend
legend( h, labels{:}, 'location', 'northwest' );

% Add titles
xlabel( 'Angle [deg]' ); ylabel( 'Signal [J/T]' );
titlestr	=	sprintf( ...
    'Final Signal vs. Angle: CA = %0.4f mM, Num Major Vessels = %d, BVF = %0.4f%%, Minor Vessel Rel-BVF = %0.4f%%',     ...
    Params.CA_Concentration, SimSettings.NumMajorVessels, 100*Params.Total_BVF, 100*Params.MinorVessel_RelBVF );

title( titlestr );

% Draw plot
drawnow

end


