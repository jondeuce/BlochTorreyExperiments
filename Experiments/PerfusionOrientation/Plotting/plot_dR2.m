function [ h_time, h_angle ] = plot_dR2( Results, AvgResults, Params, SimSettings )
%PLOT_DR2 Plots the resulting dR2 values from the simulation

PlotAnything	=   SimSettings.flags.PlotAnything;
PlotAnything	=   PlotAnything && ...
                    ~( isempty(AvgResults) && ~SimSettings.flags.PlotAllReps );
PlotAnything	=   PlotAnything && ...
                    ~( isempty(Results) && isempty(AvgResults) );

PlotTimeSeries	=   PlotAnything && SimSettings.flags.PlotR2vsTime;
PlotAngleSeries	=	PlotAnything && SimSettings.flags.PlotR2vsAngles;

h_time	=   [];
h_angle	=   [];

if PlotTimeSeries
    
    if isempty(Results)
        for ii = 1:size(AvgResults,1)
            h	=   plot_dR2_TimeSeries( [], AvgResults(ii,:), Params(ii,:), SimSettings );
            h_time	=   [ h_time; h ];
        end
    elseif isempty(AvgResults)
        for ii = 1:size(Results,1)
            h	=   plot_dR2_TimeSeries( Results(ii,:), [], Params(ii,:), SimSettings );
            h_time	=   [ h_time; h ];
        end
    else
        for ii = 1:size(Results,1)
            h	=   plot_dR2_TimeSeries( Results(ii,:), AvgResults(ii,:), Params(ii,:), SimSettings );
            h_time	=   [ h_time; h ];
        end
    end
    
end

if PlotAngleSeries

    if isempty(Results)
        for ii = 1:size(AvgResults,1)
            h	=   plot_dR2_AngleSeries( [], AvgResults(ii,:), Params(ii,:), SimSettings );
            h_angle	=   [ h_angle; h ];
        end
    elseif isempty(AvgResults)
        for ii = 1:size(Results,1)
            h	=   plot_dR2_AngleSeries( Results(ii,:), [], Params(ii,:), SimSettings );
            h_angle	=   [ h_angle; h ];
        end
    else
        for ii = 1:size(Results,1)
            h	=   plot_dR2_AngleSeries( Results(ii,:), AvgResults(ii,:), Params(ii,:), SimSettings );
            h_angle	=   [ h_angle; h ];
        end
    end
    
end

end

function H = plot_dR2_TimeSeries( Results, AvgResults, Params, SimSettings )

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

H = figure;
hold on, grid on

% Colours
Colours	=	distinguishable_colors( numel( Angles_Deg ) );

% Plot all results
if ~isempty(Results)
    for ii = 1:NumResults
        for jj = 1:NumAngles
            dR2	=   Results(ii).dR2_all{jj}(:);
            h(jj)	=   plot( Time(:), dR2, LS_Res, 'linewidth', LW_Res, ...
                'markersize', MS_Res, 'color', Colours(jj,:) );
        end
    end
end

% Overlay average results on plot
if ~isempty(AvgResults)
    for jj = 1:NumAngles
        dR2	=   AvgResults.dR2_all{jj}(:);
        h(jj)	=   plot( Time(:), dR2, LS_Avg, 'linewidth', LW_Avg, ...
            'markersize', MS_Avg, 'color', Colours(jj,:) );
    end
end

% Create Legend
labels	=   cell(1,NumAngles);
for ii = 1:length(labels)
    labels{ii}	=   sprintf('%5.2f deg',Angles_Deg(ii));
end
legend( h, labels{:}, 'location', 'northwest' );

% Add titles
xlabel( 'Time [ms]' ); ylabel( '\DeltaR2 [1/s]' );
titlestr	=	sprintf( ...
    '\\DeltaR2 vs. Time: CA = %0.4f mM, Num Major Vessels = %d, BVF = %0.4f%%, Minor Vessel Rel-BVF = %0.4f%%',     ...
    Params.CA_Concentration, SimSettings.NumMajorVessels, 100*Params.Total_BVF, 100*Params.MinorVessel_RelBVF );

title( titlestr );

% Draw plot
drawnow

end

function H = plot_dR2_AngleSeries( Results, AvgResults, Params, SimSettings )

Angles_Deg	=   SimSettings.Angles_Deg_Data(:); % [deg]

if ~isempty(Results),   Time	=	1000 * Results(1).Time; % [ms]
else                    Time	=	1000 * AvgResults(1).Time; % [ms]
end
TotNumTimes	=   numel(Time);
TimeIdx     =   round(linspace(1,TotNumTimes,min(TotNumTimes,10)));
Time        =   Time(TimeIdx);

NumAngles	=   numel(Angles_Deg);
NumTimes	=   numel(Time);
NumResults	=   numel(Results);

if ~isempty( Results )
    dR2_all	=   reshape(    cell2mat( [Results.dR2_all] ),      ...
                            [NumAngles,TotNumTimes,NumResults]	);
    dR2_all	=   dR2_all(:,TimeIdx,:);
end

if ~isempty( AvgResults )
    dR2_avg	=	cell2mat( AvgResults.dR2_all );
    dR2_avg	=   dR2_avg(:,TimeIdx,:);
end

% LS = linestyle, LW = linewidth, MS = markersize
if isempty(AvgResults)
    LS_Res	=   '-';    LW_Res	=	4;	MS_Res	=   10;
    LS_Avg	=   [];     LW_Avg	=   [];	MS_Avg	=   [];
else
    LS_Res	=   '--';	LW_Res	=	2;  MS_Res	=   5;	
    LS_Avg	=   '-';    LW_Avg	=   4;	MS_Avg	=   10;
end

H = figure;
hold on, grid on

% Colours
Colours	=	distinguishable_colors( NumTimes );

% Plot all results
if ~isempty(Results)
    for ii = 1:NumResults
        for jj = 1:NumTimes
            dR2	=   dR2_all(:,jj,ii);
            h(jj)	=   plot( Angles_Deg(:), dR2, LS_Res, 'linewidth', LW_Res, ...
                'markersize', MS_Res, 'color', Colours(jj,:) );
        end
    end
end

% Overlay average results on plot
if ~isempty(AvgResults)
    for jj = 1:NumTimes
        dR2	=   dR2_avg(:,jj);
        h(jj)	=   plot( Angles_Deg(:), dR2, LS_Avg, 'linewidth', LW_Avg, ...
            'markersize', MS_Avg, 'color', Colours(jj,:) );
    end
end

% Create Legend
labels	=   cell(1,NumTimes);
for ii = 1:length(labels)
    labels{ii}	=   sprintf('%5.2f ms',Time(ii));
end
legend( h, labels{:}, 'location', 'northwest' );

% Add titles
xlabel( 'Angle [deg]' ); ylabel( '\DeltaR2 [1/s]' );
titlestr	=	sprintf( ...
    '\\DeltaR2 vs. Angle: CA = %0.4f mM, Num Major Vessels = %d, BVF = %0.4f%%, Minor Vessel Rel-BVF = %0.4f%%',     ...
    Params.CA_Concentration, SimSettings.NumMajorVessels, 100*Params.Total_BVF, 100*Params.MinorVessel_RelBVF );

title( titlestr );

% Draw plot
drawnow

end


