%% load in data
PSLoadResults;

%% plot first 10 results
NumToPlot = min(10, numel(Results));
for ii = vec(sorted_inds(1:NumToPlot)).'
    perforientation_plot( Results(ii).dR2, Results(ii).dR2_all, Results(ii).Geometries, Results(ii).args )
end
