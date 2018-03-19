%% load in data
files = dir('*.mat');
endsWith = @(str,ending) strcmpi( str(max(end-numel(ending)+1,1):end), ending );
if endsWith(files(end).name, 'ParticleSwarmFitResults.mat')
    res_file = files(end);
    files = files(1:end-1);
else
    res_file = [];
end

r = load(files(1).name);
Results = repmat(r.Results, numel(files), 1);
for ii = 2:length(files)
    r = load(files(ii).name);
    Results(ii) = r.Results;
end
if ~isempty(res_file)
    LsqResults = load(res_file.name);
else
    LsqResults = [];
end

%% plot results by error value
dR2 = cat(1, Results.dR2);
dR2_Data = Results(1).dR2_Data;
Weights = Results(1).args.Weights;

% calc error and sort
lsqerr = sqrt( sum( bsxfun(@times, Weights, bsxfun(@minus, dR2, dR2_Data).^2), 2 ) );
[sorted_err, sorted_inds] = sort(lsqerr);

%% plot first 10 results
NumToPlot = min(10, numel(Results));
for ii = vec(sorted_inds(1:NumToPlot)).'
    perforientation_plot( Results(ii).dR2, Results(ii).dR2_all, Results(ii).Geometries, Results(ii).args )
end
