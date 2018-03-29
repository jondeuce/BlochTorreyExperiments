%% PSLoadResults
files = dir('*.mat');
endsWith = @(str,ending) strcmpi( str(max(end-numel(ending)+1,1):end), ending );
res_file = [];
if endsWith(files(end).name, 'ParticleSwarmFitResults.mat')
    res_file = files(end);
    files = files(1:end-1);
end
if endsWith(files(end).name, 'InterruptedWorkspace.mat')
    files = files(1:end-1);
end

r = load(files(1).name);
resfield = 'Results';
if isfield(r,'CompleteResults')
    resfield = 'CompleteResults';
end
Results = repmat(r.(resfield), numel(files), 1);
for ii = 2:length(files)
    r = load(files(ii).name);
    resfield = 'Results';
    if isfield(r,'CompleteResults')
        resfield = 'CompleteResults';
    end
    Results(ii) = r.(resfield);
end
if ~isempty(res_file)
    LsqResults = load(res_file.name);
else
    LsqResults = [];
end

%% Sort results by error
dR2 = cat(1, Results.dR2);
dR2_Data = Results(1).dR2_Data;
Weights = Results(1).args.Weights;

% calc error and sort
lsqerr = sqrt( sum( bsxfun(@times, Weights, bsxfun(@minus, dR2, dR2_Data).^2), 2 ) );
[sorted_err, sorted_inds] = sort(lsqerr);