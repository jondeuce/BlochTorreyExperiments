%% load in data
PSLoadResults;

%% calculate the parameter bounds of loaded results
params = {'CA', 'iBVF', 'aBVF'};
values = cell(size(params));
bounds = cell(size(params));

for ii = 1:numel(params)
    values{ii} = cat(1,Results.(params{ii}));
    bounds{ii} = [min(values{ii}), max(values{ii})];
    fprintf('%s: %s\n', params{ii}, mat2str(bounds{ii},4));
end