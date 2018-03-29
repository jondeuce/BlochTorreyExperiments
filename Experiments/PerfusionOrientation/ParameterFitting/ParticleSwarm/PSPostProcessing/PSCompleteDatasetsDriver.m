%% Load partial results: to be completed with order = 2, Nsteps = 10

files = dir('*.mat'); %all m-files in current folder
fnames = {files.name}; % [1xnfiles] cell array

%% Solve for missing values

for fnameCell = fnames
    fnameWithExt = fnameCell{1};
    fname = fnameWithExt(1:end-4);
    Results = load([fname, '.mat']);
    Results = Results.Results;
    
    tstart = tic;
    fprintf('\n%s\n\n', fname);
    try
        PSCompleteDatasets;
    catch me
        warning(me.message);
        save(['FAILED--', fname, '.mat']);
    end
    display_toc_time(toc(tstart), fname);
end
